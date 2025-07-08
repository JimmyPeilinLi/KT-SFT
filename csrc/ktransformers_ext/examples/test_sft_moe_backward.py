#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT_MOE backward_many() dump 对拍脚本
------------------------------------
1. C++ forward & backward （生成 dump）
2. PyTorch 参考实现
3. 读取 dump，六大梯度矩阵逐 expert diff
"""

import os, sys, time, torch, math
from pathlib import Path

# ---------- 1. 绑定 C++ 扩展 ----------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "../build"))      # 根据实际 build 目录调整
import cpuinfer_ext

import shutil
folder_path = "/home/lpl/KT-SFT/debug"
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)

# ---------- 2. 超参数 ----------
E  = expert_num        = 10
H  = hidden_size       = 5120
I  = intermediate_size = 1536
k  = n_routed_experts  = 2
qlen                  = 30
stride, gmin, gmax    = 32, 10, 1024
dtype_fwd  = torch.float16     # forward 输入 / 权重
dtype_grad = torch.bfloat16    # grad_type
GGML_F16   = 1
LAYER_IDX  = 0                 # 本次 dump 的 layer 号
DUMP_DIR   = Path(os.getenv("SFT_DEBUG_PATH", "debug"))
EPS = 1e-12


# ---------- 3. 构造pytorch测试 ----------
def act_fn(x):
    return x / (1.0 + torch.exp(-x))
def silu(x):        return x / (1. + torch.exp(-x))
def silu_grad(x):
    s = 1. / (1. + torch.exp(-x))
    return s * (1. + x * (1. - s))

# def mlp_torch(input, gate_proj, up_proj, down_proj, requires_grad=False):
#     gate_buf = torch.mm(input, gate_proj.t())
#     up_buf = torch.mm(input, up_proj.t())
    
#     # 使用可微的SiLU或者原来的函数，取决于是否需要梯度
#     if requires_grad:
#         intermediate = silu(gate_buf) * up_buf
#     else:
#         intermediate = act_fn(gate_buf) * up_buf
    
#     ret = torch.mm(intermediate, down_proj.t())
#     return ret

# def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=False):
#     cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
#     cnts.scatter_(1, expert_ids, 1)
#     tokens_per_expert = cnts.sum(dim=0)
#     idxs = expert_ids.view(-1).argsort()
#     sorted_tokens = input[idxs // expert_ids.shape[1]]

#     outputs = []
#     start_idx = 0
#     for i, num_tokens in enumerate(tokens_per_expert):
#         end_idx = start_idx + num_tokens
#         if num_tokens == 0:
#             continue
#         tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
#         expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i], requires_grad)
#         outputs.append(expert_out)
#         start_idx = end_idx

#     outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

#     new_x = torch.empty_like(outs)
#     new_x[idxs] = outs
#     t_output = (
#         new_x.view(*expert_ids.shape, -1)
#         .type(weights.dtype)
#         .mul_(weights.unsqueeze(dim=-1))
#         .sum(dim=1)
#         .type(new_x.dtype)
#     )
#     return t_output

# ---------- 4. 构造 C++ 模型 ----------
CPUInfer = cpuinfer_ext.CPUInfer(48)

gate_proj = torch.randn(E, I, H, dtype=dtype_fwd).contiguous()
up_proj = torch.randn(E, I, H, dtype=dtype_fwd).contiguous()
down_proj = torch.randn(E, H, I, dtype=dtype_fwd).contiguous()

cfg = cpuinfer_ext.sft_moe.SFT_MOEConfig(
    E, k, H, I, stride, gmin, gmax,
    gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(),
    GGML_F16, GGML_F16, GGML_F16, GGML_F16
)
moe_cpp = cpuinfer_ext.sft_moe.SFT_MOE(cfg)

# ---------- 5. 随机输入（两端共用） ----------
torch.manual_seed(41)
input_pt = (torch.randn(qlen, H, dtype=dtype_fwd) / 100).detach().requires_grad_(True).contiguous()
input_cpp  = input_pt.clone().detach().requires_grad_(True).contiguous()

expert_ids = torch.stack([torch.randperm(E)[:k] for _ in range(qlen)]).contiguous()
weights    = torch.rand(qlen, k, dtype=torch.float32).contiguous()
grad_out   = torch.randn(qlen, H, dtype=dtype_grad).contiguous()

# ---------- 6. C++ forward ----------
out_cpp = torch.empty_like(input_cpp)
CPUInfer.submit(
    moe_cpp.forward(
        qlen, k,
        expert_ids.data_ptr(), weights.data_ptr(),
        input_cpp.data_ptr(), out_cpp.data_ptr()
    )
)
CPUInfer.sync()

# ---------- 7. C++ backward（生成 dump） ----------
grad_in_cpp = torch.empty_like(input_cpp, dtype=dtype_grad)
CPUInfer.submit(
    moe_cpp.backward(
        LAYER_IDX,
        qlen, k,
        expert_ids.data_ptr(), weights.data_ptr(),
        input_cpp.data_ptr(),
        grad_out.data_ptr(),
        grad_in_cpp.data_ptr()
    )
)
CPUInfer.sync()

# ---------- 8. PyTorch 参考计算 ----------
gate_proj_f = gate_proj.float(); up_proj_f = up_proj.float(); down_proj_f = down_proj.float()
inp_f       = input_pt.float()
gout_f      = grad_out.float()

tok_per_exp   = torch.zeros(E, dtype=torch.int32)
exp_token_idx = [[] for _ in range(E)]
exp_pos_idx   = [[] for _ in range(E)]
for t in range(qlen):
    for j in range(k):
        eid = expert_ids[t,j].item()
        exp_token_idx[eid].append(t)
        exp_pos_idx  [eid].append(j)
        tok_per_exp[eid] += 1

ref = {}
for eid in range(E):
    rows = tok_per_exp[eid].item()
    if rows == 0: continue
    rows_idx = torch.tensor(exp_token_idx[eid])
    pos_idx  = torch.tensor(exp_pos_idx  [eid])

    x_e   = inp_f[rows_idx]                 # (rows,H)
    gout  = gout_f[rows_idx]                # (rows,H)
    w_e   = weights[rows_idx, pos_idx].unsqueeze(1)   # (rows,1)

    down_out_grad = gout.clone()
    down_in_grad = torch.matmul(gout, down_proj_f[eid]) * w_e  # (rows,I)

    gate_u = torch.matmul(x_e, gate_proj_f[eid].t())  # (rows,I)
    up_v   = torch.matmul(x_e, up_proj_f[eid].t())    # (rows,I)

    gate_out_grad = down_in_grad * up_v * silu_grad(gate_u)
    up_out_grad   = down_in_grad * silu(gate_u)

    gate_in_grad = torch.matmul(gate_out_grad, gate_proj_f[eid])  # (rows,H)
    up_in_grad   = torch.matmul(up_out_grad,   up_proj_f[eid])    # (rows,H)

    ref[(eid,"down_out_grad")] = down_out_grad
    ref[(eid,"down_in_grad")]  = down_in_grad
    ref[(eid,"gate_out_grad_fp32")] = gate_out_grad
    ref[(eid,"gate_out_grad")] = gate_out_grad.to(torch.bfloat16)
    ref[(eid,"gate_in_grad")] = gate_in_grad
    ref[(eid,"up_out_grad_fp32")]   = up_out_grad
    ref[(eid,"up_out_grad")]   = up_out_grad.to(torch.bfloat16)
    ref[(eid,"up_in_grad")]   = up_in_grad

# ---------- 9. dump 读取工具 ----------
def load_bf16(stub, shape):
    with open(stub + ".bf16", "rb") as f:
        return torch.frombuffer(f.read(), dtype=torch.bfloat16).view(shape).float()
def load_f16(stub, shape):
    with open(stub+".f16",'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.float16).view(shape).float()
def load_f32(stub, shape):
    with open(stub+".f32",'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.float32).view(shape)

def check(eid,name,shape):
    stub = DUMP_DIR / f"layer{LAYER_IDX}_E{eid}_{name}"
    if stub.with_suffix(".bf16").exists():
        cpp = load_bf16(str(stub), shape)
    elif stub.with_suffix(".f16").exists():
        cpp = load_f16(str(stub), shape)
    elif stub.with_suffix(".f32").exists():
        cpp = load_f32(str(stub), shape)
    else:
        print("dump 缺失/未知类型"); return
    pt  = ref[(eid,name)]
    abs_diff = (cpp - pt).abs()
    pct_diff = abs_diff / (pt.abs() + EPS) * 100.0     # 百分比
    print(f"[E{eid:02d} {name:18s}] cpp_max {cpp.max():.3e}  cpp_mean {cpp.mean():.3e}")
    print(f"[E{eid:02d} {name:18s}] pt_max {pt.max():.3e}  pt_mean {pt.mean():.3e}")
    print(f"[E{eid:02d} {name:18s}] " f"pct_max {pct_diff.max():8.4f}%  " f"pct_mean {pct_diff.mean():8.4f}%")

# ---------- 10. 对拍结果 ----------
for eid in range(E):
    r = tok_per_exp[eid].item()
    if r==0: continue
    check(eid,"down_out_grad",(r,H))
    check(eid,"down_in_grad" ,(r,I))
    check(eid,"gate_out_grad",(r,I))
    check(eid,"gate_out_grad_fp32",(r,I))
    check(eid,"gate_in_grad",(r,H))
    check(eid,"up_out_grad",(r,I))
    check(eid,"up_out_grad_fp32"  ,(r,I))
    check(eid,"up_in_grad"  ,(r,H))
