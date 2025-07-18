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
import numpy as np
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
    E, k, H, I, stride, gmin, gmax, 2,
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
        LAYER_IDX, qlen, k,
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
def load_uint8(stub, shape):
    with open(stub+".uint8",'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.uint8).view(shape)

# 通用加载函数
def load_dump_tensor(eid: int, name: str, shape: tuple, Ename: str = "E_Before"):
    """
    根据 eid / name / shape 读取 dump 文件，并返回 torch.Tensor
    """
    stub = DUMP_DIR / f"layer{LAYER_IDX}_{Ename}{eid}_{name}"
    if stub.with_suffix(".bf16").exists():
        return load_bf16(str(stub), shape)
    elif stub.with_suffix(".f16").exists():
        return load_f16(str(stub), shape)
    elif stub.with_suffix(".f32").exists():
        return load_f32(str(stub), shape)
    elif stub.with_suffix(".uint8").exists():
        return load_uint8(str(stub), shape)
    else:
        raise FileNotFoundError(f"{stub}（bf16/f16/f32 均不存在）")

def _format_ranges(indices: torch.Tensor) -> str:
    """
    将 N×D 的整型坐标压缩成区间表示，形如 "(0-3, 10-15)"。
    若某维度不连续，则逐个列出（或可自行再做高级压缩）。
    """
    dims = indices.size(1)
    parts = []
    for d in range(dims):
        vals = indices[:, d].unique().sort()[0]   # 排序后的唯一值
        start, end = int(vals[0]), int(vals[-1])
        # 若 (end - start + 1) == 元素数量，则说明连续
        if end - start + 1 == vals.numel():
            # 连续 → 区间
            parts.append(f"{start}-{end}" if start != end else f"{start}")
        else:
            # 不连续 → 逐个列出，可按需再切片或压缩
            parts.append(", ".join(map(str, vals.tolist())))
    return "(" + ", ".join(parts) + ")"

def check_torch_cpp(eid,name,shape):
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
    nan_mask = torch.isnan(cpp)
    if nan_mask.any():
        nan_count = int(nan_mask.sum())
        total_cnt = cpp.numel()
        print(f"[E{eid:02d} {name:18s}] cpp_nan_count {nan_count}/{total_cnt}")
        
        # 提取所有 NaN 坐标
        nan_idx = nan_mask.nonzero(as_tuple=False)
        # 压缩成区块表示
        block_str = _format_ranges(nan_idx)
        print(f"  ▸ NaN block @ {block_str}")
        print(f"CPP: {cpp}")
        print(f"PT:  {pt}")
    abs_diff = (cpp - pt).abs()
    pct_diff = abs_diff / (pt.abs() + EPS) * 100.0  # 百分比

    print(f"[E{eid:02d} {name:18s}] cpp_max {cpp.max():.3e}  cpp_mean {cpp.mean():.3e}")
    print(f"[E{eid:02d} {name:18s}] pt_max  {pt.max():.3e}  pt_mean  {pt.mean():.3e}")
    print(f"[E{eid:02d} {name:18s}] pct_max {pct_diff.max():8.4f}%  pct_mean {pct_diff.mean():8.4f}%")

import torch

def compare_tensors(cpp_bef: torch.Tensor,
                    cpp_aft: torch.Tensor,
                    name_bef,
                    name_aft,
                    rtol: float = 1e-4,
                    atol: float = 1e-6,
                    topk: int = 3) -> bool:
    """
    检查两个张量是否一致，并在不一致时打印调试信息。
    参数说明
    cpp_aft / cpp_bef : 要比较的两个张量
    rtol / atol       : torch.allclose 的相对 / 绝对误差容忍度
    max_mismatch_show : 最多打印多少个不一致的坐标和值
    返回
    bool : True → 一致；False → 不一致
    """
    
    print("=" * 60)
    if cpp_aft.shape != cpp_bef.shape:
        print(f" shape 不一致: aft={cpp_aft.shape}, bef={cpp_bef.shape}")
        return False
    else:
        print(f" shape : {cpp_aft.shape}")
    if cpp_aft.dtype != cpp_bef.dtype:
        print(f" dtype 不一致: aft={cpp_aft.dtype}, bef={cpp_bef.dtype}")
        return False
    else:
        print(f" dtype : {cpp_aft.dtype}")

    for name, t in [(f"{name_aft}", cpp_aft)]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f"  {name}含 NaN={nan_cnt}、Inf={inf_cnt}")
    for name, t in [(f"{name_bef}", cpp_bef)]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f"  {name}含 NaN={nan_cnt}、Inf={inf_cnt}")

    # 构造“有限值掩码”：仅比较两边都是有限值的位置
    finite_mask = torch.isfinite(cpp_aft) & torch.isfinite(cpp_bef)
    total = cpp_aft.numel()
    ignored = total - finite_mask.sum().item()

    # 对有限值位置执行逐元素比较
    diff_mask = ~torch.isclose(
        cpp_aft[finite_mask], cpp_bef[finite_mask], rtol=rtol, atol=atol
    )
    n_diff = diff_mask.sum().item()
    ok = n_diff == 0

    if ok:
        print(
            f"[OK] All finite elements match "
            f"(ignored {ignored} NaN/Inf out of {total})."
        )
        return True

    # ------------- 以下为调试信息 -------------
    aft_finite = cpp_aft[finite_mask]
    bef_finite = cpp_bef[finite_mask]
    abs_err = (aft_finite - bef_finite).abs()
    rel_err = abs_err / (bef_finite.abs() + 1e-12)

    print(f"[Mismatch] {n_diff}/{finite_mask.sum().item()} "
          f"finite elements differ "
          f"(ignored {ignored} NaN/Inf, total {total}).")
    print(
        f"Max Abs Err: {abs_err.max():.6g},  "
        f"Mean Abs Err: {abs_err.mean():.6g}"
    )
    print(
        f"Max Rel Err: {rel_err.max():.6g},  "
        f"Mean Rel Err: {rel_err.mean():.6g}"
    )

    # 找到不一致位置索引并打印前 topk 项
    if topk > 0:
        mis_idx_flat = diff_mask.nonzero(as_tuple=False).flatten()
        # 将展平索引还原成多维坐标
        unraveled = [
            torch.unravel_index(i, cpp_aft.shape) for i in mis_idx_flat[:topk]
        ]
        print(f"First {min(topk, n_diff)} differing positions:")
        for k, idx in enumerate(unraveled, 1):
            idx_tuple = tuple(int(j) for j in idx)
            a_val = cpp_aft[idx_tuple].item()
            b_val = cpp_bef[idx_tuple].item()
            print(f"  #{k}: idx={idx_tuple}  aft={a_val:.6g}  bef={b_val:.6g}")
    print("=" * 60)
    return False

def check_nan(eid,name,shape, Ename="E"):
    stub1 = DUMP_DIR / f"layer{LAYER_IDX}_{Ename}{eid}_{name}"
    if stub1.with_suffix(".bf16").exists():
        cpp_bef = load_bf16(str(stub1), shape)
    elif stub1.with_suffix(".f16").exists():
        cpp_bef = load_f16(str(stub1), shape)
    elif stub1.with_suffix(".f32").exists():
        cpp_bef = load_f32(str(stub1), shape)
    else:
        print("dump 缺失/未知类型"); return

    print(f"{Ename}{eid}_{name}:{cpp_bef}")
    print(f" shape : {cpp_bef.shape}")
    print(f" dtype : {cpp_bef.dtype}")

    finite_mask = torch.isfinite(cpp_bef)
    if finite_mask.any():
        t_finite = cpp_bef[finite_mask]
        t_max = t_finite.max().item()
        t_min = t_finite.min().item()
        print(f" max   : {t_max:.6e}")
        print(f" min   : {t_min:.6e}")
    else:
        print(" max/min: 所有元素均为 NaN / Inf")

    for nan_name, t in [(f"{Ename}{eid}_{name}", cpp_bef)]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f"{Ename}{eid}_{name} 含 NaN={nan_cnt}、Inf={inf_cnt}")
        else:
            print("NO NaN or Inf exist")

def check_gemm(eid: int,
               r: int,          # 行数 = batch rows
               H: int,          # hidden_size
               I: int,          # intermediate_size
               topk: int = 5):
    """
    检查 up_out_grad  @  up_proj_t.T  是否 ≈  up_in_grad

    up_out_grad : (r, I)
    up_proj_t   : (H, I)
    up_in_grad  : (r, H)
    """
    # 1. 读取三个张量
    # proj_stride = 256
    up_out_grad = load_dump_tensor(eid, "up_out_grad", (r, I), "E_Before")
    up_proj_t   = load_dump_tensor(eid, "up_proj_t",   (H, I), "E_Before")
    up_in_grad  = load_dump_tensor(eid, "up_in_grad",  (r, H), "E_aft_gemm")
    
    print(f"up_in_grad:{up_in_grad}")
    for t in [up_in_grad]:
        nan_cnt = torch.isnan(t).sum().item()
        inf_cnt = torch.isinf(t).sum().item()
        if nan_cnt or inf_cnt:
            print(f"E_aft_gemm{eid}_up_in_grad 含 NaN={nan_cnt}、Inf={inf_cnt}")
        else:
            print("NO nan or inf exist in up_in_grad.")

    # 2. 统一转 float32 避免精度差异
    up_out_grad_f = up_out_grad.float()
    up_proj_t_f   = up_proj_t.float()

    # 3. GEMM 计算 (r,I) × (I,H) → (r,H)
    gemm = torch.matmul(up_out_grad_f, up_proj_t_f.t())

    print(f"\n===== 检查 E{eid} ▸ up_out_grad × up_proj_t.T =====")
    print(f" up_out_grad.shape = {tuple(up_out_grad.shape)}")
    print(f" up_proj_t.shape   = {tuple(up_proj_t.shape)}")
    print(f"  ⇒ gemm.shape     = {tuple(gemm.shape)}  (目标)")
    print(f" up_in_grad.shape  = {tuple(up_in_grad.shape)}")

    # 4. NaN / Inf 差异
    nan_gemm  = torch.isnan(gemm) | torch.isinf(gemm)
    nan_input = torch.isnan(up_in_grad) | torch.isinf(up_in_grad)

    only_gemm_nan  = nan_gemm & ~nan_input
    only_input_nan = ~nan_gemm & nan_input
    both_nan       = nan_gemm & nan_input

    print("\n── NaN/Inf 统计 ──")
    print(f"  gemm NaN/Inf : {nan_gemm.sum().item()}")
    print(f"  up_in_grad NaN/Inf : {nan_input.sum().item()}")
    print(f"  仅 gemm 为 NaN/Inf : {only_gemm_nan.sum().item()}")
    print(f"  仅 up_in_grad 为 NaN/Inf : {only_input_nan.sum().item()}")
    print(f"  两者同为 NaN/Inf : {both_nan.sum().item()}")

    if only_gemm_nan.any():
        idxs = only_gemm_nan.nonzero(as_tuple=False)[:topk]
        print(f"  ▸ 仅 gemm 为 NaN/Inf 的前 {len(idxs)} 个坐标: {[tuple(i.tolist()) for i in idxs]}")
    if only_input_nan.any():
        idxs = only_input_nan.nonzero(as_tuple=False)[:topk]
        print(f"  ▸ 仅 up_in_grad 为 NaN/Inf 的前 {len(idxs)} 个坐标: {[tuple(i.tolist()) for i in idxs]}")

    # 5. 非 NaN 区域误差
    valid = ~(nan_gemm | nan_input)
    if valid.any():
        diff = (gemm - up_in_grad).abs()
        max_diff  = diff[valid].max().item()
        mean_diff = diff[valid].mean().item()

        print("\n── 非 NaN/Inf 区域数值差异 ──")
        print(f"  最大绝对误差 : {max_diff:.6e}")
        print(f"  平均绝对误差 : {mean_diff:.6e}")

        # 打印误差最大 top-k
        if topk > 0:
            flat_idx = torch.topk(diff.view(-1), k=min(topk, diff.numel()), largest=True).indices
            unraveled = [torch.unravel_index(i, diff.shape) for i in flat_idx]
            print(f"  ▸ 误差 Top-{len(unraveled)} 坐标与误差:")
            for j, coord in enumerate(unraveled):
                print(f"     #{j+1}: {coord} → {diff[coord].item():.6e}")
    else:
        print("\n☆ 所有元素都是 NaN/Inf，无法比较数值误差 ☆")
        
def check_consist(eid,name,shape):
    stub1 = DUMP_DIR / f"layer{LAYER_IDX}_E{eid}_{name}"
    stub2 = DUMP_DIR / f"layer{LAYER_IDX}_E_aft_gemm{eid}_{name}"
    if stub1.with_suffix(".bf16").exists():
        cpp_aft = load_bf16(str(stub1), shape)
    elif stub1.with_suffix(".f16").exists():
        cpp_aft = load_f16(str(stub1), shape)
    elif stub1.with_suffix(".f32").exists():
        cpp_aft = load_f32(str(stub1), shape)
    else:
        print("dump 缺失/未知类型"); return
    if stub2.with_suffix(".bf16").exists():
        cpp_bef = load_bf16(str(stub2), shape)
    elif stub2.with_suffix(".f16").exists():
        cpp_bef = load_f16(str(stub2), shape)
    elif stub2.with_suffix(".f32").exists():
        cpp_bef = load_f32(str(stub2), shape)
    else:
        print("dump 缺失/未知类型"); return

    print(f"cpp_bef:{cpp_bef}")
    print(f"cpp_aft:{cpp_aft}")
    
    if compare_tensors(cpp_bef, cpp_aft, "cpp_bef", "cpp_aft"):
        print("Consist!")
    else:
        print("NOT consist!")

def load_tensor_by_stub(stub, shape):
    """按 stub 自动探测 .bf16/.f16/.f32 并读取张量"""
    if stub.with_suffix(".bf16").exists():
        return load_bf16(str(stub), shape)
    elif stub.with_suffix(".f16").exists():
        return load_f16(str(stub), shape)
    elif stub.with_suffix(".f32").exists():
        return load_f32(str(stub), shape)
    else:
        raise FileNotFoundError(f"{stub}*.bf16/.f16/.f32 均不存在")

def check_transpose_diff(eid, base_name, shape_a, shape_b, Ename="E_End", name_transpose=""):
    """
    检查 base_name 与 base_name+'_t' 是否真正互为转置；
    若有 NaN/Inf，统计数量；再在有效元素上计算差值。
    """
    stub_a = DUMP_DIR / f"layer{LAYER_IDX}_{Ename}{eid}_{base_name}"
    if name_transpose == "":
        stub_t = DUMP_DIR / f"layer{LAYER_IDX}_{Ename}{eid}_{base_name}_t"
    else:
        stub_t = DUMP_DIR / f"layer{LAYER_IDX}_{Ename}{eid}_{name_transpose}"
        
    # 1. 加载两个矩阵
    A = load_tensor_by_stub(stub_a, shape_a)
    B = load_tensor_by_stub(stub_t, shape_b)

    # 2. 判断是否需要转置
    if A.shape == B.shape[::-1]:
        B_aligned = B.t()
    elif A.shape == B.shape:
        B_aligned = B
    else:
        print(f"[警告] 形状不匹配：A{A.shape}, B{B.shape}，无法比较")
        return
    print(f"原矩阵：{A}，\nTT矩阵：{B_aligned}")
    # 3. 统计 NaN / Inf
    nan_A, inf_A = torch.isnan(A).sum().item(), torch.isinf(A).sum().item()
    nan_B, inf_B = torch.isnan(B).sum().item(), torch.isinf(B).sum().item()
    print(f"{Ename}{eid}_{base_name}: NaN={nan_A}, Inf={inf_A}")
    print(f"{Ename}{eid}_{base_name}_t: NaN={nan_B}, Inf={inf_B}")

    # 4. 构造有效掩码
    valid_mask = ~(torch.isnan(A) | torch.isinf(A) |
                   torch.isnan(B_aligned) | torch.isinf(B_aligned))
    valid_cnt = valid_mask.sum().item()
    if valid_cnt == 0:
        print("全部为 NaN/Inf，无法比较")
        return

    # 5. 计算差值（仅有效元素）
    diff = (A - B_aligned)[valid_mask]
    max_diff = diff.abs().max().item()
    mean_diff = diff.mean().item()
    rmse_diff = diff.pow(2).mean().sqrt().item()

    print(f"有效元素: {valid_cnt}")
    print(f"最大差值: {max_diff:.6e}")
    print(f"平均差值: {mean_diff:.6e}")
    print(f"均方根差: {rmse_diff:.6e}")
    print("-" * 60)

def save_tensor_txt(t: torch.Tensor, file_path: str):
    """
    以科学计数法把张量保存为 .txt
    兼顾大矩阵：逐行写，避免一次性 format 巨量字符串占用内存
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 确保在 CPU、float64 精度足够；若已是 float16/bf16 转 float32/64
    arr = t.detach().cpu().to(torch.float32)

    with path.open("w") as f:
        # numpy 格式化最快
        for row in arr:
            np.savetxt(f,                   # 追加写
                       row.unsqueeze(0).numpy(),
                       fmt="%.6e",          # 科学计数，6 位小数，可按需修改
                       delimiter=" ")
    print(f"⇢ 写入 {path}  (shape={tuple(t.shape)})")


def dump_mat(eid: int, name, shape, ename="E_End", root: str = "./debug_txt/"):
    """
    加载 dump 张量并保存为 txt
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    tensor_proj  = load_dump_tensor(eid, name, shape, ename)

    save_tensor_txt(tensor_proj,  root / f"E{eid}_{name}_{ename}.txt")

# ---------- 10. 对拍结果 ----------
for eid in range(E):
    r = tok_per_exp[eid].item()
    print(f"r:{r}")
    if r==0: continue
    # check_torch_cpp(eid,"down_out_grad",(r,H))
    # check_torch_cpp(eid,"down_in_grad" ,(r,I))
    # check_torch_cpp(eid,"gate_out_grad",(r,I))
    # check_torch_cpp(eid,"gate_out_grad_fp32",(r,I))
    # check_torch_cpp(eid,"gate_in_grad",(r,H))
    # check_torch_cpp(eid,"up_out_grad",(r,I))
    # check_torch_cpp(eid,"up_out_grad_fp32"  ,(r,I))
    # check_torch_cpp(eid,"up_in_grad"  ,(r,H))
    
    # check_nan(eid, "up_in_grad", (r, H), "E_Before")
    # check_nan(eid, "up_out_grad", (r, I), "E_Before")
    # check_nan(eid, "up_proj_t", (H, I), "E_Before")
    # check_nan(eid, "up_proj_t", (I, H), "E_End")
    # check_nan(eid, "up_proj", (H, I), "E_End")
    # check_nan(eid, "gate_proj_t", (I, H), "E_End")
    # check_nan(eid, "gate_proj", (H, I), "E_End")
    # check_nan(eid, "down_proj_t", (H, I), "E_End")
    # check_nan(eid, "down_proj", (I, H), "E_End")
    # check_nan(eid, "up_in_grad", (r, H), "E_aft_gemm")
    
    # check_nan(eid, "up_proj_src_", (I, H), "E_End")
    # check_nan(eid, "up_proj_f32_", (I, H), "E_End")
    # check_nan(eid, "up_proj_out_trans_", (H, I), "E_End")
    # check_nan(eid, "up_proj_dst_", (H, I), "E_End")
    # check_nan(eid, "down_proj_src_", (H, I), "E_End")
    # check_nan(eid, "down_proj_f32_", (H, I), "E_End")
    # check_nan(eid, "down_proj_out_trans_", (I, H), "E_End")
    # check_nan(eid, "down_proj_dst_", (I, H), "E_End")
    # check_nan(eid, "gate_proj_src_", (I, H), "E_End")
    # check_nan(eid, "gate_proj_f32_", (I, H), "E_End")
    # check_nan(eid, "gate_proj_out_trans_", (H, I), "E_End")
    # check_nan(eid, "gate_proj_dst_", (H, I), "E_End")
    # check_nan(eid, "up_proj_t", (H, I), "E_End")
    # check_nan(eid, "up_proj", (I, H), "E_End")
    # check_nan(eid, "up_proj_t", (H, I), "E_aft_sgemm")
    # check_nan(eid, "up_proj", (I, H), "E_aft_sgemm")
    # check_nan(eid, "up_proj_t", (H, I), "E_bef_sgemm")
    # check_nan(eid, "up_proj", (I, H), "E_bef_sgemm")
    # check_nan(eid, "up_proj_t", (H, I), "E_bef_many")
    # check_nan(eid, "up_proj", (I, H), "E_111")
    # check_nan(eid, "up_proj_t", (H, I), "E_111")
    # check_nan(eid, "up_proj", (I, H), "E_222")
    # check_nan(eid, "up_proj_t", (H, I), "E_222")
    # check_nan(eid, "up_proj", (I, H), "E_333")
    # check_nan(eid, "up_proj", (I, H), "E_444")
    check_nan(eid, "up_proj_t", (H, I), "E_333")
    check_nan(eid, "up_proj_t", (H, I), "E_444")
    check_nan(eid, "gate_proj_t", (H, I), "E_333")
    check_nan(eid, "gate_proj_t", (H, I), "E_444")
    check_nan(eid, "down_proj_t", (I, H), "E_333")
    check_nan(eid, "down_proj_t", (I, H), "E_444")
    # check_nan(eid, "up_proj", (I, H), "E_bef_many")
    # check_nan(eid, "up_proj_t", (H, I), "E_aft_many")
    # check_nan(eid, "up_proj", (I, H), "E_aft_many")
    
    # check_nan(eid, "up_proj", (I, H), "E_Forward")
    # check_nan(eid, "gate_proj", (I, H), "E_Forward")
    # check_nan(eid, "down_proj", (H, I), "E_Forward")
    
    # check_gemm(eid=eid, r=r, H=H, I=I)
    
    dump_mat(eid, "up_proj_t", (H, I), "E_333")
    dump_mat(eid, "up_proj_t", (H, I), "E_444")
    
    # check_transpose_diff(eid, "up_proj",   (I, H), (H, I))
    # check_transpose_diff(eid, "gate_proj", (I, H), (H, I))
    # check_transpose_diff(eid, "down_proj", (H, I), (I, H))
    
    
    # check_transpose_diff(eid, "up_proj_bf16_",   (I, H), (H, I), name_transpose="up_proj_dst_")
    # check_transpose_diff(eid, "gate_proj_bf16_", (I, H), (H, I), name_transpose="gate_proj_dst_")
    # check_transpose_diff(eid, "down_proj_bf16_", (H, I), (I, H), name_transpose="down_proj_dst_")
    
    
    # not use this, WRONG!!
    # check_transpose_diff(eid, "up_proj",   (H, I), (I, H))
    # check_transpose_diff(eid, "gate_proj", (H, I), (I, H))
    # check_transpose_diff(eid, "down_proj", (I, H), (H, I))