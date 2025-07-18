#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-06 10:38:05
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

expert_num = 10
hidden_size = 5120
intermediate_size = 1536
max_len = 1024

n_routed_experts = 2
qlen = 30
layer_num = 10
num_threads = 112
validation_iter = 1

dtype = torch.bfloat16
gradtype = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = False

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

def silu_fwd(x: torch.Tensor) -> torch.Tensor:
    return x / (1. + torch.exp(-x))

def silu_grad(x: torch.Tensor) -> torch.Tensor:
    """SiLU激活函数的梯度"""
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1. + x * (1. - sigmoid_x))

class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return silu_fwd(inp)

    @staticmethod
    def backward(ctx, grad_out):
        (inp,) = ctx.saved_tensors
        sig = torch.sigmoid(inp)
        return grad_out * (sig + inp * sig * (1. - sig))

silu = SiLU.apply   # 可求导版本

# -------------------- Torch MLP / MoE 参考实现 --------------------
def mlp_torch(x, gate, up, down, req_grad=False):
    g = torch.mm(x, gate.t())
    u = torch.mm(x, up.t())
    if req_grad:
        inter = silu(g) * u
    else:
        inter = silu_fwd(g) * u
    return torch.mm(inter, down.t())

def moe_torch(x, eid, w, gate, up, down, req_grad=False):
    """eid: [T,k]  int64,  w: [T,k] float"""
    T, k = eid.shape
    tok_cnt = torch.zeros(expert_num, dtype=torch.int64)
    for e in eid.view(-1):
        tok_cnt[e] += 1
    # 打包 token
    order = eid.view(-1).argsort()
    packed = x[order // k]

    outputs, start = [], 0
    for e in range(expert_num):
        num = tok_cnt[e].item()
        if not num:
            continue
        end = start + num
        o = mlp_torch(packed[start:end], gate[e], up[e], down[e], req_grad)
        outputs.append(o)
        start = end
    if outputs:
        out_all = torch.cat(outputs, 0)
    else:
        out_all = packed.new_empty(0, hidden_size)

    # 还原顺序并做加权
    out_restore = torch.empty_like(out_all)
    out_restore[order] = out_all
    out_restore = out_restore.view(T, k, hidden_size)
    out = (out_restore * w.unsqueeze(-1)).sum(1)
    return out

def moe_backward_python(x, eid, w, gate, up, down, grad_output, gate_u_cache, up_v_cache):
    """
    Python模拟C++的MoE backward计算
    参数:
        x: 输入 [T, hidden_size]
        eid: expert_ids [T, k]
        w: weights [T, k]
        gate, up, down: 权重矩阵
        grad_output: 输出梯度 [T, hidden_size]
        gate_u_cache, up_v_cache: forward时缓存的中间结果
    返回:
        grad_input: 输入梯度 [T, hidden_size]
    """
    T, k = eid.shape
    grad_input = torch.zeros_like(x, dtype=torch.float32)
    
    # 对每个token进行处理
    for token_idx in range(T):
        token_grad_input = torch.zeros(hidden_size, dtype=torch.float32)
        
        for expert_pos in range(k):
            expert_id = int(eid[token_idx, expert_pos].item())
            weight = w[token_idx, expert_pos].item()
            
            # 从cache中获取forward时的中间结果
            gate_u = gate_u_cache[token_idx][expert_pos]  # [intermediate_size]
            up_v = up_v_cache[token_idx][expert_pos]      # [intermediate_size]
            
            # 1. 计算down_input_grad (对应C++中的down_input_grad)
            # down_input_grad = down_proj_t @ output_grad * weight
            down_input_grad = torch.mm(down[expert_id].to(torch.float32).t(), grad_output[token_idx:token_idx+1].t()).squeeze() * weight
            
            # 2. 计算gate_output_grad和up_output_grad
            # gate_output_grad = down_input_grad * up_v * act_fn_grad(gate_u)
            gate_output_grad = down_input_grad * up_v * silu_grad(gate_u)
            
            # up_output_grad = down_input_grad * act_fn(gate_u)
            up_output_grad = down_input_grad * silu_fwd(gate_u)
            
            # 3. 计算gate_input_grad和up_input_grad
            # gate_input_grad = gate_proj_t @ gate_output_grad
            gate_input_grad = torch.mm(gate[expert_id].to(torch.float32).t(), gate_output_grad.unsqueeze(1)).squeeze()
            
            # up_input_grad = up_proj_t @ up_output_grad
            up_input_grad = torch.mm(up[expert_id].to(torch.float32).t(), up_output_grad.unsqueeze(1)).squeeze()
            
            # 4. 累加到token的输入梯度
            token_grad_input += gate_input_grad + up_input_grad
        
        grad_input[token_idx] = token_grad_input
    
    return grad_input

# --------------------------- 主测试 ---------------------------
def test_amx_moe_two_round():
    # ------------ 构造权重 (BF16) ------------
    gate_proj = torch.randn(expert_num, intermediate_size, hidden_size,
                            dtype=torch.bfloat16, requires_grad=True).contiguous()
    up_proj   = torch.randn_like(gate_proj)
    down_proj = torch.randn(expert_num, hidden_size, intermediate_size,
                            dtype=torch.bfloat16, requires_grad=True).contiguous()

    # ------------ SFT-AMX 对象 ------------
    cfg = cpuinfer_ext.sft_moe.SFT_AMX_MOEConfig(
        expert_num, n_routed_experts,
        hidden_size, intermediate_size,
        max_len,          # max_len
        gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr()
    )
    moe_cpp = cpuinfer_ext.sft_moe.SFT_AMXBF16_MOE(cfg)
    
    cpu_infer = cpuinfer_ext.CPUInfer(num_threads)

    # FLOPs 估计（只看 GEMM 部分；粗略）
    flop_fwd = 6  * qlen * n_routed_experts * hidden_size * intermediate_size
    flop_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    
    cpu_infer.submit(moe_cpp.load_weights())
    cpu_infer.sync() # ATTENTION: DO NOT FORGET sync after load weights
    
    # moe_cpp.warm_up(backend)

    for rnd in range(validation_iter):
        print(f"\n=========== Round {rnd+1}/{validation_iter} ===========")
        # ---- 随机 inputs / routing ----
        expert_ids = torch.stack(
            [torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()

        weights = torch.rand(qlen, n_routed_experts, dtype=torch.float32).contiguous()

        input_pt  = (torch.randn((qlen, hidden_size), dtype=dtype) / 100)\
                    .detach().requires_grad_(True).contiguous()
        input_cpp = input_pt.detach().clone().requires_grad_(True).contiguous()

        # ------------- forward -------------
        # Torch reference
        out_ref = moe_torch(input_pt, expert_ids, weights,
                            gate_proj, up_proj, down_proj, True)
        out_ref.retain_grad()
        
        batch_size_tensor = torch.tensor([qlen], dtype=torch.int32).contiguous()

        # 缓存forward中间结果用于python backward
        gate_u_cache = []
        up_v_cache = []
        
        # 模拟forward过程并缓存中间结果
        for token_idx in range(qlen):
            token_gate_u = []
            token_up_v = []
            for expert_pos in range(n_routed_experts):
                expert_id = int(expert_ids[token_idx, expert_pos].item())
                # 计算gate和up的输出
                gate_u = torch.mm(input_pt[token_idx:token_idx+1].to(torch.float32), gate_proj[expert_id].to(torch.float32).t()).squeeze()
                up_v = torch.mm(input_pt[token_idx:token_idx+1].to(torch.float32), up_proj[expert_id].to(torch.float32).t()).squeeze()
                token_gate_u.append(gate_u)
                token_up_v.append(up_v)
            gate_u_cache.append(token_gate_u)
            up_v_cache.append(token_up_v)

        # C++ AMX forward
        out_cpp = torch.empty_like(out_ref, dtype=dtype).contiguous()
        t0 = time.time()
        cpu_infer.submit(moe_cpp.forward(
            qlen, n_routed_experts,
            expert_ids.data_ptr(), weights.data_ptr(),
            input_cpp.data_ptr(), out_cpp.data_ptr(), batch_size_tensor.data_ptr()))
        cpu_infer.sync()
        t1 = time.time()
        diff_fwd = (out_cpp.to(torch.float32) - out_ref.to(torch.float32)).abs()
        print(f"out_cpp.to(torch.float32):{out_cpp.to(torch.float32)}, out_ref.to(torch.float32):{out_ref.to(torch.float32)}")
        rel_fwd  = diff_fwd.mean() / out_ref.abs().mean()
        print(f"Forward   diff: {rel_fwd.item():.3e} | time {t1-t0:.4f}s | "
              f"TFLOPS {flop_fwd/(t1-t0)/1e12:.2f}")

        # ------------- backward -------------
        grad_out = torch.randn_like(out_ref, dtype=gradtype).contiguous()
        grad_out_cpp = grad_out.clone().contiguous()
        grad_in_cpp  = torch.zeros_like(input_cpp, dtype=gradtype).contiguous()

        # Torch backward
        for p in (gate_proj, up_proj, down_proj, input_pt):
            if p.grad is not None:
                p.grad.zero_()
        t2 = time.time()
        out_ref.backward(grad_out, retain_graph=True)
        t3 = time.time()
        print(f"PyTorch backward time {t3-t2:.4f}s | "
              f"TFLOPS {flop_bwd/(t3-t2)/1e12:.2f}")

        # Python backward（模拟C++逻辑）
        t4_py = time.time()
        grad_in_python = moe_backward_python(
            input_pt, expert_ids, weights,
            gate_proj, up_proj, down_proj,
            grad_out.to(torch.float32), gate_u_cache, up_v_cache)
        t5_py = time.time()
        print(f"Python   backward time {t5_py-t4_py:.4f}s | "
              f"TFLOPS {flop_bwd/(t5_py-t4_py)/1e12:.2f}")

        # C++ backward
        t4 = time.time()
        cpu_infer.submit(moe_cpp.backward(
            qlen, n_routed_experts,
            expert_ids.data_ptr(), weights.data_ptr(), input_cpp.data_ptr(),
            grad_out_cpp.data_ptr(),
            grad_in_cpp.data_ptr(), batch_size_tensor.data_ptr()))
        cpu_infer.sync()
        t5 = time.time()
        print(f"C++      backward time {t5-t4:.4f}s | "
              f"TFLOPS {flop_bwd/(t5-t4)/1e12:.2f}")

        # 三种backward结果对比
        gcpp = grad_in_cpp.to(torch.float32)
        gref = input_pt.grad.to(torch.float32) if input_pt.grad is not None else torch.zeros_like(input_pt, dtype=torch.float32)
        gpy = grad_in_python.to(torch.float32)
        
        print(f"C++ AMX backward:{gcpp}", '\n', f"pytorch backward:{gref}", '\n', f"python backward:{gpy}")
        
        # 对比结果
        rel_bwd_cpp = (gcpp - gref).abs().mean() / gref.abs().mean()
        rel_bwd_py = (gpy - gref).abs().mean() / gref.abs().mean()
        rel_bwd_cpp_py = (gcpp - gpy).abs().mean() / gpy.abs().mean()
        
        print(f"Torch vs C++:    {rel_bwd_cpp.item():.3e}")
        print(f"Torch vs Python: {rel_bwd_py.item():.3e}")
        print(f"C++ vs Python:   {rel_bwd_cpp_py.item():.3e}")

if __name__ == "__main__":
    torch.manual_seed(42)
    test_amx_moe_two_round()