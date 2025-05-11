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
stride = 32
group_min_len = 10
group_max_len = 1024
gate_type = 1 # ggml_type::GGML_TYPE_F16
up_type = 1 # ggml_type::GGML_TYPE_F16
down_type = 1 # ggml_type::GGML_TYPE_F16
hidden_type = 1 # ggml_type::GGML_TYPE_F16
n_routed_experts = 2
qlen = 30
layer_num = 10
CPUInfer = cpuinfer_ext.CPUInfer(48)
validation_iter = 100

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

# 定义SiLU激活函数的可微版本（带梯度）
class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input / (1.0 + torch.exp(-input))
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid = 1.0 / (1.0 + torch.exp(-input))
        return grad_output * (sigmoid + input * sigmoid * (1 - sigmoid))

silu = SiLU.apply

def mlp_torch(input, gate_proj, up_proj, down_proj, requires_grad=False):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    
    # 使用可微的SiLU或者原来的函数，取决于是否需要梯度
    if requires_grad:
        intermediate = silu(gate_buf) * up_buf
    else:
        intermediate = act_fn(gate_buf) * up_buf
    
    ret = torch.mm(intermediate, down_proj.t())
    return ret

def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=False):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i], requires_grad)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return t_output

# 前向传播验证
def test_forward():
    with torch.inference_mode(mode=True):
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
            config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
            moe = cpuinfer_ext.moe.MOE(config)
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)

        # validation
        for i in range(validation_iter):
            expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
            weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
            input = torch.randn((qlen, hidden_size), dtype=torch.float16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.float16).contiguous()
            input = input / 100
            
            moe = moes[i % layer_num]
            CPUInfer.submit(
                moe.forward( 
                    qlen,
                    n_routed_experts, 
                    expert_ids.data_ptr(), 
                    weights.data_ptr(), 
                    input.data_ptr(), 
                    output.data_ptr()
                )
            )
            CPUInfer.sync()
            # print('cpuinfer output', output)

            gate_proj = gate_projs[i%layer_num]
            up_proj = up_projs[i%layer_num]
            down_proj = down_projs[i%layer_num]
            t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj)
            # print('torch output', t_output)

            diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
            print('diff = ', diff)
            assert(diff < 0.001)

# 反向传播验证
def test_backward():
    # 先测试backward是否能正常调用
    print("\n===== Testing Backward Pass =====")
    # 创建一个单层MOE用于测试
    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), 
                           dtype=torch.float32, requires_grad=True)
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), 
                         dtype=torch.float32, requires_grad=True)
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), 
                           dtype=torch.float32, requires_grad=True)
    
    # 创建MOE实例
    config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, 
                                       stride, group_min_len, group_max_len, 
                                       gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), 
                                       0, 0, 0, 0)  # 使用float32类型(0=GGML_TYPE_F32)
    moe = cpuinfer_ext.moe.MOE(config)
    
    # 创建输入数据
    expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
    
    # 使用相同的输入进行torch和C++算子的计算
    input = torch.randn((qlen, hidden_size), dtype=torch.float32, requires_grad=True)
    input_cpp = input.clone().detach().requires_grad_(True)
    
    # 计算PyTorch参考输出
    t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=True)
    
    # 计算C++算子输出
    output_cpp = torch.empty((qlen, hidden_size), dtype=torch.float32).contiguous()
    grad_input_cpp = torch.empty_like(input_cpp)
    
    # 前向传播
    CPUInfer.submit(
        moe.forward(
            qlen,
            n_routed_experts,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_cpp.data_ptr(),
            output_cpp.data_ptr()
        )
    )
    CPUInfer.sync()
    
    # 验证前向传播结果
    forward_diff = torch.mean(torch.abs(output_cpp - t_output)) / torch.mean(torch.abs(t_output))
    print(f"Forward diff: {forward_diff.item()}")
    
    # 创建相同的梯度输入
    grad_output = torch.randn_like(t_output)
    grad_output_cpp = grad_output.clone()
    
    print("-- pytorch backward --")
    # PyTorch反向传播
    t_output.backward(grad_output, retain_graph=True)
    
    print("-- c++ backward --")
    # C++反向传播
    CPUInfer.submit(
        moe.backward(
            qlen,
            n_routed_experts,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_cpp.data_ptr(),
            grad_output_cpp.data_ptr(),
            grad_input_cpp.data_ptr()
        )
    )
    CPUInfer.sync()
    
    # 验证梯度结果
    backward_diff = torch.mean(torch.abs(grad_input_cpp - input.grad)) / torch.mean(torch.abs(input.grad))
    print(f"Backward diff: {backward_diff.item()}")
    assert backward_diff < 0.01, f"Backward diff too large: {backward_diff.item()}"
    
    print("✅ Backward pass test passed!")

if __name__ == "__main__":
    test_backward()
