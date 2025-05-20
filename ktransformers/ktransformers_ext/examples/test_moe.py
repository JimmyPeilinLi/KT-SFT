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
gate_type = 0 # ggml_type::GGML_TYPE_F16
up_type = 0 # ggml_type::GGML_TYPE_F16
down_type = 0 # ggml_type::GGML_TYPE_F16
hidden_type = 0 # ggml_type::GGML_TYPE_F16
n_routed_experts = 2
qlen = 30
layer_num = 10
CPUInfer = cpuinfer_ext.CPUInfer(48)
validation_iter = 100

ddtype = torch.float32

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
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=ddtype, device = "cuda").to("cpu").contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=ddtype, device = "cuda").to("cpu").contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=ddtype, device = "cuda").to("cpu").contiguous()
            config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type, 0)
            moe = cpuinfer_ext.moe.MOE(config)
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)

        # validation
        for i in range(validation_iter):
            expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
            weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
            input = torch.randn((qlen, hidden_size), dtype=ddtype).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=ddtype).contiguous()
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
    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=ddtype, requires_grad=True).contiguous()
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=ddtype, requires_grad=True).contiguous()
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=ddtype, requires_grad=True).contiguous()
    # 创建MOE实例
    config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, 
                                       stride, group_min_len, group_max_len, 
                                       gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), 
                                       gate_type, up_type, down_type, hidden_type)  # 使用float16类型(0=GGML_TYPE_F16)
    moe = cpuinfer_ext.moe.MOE(config)

    # 创建输入数据
    expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
    
    # 使用相同的输入进行torch和C++算子的计算
    input = torch.randn((qlen, hidden_size), dtype=ddtype, requires_grad=True).contiguous()
    input = (input / 100).detach().requires_grad_(True)
    input_cpp = input.clone().detach().requires_grad_(True).contiguous()

    # 计算PyTorch参考输出
    t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, requires_grad=True)
    # 确保非叶子张量保留梯度
    t_output.retain_grad()
    
    # 计算C++算子输出
    output_cpp = torch.empty((qlen, hidden_size), dtype=ddtype).contiguous()
    grad_input_cpp = torch.empty_like(input_cpp)
    
    # 前向传播
    forward_start_time = time.time()
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
    forward_end_time = time.time()
    print(f"C++ forward 耗时: {forward_end_time - forward_start_time:.4f} 秒")
    
    # 验证前向传播结果
    forward_diff = torch.mean(torch.abs(output_cpp - t_output)) / torch.mean(torch.abs(t_output))
    print(f"Forward diff: {forward_diff.item()}")
    assert forward_diff < 0.001, f"Forward diff too large: {forward_diff.item()}"
    print("✅ Forward test passed!")
    
    # 创建相同的梯度输入
    grad_output = torch.randn_like(t_output)
    grad_output_cpp = grad_output.clone()
    
    print("-- pytorch backward --")
    # PyTorch反向传播性能测试
    pytorch_start_time = time.time()

    t_output.backward(grad_output, retain_graph=True)

    pytorch_end_time = time.time()
    pytorch_time = (pytorch_end_time - pytorch_start_time)
    
    print("-- c++ backward --")
    # C++反向传播性能测试
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

    cpp_start_time = time.time()
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

    cpp_end_time = time.time()
    cpp_time = (cpp_end_time - cpp_start_time)
    print(f"PyTorch backward 耗时: {pytorch_time:.4f} 秒")
    print(f"C++ backward 耗时: {cpp_time:.4f} 秒")
    print(f"性能比较: PyTorch/C++ = {pytorch_time/cpp_time:.2f}x")
    
    # 验证梯度结果
    backward_diff = torch.mean(torch.abs(grad_input_cpp - input.grad)) / torch.mean(torch.abs(input.grad))
    print(f"Backward diff: {backward_diff.item()}")
    assert backward_diff < 0.001, f"Backward diff too large: {backward_diff.item()}"
    print("✅ Backward pass test passed!")

if __name__ == "__main__":
    test_backward()
 
# Ascan: ==972151==AddressSanitizer CHECK failed: ../../../../src/libsanitizer/asan/asan_interceptors.cpp:335 "((__interception::real___cxa_throw)) != (0)" (0x0, 0x0)
#     #0 0x7f3f99efc9a8 in AsanCheckFailed ../../../../src/libsanitizer/asan/asan_rtl.cpp:74
#     #1 0x7f3f99f1d32e in __sanitizer::CheckFailed(char const*, int, char const*, unsigned long long, unsigned long long) ../../../../src/libsanitizer/sanitizer_common/sanitizer_termination.cpp:78
#     #2 0x7f3f99e785a4 in __interceptor___cxa_throw ../../../../src/libsanitizer/asan/asan_interceptors.cpp:335
#     #3 0x7f3f4abfaed7 in c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libc10.so+0x28ed7)
#     #4 0x7f3f95176b6b in c10::cuda::device_count() (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libc10_cuda.so+0x56b6b)
#     #5 0x7f3f4bc43c68 in at::cuda::detail::CUDAHooks::hasCUDA() const (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so+0xf6fc68)
#     #6 0x7f3f80f17879 in at::getAccelerator(bool) (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so+0x15b7879)
#     #7 0x7f3f8498afd3 in torch::autograd::Node::stream() (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so+0x502afd3)
#     #8 0x7f3f84987889 in torch::autograd::Engine::compute_dependencies(torch::autograd::Node*, torch::autograd::GraphTask&, unsigned long) (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so+0x5027889)
#     #9 0x7f3f84987c33 in torch::autograd::Engine::execute(std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&, std::vector<at::Tensor, std::allocator<at::Tensor> > const&, bool, bool, bool, std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&) (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so+0x5027c33)
#     #10 0x7f3f941eb305 in torch::autograd::python::PythonEngine::execute(std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&, std::vector<at::Tensor, std::allocator<at::Tensor> > const&, bool, bool, bool, std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&) (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_python.so+0x78e305)
#     #11 0x7f3f941e9d5a in THPEngine_run_backward(_object*, _object*, _object*) (/home/lpl/anaconda3/envs/KSFT/lib/python3.11/site-packages/torch/lib/libtorch_python.so+0x78cd5a)
#     #12 0x528b16 in cfunction_call /usr/local/src/conda/python-3.11.11/Objects/methodobject.c:542
#     #13 0x54303c in _PyObject_Call /usr/local/src/conda/python-3.11.11/Objects/call.c:343
#     #14 0x54303c in PyObject_Call /usr/local/src/conda/python-3.11.11/Objects/call.c:355
#     #15 0x51a206 in do_call_core /usr/local/src/conda/python-3.11.11/Python/ceval.c:7321
#     #16 0x51a206 in _PyEval_EvalFrameDefault /usr/local/src/conda/python-3.11.11/Python/ceval.c:5376
#     #17 0x5cc3a9 in _PyEval_EvalFrame /usr/local/src/conda/python-3.11.11/Include/internal/pycore_ceval.h:73
#     #18 0x5cc3a9 in _PyEval_Vector /usr/local/src/conda/python-3.11.11/Python/ceval.c:6434
#     #19 0x5cba7e in PyEval_EvalCode /usr/local/src/conda/python-3.11.11/Python/ceval.c:1148
#     #20 0x5ecba6 in run_eval_code_obj /usr/local/src/conda/python-3.11.11/Python/pythonrun.c:1741
#     #21 0x5e873f in run_mod /usr/local/src/conda/python-3.11.11/Python/pythonrun.c:1762
#     #22 0x5fd5f1 in pyrun_file /usr/local/src/conda/python-3.11.11/Python/pythonrun.c:1657
#     #23 0x5fc9be in _PyRun_SimpleFileObject /usr/local/src/conda/python-3.11.11/Python/pythonrun.c:440
#     #24 0x5fc6e2 in _PyRun_AnyFileObject /usr/local/src/conda/python-3.11.11/Python/pythonrun.c:79
#     #25 0x5f73fd in pymain_run_file_obj /usr/local/src/conda/python-3.11.11/Modules/main.c:360
#     #26 0x5f73fd in pymain_run_file /usr/local/src/conda/python-3.11.11/Modules/main.c:379
#     #27 0x5f73fd in pymain_run_python /usr/local/src/conda/python-3.11.11/Modules/main.c:605
#     #28 0x5f73fd in Py_RunMain /usr/local/src/conda/python-3.11.11/Modules/main.c:684
#     #29 0x5bc148 in Py_BytesMain /usr/local/src/conda/python-3.11.11/Modules/main.c:738
#     #30 0x7f3f99b34d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)
#     #31 0x7f3f99b34e3f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x29e3f)
#     #32 0x5bbf92  (/home/lpl/anaconda3/envs/KSFT/bin/python3.11+0x5bbf92)*/