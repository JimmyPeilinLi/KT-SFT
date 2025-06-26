import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from torchviz import make_dot
from torch import nn
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import unittest
from torch.autograd import gradcheck

from ktransformers.operators.linear import KLinearTorch, KTransformersLinear
from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.operators.experts import KExpertsTorch
from ktransformers.util.utils import load_weights

gguf_loader = GGUFLoader(gguf_path="/home/yj/ktransformers/GGUF-DeepSeek-V2-Lite-Chat")
config = AutoConfig.from_pretrained("/home/yj/ktransformers/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
torch.set_default_dtype(config.torch_dtype)

class TestKExpertsTorch(unittest.TestCase):
    def setUp(self):
        # 确保计算确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 初始化模型到CPU
        self.base_device = "cpu"
        self.num_experts = 8
        # model = KExpertsTorch(
        #     key="blk.1",
        #     gguf_loader=gguf_loader,
        #     config=config,
        #     n_routed_experts=self.num_experts,
        #     device=self.base_device
        # )
        # model.load()
        
    def _run_single_device_test(self, device, seed=42):
        """在指定设备上运行前向反向传播并返回梯度"""
        # 固定随机种子
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        # 克隆模型时携带随机数状态
        model = KExpertsTorch(
            key="blk.1",
            gguf_loader=gguf_loader,
            config=config,
            n_routed_experts=self.num_experts,
            device=device
        )
        model.load(device=device)

        # 生成确定性输入数据
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            batch_size = 2
            hidden_size = model.config.hidden_size
            input_tensor = torch.randn(batch_size, hidden_size, device=device, requires_grad=True)
            expert_ids = torch.randint(0, self.num_experts, 
                                    (batch_size, model.config.num_experts_per_tok), 
                                    device=device)
            weights = torch.randn(batch_size, model.config.num_experts_per_tok, device=device)
            weights = torch.softmax(weights, dim=-1)
        
        # 添加设备验证
        print(f"input_tensor.device:{input_tensor.device}")
        print(f"torch.device(device):{torch.device(device)}")
        # assert input_tensor.device == torch.device(device)
        for p in model.parameters():
            print(f"p.device:{p.device}")

        for name, param in model.named_parameters():
            print(name, param.size())

        # 前向传播
        
        model.to(device)
        with torch.autocast(device_type=device, enabled=False):  # 强制禁用混合精度
            output = model(input_tensor, expert_ids, weights)
        
        # 反向传播
        loss = output.sum()

        
        # # 可视化计算图
        # dot = make_dot(output, params=dict(model.named_parameters()))
        # dot.render(f"origin_moe_{torch.device(device)}_graph", format="svg")

        loss.backward()
        
        # 收集梯度
        gradients = {
            "input": input_tensor.grad.clone().cpu(),
            "loss": loss.clone().cpu(),
            "model": [p.grad.clone().cpu() for p in model.parameters() if p.grad is not None]
        }
        return gradients

    def test_forward_gradient(self):
        # # 验证基础模型参数类型
        # for param in model.parameters():
        #     self.assertEqual(param.dtype, config.torch_dtype)
        
        # 在CPU上运行
        cpu_gradients = self._run_single_device_test("cpu")
        print(f"cpu_gradients: {cpu_gradients}")
        
        # 基础检查
        self.assertIsNotNone(cpu_gradients["input"])
        self.assertTrue(all(g is not None for g in cpu_gradients["model"]))
        
        # 如果GPU可用则在GPU上运行并比较
        if torch.cuda.is_available():
            gpu_gradients = self._run_single_device_test("cuda")

            print(f"gpu_gradients: {gpu_gradients}")

            
            max_diff = (cpu_gradients["input"] - gpu_gradients["input"].cpu()).abs().max()
            print(f"Input梯度最大差异: {max_diff.item()}")

            self.assertTrue(torch.allclose(cpu_gradients["input"], gpu_gradients["input"], atol=1e-4, rtol=1e-3),
                        f"Input梯度差异超出阈值，最大差异: {max_diff.item()}")

            # 模型梯度对比
            for i, (cpu_g, gpu_g) in enumerate(zip(cpu_gradients["model"], gpu_gradients["model"])):
                diff = (cpu_g - gpu_g.cpu()).abs().max()
                print(f"参数梯度 {i} 最大差异: {diff.item()}")
                self.assertTrue(torch.allclose(cpu_g, gpu_g, atol=1e-4, rtol=1e-3),
                            f"参数梯度 {i} 差异超出阈值，最大差异: {diff.item()}")

        else:
            raise ImportError("NO CUDA FOR TEST!!")

    # def test_detach_effect(self):
    #     # 确保在CPU上运行
    #     input_tensor = torch.randn(1, model.config.hidden_size, device="cpu", requires_grad=True)
    #     expert_ids = torch.tensor([[0, 1]], device="cpu")
    #     weights = torch.tensor([[0.5, 0.5]], device="cpu")

    #     output = model(input_tensor, expert_ids, weights)
        
    #     # # 可视化计算图
    #     # dot = make_dot(output, params=dict(model.named_parameters()))
    #     # dot.render("origin_moe_cpu_graph", format="svg")
        
    #     # 反向传播检查
    #     loss = output.sum()
    #     loss.backward()
        
    #     # 验证梯度存在性
    #     self.assertIsNotNone(input_tensor.grad)
    #     self.assertTrue(all(p.grad is not None for p in model.parameters()))

if __name__ == '__main__':
    unittest.main()