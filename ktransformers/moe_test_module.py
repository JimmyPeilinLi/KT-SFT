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
from ktransformers.util.custom_gguf import GGUFLoader
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
        self.num_experts = 8
        
        # 预定义固定值
        self.fixed_input = None
        self.fixed_expert_ids = None
        self.fixed_weights = None
        
    def _create_fixed_data(self, device, batch_size=2):
        """创建固定输入数据"""
        if self.fixed_input is None:
            # 使用固定种子生成可重复数据
            with torch.random.fork_rng():
                torch.manual_seed(42)
                hidden_size = config.hidden_size
                
                # 固定输入张量
                self.fixed_input = torch.randn(batch_size, hidden_size)
                
                # 固定选择的专家 (示例: 第一个样本选专家0和1，第二个样本选专家2和3)
                self.fixed_expert_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
                
                # 固定权重 (示例: 每个样本的两个专家权重相同)
                self.fixed_weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        
        # 转移到目标设备
        return (
            self.fixed_input.clone().to(device).requires_grad_(True),
            self.fixed_expert_ids.clone().to(device),
            self.fixed_weights.clone().to(device)
        )

    def _run_single_device_test(self, device, seed=42):
        """在指定设备上运行前向反向传播并返回梯度"""
        # 固定所有随机种子
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        # 初始化模型到目标设备
        model = KExpertsTorch(
            key="blk.1",
            gguf_loader=gguf_loader,
            config=config,
            n_routed_experts=self.num_experts,
            device=device
        )
        model.load(device=device)
        
        # 生成固定输入数据
        input_tensor, expert_ids, weights = self._create_fixed_data(device)
        
        # 确保模型参数在正确设备上
        model.to(device)
        
        # 强制使用全精度计算
        with torch.autocast(device_type=device, enabled=False):
            output = model(input_tensor, expert_ids, weights)
            
        # 反向传播
        loss = output.sum()
        loss.backward()
        
        # 收集梯度
        gradients = {
            "input": input_tensor.grad.detach().cpu(),
            "loss": loss.detach().cpu(),
            "model": [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None]
        }
        return gradients

    def test_forward_gradient(self):
        # CPU运行
        cpu_gradients = self._run_single_device_test("cpu")
        
        # GPU运行 (如果可用)
        if torch.cuda.is_available():
            gpu_gradients = self._run_single_device_test("cuda")

            print(f"cpu_gradients:{cpu_gradients}")
            print(f"gpu_gradients:{gpu_gradients}")
            
            # 输入梯度比较
            input_diff = torch.max(torch.abs(cpu_gradients["input"] - gpu_gradients["input"]))
            print(f"input_diff:{input_diff}")
            # self.assertLess(input_diff, 1e-5, f"Input梯度差异过大: {input_diff.item()}")
            
            # 参数梯度比较
            for i, (cpu_g, gpu_g) in enumerate(zip(cpu_gradients["model"], gpu_gradients["model"])):
                param_diff = torch.max(torch.abs(cpu_g - gpu_g))
                print(f"param_diff:{param_diff}")
                # self.assertLess(param_diff, 1e-5, f"参数 {i} 梯度差异过大: {param_diff.item()}")

            # 模型梯度对比
            for i, (cpu_g, gpu_g) in enumerate(zip(cpu_gradients["model"], gpu_gradients["model"])):
                diff = (cpu_g - gpu_g.cpu()).abs().max()
                print(f"参数梯度 {i} 最大差异: {diff.item()}")
                self.assertTrue(torch.allclose(cpu_g, gpu_g, atol=1e-4, rtol=1e-3),
                            f"参数梯度 {i} 差异超出阈值，最大差异: {diff.item()}")
                
        else:
            self.skipTest("CUDA不可用，跳过GPU测试")

if __name__ == '__main__':
    unittest.main()