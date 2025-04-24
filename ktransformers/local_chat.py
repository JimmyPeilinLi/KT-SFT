"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import argparse
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
from torchviz import make_dot
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import load_weights, prefill_and_generate, get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.sft.lora import inject_lora_layer, lora_and_load_adapter
from ktransformers.util.custom_gguf import GGUFLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# for debug
def print_module_tree(module, indent=0):
    print(" " + f"{module.__class__.__name__}(training={module.training})")
    for name, child in module.named_children():
        print(" " + f"└─{name}: ", end="")
        print_module_tree(child, indent + 4)

# for debug
def write_to_file(content, file_path: str = 'ktransformers/mark_content.txt', mode: str = 'a', encoding: str = 'utf-8') -> None:
    """
    将字符串写入指定文件 
    :param content: 要写入的字符串内容 
    :param file_path: 目标文件路径 
    :param mode: 文件打开模式（默认'w'为覆盖写入，可选'a'追加写入）
    :param encoding: 文件编码（默认utf-8）
    """
    with open(file_path, mode, encoding=encoding) as f:
        f.write(content) 

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def local_chat(
    model_path: str | None = None,
	model_config_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 300,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = False,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_prefill_size: int = 8192,
    is_sft: bool = False,
    sft_data_path: str | None = None,
    save_adapter_path: str | None = None,
    use_adapter: bool = False,
    use_adapter_path: str | None = None,
):

    # torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if model_config_path == None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    else:
	    config = AutoConfig.from_pretrained(model_config_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)

    # print_module_tree(model)  # 观察输出是否有重复模块或循环

    model.train()

    if is_sft == True:
        print(f"sft with lora in dataset: {sft_data_path} ...")
        lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path)

    if use_adapter == True:
        if is_sft == True:
            raise AttributeError("We do not support more than one adapter up to now...")
        
        # TODO: 判断如果是GGUF格式的adapter，把他跟原来的模型一起处理一下，在后面进行推理
        # 处理GGUF格式的适配器
        if use_adapter_path.endswith('.gguf'):
            inject_lora_layer(model)
            # 加载适配器权重到现有模型结构
            adapter_gguf_loader = GGUFLoader(use_adapter_path)
            # 获取模型当前设备信息
            # current_device = next(model.parameters()).device
            # 直接加载权重到模型的适配器部分（假设适配器层的名称与GGUF键名匹配）
            load_weights(model, adapter_gguf_loader, adapter_gguf=True)
            # 确保模型回到训练模式（适配器可能需要梯度）
            model.train()
        else:
            raise NotImplementedError("Currently only GGUF format adapters are supported. Please provide a .gguf file.")

    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    # for debug
    # if system == "Windows":
    #     os.system("cls")
    # else:
    #     os.system("clear")

    while True:
        content = input("Chat: ")
        if content.startswith('"""'):  # prefix """
            # multi lines input
            content = content[3:] + "\n"
            while True:
                line = input("")
                if line.endswith('"""'):
                    # end multi lines input
                    line = line[:-3]  # suffix """
                    if line:
                        content += line + "\n"
                    break
                else:
                    content += line + "\n"

        if content == "":
            if prompt_file != None:
                content = open(prompt_file, "r").read()
            else:
                content = "Please write a piece of quicksort code in C++."
        elif os.path.isfile(content):
            content = open(content, "r").read()
            
        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if force_think:
            token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
            input_tensor = torch.cat(
                [input_tensor, token_thinks], dim=1
            )
        if mode == 'long_context':
            assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        
        if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
            )
        else:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_config_path", default=None)
    parser.add_argument("--gguf_path", required=True)
    parser.add_argument("--cpu_infer", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--force_think", action="store_true")
    parser.add_argument("--optimize_config_path", required=True)
    parser.add_argument("--is_sft", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--sft_data_path", default=None)
    parser.add_argument("--save_adapter_path", default=None)
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--use_adapter_path", default=None)

    args = parser.parse_args()

    local_chat(
        model_path=args.model_path,
        model_config_path=args.model_config_path,
        gguf_path=args.gguf_path,
        cpu_infer=args.cpu_infer,
        max_new_tokens=args.max_new_tokens,
        force_think=args.force_think,
        optimize_config_path=args.optimize_config_path,
        is_sft=args.is_sft,
        sft_data_path=args.sft_data_path,
        save_adapter_path=args.save_adapter_path,
        use_adapter=args.use_adapter,
        use_adapter_path=args.use_adapter_path
    )