import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, TaskType
from ktransformers.sft.peft_utils.mapping import inject_adapter_in_model, get_peft_model
from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
# from ktransformers.sft.load_lora import get_custom_peft_model
import os
from torchviz import make_dot

def preprocess_function(examples, tokenizer):
    inputs = examples["input"]
    targets = examples["output"]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        # 改写trainer的save_model，在checkpoint的时候只存lora权重
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def inspect_device(model, write_file):
    for name, module in model.named_modules(): 
        with open(write_file, 'a') as file:
            file.write(f"Layer: {name}\n")
        # print(f"Layer: {name}")
        # 检查参数 
        for param_name, param in module.named_parameters(recurse=False): 
            with open(write_file, 'a') as file:
                file.write(f"  Parameter '{param_name}' device: {param.device}\n")
            # print(f"  Parameter '{param_name}' device: {param.device}") 
        # 检查缓冲区（如BatchNorm的running_mean）
        for buffer_name, buffer in module.named_buffers(recurse=False): 
            with open(write_file, 'a') as file:
                file.write(f"  Buffer '{buffer_name}' device: {buffer.device}\n")
            # print(f"  Buffer '{buffer_name}' device: {buffer.device}")

def print_model_params(model):
    # 遍历所有Decoder层（共27层）
    # for layer_idx in range(len(model.model.orig_module.layers)):
    for layer_idx in range(0, 3):
        layer = model.model.orig_module.layers[layer_idx]
        
        # ============= 打印注意力层参数 =============
        print(f"\n================ Layer {layer_idx} Attention ================")
        
        # 打印q_proj参数
        q_proj = layer.self_attn.orig_module.q_proj.orig_module
        print(f"\nq_proj.generate_linear.weight (shape: {q_proj.generate_linear.weight.shape})")
        print(q_proj.generate_linear.weight.cpu())
        
        # # 打印kv_a_proj参数
        # kv_a_proj = layer.self_attn.orig_module.kv_a_proj_with_mqa.orig_module
        # print(f"\nkv_a_proj.weight (shape: {kv_a_proj.weight.shape})")
        # print(kv_a_proj.weight.data[:3, :5].detach().cpu().numpy())
        
        # # 打印o_proj参数
        # o_proj = layer.self_attn.orig_module.o_proj.orig_module
        # print(f"\no_proj.weight (shape: {o_proj.weight.shape})")
        # print(o_proj.weight.data[:3, :5].detach().cpu().numpy())
        
        # # ============= 打印MLP/MoE参数 =============
        # print(f"\n================ Layer {layer_idx} MLP/MoE ================")
        
        # # 区分普通MLP和MoE层（第0层是普通MLP，其他是MoE）
        # if layer_idx == 0:
        #     # 普通MLP层参数
        #     mlp = layer.mlp
        #     for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        #         module = getattr(mlp, proj_type).orig_module
        #         print(f"\n{proj_type}.weight (shape: {module.weight.shape})")
        #         print(module.weight.data[:3, :5].detach().cpu().numpy())
        # else:
        #     # MoE层参数
        #     moe = layer.mlp.orig_module
        #     # 打印共享专家参数
        #     print("\n[Shared Experts]")
        #     for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        #         module = getattr(moe.shared_experts, proj_type).orig_module
        #         print(f"\nshared_{proj_type}.weight (shape: {module.weight.shape})")
        #         print(module.weight.data[:3, :5].detach().cpu().numpy())
            
        #     # 打印64个专家参数（采样前3个）
        #     print("\n[Experts]")
        #     for expert_idx in range(3):  # 采样前3个专家
        #         expert = moe.experts.orig_module[expert_idx]
        #         print(f"\nExpert {expert_idx}:")
        #         for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        #             module = getattr(expert, proj_type)
        #             print(f"{proj_type}.weight (shape: {module.weight.shape})")
        #             print(module.weight.data[:3, :5].detach().cpu().numpy())

def print_lora_params(model):
    # 遍历所有Decoder层 (索引0到26共27层)
    # for layer_idx in range(len(model.model.orig_module.layers)):
    for layer_idx in range(0, 3):
        # 获取当前Decoder层
        layer = model.base_model.model.model.orig_module.layers[layer_idx]
        # layer = model.model.orig_module.layers[layer_idx]
        
        # 定位到目标模块路径
        q_proj_module = layer.self_attn.orig_module.q_proj.orig_module
        
        # 提取目标矩阵参数
        linear_weight = q_proj_module.generate_linear.weight
        lora_A_weight = q_proj_module.lora_A["default"].weight
        lora_B_weight = q_proj_module.lora_B["default"].weight
        
        # 打印参数信息
        print(f"\n=================== Layer {layer_idx} ===================")
        
        # 打印原Linear矩阵参数
        print("\nOriginal Linear (first row slice):")
        print(linear_weight.cpu())  # 第一行前5个参数
        
        # 打印Lora_A参数
        print("\nLora_A (first row slice):")
        print(lora_A_weight.cpu())  # 第一行前5个参数
        
        # 打印Lora_B参数
        print("\nLora_B (first row slice):")
        print(lora_B_weight.cpu())  # 第一行前5个参数

def print_grad_fn(grad_fn, indent=0):
    """递归打印计算图节点"""
    if grad_fn is None:
        return
    # 打印当前节点信息
    print(' ' * indent, f"Node: {str(grad_fn).split('(')[0]}")
    print(' ' * indent, f"  Metadata: {grad_fn.metadata}")
    # 遍历子节点
    for child in getattr(grad_fn, 'next_functions', []):
        if child[0] is not None:
            print_grad_fn(child[0], indent + 2)

def forward_hook(module, inputs, output):
    if isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            if o is None:
                print(f"{module.__class__.__name__} output index {i} is None")
            else:
                print(f"{module.__class__.__name__} output index {i}: requires_grad={o.requires_grad}, grad_fn={o.grad_fn}")
    elif output is None:
        print(f"{module.__class__.__name__} returned None")
    else:
        print(f"{module.__class__.__name__}: requires_grad={output.requires_grad}, grad_fn={output.grad_fn}")



def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path):

    torch.autograd.set_detect_anomaly(True) # 在反向传播出错时，PyTorch 会提供更详细的堆栈信息
    
    # tokenizer = AutoTokenizer.from_pretrained('/data/model/Qwen2.5-7B-Instruct', trust_remote_code=True)

    dataset = Dataset.from_json(sft_data_path)

    # processed_dataset = dataset.map(preprocess_function, batched=True)
    processed_dataset = dataset.map(lambda  examples: preprocess_function(examples, tokenizer), batched=True)
    split_dataset = processed_dataset.train_test_split(test_size=0.1)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=8)

    # print(f"LoRA前:{model}")

    # for name, module in model.named_modules():
    #     if "q_proj" in name or "kv_a_proj" in name or "o_proj" in name:
    #         print(name)

    # print_model_params(model)

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj"
            # "kv_a_proj_with_mqa",
            # "kv_b_proj",
            # "o_proj"
        ],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    # model = inject_adapter_in_model(lora_config, model)
    model = get_peft_model(model, lora_config)
    # model = get_custom_peft_model(model, lora_config)

    # model = KTransformersLinearLora()

    # inspect_device(model, '/home/yj/ktransformers/device1.txt')
    # with open('/home/yj/ktransformers/device1.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 

    # print(f"LoRA后:{model}")

    # model = model.to('cuda')
    model.config.use_cache = False

    # for name, module in model.named_modules():
    #     module.register_forward_hook(forward_hook)

    # for name, parms in model.named_parameters():	
    #     print('-->name:', name)
    #     print('-->para:', parms)
    #     print('-->grad_requirs:',parms.requires_grad)
    #     print('-->grad_fn:',parms.grad_fn)
    #     print('-->grad_value:',parms.grad)
    #     print("===")

    # 选择特定层的输入输出
    output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    loss = output.logits.mean()
    # print_grad_fn(loss.grad_fn)
    # 生成计算图
    dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot_str = dot.source
    # with open("compute_graph.dot", "w") as f:
    #     f.write(dot_str)
    dot.render("compute_graph", format="svg")

    # inspect_device(model, '/home/yj/ktransformers/device2.txt')
    # with open('/home/yj/ktransformers/device2.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 
    
    os.environ["NCCL_P2P_DISABLE"]  = "1"
    os.environ["NCCL_IB_DISABLE"]  = "1"

    # print_lora_params(model)

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            output_dir=save_adapter_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=10,
            learning_rate=3e-4,
            fp16=False,
            logging_steps=10,
            save_steps=200,
            # 可额外添加分布式训练优化参数 
            dataloader_drop_last=True,
            ddp_find_unused_parameters=False 
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    model.print_trainable_parameters() 

    # trainer.train()


    # model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))

    model.save_pretrained(save_adapter_path)


    print_lora_params(model)

    # model = model.merge_and_unload()


def inject_lora_layer(model):

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj"
        ],
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )
    
    model = inject_adapter_in_model(lora_config, model)

    
