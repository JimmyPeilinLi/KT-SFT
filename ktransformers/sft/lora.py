import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, TaskType
import os
from torchviz import make_dot
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from transformers import TrainerCallback
import gc
from tqdm import tqdm
import os, torch, json, tempfile
from pathlib import Path
from accelerate import Accelerator

from ktransformers.sft.peft_utils.mapping import inject_adapter_in_model, get_peft_model
from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
from ktransformers.sft.flops_utils.custom_profile import custom_profile
from ktransformers.operators.experts import KExpertsTorch, KTransformersExperts
# from ktransformers.sft.load_lora import get_custom_peft_model

# FOR: not A or H GPU
os.environ["NCCL_P2P_DISABLE"]  = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

layer_data = {}  # 存储各层输入输出数据

def record_layer_io(module, input, output, layer_name):
    layer_data[layer_name] = {
        'input': input[0].detach().clone(),
        'output': output.detach().clone()
    }

# 注册钩子
hooks = []
target_layers = [
    'base_model.model.model.orig_module.layers.0.self_attn.kv_a_proj_with_mqa',
    'base_model.model.model.orig_module.layers.0.self_attn.kv_b_proj',
    # 'base_model.model.model.orig_module.layers.1.self_attn.kv_a_proj_with_mqa',
    # 'base_model.model.model.orig_module.layers.1.self_attn.kv_b_proj'
]


# 自定义回调，手动控制Profiler，避免transformer库版本太低
class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

def preprocess_function(batch, tokenizer, max_len=512):
    full_inputs = [ins + inp for ins, inp in zip(batch["instruction"], batch["input"])]
    
    # print(f"FI: {full_inputs}")
    # print(f"TFI: {type(full_inputs)}")
    tokenized_inputs = tokenizer(full_inputs, padding="max_length", truncation=True, max_length=max_len)
    tokenized_outputs = tokenizer(batch["output"], padding="max_length", truncation=True, max_length=max_len)

    # need batch=false, just for debug and test
    # print("🔹Instruction tokens:", len(tokenizer.tokenize(instruction)))
    # print("🔹Input tokens:", len(tokenizer.tokenize(inputs)))
    # print("🔹Instruction+Input tokens:", len(tokenizer.tokenize(full_input)))
    # print("🔸Output tokens (label):", len(tokenizer.tokenize(targets)))

    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    return tokenized_inputs

class KTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 只保存 LoRA adapter（含 adapter_config.json）
        self.model.save_pretrained(output_dir)
        
    def _move_model_to_device(self, model, device):
        print("[KTrainer] Due to the placement feature in KTransformers, skip moving model to", device)
        return model

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

def check_moe_gradients(model):
    moe_layer = model.base_model.model.model.orig_module.layers[1].mlp.orig_module
    for name, param in moe_layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad)
            print(f"MoE参数 {name} 梯度范数: {grad_norm}")
        else:
            print(f"MoE参数 {name} 无梯度")

def disable_all_dropout(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = 0  # 直接修改概率参数
                child.inplace = False  # 确保不影响原始数据
            disable_all_dropout(child)  # 递归处理子模块

def verify_lora_layers(model):
    for layer_path in target_layers:
        # 获取模块实例
        module = model.get_submodule(layer_path)
        orig_module = module.orig_module
        
        # 提取参数
        W = orig_module.weight.data  # [576, 2048] -> [2048, 576]
        lora_A = module.lora_A['default'].weight.data  # [8, 2048]
        lora_B = module.lora_B['default'].weight.data  # [576, 8]
        alpha_over_r = 32/8  # alpha=32, r=8
        
        # 获取记录的数据（保持batch维度）
        input_tensor = layer_data[layer_path]['input']  # [1, 512, 2048]
        
        # 手动计算流程
        # 原始部分计算
        try:
            original_output = torch.matmul(input_tensor, W)  # [1,512,2048] @ [2048,576] => [1,512,576]
        except:
            original_output = torch.matmul(input_tensor, W.T)  # [1,512,2048] @ [2048,576] => [1,512,576]
        
        # LoRA部分计算
        lora_effect = torch.matmul(
            torch.matmul(input_tensor, lora_A.T),  # [1,512,2048] @ [2048,8] => [1,512,8]
            lora_B.T  # [1,512,8] @ [8,576] => [1,512,576]
        ) * alpha_over_r
        
        # 合并结果
        manual_output = original_output + lora_effect  # [1,512,576]
        
        # 获取模型输出
        model_output = layer_data[layer_path]['output']

        print(f"manual_output:{manual_output}")
        print(f"model_output:{model_output}")
        
        # 数值比较
        if torch.allclose(manual_output, model_output, atol=1e-5):
            print(f"{layer_path} 验证通过")
        else:
            print(f"{layer_path} 验证失败！最大误差：{torch.max(torch.abs(manual_output - model_output))}")

def print_moe_stats(moe_layer: KExpertsTorch):
    print(f"Total Params: {moe_layer.total_params/1e6:.2f}M")
    
    total_time = sum(moe_layer.times)
    gflops = (moe_layer.total_flops / 1e9) / total_time if total_time !=0 else 0
    
    print(f"Total Calls: {moe_layer.call_count}")
    # print(f"Avg GFLOPS per Call: {gflops/moe_layer.call_count:.2f}")
    print(f"Overall GFLOPS: {gflops:.2f}")
    
    # 打印单次调用示例
    if moe_layer.call_count > 0:
        last_flops = moe_layer.flops_per_call[-1]
        last_time = moe_layer.times[-1]
        print(f"\nLast Call - FLOPs: {last_flops/1e9:.2f}G  Time: {last_time*1000:.2f}ms  "
              f"GFLOPS: {(last_flops/1e9)/last_time:.2f}")
        
def recursive_traverse(model, parent_name=''):
    """
    递归遍历模型，查找MoE层并调用print_moe_stats。
    """
    # 遍历模型中的所有子模块
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        # 如果是 MoE 层，调用 print_moe_stats
        if isinstance(module, KTransformersExperts):  # 检查是否为 MoE 层
            print(f"Found MoE layer: {full_name}")
            print_moe_stats(module.generate_experts)
        
        # 递归处理子模块
        recursive_traverse(module, full_name)

def log_step_state(
    step: int,
    inputs: dict,
    loss: torch.Tensor,
    model: torch.nn.Module,
    log_dir: str = "train_logs",
):
    """
    把当前 step 的输入 / loss / grad / param 保存到 log_dir/step_{step}.pt
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ① 处理输入：只保存张量，且先搬到 CPU，避免 GPU 进程间序列化问题
    logged_inputs = {
        k: v.detach().cpu()
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }

    # ② loss 一般是标量 Tensor
    loss_val = loss.detach().cpu()

    # ③ 参数与梯度
    params, grads = {}, {}
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu()
        grads[name] = p.grad.detach().cpu() if p.grad is not None else None

    torch.save(
        {
            "step": step,
            "inputs": logged_inputs,
            "loss": loss_val,
            "params": params,
            "grads": grads,
        },
        f"{log_dir}/step_{step:08d}.pt",
    )

def collect_gradients(model, input_ids):
    # 确保可复现性
    torch.manual_seed(42)
    
    output = model(input_ids=input_ids)
    
    logits = output.logits
    loss = logits.mean()
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 收集梯度信息
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(f"{name}: {param.grad.norm().item():.6f}")
    
    return grads

def report_meta_tensors(model):
    import torch, inspect
    meta_modules = []
    for mod_name, mod in model.named_modules():
        metas = []
        # 仅检查当前模块自身（不递归）挂载的参数/缓冲
        for n, p in list(mod.named_parameters(recurse=False)):
            if getattr(p, "is_meta", False) and p.is_meta:
                metas.append(("param", n, tuple(p.shape)))
        for n, b in list(mod.named_buffers(recurse=False)):
            if getattr(b, "is_meta", False) and b.is_meta:
                metas.append(("buffer", n, tuple(b.shape)))
        if metas:
            print(f"[META] {mod_name} ({type(mod).__name__}): {metas}")
            meta_modules.append((mod_name, type(mod).__name__, metas))
    return meta_modules

def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path, is_profiler=False):

    torch.autograd.set_detect_anomaly(True) # 在反向传播出错时，PyTorch 会提供更详细的堆栈信息
    
    # tokenizer = AutoTokenizer.from_pretrained('/data/model/Qwen2.5-7B-Instruct', trust_remote_code=True)

    dataset = Dataset.from_json(sft_data_path)

    processed_dataset = dataset.map(lambda  examples: preprocess_function(examples, tokenizer), batched=True)
    split_dataset = processed_dataset.train_test_split(test_size=0.1)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=8)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj"
        ],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = TrainingArguments(
        output_dir=save_adapter_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        # max_steps=4, # TODO: FOR TEST, will override any value given in num_train_epochs
        learning_rate=3e-4,
        fp16=False,
        logging_steps=10,
        save_steps=1000,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )
    
    # model = inject_adapter_in_model(lora_config, model)
    model = get_peft_model(model, lora_config)
    # model = get_custom_peft_model(model, lora_config)

    model.config.use_cache = False

    model.print_trainable_parameters() 
    
    # print(f"model:{model}")
    
    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()
        
    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_cpuinfer_moe_model_graph", format="svg")
    
    # _ = report_meta_tensors(model)
    
    trainer = KTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    trainer.accelerator = Accelerator(device_placement=False)
    # first_batch = next(iter(trainer.get_train_dataloader()))
    # print("Batch keys:", list(first_batch.keys()))

    print("-------------------------START TRAINING!!!-------------------------")

    trainer.train()

    # input_ids = torch.randint(0, 1000, (32, 128), device="cuda:0")
    # gradients = collect_gradients(model, input_ids)
    
    # with open(f"/home/lpl/KT-SFT/tmp/KSFTExpertsCPU_grads.txt", "w") as f:
    #     f.write("\n".join(gradients))
    # print(xx)
    
    # -----------------模型输入数据测试-----------------
    # total_length = 0
    # valid_count = 0
    # for batch in tqdm(train_dataloader):
    #     input_ids = batch['input_ids']
    #     # print(f"Token count per sample: {[len(ids) for ids in input_ids]}")
    #     for ids in input_ids:
    #         if not torch.equal(ids, torch.tensor([100001])):
    #             total_length += len(ids)
    #     valid_count += 1
    #     # print(f"Input tensor: {input_ids}")
    #     # print(f"total_length:{total_length}")
    #     # break

    # if valid_count > 0:
    #     average_length = total_length / valid_count
    #     print(f"平均长度: {average_length}")
    # else:
    #     print("没有有效的 input_ids 元素。")

    # print(xx)
    # -----------------模型输入数据测试-----------------
    
    # -----------------模型FLOPS测试（THOP方法）-----------------
    # 没有继续使用这种方式进行测试，原因在于需要对每个第三方模块进行添加（方法本身不认）。
    # 需要的话可以参考：https://github.com/ultralytics/thop 里面的Define Custom Rules for Third-Party Modules
    # from ktransformers.sft.flops_utils.custom_profile import custom_profile

    # for module in model.modules():
    #     if not hasattr(module, 'total_ops'):
    #         module.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
    #     if not hasattr(module, 'total_params'):
    #         module.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
            
    # # print(f"input:{input}")
    # for inputs in tqdm(train_dataloader):
    #     # input_ids = batch['input_ids']
    #     # del inputs['instruction']
    #     # del inputs['input']
    #     # del inputs['output']
    #     # output = model(**inputs)
    #     model.eval()
    #     content = inputs['instruction'][0] + inputs['input'][0]
    #     # flops,params = custom_profile(model, inputs=inputs, content=content, tokenizer=tokenizer, custom_ops={YourModule: count_your_model})
    #     # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    #     # print('Params = ' + str(params / 1000 ** 2) + 'M')

    #     messages = [{"role": "user", "content": content}]
    #     input_tensor = tokenizer.apply_chat_template(
    #         messages, add_generation_prompt=True, return_tensors="pt"
    #     )
    #     with torch.no_grad():
    #         # model(*inputs)
    #         # model.model to deal with the PeftModelForCaualLM temp
    #         prefill_and_generate(
    #             model.model, tokenizer, input_tensor.cuda(), max_new_tokens=1000, use_cuda_graph=False, mode = 'normal', force_think = False, chunk_prefill_size = 8192,
    #         )
    #     recursive_traverse(model)
    # -----------------模型FLOPS测试（THOP方法）-----------------
    
    # -----------------计算图测试-----------------
    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()
        
    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_cpuinfer_moe_model_graph", format="svg")
    # -----------------计算图测试-----------------

    # -----------------KSFT前向测试-----------------
    # with open("tmp/output_loss_KCPU.txt", "w") as file:
    #     file.write("Output (logits):\n")
    #     file.write(str(output.logits.cpu().detach().numpy()))  # 这里将张量转换为 numpy 数组后写入
    #     file.write("\n\nLoss:\n")
    #     file.write(str(loss.item()))  # 这里将 loss 的值转成字符串
    # -----------------KSFT前向测试-----------------
    
    # -----------------模型层确定性梯度测试-----------------
    # disable_all_dropout(model)

    # def print_dropout_status(module, prefix=""):
    #     for name, child in module.named_children():
    #         if isinstance(child, torch.nn.Dropout):
    #             print(f"{prefix}{name}: p={child.p}, training={child.training}")
    #         print_dropout_status(child, prefix + name + ".")
    
    # print("Dropout层状态验证：") # 空输出或者p=0就是成功验证
    # print_dropout_status(model)

    # for layer_path in target_layers:
    #     module = model.get_submodule(layer_path)
    #     hook = module.register_forward_hook(
    #         lambda m, i, o, ln=layer_path: record_layer_io(m, i, o, ln)
    #     )
    #     hooks.append(hook)
    # -----------------模型层确定性梯度测试-----------------

    
    # -----------------模型层性能初步测试-----------------
    # if is_profiler:
    #     profiler = profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(
    #             wait=1,        # 跳过第1步
    #             warmup=1,      # 预热第2步
    #             active=1,      # 仅记录接下来3步（减少显存占用）
    #             repeat=1       # 不重复
    #         ),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    #         record_shapes=False,
    #         profile_memory=False, # 关闭内存分析，避免占用大量内存（目前这个服务器CPU内存不是很大）
    #         with_stack=False
    #     )

    #     # transformer版本低不支持，不能直接在TrainingArguments里面写profiler_args
    #     # profiler_args = {
    #     #     "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 同时监控CPU和CUDA
    #     #     "record_shapes": True,         # 记录张量形状
    #     #     "profile_memory": True,        # 记录内存消耗
    #     #     "with_stack": True,            # 记录调用栈信息
    #     #     "on_trace_ready": torch.profiler.tensorboard_trace_handler('./logs'),  # 自动保存到TensorBoard
    #     #     "schedule": torch.profiler.schedule(
    #     #         wait=1,        # 跳过前1步
    #     #         warmup=1,      # 预热1步
    #     #         active=100,     # 记录接下来100步（覆盖全部训练步）
    #     #         repeat=1       # 不重复
    #     #     )
    #     # }

    #     trainer = KTrainer(
    #         model=model,
    #         train_dataset=train_dataset,
    #         args=training_args,            # 使用修改后的参数
    #         data_collator=DataCollatorForSeq2Seq(
    #             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #         ),
    #         callbacks=[ProfilerCallback(profiler)]
    #     )

    #     with profiler:
    #         trainer.train()

    #     print("Training finished. Exporting profiler data...")
    #     with open("profiler_output.txt", "w") as f:
    #         f.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    #   profiler.export_chrome_trace("trace.json")
    
    #   check_moe_gradients(model) # 调试结果：无梯度
    
    # -----------------模型层性能初步测试-----------------

    # verify_lora_layers(model)

    # model.save_pretrained(save_adapter_path)

    '''
    ----------------------- START: Lora Test -----------------------
    
    # print(f"LoRA前:{model}")

    # for name, module in model.named_modules():
    #     if "q_proj" in name or "kv_a_proj" in name or "o_proj" in name:
    #         print(name)

    # print_model_params(model)

    # model = KTransformersLinearLora()

    # inspect_device(model, '/home/yj/ktransformers/device1.txt')
    # with open('/home/yj/ktransformers/device1.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 

    # print(f"LoRA后:{model}")

    # model = model.to('cuda')

    # for name, module in model.named_modules():
    #     module.register_forward_hook(forward_hook)

    # for name, parms in model.named_parameters():	
    #     # parms.requires_grad = True
    #     print('-->name:', name)
    #     print('-->para:', parms)
    #     print('-->grad_requirs:',parms.requires_grad)
    #     print('-->grad_fn:',parms.grad_fn)
    #     print('-->grad_value:',parms.grad)
    #     print("===")

    # # 选择特定层的输入输出
    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()

    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_graph", format="svg")

    # inspect_device(model, '/home/yj/ktransformers/device2.txt')
    # with open('/home/yj/ktransformers/device2.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 

    # print_lora_params(model)

    # 被带profile的Trainer替代
    # trainer = KTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     args=transformers.TrainingArguments(
    #         output_dir=save_adapter_path,
    #         per_device_train_batch_size=1,
    #         gradient_accumulation_steps=16,
    #         num_train_epochs=10,
    #         learning_rate=3e-4,
    #         fp16=False,
    #         logging_steps=10,
    #         save_steps=200,
    #         # 可额外添加分布式训练优化参数 
    #         dataloader_drop_last=True,
    #         ddp_find_unused_parameters=False 
    #     ),
    #     data_collator=DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )

    # model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))

    # trainer.train()

    # print_lora_params(model)

    # model = model.merge_and_unload()
    ----------------------- END: Lora Test -----------------------

    '''

def inject_lora_layer(model):

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj"
        ],
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )
    
    model = inject_adapter_in_model(lora_config, model)

    
