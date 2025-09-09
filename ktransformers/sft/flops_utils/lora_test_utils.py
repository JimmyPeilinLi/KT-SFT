from torch.profiler import profile, record_function, ProfilerActivity
import os
from transformers import TrainerCallback

class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

def _short(t):
    return tuple(t.shape) if isinstance(t, torch.Tensor) else type(t)

def install_shape_probes(model):
    if os.environ.get("KT_DEBUG_MOE","0") != "1":
        print("[KT_DEBUG_MOE] off"); return

    # 0) 打印 DataLoader 配置你已经有了，这里再贴一遍保险
    try:
        acc = trainer.accelerator
        cfg = getattr(acc, "dataloader_config", None)
        if cfg is not None:
            print("[ACCEL DL CONFIG]",
                  "split_batches=", getattr(cfg,"split_batches",None),
                  "dispatch_batches=", getattr(cfg,"dispatch_batches",None),
                  "even_batches=", getattr(cfg,"even_batches",None),
                  "use_seedable_sampler=", getattr(cfg,"use_seedable_sampler",None),
                  "non_blocking=", getattr(cfg,"non_blocking",None))
    except Exception as e:
        print("[ACCEL DL CONFIG] <err>", e)

    # 1) 在 embed_tokens 入口打印 input_ids 的 (B,S)
    try:
        emb = model.base_model.model.model.embed_tokens
        def _emb_pre(mod, inp):
            x = inp[0]
            if not hasattr(mod, "_dbg_once"):
                print(f"[DBG] embed input_ids shape = {tuple(x.shape)}  (expect B,S)")
                mod._dbg_once = True
        emb.register_forward_pre_hook(_emb_pre)
    except Exception as e:
        print("[DBG] embed hook failed:", e)

    # 2) 在第 0 个 decoder layer 入口/出口打印 hidden_states 形状
    try:
        first_layer = model.base_model.model.model.layers[0]
        _orig_fwd = first_layer.forward
        def _wrap_fwd(self, *args, **kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if not hasattr(self, "_dbg_once_in"):
                print(f"[DBG] L0.in hidden_states = {_short(hs)}  (expect B,S,H)")
                self._dbg_once_in = True
            out = _orig_fwd(*args, **kwargs)
            hs_out = out[0] if isinstance(out, (tuple, list)) else out
            if not hasattr(self, "_dbg_once_out"):
                print(f"[DBG] L0.out hidden_states = {_short(hs_out)}")
                self._dbg_once_out = True
            return out
        first_layer.forward = MethodType(_wrap_fwd, first_layer)
    except Exception as e:
        print("[DBG] L0 wrap failed:", e)

    # 3) 在一个 MoE 层的 mlp 入口打印（DeepseekV3 的 mlp 是 MoE）
    try:
        # 找到第一个含 MoE 的层（名字可能是 .mlp 或你自定义类）
        moe_layer = None
        for i, lyr in enumerate(model.base_model.model.model.layers):
            if hasattr(lyr, "mlp"):
                moe_layer = lyr.mlp
                moe_idx = i
                break
        if moe_layer is not None:
            _moe_orig = moe_layer.forward
            def _moe_wrap(self, *args, **kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                if not hasattr(self, "_dbg_once"):
                    # 这里很多 MoE 实现会把 (B,S,H) 展平成 (B*S,H) 再做 gate
                    print(f"[DBG] MLP(in) @layer{moe_idx} hidden_states = {_short(x)}")
                    if isinstance(x, torch.Tensor) and x.dim() == 3:
                        B,S,H = x.shape
                        print(f"[DBG] tokens before flatten = B*S = {B}*{S} = {B*S}")
                    self._dbg_once = True
                return _moe_orig(*args, **kwargs)
            moe_layer.forward = MethodType(_moe_wrap, moe_layer)
        else:
            print("[DBG] no moe_layer found")
    except Exception as e:
        print("[DBG] moe wrap failed:", e)

    # 4) 在 KTransformersExperts 入口打印 (N, …) 与 expert_ids/weights 形状（你之前装过，这里更完整）
    try:
        from ktransformers.operators.experts import KTransformersExperts
        def _experts_pre(mod, args):
            if hasattr(mod, "_dbg_once"): return
            try:
                input_tensor, expert_ids, weights = args[:3]
                print(f"[DBG] experts.in input_tensor={tuple(input_tensor.shape)} "
                      f"expert_ids={tuple(expert_ids.shape)} weights={tuple(weights.shape)}")
                if input_tensor.dim()==2:
                    N = input_tensor.shape[0]
                    print(f"[DBG] N(input rows)={N}")
                if expert_ids.dim()==2:
                    T,K = expert_ids.shape
                    print(f"[DBG] tokens(T)={T}, K={K}, T*K={T*K}")
                mod._dbg_once = True
            except Exception as e:
                print("[DBG] experts hook parse err:", e)
        count=0
        for name,m in model.named_modules():
            if isinstance(m, KTransformersExperts):
                m.register_forward_pre_hook(_experts_pre); count+=1
        print(f"[KT_DEBUG_MOE] installed experts hook on {count} modules.")
    except Exception as e:
        print("[DBG] experts hook failed:", e)

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
            if isinstance(child, nn.Dropout):
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
    model: nn.Module,
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

# def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path, is_profiler=False):
    # show some lora test
    
    '''
    # multi-gpu dataloader test
    # _ = report_meta_tensors(model)
    
    # print("=== SAMPLE INSPECT ===")
    # for i in range(2):
    #     ex = train_dataset[i]  # HF datasets 的单条样本（已经过 preprocess_function）
    #     summary = {}
    #     for k,v in ex.items():
    #         if isinstance(v, list):
    #             if len(v)>0 and isinstance(v[0], list):
    #                 summary[k] = f"list-of-lists len={len(v)} x len0={len(v[0])}"
    #             else:
    #                 summary[k] = f"list len={len(v)}"
    #         elif torch.is_tensor(v):
    #             summary[k] = f"tensor shape={tuple(v.shape)}"
    #         else:
    #             summary[k] = str(type(v))
    #     print(f"[SAMPLE {i}]", summary)
    
    # trainer.accelerator = Accelerator(device_placement=False)
    # first_batch = next(iter(trainer.get_train_dataloader()))
    # print("Batch keys:", list(first_batch.keys()))
    
    # acc = KAccelerator(device_placement=False)
    # acc.state.device_ids = [0]
    # acc.state.num_processes = 1
    # acc.state.num_gpus = 1
    # trainer.accelerator = acc

    # print("Accelerator device_ids:", trainer.accelerator.state.device_ids)
    # print(f"type(trainer.model):{type(trainer.model)}")
    # print(f"type(trainer.accelerator):{type(trainer.accelerator)}")
    
    # print("WRAP FUNC:", KTrainer._wrap_model is Trainer._wrap_model)   # 应为 False
    # print("IS DP:", isinstance(trainer.model, nn.DataParallel))         # 应为 False
    # print("IS DP WRAPPED:", isinstance(getattr(trainer, "model_wrapped", None), nn.DataParallel))  # 应为 False
    
    # print("-------------------------START TRAINING!!!-------------------------")

    # cfg = getattr(trainer.accelerator, "dataloader_config", None)
    # print(
    #     "[ACCEL DL CONFIG]",
    #     "split_batches=", getattr(cfg, "split_batches", None),
    #     "dispatch_batches=", getattr(cfg, "dispatch_batches", None),
    #     "even_batches=", getattr(cfg, "even_batches", None),
    #     "use_seedable_sampler=", getattr(cfg, "use_seedable_sampler", None),
    #     "non_blocking=", getattr(cfg, "non_blocking", None),
    # )
    # print("--------------------NEW DEBUG--------------------")
    # install_shape_probes(trainer.model) # print some debug info about multi-gpu placement.

    # input_ids = torch.randint(0, 1000, (32, 128), device="cuda:0")
    # gradients = collect_gradients(model, input_ids)
    '''
    
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
    #         if isinstance(child, nn.Dropout):
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