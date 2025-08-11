import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_torch_hpu_available,
    is_accelerate_available,
    is_apex_available,
    logging,
)
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers import TrainerCallback
from packaging import version
import os
import inspect
import functools
from typing import Union, Any

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType
from datasets import Dataset
from torchviz import make_dot
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import torch.nn as nn
import gc
from tqdm import tqdm
import os, torch, json, tempfile
from pathlib import Path
from accelerate import Accelerator
if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration
from accelerate import __version__ as accelerate_version
if is_apex_available():
    from apex import amp
if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
if is_sagemaker_mp_enabled():
    from transformers.trainer_utils import smp_forward_backward

from ktransformers.sft.peft_utils.mapping import inject_adapter_in_model, get_peft_model
from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
from ktransformers.sft.flops_utils.custom_profile import custom_profile
from ktransformers.operators.experts import KExpertsTorch, KTransformersExperts
# from ktransformers.sft.load_lora import get_custom_peft_model

logger = logging.get_logger(__name__)

# FOR: not A or H GPU
os.environ["NCCL_P2P_DISABLE"]  = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"
os.environ["KT_DEBUG_MOE"] = "1"

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
    
class KAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("device_placement", False)
        super().__init__(*args, **kwargs)
        
    def prepare_model(self, model, *args, **kwargs):
        return model
    
    def prepare(self, *args, **kwargs):
        prepped = []
        for obj in args:
            if isinstance(obj, nn.Module):
                prepped.append(self.prepare_model(obj, **kwargs))
            else:
                prepped.append(super().prepare(obj, **kwargs))
        return tuple(prepped) if len(prepped) > 1 else prepped[0]

class KTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # only save LoRA adapter（include adapter_config.json）
        self.model.save_pretrained(output_dir)
        
    def _move_model_to_device(self, model, device):
        print("[KTrainer] Due to the placement feature in KTransformers, skip moving model to", device)
        return model
    
    # 禁止 Trainer 在 n_gpu>1 时套 DataParallel
    def _wrap_model(self, model, training=True, dataloader=None):
        self.model_wrapped = model
        return model
    
    def create_accelerator_and_postprocess(self):
        # We explicitly don't rely on the `Accelerator` to do gradient accumulation
        grad_acc_kwargs = {}
        if is_accelerate_available("0.28.0") and self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        accelerator_config = self.args.accelerator_config.to_dict()

        if is_accelerate_available("0.28.0"):
            # Extract dataloader config params from accelerator config
            dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
            dataloader_config_dict = {param: accelerator_config.pop(param) for param in dataloader_params if param in accelerator_config}
            if DataLoaderConfiguration is None:
                raise ImportError("Your accelerate does not provide DataLoaderConfiguration but Trainer expects it.")
            dataloader_config = DataLoaderConfiguration(**dataloader_config_dict)
            if is_accelerate_available("1.1.0"):
                dataloader_config.data_seed = self.args.data_seed
        else:
            dataloader_config = None

        non_blocking = accelerator_config.pop("non_blocking", False)
        if not is_accelerate_available("0.30.0"):
            if non_blocking:
                raise ImportError(
                    "`non_blocking` is only supported in accelerate v0.30.0 and above. Please upgrade accelerate to use this feature."
                )
        else:
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning("`non_blocking` is enabled but `dataloader_pin_memory` is not. For best performance, enable both.")
            if dataloader_config is not None:
                dataloader_config.non_blocking = non_blocking

        accelerator_config.pop("gradient_accumulation_kwargs", None)

        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "device_placement": False,
        }

        if is_accelerate_available("0.28.0"):
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)

        if getattr(self.args, "tp_size", 1) > 1:
            self.is_tp_enabled = True
            if version.parse(accelerate_version) > version.parse("1.3.0") and TorchTensorParallelPlugin is not None:
                args["torch_tp_plugin"] = TorchTensorParallelPlugin(tp_size=self.args.tp_size)
            else:
                raise ValueError("Requires accelerate>1.3.0 to use Tensor Parallelism.")

        self.accelerator = KAccelerator(**args)

        try:
            self.accelerator.state.device_ids = [0]
            self.accelerator.state.num_processes = 1
            self.accelerator.state.num_gpus = 1
        except Exception:
            pass

        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # `auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3
        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
            )
        if (
            self.args.save_only_model
            and self.is_fsdp_enabled
            and "SHARDED_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError("save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'")
        
        if dataloader_config is not None:
            dataloader_config.split_batches = False
            dataloader_config.dispatch_batches = False
            dataloader_config.even_batches = False
            
    # ★ 核心修正：训练 DataLoader 的 batch_size 固定用 per_device，不乘 n_gpu
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with per_device_train_batch_size
        (no implicit multipliers by number of visible GPUs).
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # 与原生一致：基于 datasets 的移除无用列；否则包一层剔列的 collator
        if is_datasets_available():
            try:
                import datasets  # 仅用于 isinstance 检查
                if isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
                else:
                    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
            except Exception:
                # datasets 不可用或版本不兼容时，退化到剔列 collator
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # 这里与原生不同：batch_size 用 per_device，不用 self._train_batch_size
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,   # ★ 不乘 n_gpu
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # 非 IterableDataset 时，补充 sampler / drop_last / worker_init_fn / prefetch_factor
        if not isinstance(train_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            # 仅当 num_workers>0 且设置了 prefetch_factor 时才传（与 torch DataLoader 要求一致）
            if self.args.dataloader_num_workers > 0 and self.args.dataloader_prefetch_factor is not None:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dl = DataLoader(train_dataset, **dataloader_params)

        # 为了完全显式，告诉 Accelerate 不要做 device_placement
        try:
            prepared = self.accelerator.prepare(dl, device_placement=[False])
        except TypeError:
            # 某些 accelerate 版本没有 device_placement 参数，直接 prepare
            prepared = self.accelerator.prepare(dl)

        return prepared
    
    # === 训练步：与原生一致，唯一改动是最后返回时把 loss 挪到 self.args.device ===
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # ★ 关键：保留原生的数据准备（会把 batch 张量放到 self.args.device，
        #  你的自定义算子/替换模块很多是据此决定内部流向的）
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            # ★ 返回值放到 args.device，直接满足 HF 的设备检查
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # 与原生一致的上下文（amp/autocast 等）
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # 释放 batch
        del inputs

        # 原生的 empty_cache 步骤（照抄）
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # LOMO 需要学习率
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # 多卡数据并行情况下做均值（你现在是模型并行，但保持兼容）
        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Apex/amp 路径
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:  # type: ignore
                scaled_loss.backward()
        else:
            # 与原生一致：当 loss 不是用户自定义计算时，按梯度累积步数缩放
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # DeepSpeed 关闭 gas 缩放
            if getattr(self.accelerator, "distributed_type", None) and \
               str(self.accelerator.distributed_type) == "DistributedType.DEEPSPEED":
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

        # ★ 唯一改动：返回给 Trainer 的 loss 必须在 self.args.device
        ret = loss.detach()
        if ret.device != self.args.device:
            ret = ret.to(self.args.device, non_blocking=True)

        # 一次性调试（可开 `KT_DBG_STEP=1` 查看）
        if os.environ.get("KT_DBG_STEP", "0") == "1" and not hasattr(self, "_kt_dbg_once"):
            try:
                print(f"[KT-DBG] args.device={self.args.device}  loss(before)={loss.device}  loss(return)={ret.device}")
            except Exception:
                pass
            self._kt_dbg_once = True

        return ret

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
        target_modules=[ # TODO: 这里需要写入到shell里面，每个模型的template是不一样的
            # "q_proj", # FOR DeepSeek-V2-Lite
            "q_a_proj", # FOR DeepSeek-V3&R1
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
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
    # return
    
    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()
        
    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_cpuinfer_moe_model_graph", format="svg")
    
    # _ = report_meta_tensors(model)
    
    print("=== SAMPLE INSPECT ===")
    for i in range(2):
        ex = train_dataset[i]  # HF datasets 的单条样本（已经过 preprocess_function）
        summary = {}
        for k,v in ex.items():
            if isinstance(v, list):
                if len(v)>0 and isinstance(v[0], list):
                    summary[k] = f"list-of-lists len={len(v)} x len0={len(v[0])}"
                else:
                    summary[k] = f"list len={len(v)}"
            elif torch.is_tensor(v):
                summary[k] = f"tensor shape={tuple(v.shape)}"
            else:
                summary[k] = str(type(v))
        print(f"[SAMPLE {i}]", summary)
    
    trainer = KTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    # trainer.accelerator = Accelerator(device_placement=False)
    # first_batch = next(iter(trainer.get_train_dataloader()))
    # print("Batch keys:", list(first_batch.keys()))
    
    # acc = KAccelerator(device_placement=False)
    # acc.state.device_ids = [0]
    # acc.state.num_processes = 1
    # acc.state.num_gpus = 1
    # trainer.accelerator = acc

    print("Accelerator device_ids:", trainer.accelerator.state.device_ids)
    print(f"type(trainer.model):{type(trainer.model)}")
    print(f"type(trainer.accelerator):{type(trainer.accelerator)}")
    
    _ = trainer._wrap_model(trainer.model, training=True)
    assert not isinstance(trainer.model, nn.DataParallel), "Model was wrapped with DataParallel unexpectedly"
    
    print("WRAP FUNC:", KTrainer._wrap_model is Trainer._wrap_model)   # 应为 False
    print("IS DP:", isinstance(trainer.model, nn.DataParallel))         # 应为 False
    print("IS DP WRAPPED:", isinstance(getattr(trainer, "model_wrapped", None), nn.DataParallel))  # 应为 False
    
    print("-------------------------START TRAINING!!!-------------------------")

    cfg = getattr(trainer.accelerator, "dataloader_config", None)
    print(
        "[ACCEL DL CONFIG]",
        "split_batches=", getattr(cfg, "split_batches", None),
        "dispatch_batches=", getattr(cfg, "dispatch_batches", None),
        "even_batches=", getattr(cfg, "even_batches", None),
        "use_seedable_sampler=", getattr(cfg, "use_seedable_sampler", None),
        "non_blocking=", getattr(cfg, "non_blocking", None),
    )
    print("--------------------NEW DEBUG--------------------")
    # install_shape_probes(trainer.model) # print some debug info about multi-gpu placement.
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

    
