import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from ktransformers.sft.peft_utils.mapping import inject_adapter_in_model
# from ktransformers.sft.load_lora import get_custom_peft_model
import os

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

def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path):
    
    # tokenizer = AutoTokenizer.from_pretrained('/data/model/Qwen2.5-7B-Instruct', trust_remote_code=True)

    dataset = Dataset.from_json(sft_data_path)

    # processed_dataset = dataset.map(preprocess_function, batched=True)
    processed_dataset = dataset.map(lambda  examples: preprocess_function(examples, tokenizer), batched=True)
    split_dataset = processed_dataset.train_test_split(test_size=0.1)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=8)

    print(f"LoRA前:{model}")

    for name, module in model.named_modules():
        if "q_proj" in name or "kv_a_proj" in name or "o_proj" in name:
            print(name)

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
    
    model = inject_adapter_in_model(lora_config, model)
    # model = get_peft_model(model, lora_config)
    # model = get_custom_peft_model(model, lora_config)

    # inspect_device(model, '/home/yj/ktransformers/device1.txt')
    # with open('/home/yj/ktransformers/device1.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 

    print(f"LoRA后:{model}")

    # model = model.to('cuda')
    model.config.use_cache = False

    # inspect_device(model, '/home/yj/ktransformers/device2.txt')
    # with open('/home/yj/ktransformers/device2.txt', 'a') as file:
    #     file.write(f"Base model device: {model.base_model.device}\n")
        # file.write(f"LoRA adapter device: {model.lora_config['target_modules'].device}\n")
    # print(f"Base model device: {model.base_model.device}") 
    # print(f"LoRA adapter device: {model.lora_config['target_modules'].device}") 
    
    os.environ["NCCL_P2P_DISABLE"]  = "1"
    os.environ["NCCL_IB_DISABLE"]  = "1"

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            output_dir=save_adapter_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=10,
            learning_rate=3e-4,
            fp16=True,
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

    # trainer.train()


    model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))

    model.save_pretrained(save_adapter_path)


    model.print_trainable_parameters() 

    model = model.merge_and_unload()