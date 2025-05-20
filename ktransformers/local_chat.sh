#!/bin/bash

python3 ktransformers/local_chat.py \
    --model_path "/mnt/data/models/DeepSeek-V2-Lite-Chat" \
    --model_config_path "/home/lpl/KT-SFT/ktransformers/configs/model_config" \
    --gguf_path "/mnt/data/models/DeepSeek-V2-Lite-Chat-GGUF-FP16/" \
    --cpu_infer 32 \
    --max_new_tokens 1000 \
    --force_think \
    --optimize_config_path "ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml" \
    --is_sft True \
    --sft_data_path "/home/lpl/KT-SFT/test_adapter/sft_translation.json" \
    --save_adapter_path "/home/lpl/KT-SFT/test_adapter/demo_adapter_KT_target_kv" \
    --use_adapter False \
    --use_adapter_path "/home/lpl/KT-SFT/test_adapter/demo_adapter_origin_target_kv"
