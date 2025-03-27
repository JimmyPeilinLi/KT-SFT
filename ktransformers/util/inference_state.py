# 不然util/utils.py: InferenceState和sft/peft_utils/lora_layer.py:KTLinearLora循环引用

import enum


class InferenceState(enum.Enum):
    UNLOAD = 0
    PREFILL = 1
    GENERATE = 2
    RESTORE = 3
