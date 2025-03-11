import torch

from peft.config import PeftConfig

from ktransformers.sft.peft_utils.lora_model import LoraModel

def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default", low_cpu_mem_usage: bool = False
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.
    """
    # tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING["LORA"]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = LoraModel(model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    return peft_model.model
