from .modeling import AutoGPTQForCausalLM, BaseQuantizeConfig, LlamaGPTQForCausalLM, OPTGPTQForCausalLM, OlmoeGPTQForCausalLM
from .utils.exllama_utils import exllama_set_max_input_length


def get_gptq_peft_model(*args, **kwargs):
    from .utils.peft_utils import get_gptq_peft_model as _get_gptq_peft_model

    return _get_gptq_peft_model(*args, **kwargs)


__version__ = "0.8.0.dev0"
