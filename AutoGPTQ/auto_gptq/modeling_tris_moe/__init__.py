from .configuration_olmoe import OlmoeConfig
from .modeling_olmoe import OlmoeForCausalLM, OlmoeModel, OlmoePreTrainedModel

__all__ = [
    "OlmoeConfig",
    "OlmoeForCausalLM",
    "OlmoeModel",
    "OlmoePreTrainedModel",
]

try:
    from .modeling_mixtral import (
        MixtralConfig,
        MixtralForCausalLM,
        MixtralModel,
        MixtralPreTrainedModel,
        MixtralSparseMoeBlock,
    )
except ImportError:
    pass
else:
    __all__[:0] = [
        "MixtralConfig",
        "MixtralForCausalLM",
        "MixtralModel",
        "MixtralPreTrainedModel",
        "MixtralSparseMoeBlock",
    ]

try:
    from .modeling_qwen2_moe import (
        Qwen2MoeConfig,
        Qwen2MoeForCausalLM,
        Qwen2MoeModel,
        Qwen2MoePreTrainedModel,
        Qwen2MoeSparseMoeBlock,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "Qwen2MoeConfig",
            "Qwen2MoeForCausalLM",
            "Qwen2MoeModel",
            "Qwen2MoePreTrainedModel",
            "Qwen2MoeSparseMoeBlock",
        ]
    )

try:
    from .modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoePreTrainedModel,
        Qwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoePreTrainedModel",
            "Qwen3_5MoeSparseMoeBlock",
            "Qwen3_5MoeTextConfig",
            "Qwen3_5MoeTextModel",
        ]
    )
