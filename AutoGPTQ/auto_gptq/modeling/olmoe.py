from ._base import BaseGPTQForCausalLM


class OlmoeGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "OlmoeDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate"],
        [f"mlp.experts.{i}.gate_proj" for i in range(64)] + \
        [f"mlp.experts.{i}.up_proj" for i in range(64)],
        [f"mlp.experts.{i}.down_proj" for i in range(64)],
    ]

__all__ = ["OlmoeGPTQForCausalLM"]


