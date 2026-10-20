from ._base import BaseGPTQForCausalLM


class Qwen2MoeGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Qwen2MoeDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate", "mlp.shared_expert_gate"],
        [f"mlp.experts.{i}.gate_proj" for i in range(60)] + [f"mlp.experts.{i}.up_proj" for i in range(60)],
        [f"mlp.experts.{i}.down_proj" for i in range(60)],
        ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"],
        ["mlp.shared_expert.down_proj"],
    ]


__all__ = ["Qwen2MoeGPTQForCausalLM"]
