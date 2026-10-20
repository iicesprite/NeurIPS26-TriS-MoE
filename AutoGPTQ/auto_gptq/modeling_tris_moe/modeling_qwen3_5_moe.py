"""Qwen3.5-MoE text modeling with expanded experts for TriSMoE grouped runtime."""

import os

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeDecoderLayer as HFQwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM as HFQwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextModel as HFQwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
    load_balancing_loss_func,
)

try:
    from .grouped_payload_sidecar import maybe_create_grouped_qgmap_cpu_staging_manager
except ImportError:
    maybe_create_grouped_qgmap_cpu_staging_manager = None


def _env_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    return False


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Qwen3.5 MoE block exposing per-expert Linear modules instead of HF 3D tensors."""

    def __init__(self, config: Qwen3_5MoeTextConfig):
        nn.Module.__init__(self)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = True
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3_5MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_dim, 1, bias=False)
        self._grouped_payload_cache = {}
        self._deployment_only_active = False
        self._grouped_payload_prefix = None
        self._grouped_payload_model_dir = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)

    def _grouped_enabled(self) -> bool:
        for name in (
            "QWEN35MOE_GROUPED_EXPERTS",
            "QWEN3_5MOE_GROUPED_EXPERTS",
            "QWEN_GROUPED_EXPERTS",
            "TRIS_GROUPED_EXPERTS",
        ):
            value = os.environ.get(name)
            if value is not None:
                return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
        return True

    def _grouped_payload_cache_enabled(self) -> bool:
        return self._deployment_only_active or _env_flag_enabled(
            "QWEN35MOE_GROUPED_PAYLOAD_CACHE",
            "QWEN3_5MOE_GROUPED_PAYLOAD_CACHE",
            "QWEN_GROUPED_PAYLOAD_CACHE",
            "TRIS_GROUPED_PAYLOAD_CACHE",
        )

    def _grouped_empty_cache_enabled(self) -> bool:
        return _env_flag_enabled(
            "QWEN35MOE_GROUPED_EMPTY_CACHE",
            "QWEN3_5MOE_GROUPED_EMPTY_CACHE",
            "QWEN_GROUPED_EMPTY_CACHE",
            "TRIS_GROUPED_EMPTY_CACHE",
        )

    def _grouped_qgmap_staging_active_for_cache_key(self, cache_key: str) -> bool:
        if cache_key not in {"gate", "up", "down"}:
            return False
        if not _env_flag_enabled(
            "QWEN35MOE_GROUPED_QGMAP_CPU_STAGING",
            "QWEN3_5MOE_GROUPED_QGMAP_CPU_STAGING",
            "QWEN_GROUPED_QGMAP_CPU_STAGING",
            "TRIS_GROUPED_QGMAP_CPU_STAGING",
        ):
            return False
        sidecar_path = self._get_grouped_qgmap_staging_sidecar_path()
        return sidecar_path is not None and os.path.exists(sidecar_path)

    def preload_grouped_payloads_from_reader(self, checkpoint_reader) -> None:
        return None

    def _get_grouped_sidecar_path(self):
        return None

    def _get_grouped_qgmap_staging_sidecar_path(self):
        return None

    def _require_deployment_only(self, label: str, reason: str | None = None):
        message = f"[Qwen3.5-MoE deployment-only] {self.__class__.__name__}.{label}"
        if reason:
            message = f"{message}: {reason}"
        raise RuntimeError(message)

    def enable_grouped_deployment_only(self):
        if _env_flag_enabled("TRIS_DEPLOYMENT_ONLY", "OLMOE_DEPLOYMENT_ONLY"):
            self._require_deployment_only(
                "enable_grouped_deployment_only",
                "Qwen3.5 grouped deployment payload binding is added in the AutoGPTQ registration phase",
            )

    def _forward_expert_loop(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[1]), dtype=output_dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states.index_select(0, top_x)
            current_hidden_states = expert_layer(current_state)
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(output_dtype))

        return final_hidden_states

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self._deployment_only_active:
            self._require_deployment_only("_forward_grouped", "grouped CUDA runtime is not initialized")
        return None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        output_dtype = hidden_states.dtype

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self._forward_grouped(
            hidden_states,
            routing_weights,
            selected_experts,
            output_dtype,
        )
        if final_hidden_states is None:
            final_hidden_states = self._forward_expert_loop(
                hidden_states,
                routing_weights,
                selected_experts,
                output_dtype,
            )

        shared_expert_output = self.shared_expert(hidden_states).to(output_dtype)
        shared_expert_gate = torch.sigmoid(self.shared_expert_gate(hidden_states)).to(output_dtype)
        final_hidden_states = final_hidden_states + shared_expert_gate * shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen3_5MoeDecoderLayer(HFQwen3_5MoeDecoderLayer):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.mlp._grouped_payload_prefix = f"model.layers.{layer_idx}.mlp"


class Qwen3_5MoeTextModel(HFQwen3_5MoeTextModel):
    config: Qwen3_5MoeTextConfig

    def __init__(self, config: Qwen3_5MoeTextConfig):
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._grouped_qgmap_cpu_staging_manager = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Qwen3_5MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_router_logits = () if kwargs.get("output_router_logits", False) else None
        grouped_qgmap_staging_manager = self._grouped_qgmap_cpu_staging_manager
        if maybe_create_grouped_qgmap_cpu_staging_manager is not None:
            if grouped_qgmap_staging_manager is None or not grouped_qgmap_staging_manager.matches_device(inputs_embeds.device):
                grouped_qgmap_staging_manager = maybe_create_grouped_qgmap_cpu_staging_manager(self.layers)
                self._grouped_qgmap_cpu_staging_manager = grouped_qgmap_staging_manager

        try:
            for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                layer_mask = linear_attn_mask if self.config.layer_types[layer_idx] == "linear_attention" else causal_mask
                if grouped_qgmap_staging_manager is not None:
                    grouped_qgmap_staging_manager.prepare_for_layer(layer_idx)
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    if all_router_logits is not None and len(layer_outputs) > 1 and layer_outputs[-1] is not None:
                        all_router_logits += (layer_outputs[-1],)
                else:
                    hidden_states = layer_outputs
        finally:
            if grouped_qgmap_staging_manager is not None:
                grouped_qgmap_staging_manager.finalize_forward()

        hidden_states = self.norm(hidden_states)

        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )


class Qwen3_5MoeForCausalLM(HFQwen3_5MoeForCausalLM):
    config: Qwen3_5MoeTextConfig

    def __init__(self, config):
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3_5MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> Qwen3_5MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits and outputs.router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoePreTrainedModel",
    "Qwen3_5MoeSparseMoeBlock",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeTextModel",
]
