# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OLMoE model."""

import math
import os
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, logging
from .configuration_olmoe import OlmoeConfig

try:
    from auto_gptq.nn_modules.qlinear.qlinear_cuda import (
        build_grouped_ns_compensation_payload,
        build_grouped_precomputed_payload,
        build_grouped_route,
        run_grouped_structured_expert_forward,
        run_grouped_structured_expert_pair_forward,
    )
except ImportError:
    build_grouped_ns_compensation_payload = None
    build_grouped_precomputed_payload = None
    build_grouped_route = None
    run_grouped_structured_expert_forward = None
    run_grouped_structured_expert_pair_forward = None

try:
    from .grouped_payload_sidecar import (
        GROUPED_QGMAP_PAYLOAD_NAMES,
        default_grouped_payload_sidecar_path,
        default_grouped_qgmap_staging_sidecar_path,
        load_grouped_payload_from_sidecar,
        maybe_create_grouped_qgmap_cpu_staging_manager,
    )
except ImportError:
    GROUPED_QGMAP_PAYLOAD_NAMES = frozenset()
    default_grouped_payload_sidecar_path = None
    default_grouped_qgmap_staging_sidecar_path = None
    load_grouped_payload_from_sidecar = None
    maybe_create_grouped_qgmap_cpu_staging_manager = None


if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


def _env_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    return False


def _deployment_only_enabled() -> bool:
    return _env_flag_enabled("OLMOE_DEPLOYMENT_ONLY", "TRIS_DEPLOYMENT_ONLY")


def _raise_deployment_only_failure(label: str, reason: Optional[str] = None):
    message = f"[OLMoE deployment-only] {label}"
    if reason:
        message = f"{message}: {reason}"
    raise RuntimeError(message)


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, routing_weights.shape[1]))
            .reshape(-1, routing_weights.shape[1])
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    rank = routing_weights.shape[1] * int(routing_weights.device.index)
    overall_loss = torch.sum(
        tokens_per_expert[:, rank : rank + routing_weights.shape[1]] * router_prob_per_expert.unsqueeze(0)
    )
    return overall_loss * num_experts


class OlmoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        OlmoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Olmoe
class OlmoeRotaryEmbedding(nn.Module):
    def __init__(self, config: OlmoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.olmo.modeling_olmo.OlmoMLP with Olmo->Olmoe
class OlmoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class OlmoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OlmoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.q_norm = OlmoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(
            (self.hidden_size // self.num_heads) * self.num_key_value_heads, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class OlmoeFlashAttention2(OlmoeAttention):
    """
    OLMoE flash attention module. This module inherits from `OlmoeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)
        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (OlmoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        device_type = query_states.device.type if query_states.device.type != "mps" else "cpu"
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = (
                    torch.get_autocast_dtype(device_type)
                    if hasattr(torch, "get_autocast_dtype")
                    else torch.get_autocast_gpu_dtype()
                )
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class OlmoeSdpaAttention(OlmoeAttention):
    """
    OLMoE attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `OlmoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from OlmoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "OlmoeModel is using OlmoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = causal_mask is None and q_len > 1

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


OLMOE_ATTENTION_CLASSES = {
    "eager": OlmoeAttention,
    "flash_attention_2": OlmoeFlashAttention2,
    "sdpa": OlmoeSdpaAttention,
}


class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])
        self._grouped_payload_cache = {}
        self._deployment_only_active = False
        self._grouped_payload_prefix = None
        self._grouped_payload_model_dir = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)

    def _grouped_enabled(self) -> bool:
        for name in ("OLMOE_GROUPED_EXPERTS", "TRIS_GROUPED_EXPERTS"):
            value = os.environ.get(name)
            if value is not None:
                return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
        return True

    def _grouped_payload_cache_enabled(self) -> bool:
        return self._deployment_only_active or os.getenv("OLMOE_GROUPED_PAYLOAD_CACHE", "0") != "0"

    def _grouped_empty_cache_enabled(self) -> bool:
        return os.getenv("OLMOE_GROUPED_EMPTY_CACHE", "0") != "0"

    def _payload_device_key(self, device: torch.device):
        return (device.type, device.index if device.index is not None else -1)

    def _cache_grouped_payload(self, cache_key: str, payload, device: torch.device, dtype: torch.dtype):
        self._grouped_payload_cache[cache_key] = {
            "device": self._payload_device_key(device),
            "dtype": dtype,
            "payload": payload,
        }
        return payload

    def _get_grouped_sidecar_path(self):
        if default_grouped_payload_sidecar_path is None:
            return None
        candidates = []
        model_dir = self._grouped_payload_model_dir
        if model_dir and os.path.isdir(model_dir):
            candidates.append(default_grouped_payload_sidecar_path(model_dir))
        candidates.append(default_grouped_payload_sidecar_path(os.path.dirname(__file__)))
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return candidates[0] if candidates else None

    def _get_grouped_qgmap_staging_sidecar_path(self):
        candidates = []
        if default_grouped_qgmap_staging_sidecar_path is not None:
            model_dir = self._grouped_payload_model_dir
            if model_dir and os.path.isdir(model_dir):
                candidates.append(default_grouped_qgmap_staging_sidecar_path(model_dir))
            candidates.append(default_grouped_qgmap_staging_sidecar_path(os.path.dirname(__file__)))
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    return candidate

        grouped_sidecar_path = self._get_grouped_sidecar_path()
        if grouped_sidecar_path and os.path.exists(grouped_sidecar_path):
            return grouped_sidecar_path
        return candidates[0] if candidates else grouped_sidecar_path

    def _grouped_qgmap_staging_active_for_cache_key(self, cache_key: str) -> bool:
        if cache_key not in {"gate", "up", "down"}:
            return False
        if not _env_flag_enabled("OLMOE_GROUPED_QGMAP_CPU_STAGING", "TRIS_GROUPED_QGMAP_CPU_STAGING"):
            return False
        sidecar_path = self._get_grouped_qgmap_staging_sidecar_path()
        return sidecar_path is not None and os.path.exists(sidecar_path)

    def _load_grouped_payload_from_sidecar(self, cache_key: str, device: torch.device, dtype: torch.dtype):
        if load_grouped_payload_from_sidecar is None or not self._grouped_payload_prefix:
            return None
        sidecar_path = self._get_grouped_sidecar_path()
        if sidecar_path is None:
            return None
        exclude_names = GROUPED_QGMAP_PAYLOAD_NAMES if self._grouped_qgmap_staging_active_for_cache_key(cache_key) else None
        payload = load_grouped_payload_from_sidecar(
            sidecar_path,
            self._grouped_payload_prefix,
            cache_key,
            device,
            dtype,
            exclude_names=exclude_names,
        )
        if payload is None:
            return None
        return self._cache_grouped_payload(cache_key, payload, device, dtype)

    def _require_deployment_only(self, label: str, reason: Optional[str] = None):
        _raise_deployment_only_failure(f"{self.__class__.__name__}.{label}", reason)

    def _infer_grouped_payload_device_dtype(self):
        if len(self.experts) == 0:
            self._require_deployment_only("enable_grouped_deployment_only", "no experts available")
        reference = self.experts[0].gate_proj
        return reference.fp_weight.device, reference.fp_weight.dtype

    def _build_grouped_payload_or_raise(self, cache_key: str, modules, device: torch.device, dtype: torch.dtype, builder=None):
        payload_builder = build_grouped_precomputed_payload if builder is None else builder
        if payload_builder is None:
            self._require_deployment_only("enable_grouped_deployment_only", f"payload builder for {cache_key} is unavailable")
        payload = payload_builder(modules, device, dtype)
        if payload is None:
            self._require_deployment_only("enable_grouped_deployment_only", f"failed to build static payload for {cache_key}")
        return self._cache_grouped_payload(cache_key, payload, device, dtype)

    def enable_grouped_deployment_only(self):
        if self._deployment_only_active:
            return
        if not _deployment_only_enabled():
            return
        if not self._grouped_enabled():
            self._require_deployment_only("enable_grouped_deployment_only", "OLMOE_GROUPED_EXPERTS must be enabled")
        if (
            build_grouped_ns_compensation_payload is None
            or build_grouped_precomputed_payload is None
            or build_grouped_route is None
            or run_grouped_structured_expert_forward is None
            or run_grouped_structured_expert_pair_forward is None
        ):
            self._require_deployment_only("enable_grouped_deployment_only", "grouped expert kernels are unavailable")
        if len(self.experts) == 0:
            self._deployment_only_active = True
            return

        device, dtype = self._infer_grouped_payload_device_dtype()
        if device.type != "cuda":
            self._require_deployment_only("enable_grouped_deployment_only", "static grouped payloads require CUDA expert weights")
        if dtype != torch.float16:
            self._require_deployment_only("enable_grouped_deployment_only", f"static grouped payloads require float16 weights, got {dtype}")

        payload_specs = (
            ("gate", [expert.gate_proj for expert in self.experts]),
            ("up", [expert.up_proj for expert in self.experts]),
            ("down", [expert.down_proj for expert in self.experts]),
        )
        self._grouped_payload_cache.clear()
        for cache_key, modules in payload_specs:
            for expert_idx, module in enumerate(modules):
                if getattr(module, "_mode_host", None) not in (3, 5):
                    self._require_deployment_only(
                        "enable_grouped_deployment_only",
                        f"{cache_key}[{expert_idx}] must use mode=3 or mode=5, got mode={getattr(module, '_mode_host', None)}",
                    )
                if not hasattr(module, "can_precompute_shared") or not module.can_precompute_shared():
                    self._require_deployment_only(
                        "enable_grouped_deployment_only",
                        f"{cache_key}[{expert_idx}] cannot precompute shared hidden",
                    )
                if not hasattr(module, "prune_expert_runtime_state"):
                    self._require_deployment_only(
                        "enable_grouped_deployment_only",
                        f"{cache_key}[{expert_idx}] does not support expert runtime pruning",
                    )
                if not hasattr(module, "prune_ns_runtime_state"):
                    self._require_deployment_only(
                        "enable_grouped_deployment_only",
                        f"{cache_key}[{expert_idx}] does not support ns runtime pruning",
                    )
            if self._load_grouped_payload_from_sidecar(cache_key, device, dtype) is None:
                self._build_grouped_payload_or_raise(cache_key, modules, device, dtype)
            if self._load_grouped_payload_from_sidecar(f"{cache_key}_ns", device, dtype) is None:
                self._build_grouped_payload_or_raise(
                    f"{cache_key}_ns",
                    modules,
                    device,
                    dtype,
                    builder=build_grouped_ns_compensation_payload,
                )

        for expert_layer in self.experts:
            expert_layer.gate_proj.prune_expert_runtime_state()
            expert_layer.up_proj.prune_expert_runtime_state()
            expert_layer.down_proj.prune_expert_runtime_state()
            expert_layer.gate_proj.prune_ns_runtime_state()
            expert_layer.up_proj.prune_ns_runtime_state()
            expert_layer.down_proj.prune_ns_runtime_state()

        self._deployment_only_active = True

    def _release_transient_grouped_payloads(self, *payloads) -> None:
        if self._grouped_payload_cache_enabled():
            return
        released = False
        for payload in payloads:
            if isinstance(payload, dict):
                payload.clear()
                released = True
        if released and self._grouped_empty_cache_enabled() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_grouped_payload(self, cache_key: str, modules, device: torch.device, dtype: torch.dtype, builder=None):
        payload_builder = build_grouped_precomputed_payload if builder is None else builder
        if payload_builder is None:
            if self._deployment_only_active:
                self._require_deployment_only("_get_grouped_payload", f"grouped payload builder for {cache_key} is unavailable")
            return None

        device_key = self._payload_device_key(device)
        cache_entry = self._grouped_payload_cache.get(cache_key)
        if self._deployment_only_active:
            if cache_entry is None:
                self._load_grouped_payload_from_sidecar(cache_key, device, dtype)
                cache_entry = self._grouped_payload_cache.get(cache_key)
            if cache_entry is None:
                self._require_deployment_only("_get_grouped_payload", f"static payload {cache_key} was not initialized")
            if cache_entry["device"] != device_key or cache_entry["dtype"] != dtype:
                self._require_deployment_only(
                    "_get_grouped_payload",
                    f"static payload {cache_key} was initialized for device={cache_entry['device']} dtype={cache_entry['dtype']}, got device={device_key} dtype={dtype}",
                )
            return cache_entry["payload"]

        if not self._grouped_payload_cache_enabled():
            sidecar_payload = self._load_grouped_payload_from_sidecar(cache_key, device, dtype)
            if sidecar_payload is not None:
                return sidecar_payload
            return payload_builder(modules, device, dtype)
        if cache_entry is not None and cache_entry["device"] == device_key and cache_entry["dtype"] == dtype:
            return cache_entry["payload"]
        sidecar_payload = self._load_grouped_payload_from_sidecar(cache_key, device, dtype)
        if sidecar_payload is not None:
            return sidecar_payload
        payload = payload_builder(modules, device, dtype)
        if payload is None:
            return None
        return self._cache_grouped_payload(cache_key, payload, device, dtype)

    def _forward_expert_loop(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        use_shared_precompute: bool,
        use_down_shared_precompute: bool,
    ) -> torch.Tensor:
        if self._deployment_only_active:
            self._require_deployment_only("_forward_expert_loop", "expert-loop fallback is disabled")

        final_hidden_states = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[1]), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        gate_shared_all = None
        up_shared_all = None
        if use_shared_precompute and len(self.experts) > 0:
            gate_shared_all = self.experts[0].gate_proj.compute_shared_hidden(hidden_states)
            up_shared_all = self.experts[0].up_proj.compute_shared_hidden(hidden_states)

        deferred_down = []

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states.index_select(0, top_x)
            if gate_shared_all is not None and up_shared_all is not None:
                gate_hidden = gate_shared_all.index_select(0, top_x)
                up_hidden = up_shared_all.index_select(0, top_x)
                gate_out = expert_layer.gate_proj.forward_from_precomputed_hidden(gate_hidden, current_state)
                up_out = expert_layer.up_proj.forward_from_precomputed_hidden(up_hidden, current_state)
                mlp_hidden = expert_layer.act_fn(gate_out) * up_out
                if use_down_shared_precompute:
                    deferred_down.append((expert_layer, top_x, idx, mlp_hidden))
                    continue
                current_hidden_states = expert_layer.down_proj(mlp_hidden)
            else:
                current_hidden_states = expert_layer(current_state)

            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if deferred_down:
            down_inputs = torch.cat([mlp_hidden for _, _, _, mlp_hidden in deferred_down], dim=0)
            down_shared_all = self.experts[0].down_proj.compute_shared_hidden(down_inputs)
            start = 0
            for expert_layer, top_x, idx, mlp_hidden in deferred_down:
                end = start + mlp_hidden.shape[0]
                down_hidden = down_shared_all[start:end]
                current_hidden_states = expert_layer.down_proj.forward_from_precomputed_hidden(down_hidden, mlp_hidden)
                current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                start = end

        return final_hidden_states

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        failure_reason = None
        if not self._grouped_enabled():
            failure_reason = "grouped experts are disabled"
        elif build_grouped_route is None:
            failure_reason = "grouped route builder is unavailable"
        elif run_grouped_structured_expert_forward is None or run_grouped_structured_expert_pair_forward is None:
            failure_reason = "grouped expert kernels are unavailable"
        elif hidden_states.device.type != "cuda":
            failure_reason = "grouped expert kernels require CUDA hidden states"
        elif hidden_states.dtype != torch.float16:
            failure_reason = f"grouped expert kernels require float16 hidden states, got {hidden_states.dtype}"

        if failure_reason is not None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", failure_reason)
            return None
        if len(self.experts) == 0:
            return hidden_states.new_zeros((hidden_states.shape[0], hidden_states.shape[1]))

        gate_modules = [expert.gate_proj for expert in self.experts]
        up_modules = [expert.up_proj for expert in self.experts]
        down_modules = [expert.down_proj for expert in self.experts]
        gate_payload = self._get_grouped_payload("gate", gate_modules, hidden_states.device, hidden_states.dtype)
        up_payload = self._get_grouped_payload("up", up_modules, hidden_states.device, hidden_states.dtype)
        gate_ns_payload = self._get_grouped_payload(
            "gate_ns",
            gate_modules,
            hidden_states.device,
            hidden_states.dtype,
            builder=build_grouped_ns_compensation_payload,
        )
        up_ns_payload = self._get_grouped_payload(
            "up_ns",
            up_modules,
            hidden_states.device,
            hidden_states.dtype,
            builder=build_grouped_ns_compensation_payload,
        )
        if gate_payload is None or up_payload is None or gate_ns_payload is None or up_ns_payload is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "gate/up grouped payload lookup returned no result")
            return None

        num_tokens = hidden_states.shape[0]
        grouped_route = build_grouped_route(selected_experts, self.num_experts)
        if grouped_route is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "failed to build grouped route")
            return None
        grouped_token_indices = grouped_route["grouped_token_indices"]
        grouped_slots = grouped_route["grouped_slots"]
        if grouped_token_indices.numel() == 0:
            return hidden_states.new_zeros((num_tokens, hidden_states.shape[1]))

        x_grouped = hidden_states.index_select(0, grouped_token_indices)
        grouped_routing = routing_weights[grouped_token_indices, grouped_slots].unsqueeze(-1)
        gate_shared_all = self.experts[0].gate_proj.compute_shared_hidden(hidden_states)
        up_shared_all = self.experts[0].up_proj.compute_shared_hidden(hidden_states)
        if gate_shared_all is None or up_shared_all is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "shared hidden precompute returned no result")
            return None
        gate_hidden_grouped = gate_shared_all.index_select(0, grouped_token_indices)
        up_hidden_grouped = up_shared_all.index_select(0, grouped_token_indices)

        gate_out_grouped, up_out_grouped = run_grouped_structured_expert_pair_forward(
            gate_payload,
            up_payload,
            gate_hidden_grouped,
            up_hidden_grouped,
            x_grouped,
            grouped_route,
            expert_modules_a=gate_modules,
            expert_modules_b=up_modules,
            ns_payload_a=gate_ns_payload,
            ns_payload_b=up_ns_payload,
        )
        self._release_transient_grouped_payloads(gate_payload, up_payload, gate_ns_payload, up_ns_payload)
        if gate_out_grouped is None or up_out_grouped is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "grouped gate/up kernel returned no result")
            return None

        mlp_hidden_grouped = self.experts[0].act_fn(gate_out_grouped) * up_out_grouped
        down_hidden_grouped = self.experts[0].down_proj.compute_shared_hidden(mlp_hidden_grouped)
        if down_hidden_grouped is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "down shared hidden precompute returned no result")
            return None
        down_payload = self._get_grouped_payload("down", down_modules, hidden_states.device, hidden_states.dtype)
        down_ns_payload = self._get_grouped_payload(
            "down_ns",
            down_modules,
            hidden_states.device,
            hidden_states.dtype,
            builder=build_grouped_ns_compensation_payload,
        )
        if down_payload is None or down_ns_payload is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "down grouped payload lookup returned no result")
            return None
        down_out_grouped = run_grouped_structured_expert_forward(
            down_payload,
            down_hidden_grouped,
            mlp_hidden_grouped,
            grouped_route,
            expert_modules=down_modules,
            ns_payload=down_ns_payload,
            prefer_ns_fused_decode=True,
        )
        self._release_transient_grouped_payloads(down_payload, down_ns_payload)
        if down_out_grouped is None:
            if self._deployment_only_active:
                self._require_deployment_only("_forward_grouped", "grouped down kernel returned no result")
            return None

        final_hidden_states = hidden_states.new_zeros((num_tokens, hidden_states.shape[1]))
        final_hidden_states.index_add_(
            0,
            grouped_token_indices,
            (down_out_grouped * grouped_routing).to(hidden_states.dtype),
        )
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if _deployment_only_enabled() and not self._deployment_only_active:
            self._require_deployment_only("forward", "static grouped payloads were not initialized during model load")

        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        use_shared_precompute = True
        use_down_shared_precompute = True
        for expert_layer in self.experts:
            gate_proj = expert_layer.gate_proj
            up_proj = expert_layer.up_proj
            down_proj = expert_layer.down_proj
            if not (
                hasattr(gate_proj, "can_precompute_shared")
                and hasattr(gate_proj, "compute_shared_hidden")
                and hasattr(gate_proj, "forward_from_precomputed_hidden")
                and hasattr(up_proj, "can_precompute_shared")
                and hasattr(up_proj, "compute_shared_hidden")
                and hasattr(up_proj, "forward_from_precomputed_hidden")
                and gate_proj.can_precompute_shared()
                and up_proj.can_precompute_shared()
            ):
                use_shared_precompute = False
            if not (
                hasattr(down_proj, "can_precompute_shared")
                and hasattr(down_proj, "compute_shared_hidden")
                and hasattr(down_proj, "forward_from_precomputed_hidden")
                and down_proj.can_precompute_shared()
            ):
                use_down_shared_precompute = False
            if not use_shared_precompute and not use_down_shared_precompute:
                break

        if self._deployment_only_active and (not use_shared_precompute or not use_down_shared_precompute):
            self._require_deployment_only("forward", "shared precompute path is unavailable for one or more experts")

        final_hidden_states = None
        if use_shared_precompute and use_down_shared_precompute:
            final_hidden_states = self._forward_grouped(hidden_states, routing_weights, selected_experts)
        if final_hidden_states is None:
            if self._deployment_only_active:
                self._require_deployment_only("forward", "grouped expert execution returned no result")
            final_hidden_states = self._forward_expert_loop(
                hidden_states,
                routing_weights,
                selected_experts,
                use_shared_precompute,
                use_down_shared_precompute,
            )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class OlmoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OlmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OLMOE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = OlmoeSparseMoeBlock(config)
        self.mlp._grouped_payload_prefix = f"model.layers.{layer_idx}.mlp"
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


@auto_docstring
class OlmoePreTrainedModel(PreTrainedModel):
    config: OlmoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OlmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, OlmoeRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@auto_docstring
class OlmoeModel(OlmoePreTrainedModel):
    def __init__(self, config: OlmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OlmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._grouped_qgmap_cpu_staging_manager = None

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        grouped_qgmap_staging_manager = self._grouped_qgmap_cpu_staging_manager
        if maybe_create_grouped_qgmap_cpu_staging_manager is not None:
            if grouped_qgmap_staging_manager is None or not grouped_qgmap_staging_manager.matches_device(inputs_embeds.device):
                grouped_qgmap_staging_manager = maybe_create_grouped_qgmap_cpu_staging_manager(self.layers)
                self._grouped_qgmap_cpu_staging_manager = grouped_qgmap_staging_manager

        try:
            for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                if grouped_qgmap_staging_manager is not None:
                    grouped_qgmap_staging_manager.prepare_for_layer(layer_idx)
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)
        finally:
            if grouped_qgmap_staging_manager is not None:
                grouped_qgmap_staging_manager.finalize_forward()

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class OlmoeForCausalLM(OlmoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OlmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoeForCausalLM

        >>> model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["OlmoeForCausalLM", "OlmoeModel", "OlmoePreTrainedModel"]
