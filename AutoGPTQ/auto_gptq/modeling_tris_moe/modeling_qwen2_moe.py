"""Qwen2-MoE modeling with TriSMoE grouped expert runtime support."""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeAttention as HFQwen2MoeAttention,
    Qwen2MoeDecoderLayer as HFQwen2MoeDecoderLayer,
    Qwen2MoeFlashAttention2 as HFQwen2MoeFlashAttention2,
    Qwen2MoeForCausalLM as HFQwen2MoeForCausalLM,
    Qwen2MoeMLP,
    Qwen2MoeModel as HFQwen2MoeModel,
    Qwen2MoePreTrainedModel,
    Qwen2MoeRMSNorm,
    Qwen2MoeRotaryEmbedding,
    Qwen2MoeSdpaAttention as HFQwen2MoeSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

try:
    from auto_gptq.nn_modules.qlinear.qlinear_cuda import (
        build_grouped_ns_compensation_payload,
        build_grouped_precomputed_payload,
        build_grouped_route,
        run_compact3_qkv_block_gr_forward,
        run_compact3_shared_expert_gate_up_forward,
        run_grouped_structured_expert_forward,
        run_grouped_structured_expert_pair_forward,
    )
except ImportError:
    build_grouped_ns_compensation_payload = None
    build_grouped_precomputed_payload = None
    build_grouped_route = None
    run_compact3_qkv_block_gr_forward = None
    run_compact3_shared_expert_gate_up_forward = None
    run_grouped_structured_expert_forward = None
    run_grouped_structured_expert_pair_forward = None

try:
    from .grouped_payload_sidecar import (
        GROUPED_QGMAP_PAYLOAD_NAMES,
        default_grouped_payload_sidecar_path,
        default_grouped_qgmap_staging_sidecar_path,
        load_grouped_payload_from_sidecar,
        load_grouped_payloads_from_reader,
        maybe_create_grouped_qgmap_cpu_staging_manager,
    )
except ImportError:
    GROUPED_QGMAP_PAYLOAD_NAMES = frozenset()
    default_grouped_payload_sidecar_path = None
    default_grouped_qgmap_staging_sidecar_path = None
    load_grouped_payload_from_sidecar = None
    load_grouped_payloads_from_reader = None
    maybe_create_grouped_qgmap_cpu_staging_manager = None


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
    message = f"[Qwen2-MoE deployment-only] {label}"
    if reason:
        message = f"{message}: {reason}"
    raise RuntimeError(message)


def _module_has_quant_linear(module: nn.Module) -> bool:
    for submodule in module.modules():
        if getattr(submodule, "QUANT_TYPE", None) == "cuda":
            return True
    return False


def _maybe_cast_kernel_input(hidden_states: torch.Tensor, module: nn.Module) -> torch.Tensor:
    # The exported Qwen grouped-runtime checkpoints are materialized in float16.
    # Keeping hidden states in float16 end-to-end avoids per-layer bf16->fp16 casts
    # before quantized kernels.
    return hidden_states


def _maybe_cast_position_embeddings(position_embeddings, dtype: torch.dtype):
    if position_embeddings is None:
        return None
    cos, sin = position_embeddings
    if cos.dtype == dtype and sin.dtype == dtype:
        return position_embeddings
    return (cos.to(dtype), sin.to(dtype))


def _qwen2_moe_qkv_projections(attn: nn.Module, hidden_states: torch.Tensor):
    if run_compact3_qkv_block_gr_forward is not None:
        fused = run_compact3_qkv_block_gr_forward(attn.q_proj, attn.k_proj, attn.v_proj, hidden_states)
        if fused is not None:
            return fused
    return attn.q_proj(hidden_states), attn.k_proj(hidden_states), attn.v_proj(hidden_states)


def _qwen2_moe_shared_expert_forward(shared_expert: nn.Module, hidden_states: torch.Tensor):
    if run_compact3_shared_expert_gate_up_forward is not None:
        fused = run_compact3_shared_expert_gate_up_forward(shared_expert.gate_proj, shared_expert.up_proj, hidden_states)
        if fused is not None:
            gate_states, up_states = fused
            return shared_expert.down_proj(shared_expert.act_fn(gate_states) * up_states)
    return shared_expert(hidden_states)


class Qwen2MoeFusedAttention(HFQwen2MoeAttention):
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
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = _qwen2_moe_qkv_projections(self, hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Qwen2MoeFusedSdpaAttention(HFQwen2MoeSdpaAttention):
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
    ):
        if output_attentions:
            return Qwen2MoeFusedAttention.forward(
                self,
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

        query_states, key_states, value_states = _qwen2_moe_qkv_projections(self, hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = bool(causal_mask is None and q_len > 1)
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


def _replace_attention_with_fused_qkv(attn: nn.Module, config: Qwen2MoeConfig, layer_idx: int):
    if isinstance(attn, HFQwen2MoeFlashAttention2):
        return attn
    if isinstance(attn, HFQwen2MoeSdpaAttention):
        return Qwen2MoeFusedSdpaAttention(config, layer_idx)
    if isinstance(attn, HFQwen2MoeAttention):
        return Qwen2MoeFusedAttention(config, layer_idx)
    return attn


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_dim, 1, bias=False)
        self._grouped_payload_cache = {}
        self._deployment_only_active = False
        self._grouped_payload_prefix = None
        self._grouped_payload_model_dir = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)

    def _grouped_enabled(self) -> bool:
        for name in (
            "QWEN2MOE_GROUPED_EXPERTS",
            "QWEN_GROUPED_EXPERTS",
            "OLMOE_GROUPED_EXPERTS",
            "TRIS_GROUPED_EXPERTS",
        ):
            value = os.environ.get(name)
            if value is not None:
                return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
        return True

    def _grouped_payload_cache_enabled(self) -> bool:
        return self._deployment_only_active or _env_flag_enabled(
            "QWEN2MOE_GROUPED_PAYLOAD_CACHE",
            "QWEN_GROUPED_PAYLOAD_CACHE",
            "OLMOE_GROUPED_PAYLOAD_CACHE",
            "TRIS_GROUPED_PAYLOAD_CACHE",
        )

    def _grouped_empty_cache_enabled(self) -> bool:
        return _env_flag_enabled(
            "QWEN2MOE_GROUPED_EMPTY_CACHE",
            "QWEN_GROUPED_EMPTY_CACHE",
            "OLMOE_GROUPED_EMPTY_CACHE",
            "TRIS_GROUPED_EMPTY_CACHE",
        )

    def _payload_device_key(self, device: torch.device):
        return (device.type, device.index if device.index is not None else -1)

    def _cache_grouped_payload(self, cache_key: str, payload, device: torch.device, dtype: torch.dtype):
        self._grouped_payload_cache[cache_key] = {
            "device": self._payload_device_key(device),
            "dtype": dtype,
            "payload": payload,
        }
        return payload

    def _cached_grouped_payload_matches(self, cache_key: str, device: torch.device, dtype: torch.dtype) -> bool:
        cache_entry = self._grouped_payload_cache.get(cache_key)
        if cache_entry is None:
            return False
        return cache_entry.get("device") == self._payload_device_key(device) and cache_entry.get("dtype") == dtype

    def preload_grouped_payloads_from_reader(self, checkpoint_reader) -> None:
        if load_grouped_payloads_from_reader is None or not self._grouped_payload_prefix:
            return
        if len(self.experts) == 0:
            return
        device, dtype = self._infer_grouped_payload_device_dtype()
        if device.type != "cuda" or dtype != torch.float16:
            return

        cache_keys = ("gate", "up", "down", "gate_ns", "up_ns", "down_ns")
        exclude_names_by_cache = {
            cache_key: GROUPED_QGMAP_PAYLOAD_NAMES
            for cache_key in ("gate", "up", "down")
            if self._grouped_qgmap_staging_active_for_cache_key(cache_key)
        }
        payloads = load_grouped_payloads_from_reader(
            checkpoint_reader,
            self._grouped_payload_prefix,
            cache_keys,
            device,
            dtype,
            exclude_names_by_cache=exclude_names_by_cache,
        )
        for cache_key, payload in payloads.items():
            self._cache_grouped_payload(cache_key, payload, device, dtype)

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
        if not _env_flag_enabled(
            "QWEN2MOE_GROUPED_QGMAP_CPU_STAGING",
            "QWEN_GROUPED_QGMAP_CPU_STAGING",
            "OLMOE_GROUPED_QGMAP_CPU_STAGING",
            "TRIS_GROUPED_QGMAP_CPU_STAGING",
        ):
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
            self._require_deployment_only("enable_grouped_deployment_only", "grouped experts must be enabled")
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
        device_key = self._payload_device_key(device)
        self._grouped_payload_cache = {
            cache_key: cache_entry
            for cache_key, cache_entry in self._grouped_payload_cache.items()
            if cache_entry.get("device") == device_key and cache_entry.get("dtype") == dtype
        }
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
            if not self._cached_grouped_payload_matches(cache_key, device, dtype) and self._load_grouped_payload_from_sidecar(cache_key, device, dtype) is None:
                self._build_grouped_payload_or_raise(cache_key, modules, device, dtype)
            ns_cache_key = f"{cache_key}_ns"
            if not self._cached_grouped_payload_matches(ns_cache_key, device, dtype) and self._load_grouped_payload_from_sidecar(ns_cache_key, device, dtype) is None:
                self._build_grouped_payload_or_raise(
                    ns_cache_key,
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
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._deployment_only_active:
            self._require_deployment_only("_forward_expert_loop", "expert-loop fallback is disabled")

        final_hidden_states = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[1]), dtype=output_dtype, device=hidden_states.device
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
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(output_dtype))

        if deferred_down:
            down_inputs = torch.cat([mlp_hidden for _, _, _, mlp_hidden in deferred_down], dim=0)
            down_shared_all = self.experts[0].down_proj.compute_shared_hidden(down_inputs)
            start = 0
            for expert_layer, top_x, idx, mlp_hidden in deferred_down:
                end = start + mlp_hidden.shape[0]
                down_hidden = down_shared_all[start:end]
                current_hidden_states = expert_layer.down_proj.forward_from_precomputed_hidden(down_hidden, mlp_hidden)
                current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(output_dtype))
                start = end

        return final_hidden_states

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        output_dtype: torch.dtype,
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

        final_hidden_states = torch.zeros(
            (num_tokens, hidden_states.shape[1]), dtype=output_dtype, device=hidden_states.device
        )
        final_hidden_states.index_add_(
            0,
            grouped_token_indices,
            (down_out_grouped * grouped_routing).to(output_dtype),
        )
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if _deployment_only_enabled() and not self._deployment_only_active:
            self._require_deployment_only("forward", "static grouped payloads were not initialized during model load")

        output_dtype = hidden_states.dtype
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_kernel_hidden_states = _maybe_cast_kernel_input(hidden_states, self.experts)
        expert_kernel_dtype = expert_kernel_hidden_states.dtype
        routing_weights = routing_weights.to(expert_kernel_dtype)

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
            final_hidden_states = self._forward_grouped(
                expert_kernel_hidden_states,
                routing_weights,
                selected_experts,
                output_dtype,
            )
        if final_hidden_states is None:
            if self._deployment_only_active:
                self._require_deployment_only("forward", "grouped expert execution returned no result")
            final_hidden_states = self._forward_expert_loop(
                expert_kernel_hidden_states,
                routing_weights,
                selected_experts,
                use_shared_precompute,
                use_down_shared_precompute,
                output_dtype,
            )

        shared_expert_hidden_states = _maybe_cast_kernel_input(hidden_states, self.shared_expert)
        shared_expert_output = _qwen2_moe_shared_expert_forward(self.shared_expert, shared_expert_hidden_states).to(output_dtype)
        shared_expert_gate = torch.sigmoid(self.shared_expert_gate(hidden_states)).to(output_dtype)
        final_hidden_states = final_hidden_states + shared_expert_gate * shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen2MoeDecoderLayer(HFQwen2MoeDecoderLayer):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = _replace_attention_with_fused_qkv(self.self_attn, config, layer_idx)
        if isinstance(self.mlp, Qwen2MoeMLP):
            return
        self.mlp = Qwen2MoeSparseMoeBlock(config)
        self.mlp._grouped_payload_prefix = f"model.layers.{layer_idx}.mlp"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_hidden_states = _maybe_cast_kernel_input(hidden_states, self.self_attn)
        attn_position_embeddings = _maybe_cast_position_embeddings(position_embeddings, attn_hidden_states.dtype)
        attn_hidden_states, self_attn_weights = self.self_attn(
            hidden_states=attn_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=attn_position_embeddings,
        )
        hidden_states = residual + attn_hidden_states.to(residual.dtype)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states.to(residual.dtype)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class Qwen2MoeModel(HFQwen2MoeModel):
    def __init__(self, config: Qwen2MoeConfig):
        Qwen2MoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._grouped_qgmap_cpu_staging_manager = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        grouped_qgmap_staging_manager = self._grouped_qgmap_cpu_staging_manager
        if maybe_create_grouped_qgmap_cpu_staging_manager is not None:
            if grouped_qgmap_staging_manager is None or not grouped_qgmap_staging_manager.matches_device(inputs_embeds.device):
                grouped_qgmap_staging_manager = maybe_create_grouped_qgmap_cpu_staging_manager(self.layers)
                self._grouped_qgmap_cpu_staging_manager = grouped_qgmap_staging_manager

        try:
            for layer_idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                if grouped_qgmap_staging_manager is not None:
                    grouped_qgmap_staging_manager.prepare_for_layer(layer_idx)

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

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class Qwen2MoeForCausalLM(HFQwen2MoeForCausalLM):
    def __init__(self, config):
        Qwen2MoePreTrainedModel.__init__(self, config)
        self.model = Qwen2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


__all__ = [
    "Qwen2MoeConfig",
    "Qwen2MoeForCausalLM",
    "Qwen2MoeModel",
    "Qwen2MoePreTrainedModel",
    "Qwen2MoeSparseMoeBlock",
]
