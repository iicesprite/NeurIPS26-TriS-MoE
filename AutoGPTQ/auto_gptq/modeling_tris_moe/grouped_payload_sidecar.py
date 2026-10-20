from bisect import bisect_left
import os
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..nn_modules.qlinear.qlinear_cuda import (
    QuantLinear,
    _compact3_block_gr_enabled,
    build_grouped_ns_compensation_payload,
    build_grouped_precomputed_payload,
    prepare_compact3_k64_groups,
)


GROUPED_PAYLOAD_SIDECAR_BASENAME = "grouped_runtime_payload.safetensors"
GROUPED_QGMAP_STAGING_SIDECAR_BASENAME = "grouped_qgmap_staging.safetensors"
GROUPED_PAYLOAD_METADATA = {"format": "olmoe_grouped_runtime_payload_v1"}
GROUPED_PAYLOAD_EMBEDDED_METADATA_KEY = "olmoe_grouped_runtime_payload"
GROUPED_PAYLOAD_DEPLOYMENT_ONLY_METADATA_KEY = "olmoe_deployment_only"
GROUPED_QGMAP_STAGING_METADATA = {"format": "olmoe_grouped_qgmap_staging_v1"}
GROUPED_QGMAP_STAGING_METADATA_KEY = "olmoe_grouped_qgmap_staging"
GROUPED_PRECOMPUTED_KEYS = ("gate", "up", "down")
GROUPED_NS_KEYS = ("gate_ns", "up_ns", "down_ns")
GROUPED_QGMAP_PAYLOAD_NAMES = frozenset({"q_gmap_sparse", "q_gmap_sparse_prefix"})
GROUPED_PAYLOAD_SCALAR_NAMES = {
    "is_compact3",
    "compact3_use_block_gr",
    "has_output_scale",
    "has_bias",
    "max_ns_rank",
    "infeatures",
    "outfeatures",
    "group_size",
}
DEPLOYMENT_ONLY_EXPERT_STATE_NAMES = frozenset(
    {
        "mode",
        "ns_mode",
        "ns_vector_row",
        "ns_vector_col",
        "expert_input_scale",
    }
)


def _unwrap_transformer_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise ValueError("Expected an MoE model or GPTQ wrapper with a .model.layers module list.")


def _grouped_moe_specs_for_layer(layer_idx: int, layer):
    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", []) if mlp is not None else []
    if mlp is not None and len(experts) > 0 and hasattr(experts[0], "gate_proj"):
        yield (
            f"model.layers.{layer_idx}.mlp",
            mlp,
            (
                ("gate", [expert.gate_proj for expert in experts]),
                ("up", [expert.up_proj for expert in experts]),
                ("down", [expert.down_proj for expert in experts]),
            ),
        )

    block_sparse_moe = getattr(layer, "block_sparse_moe", None)
    experts = getattr(block_sparse_moe, "experts", []) if block_sparse_moe is not None else []
    if block_sparse_moe is not None and len(experts) > 0 and hasattr(experts[0], "w1"):
        yield (
            f"model.layers.{layer_idx}.block_sparse_moe",
            block_sparse_moe,
            (
                ("gate", [expert.w1 for expert in experts]),
                ("up", [expert.w3 for expert in experts]),
                ("down", [expert.w2 for expert in experts]),
            ),
        )


def default_grouped_payload_sidecar_path(model_dir: str) -> str:
    override = (
        os.environ.get("QWEN2MOE_GROUPED_PAYLOAD_SIDECAR_PATH")
        or os.environ.get("QWEN_GROUPED_PAYLOAD_SIDECAR_PATH")
        or os.environ.get("MIXTRAL_GROUPED_PAYLOAD_SIDECAR_PATH")
        or os.environ.get("OLMOE_GROUPED_PAYLOAD_SIDECAR_PATH")
    )
    if override:
        return override
    return os.path.join(model_dir, GROUPED_PAYLOAD_SIDECAR_BASENAME)


def default_grouped_qgmap_staging_sidecar_path(model_dir: str) -> str:
    override = (
        os.environ.get("QWEN2MOE_GROUPED_QGMAP_STAGING_SIDECAR_PATH")
        or os.environ.get("QWEN_GROUPED_QGMAP_STAGING_SIDECAR_PATH")
        or os.environ.get("MIXTRAL_GROUPED_QGMAP_STAGING_SIDECAR_PATH")
        or os.environ.get("OLMOE_GROUPED_QGMAP_STAGING_SIDECAR_PATH")
    )
    if override:
        return override
    return os.path.join(model_dir, GROUPED_QGMAP_STAGING_SIDECAR_BASENAME)


def _payload_tensor_name(layer_prefix: str, cache_key: str, name: str) -> str:
    return f"{layer_prefix}.grouped_payload.{cache_key}.{name}"


def _is_grouped_payload_tensor_name(name: str) -> bool:
    return ".grouped_payload." in name


def _split_grouped_payload_tensor_name(name: str) -> Optional[tuple[str, str, str]]:
    marker = ".grouped_payload."
    if marker not in name:
        return None
    layer_prefix, remainder = name.split(marker, 1)
    if "." not in remainder:
        return None
    cache_key, payload_name = remainder.split(".", 1)
    if not cache_key or not payload_name:
        return None
    return layer_prefix, cache_key, payload_name


def _is_grouped_qgmap_tensor_name(name: str) -> bool:
    split_name = _split_grouped_payload_tensor_name(name)
    if split_name is None:
        return False
    _, cache_key, payload_name = split_name
    return cache_key in GROUPED_PRECOMPUTED_KEYS and payload_name in GROUPED_QGMAP_PAYLOAD_NAMES


def _env_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    return False


def _grouped_qgmap_cpu_staging_enabled() -> bool:
    return _env_flag_enabled(
        "MIXTRAL_GROUPED_QGMAP_CPU_STAGING",
        "OLMOE_GROUPED_QGMAP_CPU_STAGING",
        "TRIS_GROUPED_QGMAP_CPU_STAGING",
    )


def _device_cache_key(device: torch.device):
    return (device.type, device.index if device.index is not None else -1)


def _empty_tensor_like_rank(reference: torch.Tensor, device: torch.device) -> torch.Tensor:
    shape = (0,) if reference.ndim == 0 else (0,) * reference.ndim
    return torch.empty(shape, dtype=reference.dtype, device=device)


def _clone_checkpoint_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _encode_payload_tensor(name: str, value, reference_dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        return tensor
    if value is None:
        return torch.empty((0,), dtype=reference_dtype)
    if isinstance(value, bool):
        return torch.tensor([value], dtype=torch.bool)
    if isinstance(value, int):
        return torch.tensor([value], dtype=torch.int32)
    raise TypeError(f"Unsupported grouped payload value type for {name}: {type(value)!r}")


def _decode_payload_tensor(name: str, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        tensor = tensor.to(device=device, dtype=dtype)
    else:
        tensor = tensor.to(device=device)
    if name in GROUPED_PAYLOAD_SCALAR_NAMES and tensor.numel() == 1:
        tensor = tensor.reshape(())
    return tensor.contiguous()


def _sidecar_payload_is_compatible(cache_key: str, payload: dict) -> bool:
    if not cache_key.endswith("_ns"):
        return True
    required = {
        "ns_rank_per_expert",
        "ns_c2_rank_offsets",
        "ns_c2_rank_word_offsets",
        "ns_c1_rank_word_offsets",
        "ns_c1_group_offsets",
        "max_ns_rank",
        "group_size",
        "infeatures",
        "outfeatures",
        "ns_c2_qweight",
        "ns_c2_qzeros",
        "ns_c2_scales",
        "ns_c1_qweight",
        "ns_c1_qzeros",
        "ns_c1_scales",
    }
    if any(name not in payload for name in required):
        return False
    group_size = int(payload["group_size"].reshape(-1)[0].item()) if payload["group_size"].numel() > 0 else 0
    infeatures = int(payload["infeatures"].reshape(-1)[0].item()) if payload["infeatures"].numel() > 0 else 0
    outfeatures = int(payload["outfeatures"].reshape(-1)[0].item()) if payload["outfeatures"].numel() > 0 else 0
    if group_size <= 0 or infeatures < 0 or outfeatures < 0:
        return False

    num_experts = int(payload["ns_rank_per_expert"].numel())
    if any(
        int(payload[name].numel()) != num_experts
        for name in (
            "ns_c2_rank_offsets",
            "ns_c2_rank_word_offsets",
            "ns_c1_rank_word_offsets",
            "ns_c1_group_offsets",
        )
    ):
        return False

    packed_infeatures = (infeatures + 7) // 8 if infeatures > 0 else 0
    out_qzeros_cols = (outfeatures + 7) // 8 if outfeatures > 0 else 0
    if payload["ns_c2_qweight"].ndim != 2 or int(payload["ns_c2_qweight"].shape[0]) != packed_infeatures:
        return False
    if payload["ns_c2_qzeros"].ndim != 2 or payload["ns_c2_scales"].ndim != 2:
        return False
    if int(payload["ns_c2_qzeros"].shape[0]) != int(payload["ns_c2_scales"].shape[0]):
        return False
    expected_c2_group_rows = (infeatures + group_size - 1) // group_size if infeatures > 0 else 0
    if int(payload["ns_c2_qzeros"].shape[0]) != expected_c2_group_rows:
        return False
    if payload["ns_c1_qweight"].ndim != 2 or int(payload["ns_c1_qweight"].shape[1]) != outfeatures:
        return False
    if payload["ns_c1_qzeros"].ndim != 2 or int(payload["ns_c1_qzeros"].shape[1]) != out_qzeros_cols:
        return False
    if payload["ns_c1_scales"].ndim != 2 or int(payload["ns_c1_scales"].shape[1]) != outfeatures:
        return False

    ns_ranks = payload["ns_rank_per_expert"].reshape(-1).to(dtype=torch.int64)
    positive_ranks = ns_ranks[ns_ranks > 0]
    expected_c1_rank_words = int(torch.div(positive_ranks + 7, 8, rounding_mode="floor").sum().item()) if positive_ranks.numel() > 0 else 0
    expected_c1_group_rows = int(torch.div(positive_ranks + group_size - 1, group_size, rounding_mode="floor").sum().item()) if positive_ranks.numel() > 0 else 0
    if int(payload["ns_c1_qweight"].shape[0]) != expected_c1_rank_words:
        return False
    if int(payload["ns_c1_qzeros"].shape[0]) != expected_c1_group_rows:
        return False
    if int(payload["ns_c1_scales"].shape[0]) != expected_c1_group_rows:
        return False
    return True


def _repair_grouped_payload_compatibility(payload: dict, dtype: torch.dtype) -> dict:
    is_compact3 = payload.get("is_compact3")
    if not isinstance(is_compact3, torch.Tensor) or is_compact3.numel() == 0 or not bool(is_compact3.reshape(-1)[0].item()):
        return payload

    output_scale = payload.get("output_scale")
    if not isinstance(output_scale, torch.Tensor) or output_scale.numel() == 0:
        return payload
    if not torch.all(output_scale == 0).item():
        return payload

    repaired = dict(payload)
    repaired["output_scale"] = torch.ones_like(output_scale, dtype=dtype).contiguous()
    if "has_output_scale" in repaired:
        repaired["has_output_scale"] = torch.tensor(False, device=output_scale.device, dtype=torch.bool)
    return repaired


def _normalize_grouped_payload_runtime_scalars(payload: dict) -> dict:
    scalar_names = {
        "is_compact3",
        "compact3_use_block_gr",
        "has_output_scale",
        "has_bias",
        "max_ns_rank",
        "infeatures",
        "outfeatures",
        "group_size",
    }
    bool_names = {"is_compact3", "compact3_use_block_gr", "has_output_scale", "has_bias"}
    normalized = payload
    for name in scalar_names:
        value = normalized.get(name)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            continue
        if normalized is payload:
            normalized = dict(payload)
        scalar = value.detach().to(device="cpu").reshape(-1)[0]
        normalized[name] = bool(scalar.item()) if name in bool_names else int(scalar.item())
    return normalized


def repair_grouped_payload_tensors(tensors: dict[str, torch.Tensor]) -> int:
    grouped_names: dict[tuple[str, str], dict[str, str]] = {}
    for name in tensors:
        split_name = _split_grouped_payload_tensor_name(name)
        if split_name is None:
            continue
        layer_prefix, cache_key, payload_name = split_name
        if cache_key not in GROUPED_PRECOMPUTED_KEYS:
            continue
        grouped_names.setdefault((layer_prefix, cache_key), {})[payload_name] = name

    repaired = 0
    for names in grouped_names.values():
        compact3_name = names.get("is_compact3")
        output_scale_name = names.get("output_scale")
        if compact3_name is None or output_scale_name is None:
            continue
        is_compact3 = tensors.get(compact3_name)
        output_scale = tensors.get(output_scale_name)
        if not isinstance(is_compact3, torch.Tensor) or not isinstance(output_scale, torch.Tensor):
            continue
        if is_compact3.numel() == 0 or not bool(is_compact3.reshape(-1)[0].item()):
            continue
        if output_scale.numel() == 0 or not bool(torch.all(output_scale == 0).item()):
            continue

        tensors[output_scale_name] = torch.ones_like(output_scale).contiguous()
        has_output_scale_name = names.get("has_output_scale")
        if has_output_scale_name is not None and has_output_scale_name in tensors:
            tensors[has_output_scale_name] = torch.zeros_like(tensors[has_output_scale_name], dtype=torch.bool).contiguous()
        repaired += 1

    return repaired


def inspect_grouped_payload_sidecar_compatibility(path: str) -> tuple[bool, Optional[str]]:
    if not path or not os.path.exists(path):
        return False, f"missing sidecar: {path}"

    grouped_payloads: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            split_name = _split_grouped_payload_tensor_name(key)
            if split_name is None:
                continue
            layer_prefix, cache_key, payload_name = split_name
            grouped_payloads.setdefault((layer_prefix, cache_key), {})[payload_name] = reader.get_tensor(key).contiguous()

    if not grouped_payloads:
        return False, "sidecar contains no grouped payload tensors"

    for (layer_prefix, cache_key), payload in sorted(grouped_payloads.items()):
        if not _sidecar_payload_is_compatible(cache_key, payload):
            return False, f"incompatible grouped payload: {layer_prefix}:{cache_key}"
    return True, None


def _build_layer_grouped_payload_tensors(layer_prefix: str, payload_specs) -> dict[str, torch.Tensor]:
    if not payload_specs:
        return {}

    reference = payload_specs[0][1][0]
    device = reference.fp_weight.device
    dtype = reference.fp_weight.dtype
    payload_specs = tuple(payload_specs) + tuple(
        (f"{cache_key}_ns", modules, build_grouped_ns_compensation_payload)
        for cache_key, modules, _ in payload_specs
    )

    serialized = {}
    for cache_key, modules, builder in payload_specs:
        payload = builder(modules, device, dtype)
        if payload is None:
            raise RuntimeError(f"Failed to build grouped payload for {layer_prefix}:{cache_key}")
        for name, value in payload.items():
            # Derived validation/workspace caches must not become checkpoint data.
            if name.startswith("_"):
                continue
            tensor = _encode_payload_tensor(name, value, dtype)
            serialized[_payload_tensor_name(layer_prefix, cache_key, name)] = tensor
    return serialized


def build_grouped_payload_sidecar_tensors(model) -> dict[str, torch.Tensor]:
    transformer_model = _unwrap_transformer_model(model)
    tensors: dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate(transformer_model.layers):
        for layer_prefix, _, payload_specs in _grouped_moe_specs_for_layer(layer_idx, layer):
            tensors.update(
                _build_layer_grouped_payload_tensors(
                    layer_prefix,
                    tuple(
                        (cache_key, modules, build_grouped_precomputed_payload)
                        for cache_key, modules in payload_specs
                    ),
                )
            )
    return tensors


def build_grouped_qgmap_staging_sidecar_tensors(model) -> dict[str, torch.Tensor]:
    transformer_model = _unwrap_transformer_model(model)
    tensors: dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate(transformer_model.layers):
        for layer_prefix, _, payload_specs in _grouped_moe_specs_for_layer(layer_idx, layer):
            reference = payload_specs[0][1][0]
            device = reference.fp_weight.device
            dtype = reference.fp_weight.dtype
            for cache_key, modules in payload_specs:
                payload = build_grouped_precomputed_payload(modules, device, dtype)
                if payload is None:
                    raise RuntimeError(f"Failed to build grouped qgmap payload for {layer_prefix}:{cache_key}")
                for payload_name in GROUPED_QGMAP_PAYLOAD_NAMES:
                    tensor = payload.get(payload_name)
                    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                        continue
                    tensors[_payload_tensor_name(layer_prefix, cache_key, payload_name)] = _encode_payload_tensor(
                        payload_name,
                        tensor,
                        dtype,
                    )
    return tensors


def build_grouped_runtime_payload_tensors(model, include_model_state: bool = False) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if include_model_state:
        tensors.update({key: _clone_checkpoint_tensor(value) for key, value in model.state_dict().items()})
    tensors.update(build_grouped_payload_sidecar_tensors(model))
    return tensors


def _embedded_checkpoint_metadata(checkpoint_metadata: Optional[dict], deployment_only: bool = False) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if checkpoint_metadata:
        metadata.update({str(key): str(value) for key, value in checkpoint_metadata.items()})
    metadata["format"] = "pt"
    metadata[GROUPED_PAYLOAD_EMBEDDED_METADATA_KEY] = "v1"
    if deployment_only:
        metadata[GROUPED_PAYLOAD_DEPLOYMENT_ONLY_METADATA_KEY] = "v1"
    return metadata


def _grouped_qgmap_staging_metadata(checkpoint_metadata: Optional[dict] = None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if checkpoint_metadata:
        metadata.update({str(key): str(value) for key, value in checkpoint_metadata.items()})
    metadata["format"] = "pt"
    metadata[GROUPED_QGMAP_STAGING_METADATA_KEY] = "v1"
    return metadata


def _load_safetensors_to_cpu(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        for key in reader.keys():
            tensors[key] = reader.get_tensor(key).contiguous()
    return tensors, {str(key): str(value) for key, value in metadata.items()}


def _split_moe_expert_tensor_name(name: str) -> Optional[tuple[str, str]]:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None

    parts = name.split(".")
    if len(parts) < 7:
        return None
    if parts[0] != "model" or parts[1] != "layers" or not parts[2].isdigit():
        return None
    if parts[3] == "mlp" and parts[4] == "experts" and parts[5].isdigit():
        module_name = parts[6]
        leaf_name = ".".join(parts[7:]) if len(parts) > 7 else ""
        if module_name not in {"gate_proj", "up_proj", "down_proj"} or not leaf_name:
            return None
        return module_name, leaf_name

    if (
        len(parts) >= 8
        and parts[3] == "block_sparse_moe"
        and parts[4] == "experts"
        and parts[5].isdigit()
    ):
        module_name = parts[6]
        leaf_name = ".".join(parts[7:]) if len(parts) > 7 else ""
        if module_name not in {"w1", "w2", "w3"} or not leaf_name:
            return None
        logical_name = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[module_name]
        return logical_name, leaf_name

    return None


def _is_required_deployment_only_expert_tensor(name: str) -> bool:
    split_name = _split_moe_expert_tensor_name(name)
    if split_name is None:
        return False
    _, leaf_name = split_name
    return leaf_name in DEPLOYMENT_ONLY_EXPERT_STATE_NAMES


def build_deployment_only_checkpoint_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if _is_grouped_payload_tensor_name(name):
            filtered[name] = tensor
            continue
        split_name = _split_moe_expert_tensor_name(name)
        if split_name is None or _is_required_deployment_only_expert_tensor(name):
            filtered[name] = tensor
    return filtered


def rewrite_grouped_checkpoint_for_deployment_only(source_path: str, output_path: str) -> str:
    tensors, checkpoint_metadata = _load_safetensors_to_cpu(source_path)
    deployment_only_tensors = build_deployment_only_checkpoint_tensors(tensors)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(
        deployment_only_tensors,
        output_path,
        metadata=_embedded_checkpoint_metadata(checkpoint_metadata, deployment_only=True),
    )
    return output_path


def save_grouped_payload_sidecar(
    model,
    output_path: str,
    include_model_state: bool = False,
    deployment_only: bool = False,
    checkpoint_metadata: Optional[dict] = None,
) -> str:
    tensors = build_grouped_runtime_payload_tensors(model, include_model_state=include_model_state)
    if deployment_only:
        tensors = build_deployment_only_checkpoint_tensors(tensors)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = (
        _embedded_checkpoint_metadata(checkpoint_metadata, deployment_only=deployment_only)
        if include_model_state
        else dict(GROUPED_PAYLOAD_METADATA)
    )
    save_file(tensors, output_path, metadata=metadata)
    return output_path


def save_grouped_qgmap_staging_sidecar(
    model,
    output_path: str,
    checkpoint_metadata: Optional[dict] = None,
) -> str:
    tensors = build_grouped_qgmap_staging_sidecar_tensors(model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(tensors, output_path, metadata=_grouped_qgmap_staging_metadata(checkpoint_metadata))
    return output_path


def split_grouped_qgmap_staging_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    payload_tensors: dict[str, torch.Tensor] = {}
    qgmap_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        target = qgmap_tensors if _is_grouped_qgmap_tensor_name(name) else payload_tensors
        target[name] = tensor
    return payload_tensors, qgmap_tensors


def rewrite_grouped_payload_for_qgmap_staging(
    source_path: str,
    payload_output_path: str,
    qgmap_output_path: str,
) -> tuple[str, str]:
    tensors, checkpoint_metadata = _load_safetensors_to_cpu(source_path)
    payload_tensors, qgmap_tensors = split_grouped_qgmap_staging_tensors(tensors)
    if not qgmap_tensors:
        raise ValueError(f"No grouped qgmap tensors were found in {source_path}")

    os.makedirs(os.path.dirname(payload_output_path), exist_ok=True)
    save_file(
        payload_tensors,
        payload_output_path,
        metadata=_embedded_checkpoint_metadata(
            checkpoint_metadata,
            deployment_only=checkpoint_metadata.get(GROUPED_PAYLOAD_DEPLOYMENT_ONLY_METADATA_KEY) == "v1",
        ),
    )

    os.makedirs(os.path.dirname(qgmap_output_path), exist_ok=True)
    save_file(
        qgmap_tensors,
        qgmap_output_path,
        metadata=_grouped_qgmap_staging_metadata(checkpoint_metadata),
    )
    return payload_output_path, qgmap_output_path


def merge_grouped_payload_with_checkpoint(
    checkpoint_path: str,
    sidecar_path: str,
    output_path: str,
    deployment_only: bool = False,
) -> str:
    checkpoint_tensors, checkpoint_metadata = _load_safetensors_to_cpu(checkpoint_path)
    sidecar_tensors, _ = _load_safetensors_to_cpu(sidecar_path)
    overlap = set(checkpoint_tensors).intersection(sidecar_tensors)
    if overlap:
        overlap_preview = ", ".join(sorted(overlap)[:8])
        raise ValueError(f"Grouped payload merge found overlapping tensor keys: {overlap_preview}")

    if deployment_only:
        checkpoint_tensors = build_deployment_only_checkpoint_tensors(checkpoint_tensors)
    checkpoint_tensors.update(sidecar_tensors)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(
        checkpoint_tensors,
        output_path,
        metadata=_embedded_checkpoint_metadata(checkpoint_metadata, deployment_only=deployment_only),
    )
    return output_path


def load_grouped_payload_from_sidecar(
    sidecar_path: str,
    layer_prefix: str,
    cache_key: str,
    device: torch.device,
    dtype: torch.dtype,
    exclude_names=None,
) -> Optional[dict]:
    if not sidecar_path or not os.path.exists(sidecar_path):
        return None

    with safe_open(sidecar_path, framework="pt", device="cpu") as reader:
        return load_grouped_payload_from_reader(
            reader,
            layer_prefix,
            cache_key,
            device,
            dtype,
            exclude_names=exclude_names,
        )


def _finalize_decoded_grouped_payload(cache_key: str, payload: dict, dtype: torch.dtype) -> Optional[dict]:
    if payload is None:
        return None

    if not payload:
        return None
    if not _sidecar_payload_is_compatible(cache_key, payload):
        return None
    payload = _repair_grouped_payload_compatibility(payload, dtype)
    payload = _normalize_grouped_payload_runtime_scalars(payload)
    if "g_idx" in payload:
        prepare_compact3_k64_groups(payload)
    return payload


def load_grouped_payload_from_reader(
    reader,
    layer_prefix: str,
    cache_key: str,
    device: torch.device,
    dtype: torch.dtype,
    exclude_names=None,
) -> Optional[dict]:
    excluded = set(exclude_names or ())
    prefix = _payload_tensor_name(layer_prefix, cache_key, "")
    payload = {}
    matched = False
    for key in reader.keys():
        if not key.startswith(prefix):
            continue
        matched = True
        name = key[len(prefix):]
        if name in excluded:
            continue
        payload[name] = _decode_payload_tensor(name, reader.get_tensor(key), device, dtype)

    if not matched:
        return None
    return _finalize_decoded_grouped_payload(cache_key, payload, dtype)


def load_grouped_payloads_from_reader(
    reader,
    layer_prefix: str,
    cache_keys,
    device: torch.device,
    dtype: torch.dtype,
    exclude_names_by_cache=None,
) -> dict[str, dict]:
    requested = set(cache_keys)
    if not requested:
        return {}

    exclude_names_by_cache = exclude_names_by_cache or {}
    prefix = f"{layer_prefix}.grouped_payload."
    payloads: dict[str, dict] = {cache_key: {} for cache_key in requested}
    matched: set[str] = set()
    for key in reader.keys():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        if "." not in remainder:
            continue
        cache_key, name = remainder.split(".", 1)
        if cache_key not in requested:
            continue
        matched.add(cache_key)
        if name in set(exclude_names_by_cache.get(cache_key) or ()):
            continue
        payloads[cache_key][name] = _decode_payload_tensor(name, reader.get_tensor(key), device, dtype)

    decoded = {}
    for cache_key in requested:
        if cache_key not in matched:
            continue
        payload = _finalize_decoded_grouped_payload(cache_key, payloads[cache_key], dtype)
        if payload is not None:
            decoded[cache_key] = payload
    return decoded


def load_grouped_qgmap_from_sidecar(
    sidecar_path: str,
    layer_prefix: str,
    cache_key: str,
    device="cpu",
    pin_memory: bool = False,
) -> Optional[dict[str, torch.Tensor]]:
    if not sidecar_path or not os.path.exists(sidecar_path) or cache_key not in GROUPED_PRECOMPUTED_KEYS:
        return None

    target_device = torch.device(device)
    prefix = _payload_tensor_name(layer_prefix, cache_key, "")
    payload: dict[str, torch.Tensor] = {}
    with safe_open(sidecar_path, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name not in GROUPED_QGMAP_PAYLOAD_NAMES:
                continue
            tensor = reader.get_tensor(key).contiguous()
            if target_device.type == "cpu":
                if pin_memory:
                    tensor = tensor.pin_memory()
            else:
                tensor = tensor.to(device=target_device, dtype=torch.int32)
            payload[name] = tensor.contiguous()

    if any(name not in payload for name in GROUPED_QGMAP_PAYLOAD_NAMES):
        return None
    return payload


class GroupedQGmapCPUStagingManager:
    """Reuse one bitmap workspace across projections, experts, and decoder layers.

    Sparse grouped projections use two-slot chunking to overlap copies and
    compute within the same workspace. Whole-projection staging is reserved for
    non-expert linears and grouped payloads or routes that cannot be chunked.
    Block-GR payloads keep their existing resident path.
    """

    def __init__(self, decoder_layers):
        self.decoder_layers = list(decoder_layers)
        self.layer_entries = []
        self.device = None
        self.enabled = _grouped_qgmap_cpu_staging_enabled()
        self._initialized = False
        self.buffer = None
        self._active_entry = None
        self._last_stream = None
        self._copy_stream = None
        self._pipeline_slots = None
        self._last_use_pipeline = False
        self.pipeline_stats = {"calls": 0, "chunks": 0, "bytes": 0, "slot_reuses": 0}

    @property
    def buffer_nbytes(self) -> int:
        return 0 if self.buffer is None else self.buffer.numel() * self.buffer.element_size()

    def matches_device(self, device: torch.device) -> bool:
        if self.device is None:
            return True
        return _device_cache_key(self.device) == _device_cache_key(device)

    def _load_cpu_qgmap_payload(self, moe_block, cache_key: str, payload: dict) -> Optional[dict[str, torch.Tensor]]:
        sidecar_getter = getattr(moe_block, "_get_grouped_qgmap_staging_sidecar_path", None)
        sidecar_path = sidecar_getter() if callable(sidecar_getter) else None
        sidecar_payload = load_grouped_qgmap_from_sidecar(
            sidecar_path,
            getattr(moe_block, "_grouped_payload_prefix", ""),
            cache_key,
            device="cpu",
            pin_memory=True,
        )
        if sidecar_payload is not None:
            return sidecar_payload if sidecar_payload["q_gmap_sparse_prefix"].numel() else None

        sparse = payload.get("q_gmap_sparse")
        prefix = payload.get("q_gmap_sparse_prefix")
        if not isinstance(sparse, torch.Tensor) or not isinstance(prefix, torch.Tensor):
            return None
        if prefix.numel() == 0:
            return None
        cpu_sparse = torch.empty_like(sparse, device="cpu", pin_memory=True)
        cpu_sparse.copy_(sparse, non_blocking=False)
        cpu_prefix = torch.empty_like(prefix, device="cpu", pin_memory=True)
        cpu_prefix.copy_(prefix, non_blocking=False)
        return {
            "q_gmap_sparse": cpu_sparse.contiguous(),
            "q_gmap_sparse_prefix": cpu_prefix.contiguous(),
        }

    def initialize(self) -> bool:
        if self._initialized:
            return self.enabled

        if not self.enabled or not torch.cuda.is_available():
            self._initialized = True
            self.enabled = False
            return False
        if not any(
            getattr(moe_block, "_deployment_only_active", False)
            for layer_idx, layer in enumerate(self.decoder_layers)
            for _, moe_block, _ in _grouped_moe_specs_for_layer(layer_idx, layer)
        ):
            # Streaming requires grouped deployment payloads. Raw checkpoints
            # keep their ordinary resident expert and attention bitmaps.
            self._initialized = True
            self.enabled = False
            return False

        detected_device = None
        layer_entries = []
        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            cache_entries = []
            for _, moe_block, _ in _grouped_moe_specs_for_layer(layer_idx, decoder_layer):
                if moe_block is None or not getattr(moe_block, "_deployment_only_active", False):
                    continue
                for cache_key in GROUPED_PRECOMPUTED_KEYS:
                    cache_entry = getattr(moe_block, "_grouped_payload_cache", {}).get(cache_key)
                    if cache_entry is None:
                        continue
                    payload = cache_entry.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if not bool(payload.get("is_compact3", False)) or bool(payload.get("compact3_use_block_gr", False)):
                        continue
                    qweight1 = payload.get("qweight1")
                    if not isinstance(qweight1, torch.Tensor) or qweight1.device.type != "cuda":
                        continue
                    cpu_payload = self._load_cpu_qgmap_payload(moe_block, cache_key, payload)
                    if cpu_payload is None:
                        raise RuntimeError(f"Missing streaming bitmap for layer {layer_idx}, projection {cache_key}")
                    if detected_device is None:
                        detected_device = qweight1.device
                    if detected_device != qweight1.device:
                        raise RuntimeError("Single-buffer bitmap streaming requires decoder layers on one CUDA device")
                    cache_entries.append(
                        {
                            "payload": payload,
                            "cpu_sparse": cpu_payload["q_gmap_sparse"],
                            "cpu_prefix": cpu_payload["q_gmap_sparse_prefix"],
                        }
                    )

            # Preserve block-GR, including lazy conversion requested by its switch.
            if not _compact3_block_gr_enabled():
                for module in decoder_layer.modules():
                    if not isinstance(module, QuantLinear) or module._mode_host != 4:
                        continue
                    if module.q_gmap.numel() > 0 or (
                        module.q_gmap_gr_offsets.numel() > 0 and module.q_gmap_gr_meta.numel() > 0
                    ):
                        continue
                    if module.qweight1.device.type != "cuda" or module.q_gmap_sparse_prefix.numel() == 0:
                        continue
                    if detected_device is None:
                        detected_device = module.qweight1.device
                    if detected_device != module.qweight1.device:
                        raise RuntimeError("Single-buffer bitmap streaming requires decoder layers on one CUDA device")
                    cache_entries.append(
                        {
                            "module": module,
                            "cpu_sparse": module.q_gmap_sparse.detach().to("cpu").contiguous().pin_memory(),
                            "cpu_prefix": module.q_gmap_sparse_prefix.detach().to("cpu").contiguous().pin_memory(),
                        }
                    )
            layer_entries.append(tuple(cache_entries))

        if detected_device is None or detected_device.type != "cuda" or not any(layer_entries):
            self._initialized = True
            self.enabled = False
            return False

        entries = [entry for layer in layer_entries for entry in layer]
        # Validate and allocate before taking ownership of any payload. A bad
        # entry must not leave earlier projections unbound after a failed init.
        for entry in entries:
            for name in ("cpu_sparse", "cpu_prefix"):
                if entry[name].dtype != torch.int32:
                    raise TypeError("Streaming bitmap payloads must use int32 packing")
        # One allocation, with prefix indices packed behind the bitmap. Taking
        # max(bitmap + prefix) avoids reserving two independently sized buffers.
        capacity = max(entry["cpu_sparse"].numel() + entry["cpu_prefix"].numel() for entry in entries)
        self.buffer = torch.empty(capacity, dtype=torch.int32, device=detected_device)
        self.device = detected_device
        self.layer_entries = layer_entries
        for entry in entries:
            entry["empty_sparse"] = _empty_tensor_like_rank(entry["cpu_sparse"], self.device)
            entry["empty_prefix"] = _empty_tensor_like_rank(entry["cpu_prefix"], self.device)
            self._unbind_entry(entry)
            if "payload" in entry:
                entry["payload"]["_qgmap_staging"] = (self, entry)
            else:
                object.__setattr__(entry["module"], "_qgmap_staging", (self, entry))
        self._last_stream = torch.cuda.current_stream(device=self.device)
        self._initialized = True
        return True

    @staticmethod
    def _bind_entry(entry: dict, sparse: torch.Tensor, prefix: torch.Tensor) -> None:
        if "payload" in entry:
            entry["payload"]["q_gmap_sparse"] = sparse
            entry["payload"]["q_gmap_sparse_prefix"] = prefix
        else:
            entry["module"].q_gmap_sparse = sparse
            entry["module"].q_gmap_sparse_prefix = prefix

    def _unbind_entry(self, entry: dict) -> None:
        self._bind_entry(entry, entry["empty_sparse"], entry["empty_prefix"])

    def stage(self, entry: dict) -> None:
        """Stage a whole projection for consumers that cannot use chunking."""
        stream = torch.cuda.current_stream(device=self.device)
        # Copies and consumers share the current stream. A stream switch must
        # also wait for the previous consumer before overwriting this workspace.
        if self._last_stream is not None and self._last_stream != stream:
            stream.wait_stream(self._last_stream)
        if self._copy_stream is not None:
            stream.wait_stream(self._copy_stream)
        self._last_stream = stream
        self._last_use_pipeline = False
        self.buffer.record_stream(stream)
        if self._active_entry is entry:
            return
        if self._active_entry is not None:
            self._unbind_entry(self._active_entry)
        sparse_numel = entry["cpu_sparse"].numel()
        prefix_numel = entry["cpu_prefix"].numel()
        sparse = self.buffer[:sparse_numel].view(entry["cpu_sparse"].shape)
        prefix = self.buffer[sparse_numel:sparse_numel + prefix_numel].view(entry["cpu_prefix"].shape)
        sparse.copy_(entry["cpu_sparse"], non_blocking=True)
        prefix.copy_(entry["cpu_prefix"], non_blocking=True)
        self._bind_entry(entry, sparse, prefix)
        self._active_entry = entry

    def can_pipeline(self, entry: dict) -> bool:
        if "payload" not in entry:
            return False
        experts = entry["cpu_sparse"].size(0)
        words = entry["cpu_sparse"].numel() + entry["cpu_prefix"].numel()
        half = self.buffer.numel() // 2
        return experts > 1 and half < words and 0 < words // experts <= half

    def _pipeline_jobs(self, entry: dict, route: dict):
        experts = entry["cpu_sparse"].size(0)
        words_per_expert = (entry["cpu_sparse"].numel() + entry["cpu_prefix"].numel()) // experts
        chunk_size = (self.buffer.numel() // 2) // words_per_expert
        cache = route.setdefault("_bitmap_pipeline_jobs", {})
        key = (experts, chunk_size)
        if key in cache:
            return cache[key]
        host = route.get("_bitmap_host_route")
        if host is None:
            count = route["expert_ids"].numel()
            packed = torch.cat((route["expert_ids"], route["segment_offsets"], route["problem_offsets"])).cpu().tolist()
            host = (packed[:count], packed[count:2 * count + 1], packed[2 * count + 1:])
            route["_bitmap_host_route"] = host
        ids, rows, tiles = host
        # Custom routes can use arbitrary expert order. Preserve their original
        # full-buffer execution instead of changing the order of routed rows.
        if ids != sorted(ids) or any(expert < 0 or expert >= experts for expert in ids):
            cache[key] = None
            return None
        specs = []
        values = []
        for start in range(0, experts, chunk_size):
            end = min(start + chunk_size, experts)
            first, last = bisect_left(ids, start), bisect_left(ids, end)
            if first == last:
                continue
            row_start, row_end = rows[first], rows[last]
            tile_start = tiles[first]
            offset = len(values)
            values.extend(expert - start for expert in ids[first:last])
            values.extend(row - row_start for row in rows[first:last + 1])
            values.extend(tile - tile_start for tile in tiles[first:last + 1])
            # Keep the grouped layout, but leave inactive expert bitmap positions
            # untouched: no routed segment can read their bitmaps.
            # Adjacent active experts share a DMA rather than a kernel launch.
            ranges = []
            for expert in ids[first:last]:
                local = expert - start
                if ranges and local <= ranges[-1][1]:
                    ranges[-1] = (ranges[-1][0], max(ranges[-1][1], local + 1))
                else:
                    ranges.append((local, local + 1))
            specs.append((start, end, row_start, row_end, first, last, offset, ranges))
        # One pinned, asynchronous metadata upload per chunk layout. Creating
        # CUDA tensors from Python lists separately would synchronize every job.
        metadata = torch.tensor(values, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        jobs = []
        for start, end, row_start, row_end, first, last, offset, ranges in specs:
            count = last - first
            local_metadata = metadata[offset:offset + 3 * count + 2]
            local_route = {
                "expert_ids": local_metadata[:count],
                "segment_offsets": local_metadata[count:2 * count + 1],
                "problem_offsets": local_metadata[2 * count + 1:],
                "total_problem_tiles": torch.tensor([tiles[last] - tiles[first]], dtype=torch.int64),
                "_bitmap_copy_ranges": ranges,
            }
            jobs.append((start, end, row_start, row_end, local_route))
        cache[key] = jobs
        return jobs

    @staticmethod
    def _launch_grouped(kernel, hidden, route, payload, sparse, prefix, bias, has_bias):
        return kernel(
            hidden, route["expert_ids"], route["segment_offsets"], route["problem_offsets"],
            route["total_problem_tiles"], payload["qweight1"], payload["qweight2_sparse"],
            payload["qweight2_sparse_prefix"], sparse, prefix, payload["c_map"], payload["g_idx"],
            payload["col_pack3"], payload["row_pack3"], payload["output_scale"], bias, has_bias,
        )

    def run_grouped(self, entry, hidden, route, kernel, bias, has_bias):
        """Overlap H2D of the next expert range with the current range's GEMM."""
        jobs = self._pipeline_jobs(entry, route)
        payload = entry["payload"]
        if jobs is None:
            self.stage(entry)
            return self._launch_grouped(kernel, hidden, route, payload, payload["q_gmap_sparse"],
                                        payload["q_gmap_sparse_prefix"], bias, has_bias)
        output = hidden.new_empty((hidden.size(0), payload["qweight1"].size(2)))
        if not jobs:
            return output.zero_()
        compute = torch.cuda.current_stream(device=self.device)
        if self._last_stream is not None and self._last_stream != compute:
            compute.wait_stream(self._last_stream)
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._pipeline_slots = [
                {"ready": torch.cuda.Event(), "done": torch.cuda.Event(), "used": False}
                for _ in range(2)
            ]
        compute.wait_stream(self._copy_stream)
        copy_stream = self._copy_stream
        if not self._last_use_pipeline:
            # A whole-buffer consumer can use both slots. Between pipeline
            # calls the per-slot done events suffice, allowing the next copy
            # to overlap the preceding projection's remaining computation.
            copy_stream.wait_stream(compute)
        self._last_use_pipeline = True
        self._last_stream = compute
        self.buffer.record_stream(compute)
        self.buffer.record_stream(copy_stream)
        if self._active_entry is not None:
            self._unbind_entry(self._active_entry)
            self._active_entry = None
        half = self.buffer.numel() // 2
        chunks = entry.setdefault("_pipeline_payloads", {})
        chunk_views = entry.setdefault("_pipeline_views", {})
        views = {}

        def prefetch(index):
            start, end, _, _, local_route = jobs[index]
            slot_id = index % 2
            slot = self._pipeline_slots[slot_id]
            key = (start, end, slot_id)
            if key not in chunk_views:
                cpu_sparse = entry["cpu_sparse"][start:end]
                cpu_prefix = entry["cpu_prefix"][start:end]
                offset = slot_id * half
                sparse = self.buffer[offset:offset + cpu_sparse.numel()].view(cpu_sparse.shape)
                prefix = self.buffer[offset + cpu_sparse.numel():offset + cpu_sparse.numel() + cpu_prefix.numel()].view(cpu_prefix.shape)
                chunk_views[key] = (cpu_sparse, cpu_prefix, sparse, prefix)
            cpu_sparse, cpu_prefix, sparse, prefix = chunk_views[key]
            sparse_expert_bytes = cpu_sparse.numel() // (end - start) * 4
            with torch.cuda.stream(copy_stream):
                if slot["used"]:
                    copy_stream.wait_event(slot["done"])
                    self.pipeline_stats["slot_reuses"] += 1
                for first, last in local_route["_bitmap_copy_ranges"]:
                    sparse[first:last].copy_(cpu_sparse[first:last], non_blocking=True)
                    self.pipeline_stats["bytes"] += (last - first) * sparse_expert_bytes
                # Prefixes are tiny. One contiguous transfer is cheaper than a
                # separate DMA for every active expert/run.
                prefix.copy_(cpu_prefix, non_blocking=True)
                self.pipeline_stats["bytes"] += cpu_prefix.numel() * 4
                slot["ready"].record(copy_stream)
            views[index] = (sparse, prefix)

        self.pipeline_stats["calls"] += 1
        prefetch(0)
        for index, (start, end, row_start, row_end, local_route) in enumerate(jobs):
            slot = self._pipeline_slots[index % 2]
            compute.wait_event(slot["ready"])
            key = (start, end)
            if key not in chunks:
                chunks[key] = {name: payload[name][start:end] for name in (
                    "qweight1", "qweight2_sparse", "qweight2_sparse_prefix", "c_map", "g_idx",
                    "col_pack3", "row_pack3", "output_scale",
                )}
            sparse, prefix = views.pop(index)
            result = self._launch_grouped(
                kernel, hidden[row_start:row_end], local_route, chunks[key], sparse, prefix,
                bias[start:end] if has_bias else bias, has_bias,
            )
            output[row_start:row_end].copy_(result)
            slot["done"].record(compute)
            slot["used"] = True
            self.pipeline_stats["chunks"] += 1
            if index + 1 < len(jobs):
                prefetch(index + 1)
        return output

    def prepare_for_layer(self, layer_idx: int) -> None:
        # Loading happens immediately before each projection kernel, not here:
        # prefetching a second layer would exceed the single-projection budget.
        self.initialize()

    def finalize_forward(self) -> None:
        if self._active_entry is not None:
            self._unbind_entry(self._active_entry)
            self._active_entry = None


def maybe_create_grouped_qgmap_cpu_staging_manager(decoder_layers):
    if not _grouped_qgmap_cpu_staging_enabled():
        return None
    manager = GroupedQGmapCPUStagingManager(decoder_layers)
    return manager if manager.initialize() else None
