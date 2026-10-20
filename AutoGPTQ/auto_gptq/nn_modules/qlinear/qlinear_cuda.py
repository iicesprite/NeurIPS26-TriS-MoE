import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

TRIS_COMPACT3_KERNEL_ABI_VERSION = 2
_TRIS_COMPACT3_KERNEL_ENTRYPOINTS = (
    "compact3_linear_forward",
    "compact3_qkv_linear_forward",
    "compact3_dual_linear_forward",
    "compact3_tris_precomputed_grouped_forward_block_gr",
    "compact3_tris_precomputed_grouped_dual_forward_block_gr",
    "compact3_tris_precomputed_grouped_block_gr_predecode",
    "compact3_tris_precomputed_grouped_forward_block_gr_predecoded",
    "compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded",
    "compact3_tris_precomputed_grouped_forward_block_gr_predecoded_hybrid",
    "compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_hybrid",
)


def _require_TriS_kernel_abi(module):
    version = getattr(module, "TRIS_COMPACT3_KERNEL_ABI_VERSION", None)
    missing = [name for name in _TRIS_COMPACT3_KERNEL_ENTRYPOINTS if not callable(getattr(module, name, None))]
    if version != TRIS_COMPACT3_KERNEL_ABI_VERSION or missing:
        raise ImportError(
            f"Incompatible TriS CUDA extension at {getattr(module, '__file__', '<unknown>')}: "
            f"expected compact3 ABI {TRIS_COMPACT3_KERNEL_ABI_VERSION}, got {version}; "
            f"missing entrypoints: {missing}. Rebuild with BUILD_TRIS_ONLY=1."
        )


try:
    import autogptq_TriS_cuda

    _require_TriS_kernel_abi(autogptq_TriS_cuda)
    _autogptq_TriS_cuda_available = True
    _autogptq_TriS_cuda_load_attempted = True
    _autogptq_TriS_cuda_load_error = None
except ImportError as exc:
    sys.modules.pop("autogptq_TriS_cuda", None)
    autogptq_TriS_cuda = None
    _autogptq_TriS_cuda_available = False
    _autogptq_TriS_cuda_load_attempted = False
    _autogptq_TriS_cuda_load_error = exc


def is_TriS_cuda_available(load_if_needed: bool = False) -> bool:
    if _autogptq_TriS_cuda_available:
        return True
    if not load_if_needed:
        return False
    return _load_TriS_extension() is not None


def _import_TriS_shared_object(shared_object: Path):
    global autogptq_TriS_cuda
    spec = importlib.util.spec_from_file_location("autogptq_TriS_cuda", shared_object)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load extension from {shared_object}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require_TriS_kernel_abi(module)
    sys.modules["autogptq_TriS_cuda"] = module
    autogptq_TriS_cuda = module
    return module


def _find_existing_TriS_shared_object(root: Path):
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidates = []
    search_dirs = (
        root,
        root / "build" / "autogptq_TriS_cuda",
        root / "build",
    )
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        candidates.extend(search_dir.rglob(f"autogptq_TriS_cuda*{ext_suffix}"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_TriS_extension_with_setuptools(root: Path, ext_dir: Path):
    build_dir = root / "build" / "autogptq_TriS_cuda"
    build_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

    setup_py = build_dir / "setup.py"
    setup_code = "\n".join([
        "from setuptools import setup",
        "from torch.utils.cpp_extension import BuildExtension, CUDAExtension",
        "",
        "setup(",
        '    name="autogptq_TriS_cuda",',
        "    ext_modules=[",
        "        CUDAExtension(",
        '            name="autogptq_TriS_cuda",',
        "            sources=[",
        f'                r"{(ext_dir / "TriSMoE_cuda.cpp").as_posix()}",',
        f'                r"{(ext_dir / "TriSMoE_cuda_kernel.cu").as_posix()}",',
        "            ],",
        f'            extra_compile_args={{"cxx": ["-O3", "-DTRIS_GROUPED_M_TILE={GROUPED_SEGMENT_M_TILE}"], "nvcc": ["-O3", "-DTRIS_GROUPED_M_TILE={GROUPED_SEGMENT_M_TILE}"]}},',
        "        )",
        "    ],",
        '    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},',
        ")",
        "",
    ])
    setup_py.write_text(setup_code)

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    subprocess.run(
        [sys.executable, str(setup_py), "build_ext", "--inplace"],
        cwd=build_dir,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    built_candidates = sorted(build_dir.glob(f"autogptq_TriS_cuda*{ext_suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not built_candidates:
        raise FileNotFoundError("autogptq_TriS_cuda shared object was not produced")
    return _import_TriS_shared_object(built_candidates[0])


def _load_TriS_extension():
    global autogptq_TriS_cuda
    global _autogptq_TriS_cuda_available
    global _autogptq_TriS_cuda_load_attempted
    global _autogptq_TriS_cuda_load_error

    if _autogptq_TriS_cuda_available:
        return autogptq_TriS_cuda
    cached_module = sys.modules.get("autogptq_TriS_cuda")
    if cached_module is not None:
        try:
            _require_TriS_kernel_abi(cached_module)
        except ImportError as exc:
            sys.modules.pop("autogptq_TriS_cuda", None)
            _autogptq_TriS_cuda_load_error = exc
        else:
            autogptq_TriS_cuda = cached_module
            _autogptq_TriS_cuda_available = True
            _autogptq_TriS_cuda_load_error = None
            return autogptq_TriS_cuda
    if _autogptq_TriS_cuda_load_attempted:
        return None
    _autogptq_TriS_cuda_load_attempted = True
    if not torch.cuda.is_available():
        return None

    root = Path(__file__).resolve().parents[3]
    ext_dir = root / "autogptq_extension" / "TriSMoE"
    existing_shared_object = _find_existing_TriS_shared_object(root)

    if existing_shared_object is not None:
        try:
            autogptq_TriS_cuda = _import_TriS_shared_object(existing_shared_object)
            _autogptq_TriS_cuda_available = True
            _autogptq_TriS_cuda_load_error = None
            return autogptq_TriS_cuda
        except Exception as exc:
            _autogptq_TriS_cuda_load_error = exc

    try:
        from torch.utils.cpp_extension import load

        autogptq_TriS_cuda = load(
            name="autogptq_TriS_cuda",
            sources=[
                str(ext_dir / "TriSMoE_cuda.cpp"),
                str(ext_dir / "TriSMoE_cuda_kernel.cu"),
            ],
            extra_cflags=["-O3", f"-DTRIS_GROUPED_M_TILE={GROUPED_SEGMENT_M_TILE}"],
            extra_cuda_cflags=["-O3", f"-DTRIS_GROUPED_M_TILE={GROUPED_SEGMENT_M_TILE}"],
            verbose=False,
        )
        _require_TriS_kernel_abi(autogptq_TriS_cuda)
    except Exception as exc:
        try:
            autogptq_TriS_cuda = _build_TriS_extension_with_setuptools(root, ext_dir)
        except Exception as fallback_exc:
            _autogptq_TriS_cuda_load_error = fallback_exc
            if "ninja" not in str(exc).lower():
                _autogptq_TriS_cuda_load_error = RuntimeError(
                    f"load() failed: {exc}; setuptools fallback failed: {fallback_exc}"
                )
            return None

    _autogptq_TriS_cuda_available = True
    _autogptq_TriS_cuda_load_error = None
    return autogptq_TriS_cuda


def _can_fuse_packed_ns(module: "QuantLinear") -> bool:
    c1_quant, c2_quant = module._get_effective_ns_quantized_pair()
    return module._ns_mode_host == 2 and c1_quant is not None and c2_quant is not None


def _env_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    return False


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be positive, got {parsed}")
    return parsed


def _grouped_segment_m_tile() -> int:
    return _env_positive_int("TRIS_GROUPED_M_TILE", 4)


def _compact3_block_gr_enabled() -> bool:
    """Use one Block-GR policy for grouped experts and ordinary linears."""
    value = os.environ.get("TRIS_COMPACT3_BLOCK_GR")
    if value is None or value == "":
        return True
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _compact3_tensor_core_device_supported(device: torch.device) -> bool:
    return torch.version.hip is None and int(torch.cuda.get_device_capability(device)[0]) >= 7


def _grouped_compact3_prefill_tc_min_full_tiles() -> int:
    # A separate TC launch is not worthwhile when only a handful of expert
    # segments contain a complete M16 tile. OLMoE A100 A/B measurements cross
    # over near the 128-token prefill regime, where each layer has ~40 tiles.
    return _env_positive_int("TRIS_GROUPED_COMPACT3_PREFILL_TC_MIN_FULL_TILES", 32)


def _grouped_compact3_prefill_tc_min_routed_rows() -> int:
    # This host-known guard avoids constructing hybrid offsets and synchronizing
    # their totals back to the CPU for prompt sizes below the measured crossover.
    return _env_positive_int("TRIS_GROUPED_COMPACT3_PREFILL_TC_MIN_ROUTED_ROWS", 1024)


def _shared_expert_dual_gate_up_enabled() -> bool:
    for name in ("TRIS_SHARED_EXPERT_DUAL_GATE_UP", "QWEN_SHARED_EXPERT_DUAL_GATE_UP"):
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        return str(value).strip().lower() not in {"0", "false", "no", "off"}
    return True


def _deployment_only_enabled() -> bool:
    return _env_flag_enabled("OLMOE_DEPLOYMENT_ONLY", "TRIS_DEPLOYMENT_ONLY")


def _raise_deployment_only_failure(label: str, reason: str | None = None):
    message = f"[TriSMoE deployment-only] {label}"
    if reason:
        message = f"{message}: {reason}"
    raise RuntimeError(message)


def _return_or_raise_none(label: str, reason: str):
    if _deployment_only_enabled():
        _raise_deployment_only_failure(label, reason)
    return None


def _print_cuda_fallback(label: str, exc: Exception):
    print(f"WARNING - [{label}] CUDA path failed, falling back: {exc}", flush=True)


def _empty_precomputed_kernel_tensors(device: torch.device, dtype: torch.dtype):
    return (
        torch.empty((0,), device=device, dtype=dtype),
        torch.empty((0, 0), device=device, dtype=torch.int32),
        torch.empty((0, 0), device=device, dtype=dtype),
        torch.empty((0,), device=device, dtype=torch.int32),
    )


def _device_cache_key(device: torch.device):
    return (device.type, device.index if device.index is not None else -1)


def _tensor_is_exact_all_ones(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return False
    return bool(torch.all(tensor == 1).item())


def _payload_flag(payload: dict, key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def _payload_has_output_scale(payload: dict) -> bool:
    if "has_output_scale" in payload:
        return _payload_flag(payload, "has_output_scale", default=False)
    return not _tensor_is_exact_all_ones(payload["output_scale"])


def _normalize_runtime_payload_scalars(payload: dict) -> dict:
    bool_names = {"is_compact3", "compact3_use_block_gr", "has_output_scale", "has_bias"}
    int_names = {"max_ns_rank", "infeatures", "outfeatures", "group_size"}
    for name in bool_names | int_names:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            continue
        scalar = value.detach().to(device="cpu").reshape(-1)[0].item()
        payload[name] = bool(scalar) if name in bool_names else int(scalar)
    return payload


_COMPACT3_K64_CACHE_TOKEN = object()


def _compact3_k64_cache_matches(payload: dict, g_idx: torch.Tensor) -> bool:
    cached = payload.get("_compact3_k64_groups_cache")
    return (
        isinstance(cached, tuple)
        and len(cached) == 4
        and cached[0] is _COMPACT3_K64_CACHE_TOKEN
        and cached[1] is g_idx
        and cached[2] == (g_idx._version, g_idx.data_ptr(), tuple(g_idx.shape), tuple(g_idx.stride()))
    )


def _get_prepared_compact3_k64_groups(payload: dict) -> bool:
    """Never scan in forward; unknown or changed indices use the general kernel."""
    g_idx = payload.get("g_idx")
    uniform = bool(
        isinstance(g_idx, torch.Tensor)
        and not g_idx.is_meta
        and not g_idx.is_inference()
        and _compact3_k64_cache_matches(payload, g_idx)
        and payload["_compact3_k64_groups_cache"][3]
    )
    payload["_compact3_k64_groups_uniform"] = uniform
    return uniform


def _compact3_k64_groups_are_uniform(g_idx: torch.Tensor) -> bool:
    """Check every K position, including partial tiles and every expert row."""
    if (
        not isinstance(g_idx, torch.Tensor)
        or g_idx.is_meta
        or g_idx.ndim not in (1, 2)
        or g_idx.numel() == 0
        or g_idx.dtype not in (torch.int32, torch.int64)
    ):
        return False
    indices = g_idx.detach().cpu().reshape(-1, g_idx.shape[-1])
    full_k = (indices.shape[1] // 64) * 64
    if full_k:
        tiles = indices[:, :full_k].reshape(indices.shape[0], -1, 64)
        if not bool(torch.all(tiles == tiles[:, :, :1]).item()):
            return False
    tail = indices[:, full_k:]
    return tail.numel() == 0 or bool(torch.all(tail == tail[:, :1]).item())


def prepare_compact3_k64_groups(payload: dict) -> bool:
    """Validate at load/preparation time; reuse only for the same unchanged tensor.

    Checkpoint flags are never trusted. Inference tensors lack a version counter,
    so own a tracked copy of their small index buffer before caching the result.
    The steady-state check reads host metadata only and never scans CUDA indices.
    """
    g_idx = payload.get("g_idx")
    if not isinstance(g_idx, torch.Tensor) or g_idx.is_meta or g_idx.numel() == 0:
        payload["_compact3_k64_groups_uniform"] = False
        payload.pop("_compact3_k64_groups_cache", None)
        return False
    if g_idx.is_inference():
        with torch.inference_mode(False):
            g_idx = g_idx.clone()
        payload["g_idx"] = g_idx
    stamp = (g_idx._version, g_idx.data_ptr(), tuple(g_idx.shape), tuple(g_idx.stride()))
    if _compact3_k64_cache_matches(payload, g_idx):
        uniform = payload["_compact3_k64_groups_cache"][3]
    else:
        uniform = _compact3_k64_groups_are_uniform(g_idx)
        payload["_compact3_k64_groups_cache"] = (_COMPACT3_K64_CACHE_TOKEN, g_idx, stamp, uniform)
    payload["_compact3_k64_groups_uniform"] = uniform
    return uniform


def _cache_stacked_compact3_k64_groups(payload: dict, uniform: bool) -> None:
    # A stack of independently verified rows has the same tile-wise property.
    # This avoids a new CUDA->CPU scan when transient grouped payloads are built.
    g_idx = payload["g_idx"]
    if g_idx.is_inference():
        with torch.inference_mode(False):
            g_idx = g_idx.clone()
        payload["g_idx"] = g_idx
    stamp = (g_idx._version, g_idx.data_ptr(), tuple(g_idx.shape), tuple(g_idx.stride()))
    payload["_compact3_k64_groups_cache"] = (_COMPACT3_K64_CACHE_TOKEN, g_idx, stamp, bool(uniform))
    payload["_compact3_k64_groups_uniform"] = bool(uniform)


def _route_total_problem_tiles_host(route: dict) -> int:
    value = route.get("total_problem_tiles_host")
    if value is not None:
        return int(value)
    tensor = route.get("total_problem_tiles")
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return 0
    return int(tensor.reshape(-1)[0].detach().to(device="cpu").item())


def _reset_skip_next_load_state_dict_flag(module: nn.Module) -> bool:
    skip = bool(getattr(module, "_skip_next_load_state_dict", False))
    if skip:
        object.__setattr__(module, "_skip_next_load_state_dict", False)
    return skip


def _mark_skip_next_load_state_dict(module: nn.Module) -> None:
    object.__setattr__(module, "_skip_next_load_state_dict", True)


def _assign_loaded_tensor(module: nn.Module, name: str, value: torch.Tensor) -> None:
    current = getattr(module, name, None)
    if isinstance(current, torch.Tensor):
        target_device = torch.device("cpu") if getattr(current, "is_meta", False) else current.device
        if value.device != target_device or value.dtype != current.dtype:
            value = value.to(device=target_device, dtype=current.dtype)
        elif not value.is_contiguous():
            value = value.contiguous()
    else:
        if value.device.type != "cpu":
            value = value.to(device="cpu")
        elif not value.is_contiguous():
            value = value.contiguous()
    setattr(module, name, value)


def _infer_packed_group_size_from_g_idx(packed_matrix) -> int:
    cols = int(getattr(packed_matrix, "_cols_host", 0))
    if cols <= 0:
        return 1
    g_idx = getattr(packed_matrix, "g_idx", None)
    if not isinstance(g_idx, torch.Tensor) or g_idx.numel() == 0:
        return cols
    g_idx_host = g_idx.detach().to(device="cpu", dtype=torch.int64).flatten()
    num_groups = int(g_idx_host.max().item()) + 1 if g_idx_host.numel() > 0 else 0
    if num_groups <= 1:
        return cols
    group_size = cols // num_groups
    if group_size <= 0:
        return cols
    if cols % num_groups != 0:
        return cols
    expected = torch.div(
        torch.arange(cols, dtype=torch.int64),
        group_size,
        rounding_mode="floor",
    )
    if expected.shape == g_idx_host.shape and torch.equal(expected, g_idx_host):
        return group_size
    return cols


BLOCK_GR_TILE_K = 64
BLOCK_GR_TILE_OUT = 64
BLOCK_GR_PACKED_ROWS = BLOCK_GR_TILE_K // 32
BLOCK_GR_WORDS_PER_BLOCK = BLOCK_GR_PACKED_ROWS * BLOCK_GR_TILE_OUT
BLOCK_GR_META_K_BITS = 4
BLOCK_GR_META_K_MASK = (1 << BLOCK_GR_META_K_BITS) - 1
BLOCK_GR_META_RAW_FLAG = 1 << 4
BLOCK_GR_META_ONE_COUNT_SHIFT = 8
BLOCK_GR_MAX_K = 7

# Fixed limits for each grouped Block-GR predecode payload, not the streaming workspace.
_GROUPED_BLOCK_GR_PREDECODE_ACTIVE_CAP = 64
_GROUPED_BLOCK_GR_PREDECODE_MAX_BYTES = 64 * 1024 * 1024

_GROUPED_RUNTIME_LOGGED = set()
_GROUPED_DUAL_RUNTIME_LOGGED = set()
_GROUPED_PREFILL_TC_LOGGED = set()
_GROUPED_NS_STATIC_LOGGED = set()
_COMPACT3_RUNTIME_LOGGED = set()
_COMPACT3_QKV_RUNTIME_LOGGED = set()
_GROUPED_RUNTIME_STATS = {}


def reset_grouped_runtime_logs():
    _GROUPED_RUNTIME_LOGGED.clear()
    _GROUPED_DUAL_RUNTIME_LOGGED.clear()
    _GROUPED_PREFILL_TC_LOGGED.clear()
    _GROUPED_NS_STATIC_LOGGED.clear()
    _COMPACT3_RUNTIME_LOGGED.clear()
    _COMPACT3_QKV_RUNTIME_LOGGED.clear()
    _GROUPED_RUNTIME_STATS.clear()


def get_grouped_runtime_stats(reset: bool = False):
    stats = {phase: dict(values) for phase, values in _GROUPED_RUNTIME_STATS.items()}
    if reset:
        reset_grouped_runtime_logs()
    return stats


def _module_debug_label(owner, suffix: str) -> str:
    base = getattr(owner, "_TriS_debug_name", None)
    if not base:
        base = f"<unnamed:{owner.__class__.__name__}:{id(owner):x}>"
    return f"{base}.{suffix}"


def _grouped_decode_path_enabled(batch: int, num_segments: int, total_problem_tiles: int) -> bool:
    return batch > 0 and num_segments == batch and total_problem_tiles == batch


def _decode_phase_enabled(x: torch.Tensor) -> bool:
    if x.dim() >= 3:
        return int(x.shape[-2]) == 1
    if x.dim() == 2:
        return int(x.shape[0]) == 1
    return False


_PREFILL_AUTO_SPLIT_K_MIN_K_TILES_PER_SLICE = 4


def _resolve_auto_split_k_slices_py(num_k_tiles: int, base_tiles: int, sm_count: int, decode_phase: bool):
    target_multiplier = 8 if decode_phase else 2
    target_tiles = sm_count * target_multiplier
    if base_tiles >= target_tiles:
        return 1, f"auto:{target_multiplier}xSM,base_tiles={base_tiles}>=target={target_tiles}"

    if not decode_phase and base_tiles >= sm_count:
        return 1, f"auto:prefill-base_tiles={base_tiles}>=1xSM={sm_count}"

    needed = (target_tiles + base_tiles - 1) // base_tiles
    max_split_k = num_k_tiles
    extra_desc = ""
    if not decode_phase:
        min_k_tiles_per_slice = _PREFILL_AUTO_SPLIT_K_MIN_K_TILES_PER_SLICE
        max_split_k = num_k_tiles // min_k_tiles_per_slice
        if max_split_k <= 1:
            return 1, (
                f"auto:prefill-k-guard,num_k_tiles={num_k_tiles},"
                f"min_k_tiles_per_slice={min_k_tiles_per_slice}"
            )
        extra_desc = f",prefill_max_split_k={max_split_k},min_k_tiles_per_slice={min_k_tiles_per_slice}"

    split_k = min(needed, max_split_k)
    if split_k <= 1:
        if decode_phase:
            return 1, f"auto:{target_multiplier}xSM,needed={needed},num_k_tiles={num_k_tiles}"
        return 1, (
            f"auto:prefill-guard,needed={needed},num_k_tiles={num_k_tiles},"
            f"prefill_max_split_k={max_split_k},min_k_tiles_per_slice={_PREFILL_AUTO_SPLIT_K_MIN_K_TILES_PER_SLICE}"
        )

    return split_k, (
        f"auto:{target_multiplier}xSM,base_tiles={base_tiles},target={target_tiles},"
        f"num_k_tiles={num_k_tiles}{extra_desc}"
    )


def _resolve_grouped_split_k_slices_py(rank: int, total_problem_tiles: int, out_tiles: int, device: torch.device, decode_phase: bool):
    num_k_tiles = (rank + 63) // 64
    if num_k_tiles <= 1 or total_problem_tiles <= 0 or out_tiles <= 0:
        return 1, "disabled"

    phase_env_name = "TRIS_GROUPED_DECODE_SPLIT_K" if decode_phase else "TRIS_GROUPED_PREFILL_SPLIT_K"
    env_value = os.environ.get(phase_env_name)
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:{phase_env_name}={requested}"
        return 1, f"env:{phase_env_name}={requested}"

    env_value = os.environ.get("TRIS_GROUPED_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:TRIS_GROUPED_SPLIT_K={requested}"
        return 1, f"env:TRIS_GROUPED_SPLIT_K={requested}"

    if device.type != "cuda":
        return 1, "auto:no-cuda"

    props = torch.cuda.get_device_properties(device)
    sm_count = int(getattr(props, "multi_processor_count", 0))
    if sm_count <= 0:
        return 1, "auto:no-sm"

    base_tiles = total_problem_tiles * out_tiles
    return _resolve_auto_split_k_slices_py(num_k_tiles, base_tiles, sm_count, decode_phase)


def _resolve_grouped_block_gr_split_k_slices_py(rank: int, total_problem_tiles: int, out_tiles: int, device: torch.device, decode_phase: bool):
    num_k_tiles = (rank + 63) // 64
    if num_k_tiles <= 1 or total_problem_tiles <= 0 or out_tiles <= 0:
        return 1, "disabled"

    phase_env_name = "TRIS_GROUPED_BLOCK_GR_DECODE_SPLIT_K" if decode_phase else "TRIS_GROUPED_BLOCK_GR_PREFILL_SPLIT_K"
    env_value = os.environ.get(phase_env_name)
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:{phase_env_name}={requested}"
        return 1, f"env:{phase_env_name}={requested}"

    env_value = os.environ.get("TRIS_GROUPED_BLOCK_GR_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:TRIS_GROUPED_BLOCK_GR_SPLIT_K={requested}"
        return 1, f"env:TRIS_GROUPED_BLOCK_GR_SPLIT_K={requested}"

    env_value = os.environ.get("TRIS_GROUPED_DECODE_SPLIT_K" if decode_phase else "TRIS_GROUPED_PREFILL_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            phase_env_name = "TRIS_GROUPED_DECODE_SPLIT_K" if decode_phase else "TRIS_GROUPED_PREFILL_SPLIT_K"
            return min(requested, num_k_tiles), f"env:{phase_env_name}={requested}"
        phase_env_name = "TRIS_GROUPED_DECODE_SPLIT_K" if decode_phase else "TRIS_GROUPED_PREFILL_SPLIT_K"
        return 1, f"env:{phase_env_name}={requested}"

    env_value = os.environ.get("TRIS_GROUPED_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:TRIS_GROUPED_SPLIT_K={requested}"
        return 1, f"env:TRIS_GROUPED_SPLIT_K={requested}"

    if device.type != "cuda":
        return 1, "auto:no-cuda"

    props = torch.cuda.get_device_properties(device)
    sm_count = int(getattr(props, "multi_processor_count", 0))
    if sm_count <= 0:
        return 1, "auto:no-sm"

    base_tiles = total_problem_tiles * out_tiles
    if not decode_phase:
        return _resolve_auto_split_k_slices_py(num_k_tiles, base_tiles, sm_count, False)

    target_multiplier = 32
    target_tiles = sm_count * target_multiplier
    if base_tiles >= target_tiles:
        return 1, f"auto:block-gr,{target_multiplier}xSM,base_tiles={base_tiles}>=target={target_tiles}"

    needed = (target_tiles + base_tiles - 1) // base_tiles
    split_k = 1
    while split_k < needed and split_k < num_k_tiles:
        split_k <<= 1
    split_k = min(split_k, num_k_tiles)
    return split_k, (
        f"auto:block-gr,{target_multiplier}xSM,pow2,base_tiles={base_tiles},target={target_tiles},"
        f"num_k_tiles={num_k_tiles}"
    )


def _resolve_compact3_block_gr_split_k_slices_py(rank: int, batch: int, out_tiles: int, device: torch.device, decode_phase: bool):
    num_k_tiles = (rank + 63) // 64
    if num_k_tiles <= 1 or batch <= 0 or out_tiles <= 0:
        return 1, "disabled"

    phase_env_name = "TRIS_COMPACT3_BLOCK_GR_DECODE_SPLIT_K" if decode_phase else "TRIS_COMPACT3_BLOCK_GR_PREFILL_SPLIT_K"
    env_value = os.environ.get(phase_env_name)
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:{phase_env_name}={requested}"
        return 1, f"env:{phase_env_name}={requested}"

    env_value = os.environ.get("TRIS_COMPACT3_BLOCK_GR_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:TRIS_COMPACT3_BLOCK_GR_SPLIT_K={requested}"
        return 1, f"env:TRIS_COMPACT3_BLOCK_GR_SPLIT_K={requested}"

    return _resolve_grouped_block_gr_split_k_slices_py(rank, batch, out_tiles, device, decode_phase)


def _resolve_shared_expert_compact3_block_gr_split_k_slices_py(
    *,
    rank: int,
    batch: int,
    out_tiles: int,
    device: torch.device,
    decode_phase: bool,
    label: str,
):
    if not decode_phase or batch <= 0:
        return None
    if ".shared_expert." not in label:
        return None

    num_k_tiles = (rank + 63) // 64
    if num_k_tiles <= 1 or out_tiles <= 0:
        return 1, "disabled"

    env_value = os.environ.get("TRIS_SHARED_EXPERT_COMPACT3_BLOCK_GR_DECODE_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:TRIS_SHARED_EXPERT_COMPACT3_BLOCK_GR_DECODE_SPLIT_K={requested}"
        return 1, f"env:TRIS_SHARED_EXPERT_COMPACT3_BLOCK_GR_DECODE_SPLIT_K={requested}"

    # Qwen shared_expert decode uses fixed compact3 shapes:
    # gate/up: 2048 -> 5632  (out_tiles=88, num_k_tiles=32)
    # down:    5632 -> 2048  (out_tiles=32, num_k_tiles=88)
    #
    # The generic block-GR decode heuristic targets very high CTA counts and can
    # overslice K badly on these small-batch shapes, causing repeated gmap decode
    # and partial accumulation overhead. Keep split-K small and shape-aware here.
    if out_tiles >= 64:
        if batch <= 1:
            split_k = 2
        else:
            split_k = 1
    else:
        if batch <= 1:
            split_k = 4
        elif batch <= 2:
            split_k = 2
        else:
            split_k = 1

    split_k = min(split_k, num_k_tiles)
    return split_k, f"auto:shared-expert-compact3,out_tiles={out_tiles},batch={batch},num_k_tiles={num_k_tiles}"


def _maybe_log_grouped_runtime(label: str, batch: int, num_segments: int, total_problem_tiles: int, split_k_values: dict[str, tuple[int, str]]):
    decode_hit = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
    phase = "decode" if decode_hit else "prefill"
    grouped_m_tile = 1 if decode_hit else GROUPED_SEGMENT_M_TILE
    phase_stats = _GROUPED_RUNTIME_STATS.setdefault(
        phase,
        {
            "calls": 0,
            "max_batch": 0,
            "max_segments": 0,
            "max_total_problem_tiles": 0,
            "grouped_m_tile": grouped_m_tile,
            "last_label": label,
        },
    )
    phase_stats["calls"] += 1
    phase_stats["max_batch"] = max(int(phase_stats["max_batch"]), int(batch))
    phase_stats["max_segments"] = max(int(phase_stats["max_segments"]), int(num_segments))
    phase_stats["max_total_problem_tiles"] = max(int(phase_stats["max_total_problem_tiles"]), int(total_problem_tiles))
    phase_stats["grouped_m_tile"] = grouped_m_tile
    phase_stats["last_label"] = label
    if phase in _GROUPED_RUNTIME_LOGGED:
        return
    _GROUPED_RUNTIME_LOGGED.add(phase)
    split_desc = ", ".join(f"{name}={value} ({source})" for name, (value, source) in split_k_values.items())
    print(
        f"INFO - [grouped-runtime] phase={phase}, label={label}: batch={batch}, segments={num_segments}, total_problem_tiles={total_problem_tiles}, "
        f"decode_path_hit={'yes' if decode_hit else 'no'}, decode_m_tile={grouped_m_tile}, {split_desc}",
        flush=True,
    )


def _maybe_log_grouped_dual_runtime(
    label: str,
    batch: int,
    num_segments: int,
    total_problem_tiles: int,
    *,
    variant: str,
    kernel_name: str,
    rank_a: int,
    rank_b: int,
    outfeatures_a: int,
    outfeatures_b: int,
    split_k_values: dict[str, tuple[int, str]],
):
    decode_hit = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
    phase = "decode" if decode_hit else "prefill"
    log_key = (phase, variant)
    if log_key in _GROUPED_DUAL_RUNTIME_LOGGED:
        return
    _GROUPED_DUAL_RUNTIME_LOGGED.add(log_key)
    out_tiles_max = (max(int(outfeatures_a), int(outfeatures_b)) + 31) // 32
    split_desc = ", ".join(f"{name}={value} ({source})" for name, (value, source) in split_k_values.items())
    print(
        f"INFO - [grouped-dual-runtime] phase={phase}, label={label}: variant={variant}, kernel={kernel_name}, "
        f"batch={batch}, segments={num_segments}, total_problem_tiles={total_problem_tiles}, "
        f"decode_path_hit={'yes' if decode_hit else 'no'}, rank_a={rank_a}, rank_b={rank_b}, "
        f"outfeatures_a={outfeatures_a}, outfeatures_b={outfeatures_b}, out_tiles_max={out_tiles_max}, {split_desc}",
        flush=True,
    )


def _maybe_log_compact3_runtime(
    label: str,
    batch: int,
    out_tiles: int,
    decode_phase: bool,
    use_block_gr: bool,
    split_k: int,
    split_k_source: str,
    fast_path_hit: bool,
    prefill_tc_hit: bool,
):
    phase = "decode" if decode_phase else "prefill"
    if phase in _COMPACT3_RUNTIME_LOGGED:
        return
    _COMPACT3_RUNTIME_LOGGED.add(phase)
    print(
        f"INFO - [compact3-runtime] phase={phase}, label={label}: batch={batch}, out_tiles={out_tiles}, "
        f"block_gr_hit={'yes' if use_block_gr else 'no'}, fast_path_hit={'yes' if fast_path_hit else 'no'}, "
        f"prefill_tc_hit={'yes' if prefill_tc_hit else 'no'}, split_k={split_k} ({split_k_source})",
        flush=True,
    )


def _compact3_qkv_projection_ready(module, device: torch.device, dtype: torch.dtype, outfeatures: int | None = None) -> bool:
    if getattr(module, "QUANT_TYPE", None) != "cuda":
        return False
    if getattr(module, "_mode_host", None) != 4:
        return False
    if getattr(module, "_ns_mode_host", None) != 0:
        return False
    if _can_fuse_packed_ns(module):
        return False
    if module.bias is None:
        return False
    if outfeatures is not None and int(module.outfeatures) != int(outfeatures):
        return False

    module._ensure_legacy_bitpacked_buffers(require_dense_qweight2=False)
    module._ensure_block_gr_gmap_buffers()

    required = (
        module.qweight1,
        module.qweight2_sparse,
        module.qweight2_sparse_prefix,
        module.q_gmap_gr_words,
        module.q_gmap_gr_offsets,
        module.q_gmap_gr_meta,
        module.c_map,
        module.g_idx,
        module.col_pack3,
        module.row_pack3,
    )
    if any(t.numel() == 0 for t in required):
        return False
    if module.q_gmap_gr_offsets.numel() == 0 or module.q_gmap_gr_meta.numel() == 0:
        return False
    if any(t.device != device for t in required if t.is_cuda):
        return False
    return module.bias.device == device and module.bias.dtype == dtype


def _compact3_dual_projection_ready(module, device: torch.device, outfeatures: int | None = None) -> bool:
    if getattr(module, "QUANT_TYPE", None) != "cuda":
        return False
    if getattr(module, "_mode_host", None) != 4:
        return False
    if getattr(module, "_ns_mode_host", None) != 0:
        return False
    if _can_fuse_packed_ns(module):
        return False
    if module.bias is not None:
        return False
    if outfeatures is not None and int(module.outfeatures) != int(outfeatures):
        return False

    module._ensure_legacy_bitpacked_buffers(require_dense_qweight2=False)
    module._ensure_block_gr_gmap_buffers()

    required = (
        module.qweight1,
        module.qweight2_sparse,
        module.qweight2_sparse_prefix,
        module.q_gmap_gr_words,
        module.q_gmap_gr_offsets,
        module.q_gmap_gr_meta,
        module.c_map,
        module.g_idx,
        module.col_pack3,
        module.row_pack3,
    )
    if any(t.numel() == 0 for t in required):
        return False
    if module.q_gmap_gr_offsets.numel() == 0 or module.q_gmap_gr_meta.numel() == 0:
        return False
    if any(t.device != device for t in required if t.is_cuda):
        return False
    return True


def run_compact3_qkv_block_gr_forward(q_proj, k_proj, v_proj, hidden_states: torch.Tensor):
    if not _compact3_block_gr_enabled():
        return None
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    if ext is None:
        return None
    kernel = getattr(ext, "compact3_qkv_linear_forward", None)
    if kernel is None:
        return None
    if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.float16:
        return None
    if hidden_states.shape[-1] != getattr(q_proj, "infeatures", None):
        return None
    if not _compact3_qkv_projection_ready(q_proj, hidden_states.device, hidden_states.dtype):
        return None
    outfeatures = int(q_proj.outfeatures)
    if not _compact3_qkv_projection_ready(k_proj, hidden_states.device, hidden_states.dtype, outfeatures):
        return None
    if not _compact3_qkv_projection_ready(v_proj, hidden_states.device, hidden_states.dtype, outfeatures):
        return None

    out_shape = hidden_states.shape[:-1] + (outfeatures,)
    x_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    if x_flat.shape[0] == 0:
        empty = x_flat.new_empty(out_shape)
        return empty, empty, empty

    batch = int(x_flat.size(0))
    out_tiles = (outfeatures + 63) // 64
    decode_phase = _decode_phase_enabled(hidden_states)
    split_k_slices, split_k_source = _resolve_compact3_block_gr_split_k_slices_py(
        int(q_proj.infeatures),
        batch,
        out_tiles,
        x_flat.device,
        decode_phase,
    )
    phase = "decode" if decode_phase else "prefill"
    if phase not in _COMPACT3_QKV_RUNTIME_LOGGED:
        _COMPACT3_QKV_RUNTIME_LOGGED.add(phase)
        print(
            f"INFO - [compact3-qkv-runtime] phase={phase}, batch={batch}, out_tiles={out_tiles}, "
            f"split_k={split_k_slices} ({split_k_source})",
            flush=True,
        )

    try:
        fused = kernel(
            x_flat,
            q_proj.qweight1.contiguous(),
            q_proj.qweight2_sparse.contiguous(),
            q_proj.qweight2_sparse_prefix.contiguous(),
            q_proj.q_gmap_gr_words.contiguous(),
            q_proj.q_gmap_gr_offsets.contiguous(),
            q_proj.q_gmap_gr_meta.contiguous(),
            q_proj.c_map.contiguous(),
            q_proj.g_idx.contiguous(),
            q_proj.col_pack3.contiguous(),
            q_proj.row_pack3.contiguous(),
            q_proj.bias.contiguous(),
            k_proj.qweight1.contiguous(),
            k_proj.qweight2_sparse.contiguous(),
            k_proj.qweight2_sparse_prefix.contiguous(),
            k_proj.q_gmap_gr_words.contiguous(),
            k_proj.q_gmap_gr_offsets.contiguous(),
            k_proj.q_gmap_gr_meta.contiguous(),
            k_proj.c_map.contiguous(),
            k_proj.g_idx.contiguous(),
            k_proj.col_pack3.contiguous(),
            k_proj.row_pack3.contiguous(),
            k_proj.bias.contiguous(),
            v_proj.qweight1.contiguous(),
            v_proj.qweight2_sparse.contiguous(),
            v_proj.qweight2_sparse_prefix.contiguous(),
            v_proj.q_gmap_gr_words.contiguous(),
            v_proj.q_gmap_gr_offsets.contiguous(),
            v_proj.q_gmap_gr_meta.contiguous(),
            v_proj.c_map.contiguous(),
            v_proj.g_idx.contiguous(),
            v_proj.col_pack3.contiguous(),
            v_proj.row_pack3.contiguous(),
            v_proj.bias.contiguous(),
            split_k_slices,
            all(proj._compact3_k64_groups_uniform() for proj in (q_proj, k_proj, v_proj)),
        )
    except RuntimeError as exc:
        _print_cuda_fallback(_module_debug_label(q_proj, "compact3_qkv_linear_forward"), exc)
        return None

    return tuple(fused[idx].reshape(out_shape) for idx in range(3))


def run_compact3_shared_expert_gate_up_forward(gate_proj, up_proj, hidden_states: torch.Tensor):
    if not _compact3_block_gr_enabled() or not _shared_expert_dual_gate_up_enabled():
        return None
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    if ext is None:
        return None
    kernel = getattr(ext, "compact3_dual_linear_forward", None)
    if kernel is None:
        return None
    if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.float16:
        return None
    if hidden_states.shape[-1] != getattr(gate_proj, "infeatures", None):
        return None
    if not _compact3_dual_projection_ready(gate_proj, hidden_states.device):
        return None
    outfeatures = int(gate_proj.outfeatures)
    if not _compact3_dual_projection_ready(up_proj, hidden_states.device, outfeatures):
        return None

    out_shape = hidden_states.shape[:-1] + (outfeatures,)
    x_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    if x_flat.shape[0] == 0:
        empty = x_flat.new_empty(out_shape)
        return empty, empty

    batch = int(x_flat.size(0))
    out_tiles = (outfeatures + 63) // 64
    decode_phase = _decode_phase_enabled(hidden_states)
    explicit_shared_split = os.environ.get("TRIS_SHARED_EXPERT_COMPACT3_BLOCK_GR_DECODE_SPLIT_K")
    num_k_tiles = (int(gate_proj.infeatures) + 63) // 64
    if decode_phase and explicit_shared_split is None and out_tiles >= 64 and batch <= 1:
        split_k_slices = min(8, num_k_tiles)
        split_k_source = f"auto:shared-expert-dual,out_tiles={out_tiles},batch={batch},num_k_tiles={num_k_tiles}"
    else:
        override = _resolve_shared_expert_compact3_block_gr_split_k_slices_py(
            rank=int(gate_proj.infeatures),
            batch=batch,
            out_tiles=out_tiles,
            device=x_flat.device,
            decode_phase=decode_phase,
            label=_module_debug_label(gate_proj, "shared_expert.gate_up"),
        )
        if override is not None:
            split_k_slices, split_k_source = override
        else:
            split_k_slices, split_k_source = _resolve_compact3_block_gr_split_k_slices_py(
                int(gate_proj.infeatures),
                batch,
                out_tiles,
                x_flat.device,
                decode_phase,
            )
    phase = "decode" if decode_phase else "prefill"
    log_key = (phase, "shared-expert-dual")
    if log_key not in _COMPACT3_QKV_RUNTIME_LOGGED:
        _COMPACT3_QKV_RUNTIME_LOGGED.add(log_key)
        print(
            f"INFO - [compact3-dual-runtime] phase={phase}, batch={batch}, out_tiles={out_tiles}, "
            f"split_k={split_k_slices} ({split_k_source})",
            flush=True,
        )

    try:
        fused = kernel(
            x_flat,
            gate_proj.qweight1.contiguous(),
            gate_proj.qweight2_sparse.contiguous(),
            gate_proj.qweight2_sparse_prefix.contiguous(),
            gate_proj.q_gmap_gr_words.contiguous(),
            gate_proj.q_gmap_gr_offsets.contiguous(),
            gate_proj.q_gmap_gr_meta.contiguous(),
            gate_proj.c_map.contiguous(),
            gate_proj.g_idx.contiguous(),
            gate_proj.col_pack3.contiguous(),
            gate_proj.row_pack3.contiguous(),
            up_proj.qweight1.contiguous(),
            up_proj.qweight2_sparse.contiguous(),
            up_proj.qweight2_sparse_prefix.contiguous(),
            up_proj.q_gmap_gr_words.contiguous(),
            up_proj.q_gmap_gr_offsets.contiguous(),
            up_proj.q_gmap_gr_meta.contiguous(),
            up_proj.c_map.contiguous(),
            up_proj.g_idx.contiguous(),
            up_proj.col_pack3.contiguous(),
            up_proj.row_pack3.contiguous(),
            split_k_slices,
            gate_proj._compact3_k64_groups_uniform() and up_proj._compact3_k64_groups_uniform(),
        )
    except RuntimeError as exc:
        _print_cuda_fallback(_module_debug_label(gate_proj, "compact3_dual_linear_forward"), exc)
        return None

    return fused[0].reshape(out_shape), fused[1].reshape(out_shape)


def _resolve_grouped_ns_split_k_slices_py(rank: int, batch: int, out_tiles: int, device: torch.device, decode_phase: bool):
    num_k_tiles = (rank + 63) // 64
    if num_k_tiles <= 1 or batch <= 0 or out_tiles <= 0:
        return 1, "disabled"

    phase_env_name = "TRIS_GROUPED_NS_DECODE_SPLIT_K" if decode_phase else "TRIS_GROUPED_NS_PREFILL_SPLIT_K"
    env_value = os.environ.get(phase_env_name)
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:{phase_env_name}={requested}"
        return 1, f"env:{phase_env_name}={requested}"

    env_value = os.environ.get("TRIS_GROUPED_NS_SPLIT_K")
    if env_value is not None and env_value != "":
        requested = int(env_value)
        if requested > 1:
            return min(requested, num_k_tiles), f"env:TRIS_GROUPED_NS_SPLIT_K={requested}"
        return 1, f"env:TRIS_GROUPED_NS_SPLIT_K={requested}"

    if device.type != "cuda":
        return 1, "auto:no-cuda"

    props = torch.cuda.get_device_properties(device)
    sm_count = int(getattr(props, "multi_processor_count", 0))
    if sm_count <= 0:
        return 1, "auto:no-sm"

    base_tiles = batch * out_tiles
    return _resolve_auto_split_k_slices_py(num_k_tiles, base_tiles, sm_count, decode_phase)


def _maybe_log_grouped_ns_static_runtime(label: str, batch: int, num_segments: int, total_problem_tiles: int, max_ns_rank: int, split_k_desc: str, split_k_source: str):
    decode_hit = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
    phase = "decode" if decode_hit else "prefill"
    if phase in _GROUPED_NS_STATIC_LOGGED:
        return
    _GROUPED_NS_STATIC_LOGGED.add(phase)
    print(
        f"INFO - [grouped-ns-runtime] phase={phase}, label={label}: hit grouped NS payload static path=yes, "
        f"batch={batch}, segments={num_segments}, total_problem_tiles={total_problem_tiles}, max_ns_rank={max_ns_rank}, "
        f"split_k={split_k_desc} ({split_k_source})",
        flush=True,
    )


def _run_packed_4bit_linear_cuda(input_tensor: torch.Tensor, packed_matrix: "PackedQuantMatrix4bit", debug_label: str | None = None):
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    label = debug_label or "packed4_linear_forward"
    if ext is None or not packed_matrix.is_quantized():
        return _return_or_raise_none(label, "packed4 kernel unavailable or packed matrix is not quantized")
    if input_tensor.device.type != "cuda" or input_tensor.dtype != torch.float16:
        return _return_or_raise_none(label, "packed4 kernel requires CUDA float16 inputs")
    try:
        args = (
            input_tensor.contiguous(),
            packed_matrix.qweight.contiguous(),
            packed_matrix.qzeros.contiguous(),
            packed_matrix.scales.contiguous(),
            packed_matrix.g_idx.contiguous(),
        )
        return ext.packed4_linear_forward(*args)
    except RuntimeError as exc:
        if _deployment_only_enabled():
            _raise_deployment_only_failure(label, repr(exc))
        _print_cuda_fallback(label, exc)
        return None


def _shared_precompute_v1_enabled() -> bool:
    return _env_flag_enabled("TRIS_FUSED_SHARED_PRECOMPUTE_V1", "OLMOE_FUSED_SHARED_PRECOMPUTE_V1")


def _grouped_ns_fused_decode_enabled() -> bool:
    for name in ("TRIS_GROUPED_NS_FUSED_DECODE", "OLMOE_GROUPED_NS_FUSED_DECODE"):
        value = os.environ.get(name)
        if value is None:
            continue
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}
    return False


def _unpack_packed4_qzeros(qzeros: torch.Tensor, rows: int) -> torch.Tensor:
    if qzeros.numel() == 0 or rows <= 0:
        return torch.zeros((qzeros.shape[0], max(rows, 0)), device=qzeros.device, dtype=torch.int32)
    shifts = (torch.arange(8, device=qzeros.device, dtype=torch.int64) * 4).view(1, 1, 8)
    unpacked = ((qzeros.to(dtype=torch.int64).unsqueeze(-1) >> shifts) & 0xF).reshape(qzeros.shape[0], -1)
    return unpacked[:, :rows].to(dtype=torch.int32).contiguous()


def _pack_packed4_qzeros(unpacked_qzeros: torch.Tensor) -> torch.Tensor:
    if unpacked_qzeros.numel() == 0:
        return torch.zeros((unpacked_qzeros.shape[0], 0), device=unpacked_qzeros.device, dtype=torch.int32)
    groups, rows = unpacked_qzeros.shape
    packed_cols = (rows + 7) // 8
    padded = torch.zeros((groups, packed_cols * 8), device=unpacked_qzeros.device, dtype=torch.int64)
    padded[:, :rows] = unpacked_qzeros.to(dtype=torch.int64) & 0xF
    shifts = (torch.arange(8, device=unpacked_qzeros.device, dtype=torch.int64) * 4).view(1, 1, 8)
    packed = torch.sum(padded.view(groups, packed_cols, 8) << shifts, dim=-1)
    return packed.to(dtype=torch.int32).contiguous()


def _can_fuse_packed4_rowwise(lhs: "PackedQuantMatrix4bit", rhs: "PackedQuantMatrix4bit") -> bool:
    if not lhs.is_quantized() or not rhs.is_quantized():
        return False
    if lhs._cols_host != rhs._cols_host:
        return False
    if lhs.qweight.shape[0] != rhs.qweight.shape[0]:
        return False
    if lhs.qzeros.shape[0] != rhs.qzeros.shape[0]:
        return False
    if lhs.scales.shape[0] != rhs.scales.shape[0]:
        return False
    if lhs.g_idx.shape != rhs.g_idx.shape:
        return False
    return bool(torch.equal(lhs.g_idx, rhs.g_idx))


def _build_fused_packed4_rowwise_payload(lhs: "PackedQuantMatrix4bit", rhs: "PackedQuantMatrix4bit"):
    if not _can_fuse_packed4_rowwise(lhs, rhs):
        return None
    device = lhs.qweight.device
    lhs_rows = int(lhs._rows_host)
    rhs_rows = int(rhs._rows_host)
    lhs_qzeros = _unpack_packed4_qzeros(lhs.qzeros.to(device=device, dtype=torch.int32), lhs_rows)
    rhs_qzeros = _unpack_packed4_qzeros(rhs.qzeros.to(device=device, dtype=torch.int32), rhs_rows)
    fused_qzeros = _pack_packed4_qzeros(torch.cat([lhs_qzeros, rhs_qzeros], dim=1))
    return {
        "qweight": torch.cat([lhs.qweight.to(device=device, dtype=torch.int32), rhs.qweight.to(device=device, dtype=torch.int32)], dim=1).contiguous(),
        "qzeros": fused_qzeros,
        "scales": torch.cat([lhs.scales.to(device=device), rhs.scales.to(device=device)], dim=1).contiguous(),
        "g_idx": lhs.g_idx.to(device=device, dtype=torch.int32).contiguous(),
        "rows": lhs_rows + rhs_rows,
        "cols": int(lhs._cols_host),
    }


GROUPED_SEGMENT_M_TILE = _grouped_segment_m_tile()


def _empty_grouped_route(device: torch.device):
    empty_int = torch.empty((0,), device=device, dtype=torch.int32)
    empty_long = torch.empty((0,), device=device, dtype=torch.long)
    return {
        "perm": empty_long,
        "grouped_token_indices": empty_long,
        "grouped_slots": empty_long,
        "grouped_expert_ids": empty_int,
        "expert_ids": empty_int,
        "segment_offsets": torch.zeros((1,), device=device, dtype=torch.int32),
        "problem_offsets": torch.zeros((1,), device=device, dtype=torch.int32),
        "total_problem_tiles": torch.zeros((1,), device="cpu", dtype=torch.int64),
        "total_problem_tiles_host": 0,
    }


def _single_token_decode_route_host_total(selected_experts: torch.Tensor, m_tile: int):
    if selected_experts.dim() != 2 or int(selected_experts.shape[0]) != 1 or m_tile <= 0:
        return None
    # For decode batch=1 each top-k slot contributes one one-token segment.
    # The number of grouped problem tiles is therefore exactly top_k for any positive m_tile.
    return int(selected_experts.shape[1])


def _build_grouped_segment_metadata(segment_lengths, device: torch.device, expert_ids=None, m_tile: int = GROUPED_SEGMENT_M_TILE):
    if m_tile <= 0:
        return _empty_grouped_route(device)

    lengths_tensor = torch.as_tensor(segment_lengths, device=device, dtype=torch.long).contiguous()
    if lengths_tensor.numel() == 0:
        return _empty_grouped_route(device)
    if lengths_tensor.device.type == "cpu" and bool((lengths_tensor < 0).any().item()):
        return _empty_grouped_route(device)

    if expert_ids is None:
        expert_ids_tensor = torch.arange(lengths_tensor.numel(), device=device, dtype=torch.int32)
    else:
        expert_ids_tensor = torch.as_tensor(expert_ids, device=device, dtype=torch.int32).contiguous()
        if expert_ids_tensor.numel() != lengths_tensor.numel():
            return _empty_grouped_route(device)

    zero_long = torch.zeros((1,), device=device, dtype=torch.long)
    segment_offsets = torch.cat([zero_long, torch.cumsum(lengths_tensor, dim=0)], dim=0).to(torch.int32).contiguous()
    problem_lengths = torch.div(lengths_tensor + (m_tile - 1), m_tile, rounding_mode="floor")
    problem_offsets_long = torch.cat([zero_long, torch.cumsum(problem_lengths, dim=0)], dim=0)
    total_problem_tiles = problem_offsets_long[-1:].to(device="cpu", dtype=torch.int64).contiguous()
    return {
        "perm": torch.empty((0,), device=device, dtype=torch.long),
        "grouped_token_indices": torch.empty((0,), device=device, dtype=torch.long),
        "grouped_slots": torch.empty((0,), device=device, dtype=torch.long),
        "grouped_expert_ids": torch.repeat_interleave(expert_ids_tensor.to(torch.int32), lengths_tensor.to(torch.int32)).contiguous(),
        "expert_ids": expert_ids_tensor.contiguous(),
        "segment_offsets": segment_offsets,
        "problem_offsets": problem_offsets_long.to(torch.int32).contiguous(),
        "total_problem_tiles": total_problem_tiles,
        "total_problem_tiles_host": int(total_problem_tiles.reshape(-1)[0].item()) if total_problem_tiles.numel() > 0 else 0,
    }


def build_grouped_route(selected_experts: torch.Tensor, num_experts: int, m_tile: int = GROUPED_SEGMENT_M_TILE):
    if selected_experts.dim() != 2:
        return None
    if m_tile <= 0:
        return _empty_grouped_route(selected_experts.device)
    single_token_host_total = _single_token_decode_route_host_total(selected_experts, m_tile)

    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    if selected_experts.device.type == "cuda" and ext is not None:
        kernel = getattr(ext, "build_grouped_route", None)
        if kernel is not None:
            try:
                perm, grouped_token_indices, grouped_slots, expert_ids, segment_offsets, problem_offsets, total_problem_tiles = kernel(
                    selected_experts.contiguous(),
                    int(num_experts),
                    int(m_tile),
                )
                total_problem_tiles = total_problem_tiles.contiguous()
                if single_token_host_total is not None:
                    total_problem_tiles_host = single_token_host_total
                    grouped_expert_ids = expert_ids.contiguous()
                else:
                    total_problem_tiles_host = int(total_problem_tiles.reshape(-1)[0].detach().to(device="cpu").item()) if total_problem_tiles.numel() > 0 else 0
                    grouped_expert_ids = torch.repeat_interleave(
                        expert_ids.to(device=selected_experts.device, dtype=torch.int32),
                        (segment_offsets[1:].to(torch.long) - segment_offsets[:-1].to(torch.long)).to(torch.int32),
                    ).contiguous()
                return {
                    "perm": perm.contiguous(),
                    "grouped_token_indices": grouped_token_indices.contiguous(),
                    "grouped_slots": grouped_slots.contiguous(),
                    "grouped_expert_ids": grouped_expert_ids,
                    "expert_ids": expert_ids.contiguous(),
                    "segment_offsets": segment_offsets.contiguous(),
                    "problem_offsets": problem_offsets.contiguous(),
                    "total_problem_tiles": total_problem_tiles,
                    "total_problem_tiles_host": total_problem_tiles_host,
                }
            except RuntimeError:
                pass

    device = selected_experts.device
    num_tokens, top_k = selected_experts.shape
    flat_selected_experts = selected_experts.reshape(-1).to(torch.long)
    if flat_selected_experts.numel() == 0:
        return _empty_grouped_route(device)

    grouped_experts, perm = torch.sort(flat_selected_experts)
    token_template = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(1).expand(-1, top_k).reshape(-1)
    slot_template = torch.arange(top_k, device=device, dtype=torch.long).unsqueeze(0).expand(num_tokens, -1).reshape(-1)
    grouped_token_indices = token_template.index_select(0, perm).contiguous()
    grouped_slots = slot_template.index_select(0, perm).contiguous()

    expert_counts = torch.bincount(grouped_experts, minlength=int(num_experts))
    active_expert_ids_long = torch.nonzero(expert_counts > 0, as_tuple=False).flatten().contiguous()
    segment_lengths = expert_counts.index_select(0, active_expert_ids_long)
    zero_long = torch.zeros((1,), device=device, dtype=torch.long)
    segment_offsets = torch.cat([zero_long, torch.cumsum(segment_lengths, dim=0)], dim=0).to(torch.int32).contiguous()
    problem_lengths = torch.div(segment_lengths + (m_tile - 1), m_tile, rounding_mode="floor")
    problem_offsets_long = torch.cat([zero_long, torch.cumsum(problem_lengths, dim=0)], dim=0)

    total_problem_tiles = problem_offsets_long[-1:].to(device="cpu", dtype=torch.int64).contiguous()
    total_problem_tiles_host = (
        single_token_host_total
        if single_token_host_total is not None
        else int(total_problem_tiles.reshape(-1)[0].item()) if total_problem_tiles.numel() > 0 else 0
    )
    return {
        "perm": perm.contiguous(),
        "grouped_token_indices": grouped_token_indices,
        "grouped_slots": grouped_slots,
        "grouped_expert_ids": grouped_experts.to(torch.int32).contiguous(),
        "expert_ids": active_expert_ids_long.to(torch.int32).contiguous(),
        "segment_offsets": segment_offsets,
        "problem_offsets": problem_offsets_long.to(torch.int32).contiguous(),
        "total_problem_tiles": total_problem_tiles,
        "total_problem_tiles_host": total_problem_tiles_host,
    }


def _coerce_grouped_route(grouped_route_or_segment_lengths, device: torch.device, expert_ids=None, m_tile: int = GROUPED_SEGMENT_M_TILE):
    if isinstance(grouped_route_or_segment_lengths, dict):
        required_keys = ("expert_ids", "segment_offsets", "problem_offsets", "total_problem_tiles")
        if any(key not in grouped_route_or_segment_lengths for key in required_keys):
            return None
        route = grouped_route_or_segment_lengths
        if route["expert_ids"].device != device or route["segment_offsets"].device != device or route["problem_offsets"].device != device:
            return None
        if route["expert_ids"].dtype != torch.int32 or route["segment_offsets"].dtype != torch.int32 or route["problem_offsets"].dtype != torch.int32:
            return None
        if "total_problem_tiles_host" not in route:
            route = dict(route)
            route["total_problem_tiles_host"] = _route_total_problem_tiles_host(route)
        grouped_expert_ids = route.get("grouped_expert_ids")
        if grouped_expert_ids is None:
            grouped_expert_ids = torch.repeat_interleave(
                route["expert_ids"],
                (route["segment_offsets"][1:].to(torch.long) - route["segment_offsets"][:-1].to(torch.long)).to(torch.int32),
            ).contiguous()
            route = dict(route)
            route["grouped_expert_ids"] = grouped_expert_ids
        elif grouped_expert_ids.device != device or grouped_expert_ids.dtype != torch.int32:
            return None
        return route
    return _build_grouped_segment_metadata(grouped_route_or_segment_lengths, device, expert_ids=expert_ids, m_tile=m_tile)


def _get_grouped_m16_hybrid_metadata(route: dict):
    cached = route.get("_compact3_m16_hybrid")
    if isinstance(cached, dict):
        return cached
    segment_offsets = route.get("segment_offsets")
    if not isinstance(segment_offsets, torch.Tensor) or segment_offsets.numel() <= 1:
        return None
    segment_lengths = segment_offsets[1:].to(torch.long) - segment_offsets[:-1].to(torch.long)
    full_tile_counts = torch.div(segment_lengths, 16, rounding_mode="floor")
    tail_rows = segment_lengths - full_tile_counts * 16
    tail_tile_counts = torch.div(tail_rows + (GROUPED_SEGMENT_M_TILE - 1), GROUPED_SEGMENT_M_TILE, rounding_mode="floor")
    zero = torch.zeros((1,), device=segment_offsets.device, dtype=torch.long)
    full_problem_offsets = torch.cat([zero, torch.cumsum(full_tile_counts, dim=0)], dim=0).to(torch.int32).contiguous()
    tail_problem_offsets = torch.cat([zero, torch.cumsum(tail_tile_counts, dim=0)], dim=0).to(torch.int32).contiguous()
    totals = torch.stack([full_problem_offsets[-1], tail_problem_offsets[-1]]).to(device="cpu", dtype=torch.int64)
    metadata = {
        "full_problem_offsets": full_problem_offsets,
        "full_problem_tiles": totals[0:1].contiguous(),
        "full_problem_tiles_host": int(totals[0].item()),
        "tail_problem_offsets": tail_problem_offsets,
        "tail_problem_tiles": totals[1:2].contiguous(),
        "tail_problem_tiles_host": int(totals[1].item()),
    }
    route["_compact3_m16_hybrid"] = metadata
    return metadata


def _maybe_log_grouped_prefill_tc(label: str, variant: str, hit: bool, metadata: dict | None = None, reason: str = ""):
    # A module can miss on a short prompt and hit once a later prompt creates
    # at least one full 16-row expert tile, so retain one log for each state.
    key = (variant, label, bool(hit))
    if key in _GROUPED_PREFILL_TC_LOGGED:
        return
    _GROUPED_PREFILL_TC_LOGGED.add(key)
    full_tiles = metadata["full_problem_tiles_host"] if metadata is not None else 0
    tail_tiles = metadata["tail_problem_tiles_host"] if metadata is not None else 0
    print(
        f"INFO - [grouped-prefill-tc] label={label}: variant={variant}, "
        f"grouped_prefill_tc_hit={'yes' if hit else 'no'}, full_m16_tiles={full_tiles}, "
        f"tail_m4_tiles={tail_tiles}, reason={reason or ('enabled' if hit else 'disabled')}",
        flush=True,
    )


def _maybe_get_grouped_block_gr_predecoded_words(
    payload: dict,
    route: dict,
    *,
    rank: int,
    outfeatures: int,
):
    if not _compact3_block_gr_enabled():
        return None
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    if ext is None:
        return None
    predecode_kernel = getattr(ext, "compact3_tris_precomputed_grouped_block_gr_predecode", None)
    if predecode_kernel is None:
        return None
    expert_ids = route.get("expert_ids")
    if not isinstance(expert_ids, torch.Tensor) or expert_ids.numel() == 0:
        return None
    active_count = int(expert_ids.numel())
    if active_count > _GROUPED_BLOCK_GR_PREDECODE_ACTIVE_CAP:
        return None
    packed_rows = (int(rank) + 31) // 32
    gmap_block_count = ((packed_rows + 1) // 2) * ((int(outfeatures) + 63) // 64)
    required_bytes = active_count * gmap_block_count * 128 * 4
    if required_bytes > _GROUPED_BLOCK_GR_PREDECODE_MAX_BYTES:
        return None

    cached = payload.get("_grouped_block_gr_predecode_workspace")
    if (
        not isinstance(cached, dict)
        or not isinstance(cached.get("decoded_words"), torch.Tensor)
        or cached["decoded_words"].device != expert_ids.device
        or cached["decoded_words"].dtype != torch.int32
        or tuple(cached["decoded_words"].shape) != (active_count, gmap_block_count, 128)
    ):
        cached = {
            "decoded_words": torch.empty(
                (active_count, gmap_block_count, 128),
                device=expert_ids.device,
                dtype=torch.int32,
            ).contiguous(),
            "expert_ids": torch.empty((0,), device=expert_ids.device, dtype=torch.int32),
        }
        payload["_grouped_block_gr_predecode_workspace"] = cached

    cached_ids = cached.get("expert_ids")
    if not isinstance(cached_ids, torch.Tensor) or not torch.equal(cached_ids, expert_ids):
        predecode_kernel(
            cached["decoded_words"],
            expert_ids.contiguous(),
            payload["q_gmap_gr_words"],
            payload["q_gmap_gr_offsets"],
            payload["q_gmap_gr_meta"],
            int(packed_rows),
            int(outfeatures),
        )
        cached["expert_ids"] = expert_ids.clone()
    return cached["decoded_words"]


def build_grouped_precomputed_payload(modules, device: torch.device, dtype: torch.dtype):
    if len(modules) == 0:
        return None

    for module in modules:
        if not isinstance(module, QuantLinear) or module._mode_host != 5:
            return None

    for module in modules:
        module._ensure_legacy_bitpacked_buffers(require_dense_qweight2=False)
        module._ensure_block_gr_gmap_buffers()
        if module.expert_col_pack3.numel() == 0 or module.expert_row_pack3.numel() == 0:
            return None

    try:
        qweight1 = torch.stack([m.expert_qweight1.to(device=device, dtype=torch.int32) for m in modules], dim=0).contiguous()
        c_map = torch.stack([m.expert_c_map.to(device=device, dtype=torch.int32) for m in modules], dim=0).contiguous()
        g_idx = torch.stack([m.expert_g_idx.to(device=device, dtype=torch.int32) for m in modules], dim=0).contiguous()
        output_scale = torch.stack([m.ns_vector_row.to(device=device, dtype=dtype) for m in modules], dim=0).contiguous()
        if output_scale.numel() > 0 and bool(torch.all(output_scale == 0).item()):
            # mode=5 expert row scale is folded into expert_row_pack3; missing
            # deployment ns_vector_row must be treated as identity, not zero.
            output_scale = torch.ones_like(output_scale).contiguous()
    except RuntimeError as exc:
        label = "grouped_compact3_tris_precomputed_payload"
        if _deployment_only_enabled():
            _raise_deployment_only_failure(label, repr(exc))
        _print_cuda_fallback(label, exc)
        return None

    has_output_scale = not _tensor_is_exact_all_ones(output_scale)
    has_bias = any(m.bias is not None for m in modules)
    bias = None
    if has_bias:
        bias = torch.stack(
            [
                m.bias.to(device=device, dtype=dtype)
                if m.bias is not None
                else torch.zeros((m.outfeatures,), device=device, dtype=dtype)
                for m in modules
            ],
            dim=0,
        ).contiguous()

    try:
        max_qweight2_sparse_rows = max(int(m.expert_qweight2_sparse.shape[0]) for m in modules)
        max_q_gmap_sparse_rows = max(int(m.expert_q_gmap_sparse.shape[0]) for m in modules)
        qweight2_sparse = torch.stack(
            [
                F.pad(m.expert_qweight2_sparse.to(device=device, dtype=torch.int32), (0, 0, 0, max_qweight2_sparse_rows - int(m.expert_qweight2_sparse.shape[0])))
                for m in modules
            ],
            dim=0,
        ).contiguous()
        qweight2_sparse_prefix = torch.stack(
            [m.expert_qweight2_sparse_prefix.to(device=device, dtype=torch.int32) for m in modules],
            dim=0,
        ).contiguous()
        q_gmap_sparse = torch.stack(
            [
                F.pad(m.expert_q_gmap_sparse.to(device=device, dtype=torch.int32), (0, 0, 0, max_q_gmap_sparse_rows - int(m.expert_q_gmap_sparse.shape[0])))
                for m in modules
            ],
            dim=0,
        ).contiguous()
        q_gmap_sparse_prefix = torch.stack(
            [m.expert_q_gmap_sparse_prefix.to(device=device, dtype=torch.int32) for m in modules],
            dim=0,
        ).contiguous()
        col_pack3 = torch.stack([m.expert_col_pack3.to(device=device, dtype=torch.int64) for m in modules], dim=0).contiguous()
        row_pack3 = torch.stack([m.expert_row_pack3.to(device=device, dtype=torch.int64) for m in modules], dim=0).contiguous()
        compact3_use_block_gr = _compact3_block_gr_enabled() and all(
            m.expert_q_gmap_gr_offsets.numel() > 0 and m.expert_q_gmap_gr_meta.numel() > 0 for m in modules
        )
        if compact3_use_block_gr:
            max_gmap_gr_words = max(int(m.expert_q_gmap_gr_words.shape[0]) for m in modules)
            q_gmap_gr_words = torch.stack(
                [
                    F.pad(m.expert_q_gmap_gr_words.to(device=device, dtype=torch.int32), (0, max_gmap_gr_words - int(m.expert_q_gmap_gr_words.shape[0])))
                    for m in modules
                ],
                dim=0,
            ).contiguous()
            q_gmap_gr_offsets = torch.stack(
                [m.expert_q_gmap_gr_offsets.to(device=device, dtype=torch.int32) for m in modules],
                dim=0,
            ).contiguous()
            q_gmap_gr_meta = torch.stack(
                [m.expert_q_gmap_gr_meta.to(device=device, dtype=torch.int32) for m in modules],
                dim=0,
            ).contiguous()
        else:
            q_gmap_gr_words = torch.empty((0, 0), device=device, dtype=torch.int32)
            q_gmap_gr_offsets = torch.empty((0, 0), device=device, dtype=torch.int32)
            q_gmap_gr_meta = torch.empty((0, 0), device=device, dtype=torch.int32)
    except RuntimeError as exc:
        if _deployment_only_enabled():
            _raise_deployment_only_failure("grouped_compact3_tris_precomputed_payload", repr(exc))
        _print_cuda_fallback("grouped_compact3_tris_precomputed_payload", exc)
        return None

    payload = {
        "is_compact3": True,
        "qweight1": qweight1,
        "qweight2_sparse": qweight2_sparse,
        "qweight2_sparse_prefix": qweight2_sparse_prefix,
        "q_gmap_sparse": q_gmap_sparse,
        "q_gmap_sparse_prefix": q_gmap_sparse_prefix,
        "compact3_use_block_gr": compact3_use_block_gr,
        "q_gmap_gr_words": q_gmap_gr_words,
        "q_gmap_gr_offsets": q_gmap_gr_offsets,
        "q_gmap_gr_meta": q_gmap_gr_meta,
        "c_map": c_map,
        "g_idx": g_idx,
        "col_pack3": col_pack3,
        "row_pack3": row_pack3,
        "output_scale": output_scale,
        "has_output_scale": has_output_scale,
        "bias": bias,
        "has_bias": has_bias,
    }
    _cache_stacked_compact3_k64_groups(
        payload, all(module._compact3_k64_groups_uniform(expert=True) for module in modules)
    )
    return payload


def _resolve_grouped_precomputed_payload(payload_or_modules, device: torch.device, dtype: torch.dtype):
    if isinstance(payload_or_modules, dict):
        payload_or_modules = _normalize_runtime_payload_scalars(payload_or_modules)
        common_required_keys = (
            "is_compact3",
            "qweight1",
            "c_map",
            "g_idx",
            "output_scale",
            "has_bias",
        )
        if any(key not in payload_or_modules for key in common_required_keys):
            return None
        if payload_or_modules["qweight1"].device != device or payload_or_modules["output_scale"].device != device:
            return None
        if payload_or_modules["output_scale"].dtype != dtype:
            return None
        if not payload_or_modules["is_compact3"]:
            return None
        if payload_or_modules.get("compact3_use_block_gr", False):
            required_keys = (
                "qweight2_sparse",
                "qweight2_sparse_prefix",
                "q_gmap_gr_words",
                "q_gmap_gr_offsets",
                "q_gmap_gr_meta",
                "col_pack3",
                "row_pack3",
            )
        else:
            required_keys = (
                "qweight2_sparse",
                "qweight2_sparse_prefix",
                "q_gmap_sparse",
                "q_gmap_sparse_prefix",
                "col_pack3",
                "row_pack3",
            )
        if any(key not in payload_or_modules for key in required_keys):
            return None
        payload_or_modules["has_output_scale"] = _payload_has_output_scale(payload_or_modules)
        _get_prepared_compact3_k64_groups(payload_or_modules)
        return payload_or_modules
    return build_grouped_precomputed_payload(payload_or_modules, device, dtype)


def _make_empty_grouped_packed4_payload(target_rows: int, target_cols: int, group_size: int, device: torch.device, dtype: torch.dtype):
    packed_cols = (target_cols + 7) // 8
    group_count = (target_cols + group_size - 1) // group_size if target_cols > 0 else 0
    qzeros_cols = (target_rows + 7) // 8
    return {
        "qweight": torch.zeros((packed_cols, target_rows), device=device, dtype=torch.int32),
        "qzeros": torch.zeros((group_count, qzeros_cols), device=device, dtype=torch.int32),
        "scales": torch.zeros((group_count, target_rows), device=device, dtype=dtype),
        "g_idx": torch.zeros((target_cols,), device=device, dtype=torch.int32),
        "rows": torch.tensor(target_rows, device=device, dtype=torch.int32),
        "cols": torch.tensor(target_cols, device=device, dtype=torch.int32),
    }


def _pad_packed4_payload(matrix: "PackedQuantMatrix4bit", target_rows: int, target_cols: int, group_size: int, device: torch.device, dtype: torch.dtype):
    if not matrix.is_quantized():
        return _make_empty_grouped_packed4_payload(target_rows, target_cols, group_size, device, dtype)

    qweight = matrix.qweight.to(device=device, dtype=torch.int32)
    qzeros = matrix.qzeros.to(device=device, dtype=torch.int32)
    scales = matrix.scales.to(device=device, dtype=dtype)
    g_idx = matrix.g_idx.to(device=device, dtype=torch.int32)

    packed_cols = (target_cols + 7) // 8
    group_count = (target_cols + group_size - 1) // group_size if target_cols > 0 else 0
    qzeros_cols = (target_rows + 7) // 8

    qweight_padded = torch.zeros((packed_cols, target_rows), device=device, dtype=torch.int32)
    qweight_padded[: qweight.shape[0], : qweight.shape[1]] = qweight

    qzeros_padded = torch.zeros((group_count, qzeros_cols), device=device, dtype=torch.int32)
    qzeros_padded[: qzeros.shape[0], : qzeros.shape[1]] = qzeros

    scales_padded = torch.zeros((group_count, target_rows), device=device, dtype=dtype)
    scales_padded[: scales.shape[0], : scales.shape[1]] = scales

    g_idx_padded = torch.zeros((target_cols,), device=device, dtype=torch.int32)
    g_idx_padded[: g_idx.shape[0]] = g_idx

    return {
        "qweight": qweight_padded.contiguous(),
        "qzeros": qzeros_padded.contiguous(),
        "scales": scales_padded.contiguous(),
        "g_idx": g_idx_padded.contiguous(),
        "rows": torch.tensor(target_rows, device=device, dtype=torch.int32),
        "cols": torch.tensor(target_cols, device=device, dtype=torch.int32),
    }


def build_grouped_ns_compensation_payload(modules, device: torch.device, dtype: torch.dtype):
    if len(modules) == 0:
        return None

    first_module = modules[0]
    if not isinstance(first_module, QuantLinear):
        return None
    group_size = None
    infeatures = int(first_module.infeatures)
    outfeatures = int(first_module.outfeatures)

    ns_ranks: list[int] = []
    ns_pairs = []
    for module in modules:
        if not isinstance(module, QuantLinear):
            return None
        if int(module.infeatures) != infeatures or int(module.outfeatures) != outfeatures:
            return None
        if module._ns_mode_host != 2:
            ns_ranks.append(0)
            ns_pairs.append((None, None, None))
            continue
        c1_quant, c2_quant = module._get_effective_ns_quantized_pair()
        if c1_quant is None or c2_quant is None:
            return None
        module_group_size = _infer_packed_group_size_from_g_idx(c2_quant)
        if group_size is None:
            group_size = module_group_size
        elif group_size != module_group_size:
            return None
        ns_rank = int(c2_quant._rows_host)
        if ns_rank != int(c1_quant._cols_host):
            return None
        ns_ranks.append(ns_rank)
        ns_pairs.append((c1_quant, c2_quant, module._get_effective_ns_c2_source()))

    if group_size is None:
        group_size = infeatures
    max_ns_rank = max(ns_ranks) if ns_ranks else 0
    in_group_count = (infeatures + group_size - 1) // group_size if infeatures > 0 else 0
    out_qzeros_cols = (outfeatures + 7) // 8
    packed_infeatures = (infeatures + 7) // 8

    ns_c2_rank_offsets: list[int] = []
    ns_c2_rank_word_offsets: list[int] = []
    ns_c1_rank_word_offsets: list[int] = []
    ns_c1_group_offsets: list[int] = []
    total_ns_rank = 0
    total_ns_rank_words = 0
    total_ns_c1_rank_words = 0
    total_ns_c1_groups = 0

    ns_c2_qweight_chunks = []
    ns_c2_qzeros_chunks = []
    ns_c2_scales_chunks = []
    ns_c1_qweight_chunks = []
    ns_c1_qzeros_chunks = []
    ns_c1_scales_chunks = []
    shared_c2_offsets = {}

    for module, ns_rank, ns_pair in zip(modules, ns_ranks, ns_pairs):
        ns_c1_rank_word_offsets.append(total_ns_c1_rank_words)
        ns_c1_group_offsets.append(total_ns_c1_groups)
        if ns_rank <= 0:
            ns_c2_rank_offsets.append(total_ns_rank)
            ns_c2_rank_word_offsets.append(total_ns_rank_words)
            continue

        c1_quant, c2_quant, c2_source = ns_pair
        if c1_quant is None or c2_quant is None:
            return None

        if c2_source is not None and c2_source is not module:
            c2_key = ("shared", id(c2_source))
        else:
            c2_key = ("local", id(module))

        shared_entry = shared_c2_offsets.get(c2_key)
        if shared_entry is None:
            rank_offset = total_ns_rank
            rank_word_offset = total_ns_rank_words
            shared_c2_offsets[c2_key] = (rank_offset, rank_word_offset, ns_rank)
            try:
                ns_c2_qweight_chunks.append(c2_quant.qweight.to(device=device, dtype=torch.int32))
                ns_c2_qzeros_chunks.append(c2_quant.qzeros.to(device=device, dtype=torch.int32))
                ns_c2_scales_chunks.append(c2_quant.scales.to(device=device, dtype=dtype))
            except RuntimeError as exc:
                if _deployment_only_enabled():
                    _raise_deployment_only_failure("grouped_ns_compensation_payload", repr(exc))
                _print_cuda_fallback("grouped_ns_compensation_payload", exc)
                return None
            total_ns_rank += ns_rank
            total_ns_rank_words += int(c2_quant.qzeros.shape[1])
        else:
            rank_offset, rank_word_offset, shared_rank = shared_entry
            if shared_rank != ns_rank:
                return None

        ns_c2_rank_offsets.append(rank_offset)
        ns_c2_rank_word_offsets.append(rank_word_offset)

        try:
            ns_c1_qweight_chunks.append(c1_quant.qweight.to(device=device, dtype=torch.int32))
            ns_c1_qzeros_chunks.append(c1_quant.qzeros.to(device=device, dtype=torch.int32))
            ns_c1_scales_chunks.append(c1_quant.scales.to(device=device, dtype=dtype))
        except RuntimeError as exc:
            if _deployment_only_enabled():
                _raise_deployment_only_failure("grouped_ns_compensation_payload", repr(exc))
            _print_cuda_fallback("grouped_ns_compensation_payload", exc)
            return None

        total_ns_c1_rank_words += int(c1_quant.qweight.shape[0])
        total_ns_c1_groups += int(c1_quant.qzeros.shape[0])

    if ns_c2_qweight_chunks:
        ns_c2_qweight = torch.cat(ns_c2_qweight_chunks, dim=1).contiguous()
        ns_c2_qzeros = torch.cat(ns_c2_qzeros_chunks, dim=1).contiguous()
        ns_c2_scales = torch.cat(ns_c2_scales_chunks, dim=1).contiguous()
    else:
        ns_c2_qweight = torch.zeros((packed_infeatures, 0), device=device, dtype=torch.int32)
        ns_c2_qzeros = torch.zeros((in_group_count, 0), device=device, dtype=torch.int32)
        ns_c2_scales = torch.zeros((in_group_count, 0), device=device, dtype=dtype)

    if ns_c1_qweight_chunks:
        ns_c1_qweight = torch.cat(ns_c1_qweight_chunks, dim=0).contiguous()
        ns_c1_qzeros = torch.cat(ns_c1_qzeros_chunks, dim=0).contiguous()
        ns_c1_scales = torch.cat(ns_c1_scales_chunks, dim=0).contiguous()
    else:
        ns_c1_qweight = torch.zeros((0, outfeatures), device=device, dtype=torch.int32)
        ns_c1_qzeros = torch.zeros((0, out_qzeros_cols), device=device, dtype=torch.int32)
        ns_c1_scales = torch.zeros((0, outfeatures), device=device, dtype=dtype)

    return {
        "ns_rank_per_expert": torch.tensor(ns_ranks, device=device, dtype=torch.int32),
        "ns_c2_rank_offsets": torch.tensor(ns_c2_rank_offsets, device=device, dtype=torch.int32),
        "ns_c2_rank_word_offsets": torch.tensor(ns_c2_rank_word_offsets, device=device, dtype=torch.int32),
        "ns_c1_rank_word_offsets": torch.tensor(ns_c1_rank_word_offsets, device=device, dtype=torch.int32),
        "ns_c1_group_offsets": torch.tensor(ns_c1_group_offsets, device=device, dtype=torch.int32),
        "max_ns_rank": max_ns_rank,
        "infeatures": infeatures,
        "outfeatures": outfeatures,
        "group_size": group_size,
        "ns_c2_qweight": ns_c2_qweight,
        "ns_c2_qzeros": ns_c2_qzeros,
        "ns_c2_scales": ns_c2_scales,
        "ns_c1_qweight": ns_c1_qweight,
        "ns_c1_qzeros": ns_c1_qzeros,
        "ns_c1_scales": ns_c1_scales,
    }


def _resolve_grouped_ns_compensation_payload(payload_or_modules, device: torch.device, dtype: torch.dtype):
    if payload_or_modules is None:
        return None
    if isinstance(payload_or_modules, dict):
        payload_or_modules = _normalize_runtime_payload_scalars(payload_or_modules)
        required_keys = (
            "ns_rank_per_expert",
            "ns_c2_rank_offsets",
            "ns_c2_rank_word_offsets",
            "ns_c1_rank_word_offsets",
            "ns_c1_group_offsets",
            "max_ns_rank",
            "infeatures",
            "outfeatures",
            "group_size",
            "ns_c2_qweight",
            "ns_c2_qzeros",
            "ns_c2_scales",
            "ns_c1_qweight",
            "ns_c1_qzeros",
            "ns_c1_scales",
        )
        if any(key not in payload_or_modules for key in required_keys):
            return None
        if payload_or_modules["ns_rank_per_expert"].device != device:
            return None
        if payload_or_modules["ns_c2_scales"].dtype != dtype or payload_or_modules["ns_c1_scales"].dtype != dtype:
            return None
        return payload_or_modules
    return build_grouped_ns_compensation_payload(payload_or_modules, device, dtype)


def _apply_grouped_ns_compensation(modules, x_grouped: torch.Tensor, grouped_route, outputs: list[torch.Tensor]):
    if modules is None:
        return outputs
    route = _coerce_grouped_route(grouped_route, x_grouped.device)
    if route is None or route["expert_ids"].numel() == 0:
        return outputs

    expert_ids_host = route["expert_ids"].to(device="cpu", dtype=torch.int32)
    segment_offsets_host = route["segment_offsets"].to(device="cpu", dtype=torch.int32)
    direct_index_modules = len(modules) > int(expert_ids_host[-1].item()) if expert_ids_host.numel() > 0 else False

    for seg_idx in range(int(expert_ids_host.numel())):
        expert_idx = int(expert_ids_host[seg_idx].item())
        module = modules[expert_idx] if direct_index_modules else modules[seg_idx]
        start = int(segment_offsets_host[seg_idx].item())
        end = int(segment_offsets_host[seg_idx + 1].item())
        if end <= start or module._ns_mode_host != 2:
            continue
        x_slice = x_grouped[start:end].contiguous()
        compensation = module._run_ns_compensation_cuda(x_slice)
        if compensation is None:
            if _deployment_only_enabled():
                _raise_deployment_only_failure(
                    _module_debug_label(module, "grouped_ns_compensation"),
                    "ns compensation kernel returned no result",
                )
            c1, c2 = module._get_ns_matrices(x_slice.device, x_slice.dtype)
            if c1 is not None and c2 is not None:
                compensation = F.linear(F.linear(x_slice, c2), c1)
        if compensation is not None:
            for out in outputs:
                out[start:end].add_(compensation)
    return outputs


def _resolve_grouped_precomputed_shared_ns_hidden(expert_modules, route, device: torch.device, dtype: torch.dtype):
    if not _shared_precompute_v1_enabled() or expert_modules is None or len(expert_modules) == 0:
        return None
    grouped_token_indices = route.get("grouped_token_indices")
    if grouped_token_indices is None or grouped_token_indices.numel() == 0:
        return None
    shared_module = getattr(expert_modules[0], "_shared_matrix_module", None)
    if shared_module is None:
        return None
    return shared_module.get_cached_shared_ns_hidden(grouped_token_indices=grouped_token_indices, device=device, dtype=dtype)


def run_grouped_ns_c1_forward(payload_or_modules, hidden_grouped: torch.Tensor, grouped_route_or_segment_lengths, expert_modules=None, expert_ids=None):
    route = _coerce_grouped_route(grouped_route_or_segment_lengths, hidden_grouped.device, expert_ids=expert_ids)
    if route is None:
        return None
    if hidden_grouped.device.type != "cuda" or hidden_grouped.dtype != torch.float16 or hidden_grouped.dim() != 2:
        return None
    grouped_expert_ids = route.get("grouped_expert_ids")
    if grouped_expert_ids is None or grouped_expert_ids.numel() != hidden_grouped.size(0):
        return None

    payload = _resolve_grouped_ns_compensation_payload(payload_or_modules, hidden_grouped.device, hidden_grouped.dtype)
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    kernel = getattr(ext, "grouped_ns_c1_forward", None) if ext is not None else None
    if payload is not None and kernel is not None:
        try:
            batch = int(hidden_grouped.size(0))
            num_segments = int(route["expert_ids"].numel())
            total_problem_tiles = _route_total_problem_tiles_host(route)
            decode_phase = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
            return kernel(
                hidden_grouped.contiguous(),
                route["expert_ids"].contiguous(),
                route["segment_offsets"].contiguous(),
                route["problem_offsets"].contiguous(),
                int(total_problem_tiles),
                payload["ns_rank_per_expert"].contiguous(),
                payload["ns_c1_rank_word_offsets"].contiguous(),
                payload["ns_c1_group_offsets"].contiguous(),
                payload["ns_c1_qweight"].contiguous(),
                payload["ns_c1_qzeros"].contiguous(),
                payload["ns_c1_scales"].contiguous(),
                int(payload["max_ns_rank"]),
                int(payload["group_size"]),
                decode_phase,
            )
        except RuntimeError as exc:
            if _deployment_only_enabled():
                _raise_deployment_only_failure("grouped_ns_c1_forward", repr(exc))
            _print_cuda_fallback("grouped_ns_c1_forward", exc)

    if expert_modules is None:
        return None
    output = torch.zeros((hidden_grouped.size(0), int(payload["outfeatures"])) if payload is not None else (hidden_grouped.size(0), expert_modules[0].outfeatures), device=hidden_grouped.device, dtype=hidden_grouped.dtype)
    expert_ids_host = route["expert_ids"].to(device="cpu", dtype=torch.int32)
    segment_offsets_host = route["segment_offsets"].to(device="cpu", dtype=torch.int32)
    direct_index_modules = len(expert_modules) > int(expert_ids_host[-1].item()) if expert_ids_host.numel() > 0 else False
    for seg_idx in range(int(expert_ids_host.numel())):
        expert_idx = int(expert_ids_host[seg_idx].item())
        module = expert_modules[expert_idx] if direct_index_modules else expert_modules[seg_idx]
        start = int(segment_offsets_host[seg_idx].item())
        end = int(segment_offsets_host[seg_idx + 1].item())
        if end <= start or module._ns_mode_host != 2:
            continue
        hidden_slice = hidden_grouped[start:end].contiguous()
        compensation = module._run_ns_c1_only_cuda(hidden_slice)
        if compensation is not None:
            output[start:end].copy_(compensation)
    return output


def run_grouped_ns_compensation_forward(payload_or_modules, x_grouped: torch.Tensor, grouped_route_or_segment_lengths, expert_modules=None, expert_ids=None, prefer_fused_decode: bool = False):
    route = _coerce_grouped_route(grouped_route_or_segment_lengths, x_grouped.device, expert_ids=expert_ids)
    if route is None:
        return None
    if x_grouped.device.type != "cuda" or x_grouped.dtype != torch.float16 or x_grouped.dim() != 2:
        return None
    grouped_expert_ids = route.get("grouped_expert_ids")
    if grouped_expert_ids is None or grouped_expert_ids.numel() != x_grouped.size(0):
        return None

    precomputed_hidden = _resolve_grouped_precomputed_shared_ns_hidden(expert_modules, route, x_grouped.device, x_grouped.dtype)
    if precomputed_hidden is not None:
        return run_grouped_ns_c1_forward(
            payload_or_modules,
            precomputed_hidden,
            route,
            expert_modules=expert_modules,
            expert_ids=expert_ids,
        )

    payload = _resolve_grouped_ns_compensation_payload(payload_or_modules, x_grouped.device, x_grouped.dtype)
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    kernel = getattr(ext, "grouped_ns_compensation_forward", None) if ext is not None else None
    fused_decode_kernel = (
        getattr(ext, "grouped_ns_compensation_fused_decode_forward", None)
        if ext is not None and prefer_fused_decode and _grouped_ns_fused_decode_enabled()
        else None
    )
    if payload is not None and kernel is not None:
        try:
            batch = int(x_grouped.size(0))
            num_segments = int(route["expert_ids"].numel())
            total_problem_tiles = _route_total_problem_tiles_host(route)
            decode_phase = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
            if fused_decode_kernel is not None and decode_phase:
                try:
                    return fused_decode_kernel(
                        x_grouped.contiguous(),
                        route["expert_ids"].contiguous(),
                        route["segment_offsets"].contiguous(),
                        route["problem_offsets"].contiguous(),
                        int(total_problem_tiles),
                        payload["ns_rank_per_expert"].contiguous(),
                        payload["ns_c2_rank_offsets"].contiguous(),
                        payload["ns_c2_rank_word_offsets"].contiguous(),
                        payload["ns_c1_rank_word_offsets"].contiguous(),
                        payload["ns_c1_group_offsets"].contiguous(),
                        payload["ns_c2_qweight"].contiguous(),
                        payload["ns_c2_qzeros"].contiguous(),
                        payload["ns_c2_scales"].contiguous(),
                        payload["ns_c1_qweight"].contiguous(),
                        payload["ns_c1_qzeros"].contiguous(),
                        payload["ns_c1_scales"].contiguous(),
                        int(payload["max_ns_rank"]),
                        int(payload["group_size"]),
                    )
                except RuntimeError as exc:
                    if _deployment_only_enabled():
                        _print_cuda_fallback("grouped_ns_compensation_fused_decode_forward", exc)
                    else:
                        _print_cuda_fallback("grouped_ns_compensation_fused_decode_forward", exc)
            max_ns_rank = int(payload["max_ns_rank"])
            outfeatures = int(payload["outfeatures"])
            c2_out_tiles = (max_ns_rank + 63) // 64
            c1_out_tiles = (outfeatures + 63) // 64
            c2_split_k, c2_split_k_source = _resolve_grouped_ns_split_k_slices_py(int(payload["infeatures"]), total_problem_tiles, c2_out_tiles, x_grouped.device, decode_phase)
            c1_split_k, c1_split_k_source = _resolve_grouped_ns_split_k_slices_py(max_ns_rank, total_problem_tiles, c1_out_tiles, x_grouped.device, decode_phase)
            split_k_desc = f"c2={c2_split_k},c1={c1_split_k}"
            split_k_source = f"c2:{c2_split_k_source}; c1:{c1_split_k_source}"
            label = _module_debug_label(expert_modules[0], 'grouped_ns_compensation_forward') if expert_modules else 'grouped_ns_compensation_forward'
            _maybe_log_grouped_ns_static_runtime(label, batch, num_segments, total_problem_tiles, max_ns_rank, split_k_desc, split_k_source)
            return kernel(
                x_grouped.contiguous(),
                route["expert_ids"].contiguous(),
                route["segment_offsets"].contiguous(),
                route["problem_offsets"].contiguous(),
                int(total_problem_tiles),
                payload["ns_rank_per_expert"].contiguous(),
                payload["ns_c2_rank_offsets"].contiguous(),
                payload["ns_c2_rank_word_offsets"].contiguous(),
                payload["ns_c1_rank_word_offsets"].contiguous(),
                payload["ns_c1_group_offsets"].contiguous(),
                payload["ns_c2_qweight"].contiguous(),
                payload["ns_c2_qzeros"].contiguous(),
                payload["ns_c2_scales"].contiguous(),
                payload["ns_c1_qweight"].contiguous(),
                payload["ns_c1_qzeros"].contiguous(),
                payload["ns_c1_scales"].contiguous(),
                int(payload["max_ns_rank"]),
                int(payload["group_size"]),
                decode_phase,
            )
        except RuntimeError as exc:
            if _deployment_only_enabled():
                _raise_deployment_only_failure("grouped_ns_compensation_forward", repr(exc))
            _print_cuda_fallback("grouped_ns_compensation_forward", exc)

    if expert_modules is None and not isinstance(payload_or_modules, dict):
        expert_modules = payload_or_modules
    if expert_modules is None:
        return None
    outfeatures = int(expert_modules[0].outfeatures) if len(expert_modules) > 0 else int(x_grouped.size(1))
    compensation = x_grouped.new_zeros((x_grouped.size(0), outfeatures))
    _apply_grouped_ns_compensation(expert_modules, x_grouped, route, [compensation])
    return compensation


def run_grouped_structured_expert_forward(payload_or_modules, hidden_grouped: torch.Tensor, x_grouped: torch.Tensor, grouped_route_or_segment_lengths, expert_modules=None, expert_ids=None, ns_payload=None, prefer_ns_fused_decode: bool = False):
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    if ext is None:
        return None
    if hidden_grouped.device.type != "cuda" or hidden_grouped.dtype != torch.float16:
        return None
    if x_grouped.device != hidden_grouped.device or x_grouped.dtype != hidden_grouped.dtype:
        return None
    if hidden_grouped.dim() != 2 or x_grouped.dim() != 2 or hidden_grouped.size(0) != x_grouped.size(0):
        return None
    route = _coerce_grouped_route(grouped_route_or_segment_lengths, hidden_grouped.device, expert_ids=expert_ids)
    if route is None:
        return None
    grouped_token_indices = route.get("grouped_token_indices")
    if grouped_token_indices is not None and grouped_token_indices.numel() not in (0, hidden_grouped.size(0)):
        return None

    payload = _resolve_grouped_precomputed_payload(payload_or_modules, hidden_grouped.device, hidden_grouped.dtype)
    if payload is None:
        return None
    batch = int(hidden_grouped.size(0))
    num_segments = int(route['expert_ids'].numel())
    total_problem_tiles = _route_total_problem_tiles_host(route)
    outfeatures = int(payload['qweight1'].size(2))
    out_tiles = (outfeatures + 63) // 64
    decode_phase = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
    use_compact3_block_gr = bool(payload.get("compact3_use_block_gr", False))
    if use_compact3_block_gr:
        split_k, split_k_source = _resolve_grouped_block_gr_split_k_slices_py(
            int(hidden_grouped.size(1)),
            total_problem_tiles,
            out_tiles,
            hidden_grouped.device,
            decode_phase,
        )
    else:
        split_k, split_k_source = _resolve_grouped_split_k_slices_py(
            int(hidden_grouped.size(1)),
            total_problem_tiles,
            out_tiles,
            hidden_grouped.device,
            decode_phase,
        )
    label_name = 'grouped_compact3_tris_precomputed_forward'
    label = _module_debug_label(expert_modules[0], label_name) if expert_modules else label_name
    _maybe_log_grouped_runtime(label, batch, num_segments, total_problem_tiles, {'split_k': (split_k, split_k_source)})
    if expert_modules is None and not isinstance(payload_or_modules, dict):
        expert_modules = payload_or_modules
    defer_bias = bool(payload['has_bias']) and (ns_payload is not None or expert_modules is not None)
    bias_arg = payload['bias'] if payload['has_bias'] and not defer_bias else hidden_grouped.new_empty((0, 0))
    has_bias_arg = bool(payload['has_bias']) and not defer_bias
    has_output_scale_arg = _payload_has_output_scale(payload)
    predecoded_words = None
    predecoded_kernel = None
    use_predecoded_block_gr = False
    grouped_tc_metadata = None
    use_grouped_prefill_tc = False
    grouped_tc_reason = "predecode-unavailable"
    if (
        use_compact3_block_gr
        and split_k == 1
        and not decode_phase
        and not has_output_scale_arg
        and not has_bias_arg
    ):
        predecoded_kernel = getattr(ext, "compact3_tris_precomputed_grouped_forward_block_gr_predecoded", None)
        if predecoded_kernel is not None:
            predecoded_words = _maybe_get_grouped_block_gr_predecoded_words(
                payload,
                route,
                rank=int(hidden_grouped.size(1)),
                outfeatures=outfeatures,
            )
            use_predecoded_block_gr = predecoded_words is not None
            if use_predecoded_block_gr and _env_flag_enabled("TRIS_DISABLE_GROUPED_COMPACT3_PREFILL_TC"):
                grouped_tc_reason = "kill-switch"
            elif use_predecoded_block_gr and not _compact3_tensor_core_device_supported(hidden_grouped.device):
                grouped_tc_reason = "unsupported-device"
            elif use_predecoded_block_gr and batch < _grouped_compact3_prefill_tc_min_routed_rows():
                grouped_tc_reason = "insufficient-routed-rows"
            elif use_predecoded_block_gr:
                hybrid_kernel = getattr(ext, "compact3_tris_precomputed_grouped_forward_block_gr_predecoded_hybrid", None)
                grouped_tc_metadata = _get_grouped_m16_hybrid_metadata(route)
                if (
                    hybrid_kernel is not None
                    and grouped_tc_metadata is not None
                    and grouped_tc_metadata["full_problem_tiles_host"] >= _grouped_compact3_prefill_tc_min_full_tiles()
                ):
                    predecoded_kernel = hybrid_kernel
                    use_grouped_prefill_tc = True
                    grouped_tc_reason = "enabled"
                elif hybrid_kernel is None:
                    grouped_tc_reason = "extension-missing"
                elif grouped_tc_metadata is not None and grouped_tc_metadata["full_problem_tiles_host"] > 0:
                    grouped_tc_reason = "insufficient-full-m16-tiles"
                else:
                    grouped_tc_reason = "no-full-m16-tiles"
        if not use_grouped_prefill_tc:
            _maybe_log_grouped_prefill_tc(label, "single", False, grouped_tc_metadata, grouped_tc_reason)
    if use_compact3_block_gr:
        kernel_name = "compact3_tris_precomputed_grouped_forward_block_gr"
    else:
        kernel_name = 'compact3_tris_precomputed_grouped_forward'
    kernel = getattr(ext, kernel_name, None)
    if kernel is None:
        return None
    staging = payload.get("_qgmap_staging")
    use_bitmap_pipeline = not use_compact3_block_gr and staging is not None and staging[0].can_pipeline(staging[1])
    if staging is not None and not use_bitmap_pipeline:
        staging[0].stage(staging[1])
    try:
        if use_compact3_block_gr:
            if use_predecoded_block_gr:
                if use_grouped_prefill_tc:
                    _maybe_log_grouped_prefill_tc(label, "single", True, grouped_tc_metadata, grouped_tc_reason)
                    out = predecoded_kernel(
                        hidden_grouped.contiguous(), route["expert_ids"], route["segment_offsets"],
                        grouped_tc_metadata["full_problem_offsets"], grouped_tc_metadata["full_problem_tiles"],
                        grouped_tc_metadata["tail_problem_offsets"], grouped_tc_metadata["tail_problem_tiles"],
                        predecoded_words, payload["qweight1"], payload["qweight2_sparse"],
                        payload["qweight2_sparse_prefix"], payload["c_map"], payload["g_idx"],
                        payload["col_pack3"], payload["row_pack3"],
                        payload["_compact3_k64_groups_uniform"],
                    )
                else:
                    out = predecoded_kernel(
                        hidden_grouped.contiguous(),
                        route["expert_ids"],
                        route["segment_offsets"],
                        route["problem_offsets"],
                        route["total_problem_tiles"],
                        predecoded_words,
                        payload["qweight1"],
                        payload["qweight2_sparse"],
                        payload["qweight2_sparse_prefix"],
                        payload["c_map"],
                        payload["g_idx"],
                        payload["col_pack3"],
                        payload["row_pack3"],
                        payload["_compact3_k64_groups_uniform"],
                    )
            else:
                out = kernel(
                    hidden_grouped.contiguous(),
                    route["expert_ids"],
                    route["segment_offsets"],
                    route["problem_offsets"],
                    route["total_problem_tiles"],
                    payload["qweight1"],
                    payload["qweight2_sparse"],
                    payload["qweight2_sparse_prefix"],
                    payload["q_gmap_gr_words"],
                    payload["q_gmap_gr_offsets"],
                    payload["q_gmap_gr_meta"],
                    payload["c_map"],
                    payload["g_idx"],
                    payload["col_pack3"],
                    payload["row_pack3"],
                    payload["output_scale"],
                    has_output_scale_arg,
                    bias_arg,
                    has_bias_arg,
                    payload["_compact3_k64_groups_uniform"],
                )
        elif use_bitmap_pipeline:
            out = staging[0].run_grouped(
                staging[1], hidden_grouped.contiguous(), route, kernel, bias_arg, has_bias_arg,
            )
        else:
            out = kernel(
                hidden_grouped.contiguous(),
                route["expert_ids"],
                route["segment_offsets"],
                route["problem_offsets"],
                route["total_problem_tiles"],
                payload["qweight1"],
                payload["qweight2_sparse"],
                payload["qweight2_sparse_prefix"],
                payload["q_gmap_sparse"],
                payload["q_gmap_sparse_prefix"],
                payload["c_map"],
                payload["g_idx"],
                payload["col_pack3"],
                payload["row_pack3"],
                payload["output_scale"],
                bias_arg,
                has_bias_arg,
            )
    except RuntimeError as exc:
        if _deployment_only_enabled():
            _raise_deployment_only_failure(label_name, repr(exc))
        return None

    compensation = run_grouped_ns_compensation_forward(
        ns_payload if ns_payload is not None else expert_modules,
        x_grouped,
        route,
        expert_modules=expert_modules,
        prefer_fused_decode=prefer_ns_fused_decode,
    )
    if compensation is not None:
        out.add_(compensation)
    if defer_bias:
        out.add_(payload['bias'].index_select(0, route['grouped_expert_ids'].to(dtype=torch.long)))
    return out


def run_grouped_structured_expert_pair_forward(payload_or_modules_a, payload_or_modules_b, hidden_a_grouped: torch.Tensor, hidden_b_grouped: torch.Tensor, x_grouped: torch.Tensor, grouped_route_or_segment_lengths, expert_modules_a=None, expert_modules_b=None, expert_ids=None, ns_payload_a=None, ns_payload_b=None):
    ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
    if ext is None:
        return None, None
    if hidden_a_grouped.device.type != "cuda" or hidden_a_grouped.dtype != torch.float16:
        return None, None
    if hidden_b_grouped.device != hidden_a_grouped.device or hidden_b_grouped.dtype != hidden_a_grouped.dtype:
        return None, None
    if x_grouped.device != hidden_a_grouped.device or x_grouped.dtype != hidden_a_grouped.dtype:
        return None, None
    if hidden_a_grouped.dim() != 2 or hidden_b_grouped.dim() != 2 or x_grouped.dim() != 2:
        return None, None
    if hidden_a_grouped.size(0) != hidden_b_grouped.size(0) or hidden_a_grouped.size(0) != x_grouped.size(0):
        return None, None
    route = _coerce_grouped_route(grouped_route_or_segment_lengths, hidden_a_grouped.device, expert_ids=expert_ids)
    if route is None:
        return None, None
    grouped_token_indices = route.get("grouped_token_indices")
    if grouped_token_indices is not None and grouped_token_indices.numel() not in (0, hidden_a_grouped.size(0)):
        return None, None

    payload_a = _resolve_grouped_precomputed_payload(payload_or_modules_a, hidden_a_grouped.device, hidden_a_grouped.dtype)
    payload_b = _resolve_grouped_precomputed_payload(payload_or_modules_b, hidden_a_grouped.device, hidden_a_grouped.dtype)
    if payload_a is None or payload_b is None:
        return None, None
    batch = int(hidden_a_grouped.size(0))
    num_segments = int(route['expert_ids'].numel())
    total_problem_tiles = _route_total_problem_tiles_host(route)
    rank_a = int(hidden_a_grouped.size(1))
    rank_b = int(hidden_b_grouped.size(1))
    outfeatures_a = int(payload_a['qweight1'].size(2))
    outfeatures_b = int(payload_b['qweight1'].size(2))
    outfeatures_max = max(outfeatures_a, outfeatures_b)
    out_tiles_max = (outfeatures_max + 31) // 32
    decode_phase = _grouped_decode_path_enabled(batch, num_segments, total_problem_tiles)
    if not bool(payload_a.get("is_compact3", False)) or not bool(payload_b.get("is_compact3", False)):
        return None, None

    compact3_dual_label = (
        _module_debug_label(expert_modules_a[0], 'grouped_compact3_tris_precomputed_dual_forward_block_gr')
        if expert_modules_a
        else 'grouped_compact3_tris_precomputed_dual_forward_block_gr'
    )
    can_use_compact3_dual_block_gr = (
        bool(payload_a.get("compact3_use_block_gr", False))
        and bool(payload_b.get("compact3_use_block_gr", False))
    )
    compact3_dual_kernel = (
        getattr(ext, "compact3_tris_precomputed_grouped_dual_forward_block_gr", None)
        if can_use_compact3_dual_block_gr
        else None
    )
    if compact3_dual_kernel is None:
        return (
            run_grouped_structured_expert_forward(
                payload_a,
                hidden_a_grouped,
                x_grouped,
                route,
                expert_modules=expert_modules_a,
                expert_ids=expert_ids,
                ns_payload=ns_payload_a,
            ),
            run_grouped_structured_expert_forward(
                payload_b,
                hidden_b_grouped,
                x_grouped,
                route,
                expert_modules=expert_modules_b,
                expert_ids=expert_ids,
                ns_payload=ns_payload_b,
            ),
        )

    split_k_a, split_k_source_a = _resolve_grouped_block_gr_split_k_slices_py(rank_a, total_problem_tiles, out_tiles_max, hidden_a_grouped.device, decode_phase)
    split_k_b, split_k_source_b = _resolve_grouped_block_gr_split_k_slices_py(rank_b, total_problem_tiles, out_tiles_max, hidden_b_grouped.device, decode_phase)
    effective_split_k = max(split_k_a, split_k_b)
    _maybe_log_grouped_dual_runtime(
        compact3_dual_label,
        batch,
        num_segments,
        total_problem_tiles,
        variant="compact3-block-gr",
        kernel_name="compact3_tris_precomputed_grouped_dual_forward_block_gr",
        rank_a=rank_a,
        rank_b=rank_b,
        outfeatures_a=outfeatures_a,
        outfeatures_b=outfeatures_b,
        split_k_values={
            'split_k_a': (split_k_a, split_k_source_a),
            'split_k_b': (split_k_b, split_k_source_b),
            'effective_split_k': (effective_split_k, 'max(a,b)'),
        },
    )
    defer_bias_a = bool(payload_a['has_bias']) and (ns_payload_a is not None or expert_modules_a is not None)
    defer_bias_b = bool(payload_b['has_bias']) and (ns_payload_b is not None or expert_modules_b is not None)
    bias_arg_a = payload_a['bias'] if payload_a['has_bias'] and not defer_bias_a else hidden_a_grouped.new_empty((0, 0))
    bias_arg_b = payload_b['bias'] if payload_b['has_bias'] and not defer_bias_b else hidden_b_grouped.new_empty((0, 0))
    has_bias_arg_a = bool(payload_a['has_bias']) and not defer_bias_a
    has_bias_arg_b = bool(payload_b['has_bias']) and not defer_bias_b
    has_output_scale_a = _payload_has_output_scale(payload_a)
    has_output_scale_b = _payload_has_output_scale(payload_b)
    predecoded_dual_kernel = None
    predecoded_words_a = None
    predecoded_words_b = None
    grouped_tc_metadata = None
    use_grouped_prefill_tc = False
    grouped_tc_reason = "predecode-unavailable"
    use_predecoded_block_gr = (
        effective_split_k == 1
        and not decode_phase
        and not has_output_scale_a
        and not has_bias_arg_a
        and not has_output_scale_b
        and not has_bias_arg_b
    )
    if use_predecoded_block_gr:
        predecoded_dual_kernel = getattr(ext, "compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded", None)
        if predecoded_dual_kernel is not None:
            predecoded_words_a = _maybe_get_grouped_block_gr_predecoded_words(
                payload_a,
                route,
                rank=rank_a,
                outfeatures=outfeatures_a,
            )
            predecoded_words_b = _maybe_get_grouped_block_gr_predecoded_words(
                payload_b,
                route,
                rank=rank_b,
                outfeatures=outfeatures_b,
            )
        use_predecoded_block_gr = (
            predecoded_dual_kernel is not None
            and predecoded_words_a is not None
            and predecoded_words_b is not None
        )
        if use_predecoded_block_gr and _env_flag_enabled("TRIS_DISABLE_GROUPED_COMPACT3_PREFILL_TC"):
            grouped_tc_reason = "kill-switch"
        elif use_predecoded_block_gr and not _compact3_tensor_core_device_supported(hidden_a_grouped.device):
            grouped_tc_reason = "unsupported-device"
        elif use_predecoded_block_gr and batch < _grouped_compact3_prefill_tc_min_routed_rows():
            grouped_tc_reason = "insufficient-routed-rows"
        elif use_predecoded_block_gr:
            hybrid_dual_kernel = getattr(
                ext,
                "compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_hybrid",
                None,
            )
            grouped_tc_metadata = _get_grouped_m16_hybrid_metadata(route)
            if (
                hybrid_dual_kernel is not None
                and grouped_tc_metadata is not None
                and grouped_tc_metadata["full_problem_tiles_host"] >= _grouped_compact3_prefill_tc_min_full_tiles()
            ):
                predecoded_dual_kernel = hybrid_dual_kernel
                use_grouped_prefill_tc = True
                grouped_tc_reason = "enabled"
            elif hybrid_dual_kernel is None:
                grouped_tc_reason = "extension-missing"
            elif grouped_tc_metadata is not None and grouped_tc_metadata["full_problem_tiles_host"] > 0:
                grouped_tc_reason = "insufficient-full-m16-tiles"
            else:
                grouped_tc_reason = "no-full-m16-tiles"
        if not use_grouped_prefill_tc:
            _maybe_log_grouped_prefill_tc(
                compact3_dual_label, "dual", False, grouped_tc_metadata, grouped_tc_reason
            )
    try:
        if use_predecoded_block_gr:
            if use_grouped_prefill_tc:
                _maybe_log_grouped_prefill_tc(
                    compact3_dual_label, "dual", True, grouped_tc_metadata, grouped_tc_reason
                )
                outputs = predecoded_dual_kernel(
                    hidden_a_grouped.contiguous(), hidden_b_grouped.contiguous(),
                    route["expert_ids"], route["segment_offsets"],
                    grouped_tc_metadata["full_problem_offsets"], grouped_tc_metadata["full_problem_tiles"],
                    grouped_tc_metadata["tail_problem_offsets"], grouped_tc_metadata["tail_problem_tiles"],
                    predecoded_words_a, payload_a["qweight1"], payload_a["qweight2_sparse"],
                    payload_a["qweight2_sparse_prefix"], payload_a["c_map"], payload_a["g_idx"],
                    payload_a["col_pack3"], payload_a["row_pack3"],
                    predecoded_words_b, payload_b["qweight1"], payload_b["qweight2_sparse"],
                    payload_b["qweight2_sparse_prefix"], payload_b["c_map"], payload_b["g_idx"],
                    payload_b["col_pack3"], payload_b["row_pack3"],
                    payload_a["_compact3_k64_groups_uniform"] and payload_b["_compact3_k64_groups_uniform"],
                )
            else:
                outputs = predecoded_dual_kernel(
                    hidden_a_grouped.contiguous(),
                    hidden_b_grouped.contiguous(),
                    route["expert_ids"],
                    route["segment_offsets"],
                    route["problem_offsets"],
                    route["total_problem_tiles"],
                    predecoded_words_a,
                    payload_a["qweight1"],
                    payload_a["qweight2_sparse"],
                    payload_a["qweight2_sparse_prefix"],
                    payload_a["c_map"],
                    payload_a["g_idx"],
                    payload_a["col_pack3"],
                    payload_a["row_pack3"],
                    predecoded_words_b,
                    payload_b["qweight1"],
                    payload_b["qweight2_sparse"],
                    payload_b["qweight2_sparse_prefix"],
                    payload_b["c_map"],
                    payload_b["g_idx"],
                    payload_b["col_pack3"],
                    payload_b["row_pack3"],
                    payload_a["_compact3_k64_groups_uniform"] and payload_b["_compact3_k64_groups_uniform"],
                )
        else:
            outputs = compact3_dual_kernel(
                hidden_a_grouped.contiguous(),
                hidden_b_grouped.contiguous(),
                route["expert_ids"],
                route["segment_offsets"],
                route["problem_offsets"],
                route["total_problem_tiles"],
                payload_a["qweight1"],
                payload_a["qweight2_sparse"],
                payload_a["qweight2_sparse_prefix"],
                payload_a["q_gmap_gr_words"],
                payload_a["q_gmap_gr_offsets"],
                payload_a["q_gmap_gr_meta"],
                payload_a["c_map"],
                payload_a["g_idx"],
                payload_a["col_pack3"],
                payload_a["row_pack3"],
                payload_a["output_scale"],
                has_output_scale_a,
                bias_arg_a,
                payload_b["qweight1"],
                payload_b["qweight2_sparse"],
                payload_b["qweight2_sparse_prefix"],
                payload_b["q_gmap_gr_words"],
                payload_b["q_gmap_gr_offsets"],
                payload_b["q_gmap_gr_meta"],
                payload_b["c_map"],
                payload_b["g_idx"],
                payload_b["col_pack3"],
                payload_b["row_pack3"],
                payload_b["output_scale"],
                has_output_scale_b,
                bias_arg_b,
                payload_a["_compact3_k64_groups_uniform"] and payload_b["_compact3_k64_groups_uniform"],
            )
    except RuntimeError as exc:
        if _deployment_only_enabled():
            _raise_deployment_only_failure("grouped_compact3_tris_precomputed_dual_forward", repr(exc))
        _print_cuda_fallback("grouped_compact3_tris_precomputed_dual_forward", exc)
        return (
            run_grouped_structured_expert_forward(
                payload_a,
                hidden_a_grouped,
                x_grouped,
                route,
                expert_modules=expert_modules_a,
                expert_ids=expert_ids,
                ns_payload=ns_payload_a,
            ),
            run_grouped_structured_expert_forward(
                payload_b,
                hidden_b_grouped,
                x_grouped,
                route,
                expert_modules=expert_modules_b,
                expert_ids=expert_ids,
                ns_payload=ns_payload_b,
            ),
        )
    if outputs is None or len(outputs) != 2:
        return None, None

    out_a, out_b = outputs[0], outputs[1]
    compensation_a = run_grouped_ns_compensation_forward(
        ns_payload_a if ns_payload_a is not None else expert_modules_a,
        x_grouped,
        route,
        expert_modules=expert_modules_a,
    )
    if compensation_a is not None:
        out_a.add_(compensation_a)
    if defer_bias_a:
        out_a.add_(payload_a['bias'].index_select(0, route['grouped_expert_ids'].to(dtype=torch.long)))
    compensation_b = run_grouped_ns_compensation_forward(
        ns_payload_b if ns_payload_b is not None else expert_modules_b,
        x_grouped,
        route,
        expert_modules=expert_modules_b,
    )
    if compensation_b is not None:
        out_b.add_(compensation_b)
    if defer_bias_b:
        out_b.add_(payload_b['bias'].index_select(0, route['grouped_expert_ids'].to(dtype=torch.long)))
    return out_a, out_b


class PackedQuantMatrix4bit(nn.Module):
    def __init__(self, weight_dtype=torch.float16):
        super().__init__()
        self.register_buffer("qweight", torch.zeros((0, 0), dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros((0, 0), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros((0, 0), dtype=weight_dtype))
        self.register_buffer("g_idx", torch.zeros((0,), dtype=torch.int32))
        self.register_buffer("rows", torch.zeros((), dtype=torch.int32))
        self.register_buffer("cols", torch.zeros((), dtype=torch.int32))
        object.__setattr__(self, "_rows_host", 0)
        object.__setattr__(self, "_cols_host", 0)
        object.__setattr__(self, "_is_quantized_host", False)
        object.__setattr__(self, "_skip_next_load_state_dict", False)
        self._cached_weight = None
        self._cached_weight_key = None
        self._refresh_scalar_cache()

    def _refresh_scalar_cache(self):
        rows = 0
        cols = 0
        if not getattr(self.rows, "is_meta", False):
            rows = int(self.rows.detach().cpu().item())
        if not getattr(self.cols, "is_meta", False):
            cols = int(self.cols.detach().cpu().item())
        object.__setattr__(self, "_rows_host", rows)
        object.__setattr__(self, "_cols_host", cols)
        object.__setattr__(self, "_is_quantized_host", self.qweight.numel() > 0 and rows > 0 and cols > 0)

    def _apply(self, fn):
        self._materialize_meta_buffers()
        super()._apply(fn)
        self._refresh_scalar_cache()
        self.invalidate_cache()
        return self

    def _materialize_meta_buffers(self):
        for name in ("qweight", "qzeros", "scales", "g_idx", "rows", "cols"):
            value = getattr(self, name)
            if not getattr(value, "is_meta", False):
                continue
            setattr(self, name, torch.zeros(value.shape, dtype=value.dtype))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if _reset_skip_next_load_state_dict_flag(self):
            return
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            key = prefix + name
            if key in state_dict and getattr(self, name).shape != state_dict[key].shape:
                setattr(self, name, state_dict[key].clone())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._refresh_scalar_cache()
        self.invalidate_cache()

    def _fast_load_from_local_state_dict(self, local_state_dict) -> bool:
        supported = ("qweight", "qzeros", "scales", "g_idx", "rows", "cols")
        if any(name not in supported for name in local_state_dict):
            return False

        self._materialize_meta_buffers()
        for name in supported:
            value = local_state_dict.get(name)
            if value is not None:
                _assign_loaded_tensor(self, name, value)
        self._refresh_scalar_cache()
        self.invalidate_cache()
        _mark_skip_next_load_state_dict(self)
        return True

    def invalidate_cache(self):
        self._cached_weight = None
        self._cached_weight_key = None

    def set_empty(self):
        dtype = self.scales.dtype
        self.qweight = torch.zeros((0, 0), dtype=torch.int32)
        self.qzeros = torch.zeros((0, 0), dtype=torch.int32)
        self.scales = torch.zeros((0, 0), dtype=dtype)
        self.g_idx = torch.zeros((0,), dtype=torch.int32)
        self.rows = torch.zeros((), dtype=torch.int32)
        self.cols = torch.zeros((), dtype=torch.int32)
        self._refresh_scalar_cache()
        self.invalidate_cache()

    def is_quantized(self):
        return self._is_quantized_host

    def set_quantized(self, qweight, qzeros, scales, g_idx, rows=None, cols=None):
        self.qweight = qweight.to(torch.int32).contiguous()
        self.qzeros = qzeros.to(torch.int32).contiguous()
        self.scales = scales.to(dtype=self.scales.dtype).contiguous()
        self.g_idx = g_idx.to(torch.int32).contiguous()
        n_rows = int(rows.item()) if isinstance(rows, torch.Tensor) else rows
        n_cols = int(cols.item()) if isinstance(cols, torch.Tensor) else cols
        if n_rows is None:
            n_rows = self.scales.shape[1] if self.scales.dim() == 2 else 0
        if n_cols is None:
            n_cols = self.g_idx.numel()
        self.rows = torch.tensor(n_rows, dtype=torch.int32)
        self.cols = torch.tensor(n_cols, dtype=torch.int32)
        self._refresh_scalar_cache()
        self.invalidate_cache()

    def set_from_payload(self, payload, device=None):
        if payload is None or not isinstance(payload, dict) or "qweight" not in payload:
            self.set_empty()
            return
        if device is None:
            device = self.qweight.device
        self.set_quantized(
            payload["qweight"].to(device=device),
            payload["qzeros"].to(device=device),
            payload["scales"].to(device=device),
            payload["g_idx"].to(device=device),
            payload.get("rows"),
            payload.get("cols"),
        )

    def dequantize(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (device.type, device.index, dtype)
        if self._cached_weight is not None and self._cached_weight_key == cache_key:
            return self._cached_weight

        rows = self._rows_host
        cols = self._cols_host
        if rows == 0 or cols == 0:
            weight = torch.zeros((rows, cols), device=device, dtype=dtype)
            self._cached_weight = weight
            self._cached_weight_key = cache_key
            return weight

        shifts = (torch.arange(8, device=device, dtype=torch.int64) * 4).view(1, 1, 8)
        qzeros = self.qzeros.to(device=device, dtype=torch.int64)
        zeros = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).reshape(qzeros.shape[0], -1)[:, :rows] + 1

        qweight = self.qweight.to(device=device, dtype=torch.int64)
        intweight = ((qweight.unsqueeze(-1) >> shifts) & 0xF).permute(0, 2, 1).reshape(-1, qweight.shape[1])[:cols]

        scales = self.scales.to(device=device, dtype=dtype)
        g_idx = self.g_idx.to(device=device, dtype=torch.long)
        scale_mat = scales[g_idx]
        zero_mat = zeros.to(device=device, dtype=dtype)[g_idx]
        weight = ((intweight.to(dtype=dtype) - zero_mat) * scale_mat).t().contiguous()

        self._cached_weight = weight
        self._cached_weight_key = cache_key
        return weight


class ExpertSharedMatrix(nn.Module):
    def __init__(self, weight_dtype=torch.float16):
        super().__init__()
        self._TriS_debug_name = None
        self.register_buffer("shared_sv", torch.zeros((0, 0), dtype=weight_dtype))
        self.shared_sv_quant = PackedQuantMatrix4bit(weight_dtype=weight_dtype)
        self.register_buffer("shared_ns_c2", torch.zeros((0, 0), dtype=weight_dtype))
        self.shared_ns_c2_quant = PackedQuantMatrix4bit(weight_dtype=weight_dtype)
        self.shared_precompute_quant = PackedQuantMatrix4bit(weight_dtype=weight_dtype)
        self.register_buffer("shared_precompute_sv_rank", torch.zeros((), dtype=torch.int32))
        self.register_buffer("shared_precompute_ns_rank", torch.zeros((), dtype=torch.int32))
        object.__setattr__(self, "_skip_next_load_state_dict", False)
        object.__setattr__(self, "_latest_shared_ns_hidden", None)
        object.__setattr__(self, "_shared_precompute_sv_rank_host", 0)
        object.__setattr__(self, "_shared_precompute_ns_rank_host", 0)

    def _refresh_shared_precompute_rank_host(self):
        sv_rank = getattr(self, "_shared_precompute_sv_rank_host", 0)
        ns_rank = getattr(self, "_shared_precompute_ns_rank_host", 0)
        if not getattr(self.shared_precompute_sv_rank, "is_meta", False):
            sv_rank = int(self.shared_precompute_sv_rank.detach().cpu().item())
        if not getattr(self.shared_precompute_ns_rank, "is_meta", False):
            ns_rank = int(self.shared_precompute_ns_rank.detach().cpu().item())
        object.__setattr__(self, "_shared_precompute_sv_rank_host", sv_rank)
        object.__setattr__(self, "_shared_precompute_ns_rank_host", ns_rank)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if _reset_skip_next_load_state_dict_flag(self):
            return
        for name in ("shared_sv", "shared_ns_c2"):
            key = prefix + name
            current = getattr(self, name)
            if key in state_dict and current.shape != state_dict[key].shape:
                setattr(self, name, state_dict[key].clone())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._refresh_shared_precompute_rank_host()
        optional_keys = [
            prefix + "shared_ns_c2",
            prefix + "shared_ns_c2_quant.qweight",
            prefix + "shared_ns_c2_quant.qzeros",
            prefix + "shared_ns_c2_quant.scales",
            prefix + "shared_ns_c2_quant.g_idx",
            prefix + "shared_ns_c2_quant.rows",
            prefix + "shared_ns_c2_quant.cols",
            prefix + "shared_precompute_quant.qweight",
            prefix + "shared_precompute_quant.qzeros",
            prefix + "shared_precompute_quant.scales",
            prefix + "shared_precompute_quant.g_idx",
            prefix + "shared_precompute_quant.rows",
            prefix + "shared_precompute_quant.cols",
            prefix + "shared_precompute_sv_rank",
            prefix + "shared_precompute_ns_rank",
        ]
        for key in optional_keys:
            if key not in state_dict and key in missing_keys:
                missing_keys.remove(key)

    def _fast_load_from_local_state_dict(self, local_state_dict) -> bool:
        direct_names = {
            "shared_sv",
            "shared_ns_c2",
            "shared_precompute_sv_rank",
            "shared_precompute_ns_rank",
        }
        child_modules = {
            "shared_sv_quant.": self.shared_sv_quant,
            "shared_ns_c2_quant.": self.shared_ns_c2_quant,
            "shared_precompute_quant.": self.shared_precompute_quant,
        }
        for key in local_state_dict:
            if key in direct_names:
                continue
            if any(key.startswith(prefix) for prefix in child_modules):
                continue
            return False

        child_state_dicts = {prefix: {} for prefix in child_modules}
        for key, value in local_state_dict.items():
            for prefix in child_modules:
                if key.startswith(prefix):
                    child_state_dicts[prefix][key[len(prefix):]] = value
                    break

        for prefix, child_module in child_modules.items():
            if not child_module._fast_load_from_local_state_dict(child_state_dicts[prefix]):
                return False

        for name in direct_names:
            value = local_state_dict.get(name)
            if value is not None:
                _assign_loaded_tensor(self, name, value)

        self.clear_cached_shared_ns_hidden()
        self._refresh_shared_precompute_rank_host()
        _mark_skip_next_load_state_dict(self)
        return True

    def clear_cached_shared_ns_hidden(self):
        object.__setattr__(self, "_latest_shared_ns_hidden", None)

    def cache_shared_ns_hidden(self, hidden: torch.Tensor | None):
        if hidden is None:
            self.clear_cached_shared_ns_hidden()
            return
        object.__setattr__(self, "_latest_shared_ns_hidden", hidden.contiguous())

    def get_cached_shared_ns_hidden(self, grouped_token_indices: torch.Tensor | None = None, device=None, dtype=None):
        hidden = getattr(self, "_latest_shared_ns_hidden", None)
        if hidden is None:
            return None
        if grouped_token_indices is not None:
            if grouped_token_indices.numel() == 0:
                hidden = hidden[:0]
            else:
                grouped_token_indices = grouped_token_indices.to(device=hidden.device, dtype=torch.long)
                if _env_flag_enabled("TRIS_VALIDATE_CACHED_SHARED_NS_HIDDEN") and int(grouped_token_indices.max().item()) >= hidden.shape[0]:
                    return None
                hidden = hidden.index_select(0, grouped_token_indices)
        if device is not None or dtype is not None:
            hidden = hidden.to(device=device if device is not None else hidden.device, dtype=dtype if dtype is not None else hidden.dtype)
        return hidden.contiguous()

    def clear_shared_precompute(self):
        device = self.shared_precompute_sv_rank.device
        self.shared_precompute_quant.set_empty()
        self.shared_precompute_sv_rank = torch.zeros((), device=device, dtype=torch.int32)
        self.shared_precompute_ns_rank = torch.zeros((), device=device, dtype=torch.int32)
        object.__setattr__(self, "_shared_precompute_sv_rank_host", 0)
        object.__setattr__(self, "_shared_precompute_ns_rank_host", 0)
        self.clear_cached_shared_ns_hidden()

    def maybe_pack_fused_shared_precompute(self) -> bool:
        self.clear_shared_precompute()
        if not _shared_precompute_v1_enabled():
            return False
        payload = _build_fused_packed4_rowwise_payload(self.shared_sv_quant, self.shared_ns_c2_quant)
        if payload is None:
            return False
        device = self.shared_sv.device
        self.shared_precompute_quant.set_from_payload(payload, device=device)
        sv_rank = int(self.shared_sv_quant._rows_host)
        ns_rank = int(self.shared_ns_c2_quant._rows_host)
        self.shared_precompute_sv_rank = torch.tensor(sv_rank, device=device, dtype=torch.int32)
        self.shared_precompute_ns_rank = torch.tensor(ns_rank, device=device, dtype=torch.int32)
        object.__setattr__(self, "_shared_precompute_sv_rank_host", sv_rank)
        object.__setattr__(self, "_shared_precompute_ns_rank_host", ns_rank)
        return True

    def set_shared_sv(self, payload):
        device = self.shared_sv.device
        self.clear_shared_precompute()
        if isinstance(payload, dict) and "qweight" in payload:
            self.shared_sv = torch.zeros((0, 0), dtype=self.shared_sv.dtype, device=device)
            self.shared_sv_quant.set_from_payload(payload, device=device)
        else:
            self.shared_sv_quant.set_empty()
            self.shared_sv = payload.to(device=device, dtype=self.shared_sv.dtype).contiguous()

    def get_shared_sv(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.shared_sv_quant.is_quantized():
            return self.shared_sv_quant.dequantize(device, dtype)
        return self.shared_sv.to(device=device, dtype=dtype)

    def set_shared_ns_c2(self, payload):
        device = self.shared_ns_c2.device
        self.clear_shared_precompute()
        if payload is None:
            self.shared_ns_c2_quant.set_empty()
            self.shared_ns_c2 = torch.zeros((0, 0), dtype=self.shared_ns_c2.dtype, device=device)
        elif isinstance(payload, dict) and "qweight" in payload:
            self.shared_ns_c2 = torch.zeros((0, 0), dtype=self.shared_ns_c2.dtype, device=device)
            self.shared_ns_c2_quant.set_from_payload(payload, device=device)
        else:
            self.shared_ns_c2_quant.set_empty()
            self.shared_ns_c2 = payload.to(device=device, dtype=self.shared_ns_c2.dtype).contiguous()

    def get_shared_ns_c2(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.shared_ns_c2_quant.is_quantized():
            return self.shared_ns_c2_quant.dequantize(device, dtype)
        return self.shared_ns_c2.to(device=device, dtype=dtype)


class QuantLinear(nn.Module):
    QUANT_TYPE = "cuda"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        qweight_height=None,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** max(bits, 1) - 1
        self.n_groups = math.ceil(infeatures / self.group_size)
        self.qweight_height = qweight_height or math.ceil(infeatures / 32)
        self.kernel_switch_threshold = kernel_switch_threshold
        self.trainable = trainable
        self.autogptq_cuda_available = is_TriS_cuda_available(load_if_needed=True)
        self._TriS_debug_name = None

        self.register_buffer("mode", torch.zeros((), dtype=torch.int8))
        self.register_buffer("fp_weight", torch.zeros((outfeatures, infeatures), dtype=weight_dtype))
        self.register_buffer("qweight1", torch.zeros((self.qweight_height, outfeatures), dtype=torch.int32))
        self.register_buffer("qweight2", torch.zeros((self.qweight_height, outfeatures), dtype=torch.int32))
        self.register_buffer("qweight2_sparse", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("qweight2_sparse_prefix", torch.zeros((0,), dtype=torch.int32))
        self.register_buffer("q_gmap", torch.zeros((self.qweight_height, outfeatures), dtype=torch.int32))
        self.register_buffer("q_gmap_gr_words", torch.zeros((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("q_gmap_gr_offsets", torch.zeros((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("q_gmap_gr_meta", torch.zeros((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("c_map", torch.zeros((math.ceil(infeatures / 32),), dtype=torch.int32))
        self.register_buffer("g_idx", torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32))

        self.register_buffer("scales_s_d_c1", torch.zeros((infeatures,), dtype=weight_dtype))
        self.register_buffer("scales_s_d_c2", torch.zeros((infeatures,), dtype=weight_dtype))
        self.register_buffer("scales_s_nd_c1", torch.zeros((infeatures,), dtype=weight_dtype))
        self.register_buffer("scales_s_nd_c2", torch.zeros((infeatures,), dtype=weight_dtype))
        self.register_buffer("scales_ns_d_c1", torch.zeros((infeatures,), dtype=weight_dtype))
        self.register_buffer("scales_ns_nd_c1", torch.zeros((infeatures,), dtype=weight_dtype))
        self.register_buffer("row_pack1", torch.zeros((outfeatures, 0), dtype=torch.int64))
        self.register_buffer("row_pack2", torch.zeros((outfeatures, 0), dtype=torch.int32))
        self.register_buffer("col_pack3", torch.zeros((0,), dtype=torch.int64))
        self.register_buffer("row_pack3", torch.zeros((outfeatures, 0), dtype=torch.int64))
        self.register_buffer("q_gmap_sparse", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("q_gmap_sparse_prefix", torch.zeros((0,), dtype=torch.int32))

        self.register_buffer("ns_vector_row", torch.ones((outfeatures,), dtype=weight_dtype))
        self.register_buffer("ns_vector_col", torch.ones((infeatures,), dtype=weight_dtype))
        self.register_buffer("expert_input_scale", torch.ones((infeatures,), dtype=weight_dtype))
        self.register_buffer("ns_mode", torch.zeros((), dtype=torch.int8))
        self.register_buffer("ns_c1", torch.zeros((0, 0), dtype=weight_dtype))
        self.register_buffer("ns_c2", torch.zeros((0, 0), dtype=weight_dtype))
        self.ns_c1_quant = PackedQuantMatrix4bit(weight_dtype=weight_dtype)
        self.ns_c2_quant = PackedQuantMatrix4bit(weight_dtype=weight_dtype)
        self.register_buffer("expert_qweight1", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("expert_qweight2", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("expert_qweight2_sparse", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("expert_qweight2_sparse_prefix", torch.zeros((0,), dtype=torch.int32))
        self.register_buffer("expert_q_gmap", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("expert_q_gmap_gr_words", torch.zeros((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("expert_q_gmap_gr_offsets", torch.zeros((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("expert_q_gmap_gr_meta", torch.zeros((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("expert_c_map", torch.zeros((0,), dtype=torch.int32))
        self.register_buffer("expert_g_idx", torch.zeros((0,), dtype=torch.int32))
        self.register_buffer("expert_scales_s_d_c1", torch.zeros((0,), dtype=weight_dtype))
        self.register_buffer("expert_scales_s_d_c2", torch.zeros((0,), dtype=weight_dtype))
        self.register_buffer("expert_scales_s_nd_c1", torch.zeros((0,), dtype=weight_dtype))
        self.register_buffer("expert_scales_s_nd_c2", torch.zeros((0,), dtype=weight_dtype))
        self.register_buffer("expert_scales_ns_d_c1", torch.zeros((0,), dtype=weight_dtype))
        self.register_buffer("expert_scales_ns_nd_c1", torch.zeros((0,), dtype=weight_dtype))
        self.register_buffer("expert_row_pack1", torch.zeros((outfeatures, 0), dtype=torch.int64))
        self.register_buffer("expert_row_pack2", torch.zeros((outfeatures, 0), dtype=torch.int32))
        self.register_buffer("expert_col_pack3", torch.zeros((0,), dtype=torch.int64))
        self.register_buffer("expert_row_pack3", torch.zeros((outfeatures, 0), dtype=torch.int64))
        self.register_buffer("expert_q_gmap_sparse", torch.zeros((0, outfeatures), dtype=torch.int32))
        self.register_buffer("expert_q_gmap_sparse_prefix", torch.zeros((0,), dtype=torch.int32))


        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures,), dtype=weight_dtype))
        else:
            self.bias = None

        self._cached_weight = None
        self._cached_weight_key = None
        object.__setattr__(self, "_shared_matrix_module", None)
        object.__setattr__(self, "_mode_host", 0)
        object.__setattr__(self, "_ns_mode_host", 0)
        object.__setattr__(self, "_expert_runtime_pruned", False)
        object.__setattr__(self, "_ns_runtime_pruned", False)
        object.__setattr__(self, "_skip_next_load_state_dict", False)
        object.__setattr__(self, "_compact3_group_caches", {})
        self._refresh_runtime_scalar_cache()

    def _refresh_runtime_scalar_cache(self):
        mode = self._mode_host
        ns_mode = self._ns_mode_host
        if not getattr(self.mode, "is_meta", False):
            mode = int(self.mode.detach().cpu().item())
        if not getattr(self.ns_mode, "is_meta", False):
            ns_mode = int(self.ns_mode.detach().cpu().item())
        object.__setattr__(self, "_mode_host", mode)
        object.__setattr__(self, "_ns_mode_host", ns_mode)

    def _apply(self, fn, recurse=True):
        self._materialize_meta_placeholders()
        super()._apply(fn, recurse=recurse)
        self._refresh_runtime_scalar_cache()
        self.invalidate_cache()
        # Module.to_empty also uses _apply, before checkpoint copies complete.
        # Only standard device conversions are preparation points here; custom
        # transforms remain unknown until an explicit prepare/load completes.
        conversion = getattr(fn, "__qualname__", "")
        if self._mode_host in (4, 5) and any(
            conversion.startswith(f"Module.{name}.<locals>.") for name in ("to", "cpu", "cuda")
        ):
            self.prepare_compact3_k64_groups()
        return self

    @classmethod
    def warmup(cls, model, seqlen=None):
        return

    def _empty_placeholder(self, name):
        if name == "fp_weight":
            return torch.zeros((0, 0), dtype=self.fp_weight.dtype)
        if name in {"qweight1", "qweight2", "qweight2_sparse", "q_gmap", "q_gmap_sparse", "expert_qweight1", "expert_qweight2", "expert_qweight2_sparse", "expert_q_gmap", "expert_q_gmap_sparse"}:
            return torch.zeros((0, self.outfeatures), dtype=torch.int32)
        if name in {"qweight2_sparse_prefix", "expert_qweight2_sparse_prefix", "q_gmap_sparse_prefix", "expert_q_gmap_sparse_prefix", "q_gmap_gr_words", "q_gmap_gr_offsets", "q_gmap_gr_meta", "expert_q_gmap_gr_words", "expert_q_gmap_gr_offsets", "expert_q_gmap_gr_meta"}:
            return torch.zeros((0,), dtype=torch.int32)
        if name in {"c_map", "expert_c_map"}:
            return torch.zeros((0,), dtype=torch.int32)
        if name in {"g_idx", "expert_g_idx"}:
            return torch.zeros((0,), dtype=torch.int32)
        if name == "ns_vector_row":
            return torch.ones((self.outfeatures,), dtype=self.fp_weight.dtype)
        if name == "ns_vector_col":
            return torch.ones((self.infeatures,), dtype=self.fp_weight.dtype)
        if name == "expert_input_scale":
            return torch.ones((self.infeatures,), dtype=self.fp_weight.dtype)
        if name.startswith("expert_scales_") and "_r" in name:
            return torch.zeros((self.outfeatures, 0), dtype=self.fp_weight.dtype)
        if name.startswith("expert_scales_"):
            return torch.zeros((0,), dtype=self.fp_weight.dtype)
        if name == "row_pack1":
            return torch.zeros((self.outfeatures, 0), dtype=torch.int64)
        if name == "row_pack2":
            return torch.zeros((self.outfeatures, 0), dtype=torch.int32)
        if name == "col_pack3":
            return torch.zeros((0,), dtype=torch.int64)
        if name == "row_pack3":
            return torch.zeros((self.outfeatures, 0), dtype=torch.int64)
        if name == "expert_row_pack1":
            return torch.zeros((self.outfeatures, 0), dtype=torch.int64)
        if name == "expert_row_pack2":
            return torch.zeros((self.outfeatures, 0), dtype=torch.int32)
        if name == "expert_col_pack3":
            return torch.zeros((0,), dtype=torch.int64)
        if name == "expert_row_pack3":
            return torch.zeros((self.outfeatures, 0), dtype=torch.int64)
        if name.startswith("scales_") and "_r" in name:
            return torch.zeros((self.outfeatures, 0), dtype=self.fp_weight.dtype)
        if name.startswith("scales_"):
            return torch.zeros((0,), dtype=self.fp_weight.dtype)
        return None

    def _materialize_meta_placeholders(self):
        names = set()
        if getattr(self.mode, "is_meta", False):
            for mode in (0, 4, 5):
                names.update(self._deployment_skip_state_names(mode))
        else:
            names.update(self._deployment_skip_state_names(self._mode_host))
        names.update(name for name, value in self._buffers.items() if value is not None and getattr(value, "is_meta", False))
        for name in names:
            value = getattr(self, name, None)
            if value is None or not getattr(value, "is_meta", False):
                continue
            replacement = self._empty_placeholder(name)
            if replacement is None:
                replacement = torch.zeros(value.shape, dtype=value.dtype)
            setattr(self, name, replacement)

    def post_init(self):
        self._ensure_legacy_bitpacked_buffers()
        self._ensure_block_gr_gmap_buffers()
        self._refresh_runtime_scalar_cache()
        self.invalidate_cache()
        if self._mode_host in (4, 5):
            self.prepare_compact3_k64_groups()

    @staticmethod
    def _base_common_bitpacked_state_names():
        return (
            "qweight1",
            "qweight2_sparse",
            "qweight2_sparse_prefix",
            "c_map",
            "g_idx",
        )

    @staticmethod
    def _base_compact3_state_names():
        return (
            "q_gmap_sparse",
            "q_gmap_sparse_prefix",
            "col_pack3",
            "row_pack3",
        )

    @staticmethod
    def _expert_common_bitpacked_state_names():
        return (
            "expert_qweight1",
            "expert_qweight2_sparse",
            "expert_qweight2_sparse_prefix",
            "expert_c_map",
            "expert_g_idx",
        )

    @staticmethod
    def _expert_compact3_state_names():
        return (
            "expert_q_gmap_sparse",
            "expert_q_gmap_sparse_prefix",
            "expert_col_pack3",
            "expert_row_pack3",
        )

    @classmethod
    def _deployment_skip_state_names(cls, mode: int):
        names = []
        removed_base_mode_names = (
            "qweight2",
            "q_gmap",
            "scales_s_d_c1",
            "scales_s_d_c2",
            "scales_s_nd_c1",
            "scales_s_nd_c2",
            "scales_ns_d_c1",
            "scales_ns_nd_c1",
            "row_pack1",
            "row_pack2",
        )
        removed_expert_mode_names = (
            "expert_qweight2",
            "expert_q_gmap",
            "expert_scales_s_d_c1",
            "expert_scales_s_d_c2",
            "expert_scales_s_nd_c1",
            "expert_scales_s_nd_c2",
            "expert_scales_ns_d_c1",
            "expert_scales_ns_nd_c1",
            "expert_row_pack1",
            "expert_row_pack2",
        )
        if mode != 0:
            names.append("fp_weight")
        if mode != 5:
            names.append("expert_input_scale")
        if mode != 4:
            names.extend(cls._base_common_bitpacked_state_names())
        names.extend(removed_base_mode_names)
        if mode != 4:
            names.extend(cls._base_compact3_state_names())
        if mode != 5:
            names.extend(cls._expert_common_bitpacked_state_names())
        names.extend(removed_expert_mode_names)
        if mode != 5:
            names.extend(cls._expert_compact3_state_names())
        return tuple(names)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self.mode, "is_meta", False):
            return
        mode = self._mode_host
        for name in self._deployment_skip_state_names(mode):
            destination.pop(prefix + name, None)
        if mode == 4:
            destination.pop(prefix + "ns_vector_row", None)
            destination.pop(prefix + "ns_vector_col", None)
        elif mode == 5:
            destination.pop(prefix + "ns_vector_row", None)
            destination.pop(prefix + "ns_vector_col", None)
        if mode == 4:
            destination.pop(prefix + "qweight2", None)
            destination.pop(prefix + "q_gmap", None)
        if mode == 5:
            destination.pop(prefix + "expert_qweight2", None)
            destination.pop(prefix + "expert_q_gmap", None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if _reset_skip_next_load_state_dict_flag(self):
            return
        base_block_gr_names = (
            "q_gmap_gr_words",
            "q_gmap_gr_offsets",
            "q_gmap_gr_meta",
        )
        expert_block_gr_names = (
            "expert_q_gmap_gr_words",
            "expert_q_gmap_gr_offsets",
            "expert_q_gmap_gr_meta",
        )
        has_base_block_gr = all((prefix + name) in state_dict for name in base_block_gr_names)
        has_expert_block_gr = all((prefix + name) in state_dict for name in expert_block_gr_names)
        for name in ("c_map", "expert_c_map"):
            key = prefix + name
            if key in state_dict and state_dict[key].dtype == torch.bool:
                state_dict[key] = self._pack_bool_vector(state_dict[key].to(device=self.fp_weight.device)).cpu()
        mode_key = prefix + "mode"
        saved_mode = None
        if mode_key in state_dict:
            saved_mode = int(state_dict[mode_key].item())
        elif not getattr(self.mode, "is_meta", False):
            saved_mode = self._mode_host
        if saved_mode in (1, 2, 3):
            raise RuntimeError("QuantLinear structured modes have been removed; please regenerate the checkpoint with compact3 or expert_compact3.")
        if saved_mode == 5:
            expert_input_scale_key = prefix + "expert_input_scale"
            legacy_ns_col_key = prefix + "ns_vector_col"
            if expert_input_scale_key not in state_dict and legacy_ns_col_key in state_dict:
                state_dict[expert_input_scale_key] = state_dict[legacy_ns_col_key].clone()
        for name in (
            "qweight1",
            "qweight2",
            "qweight2_sparse",
            "qweight2_sparse_prefix",
            "q_gmap",
            "q_gmap_gr_words",
            "q_gmap_gr_offsets",
            "q_gmap_gr_meta",
            "q_gmap_sparse",
            "q_gmap_sparse_prefix",
            "c_map",
            "g_idx",
            "row_pack1",
            "row_pack2",
            "col_pack3",
            "row_pack3",
            "ns_mode",
            "expert_input_scale",
            "ns_c1",
            "ns_c2",
            "expert_qweight1",
            "expert_qweight2",
            "expert_qweight2_sparse",
            "expert_qweight2_sparse_prefix",
            "expert_q_gmap",
            "expert_q_gmap_gr_words",
            "expert_q_gmap_gr_offsets",
            "expert_q_gmap_gr_meta",
            "expert_q_gmap_sparse",
            "expert_q_gmap_sparse_prefix",
            "expert_c_map",
            "expert_g_idx",
            "expert_scales_s_d_c1",
            "expert_scales_s_d_c2",
            "expert_scales_s_nd_c1",
            "expert_scales_s_nd_c2",
            "expert_scales_ns_d_c1",
            "expert_scales_ns_nd_c1",
            "expert_row_pack1",
            "expert_row_pack2",
            "expert_col_pack3",
            "expert_row_pack3",
        ):
            key = prefix + name
            if key in state_dict and getattr(self, name).shape != state_dict[key].shape:
                setattr(self, name, state_dict[key].clone())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if saved_mode is not None:
            for name in self._deployment_skip_state_names(saved_mode):
                key = prefix + name
                if key not in state_dict and key in missing_keys:
                    missing_keys.remove(key)
            if saved_mode == 4:
                for name in ("ns_vector_row", "ns_vector_col"):
                    key = prefix + name
                    if key not in state_dict and key in missing_keys:
                        missing_keys.remove(key)
            elif saved_mode == 5:
                for name in ("ns_vector_row", "ns_vector_col"):
                    key = prefix + name
                    if key not in state_dict and key in missing_keys:
                        missing_keys.remove(key)
        ns_mode_key = prefix + "ns_mode"
        if ns_mode_key not in state_dict and ns_mode_key in missing_keys:
            missing_keys.remove(ns_mode_key)
        device = self.fp_weight.device
        self._maybe_restore_sparse_qweight2_buffers()
        # Keep prepacked block-GR payloads when they are present in the checkpoint.
        if not has_base_block_gr:
            self._clear_base_block_gr_buffers(device=device)
        if not has_expert_block_gr:
            self._clear_expert_block_gr_buffers(device=device)
        self._refresh_runtime_scalar_cache()
        self._trim_runtime_state_for_mode()
        self.invalidate_cache()
        if self._mode_host in (4, 5):
            self.prepare_compact3_k64_groups()

    def _fast_load_from_local_state_dict(self, local_state_dict) -> bool:
        child_modules = {
            "ns_c1_quant.": self.ns_c1_quant,
            "ns_c2_quant.": self.ns_c2_quant,
        }
        supported_names = set(self._buffers.keys())

        for key in local_state_dict:
            if key in supported_names:
                continue
            if any(key.startswith(prefix) for prefix in child_modules):
                continue
            return False

        self._materialize_meta_placeholders()

        child_state_dicts = {prefix: {} for prefix in child_modules}
        direct_state = {}
        for key, value in local_state_dict.items():
            matched_child = False
            for prefix in child_modules:
                if key.startswith(prefix):
                    child_state_dicts[prefix][key[len(prefix):]] = value
                    matched_child = True
                    break
            if not matched_child:
                direct_state[key] = value

        for prefix, child_module in child_modules.items():
            if not child_module._fast_load_from_local_state_dict(child_state_dicts[prefix]):
                return False

        has_base_block_gr = all(
            name in direct_state
            for name in ("q_gmap_gr_words", "q_gmap_gr_offsets", "q_gmap_gr_meta")
        )
        has_expert_block_gr = all(
            name in direct_state
            for name in ("expert_q_gmap_gr_words", "expert_q_gmap_gr_offsets", "expert_q_gmap_gr_meta")
        )

        for name in ("c_map", "expert_c_map"):
            value = direct_state.get(name)
            if value is not None and value.dtype == torch.bool:
                packed = self._pack_bool_vector(value.to(device=self.fp_weight.device))
                direct_state[name] = packed.cpu()

        for name in self._buffers:
            value = direct_state.get(name)
            if value is not None:
                _assign_loaded_tensor(self, name, value)

        object.__setattr__(self, "_expert_runtime_pruned", False)
        object.__setattr__(self, "_ns_runtime_pruned", False)
        self._refresh_runtime_scalar_cache()
        self._maybe_restore_sparse_qweight2_buffers()
        # Keep prepacked block-GR payloads when they are present in the checkpoint.
        if not has_base_block_gr:
            self._clear_base_block_gr_buffers(device=self.fp_weight.device)
        if not has_expert_block_gr:
            self._clear_expert_block_gr_buffers(device=self.fp_weight.device)
        self._refresh_runtime_scalar_cache()
        self._trim_runtime_state_for_mode()
        self.invalidate_cache()
        if self._mode_host in (4, 5):
            self.prepare_compact3_k64_groups()
        _mark_skip_next_load_state_dict(self)
        return True

    def invalidate_cache(self):
        self._cached_weight = None
        self._cached_weight_key = None
        self.ns_c1_quant.invalidate_cache()
        self.ns_c2_quant.invalidate_cache()
        if self._mode_host == 4:
            self._compact3_group_caches.pop("expert_g_idx", None)
            self._compact3_k64_groups_uniform()
        elif self._mode_host == 5:
            self._compact3_group_caches.pop("g_idx", None)
            self._compact3_k64_groups_uniform(expert=True)
        else:
            self._compact3_group_caches.clear()

    def _compact3_k64_groups_uniform(self, expert: bool = False) -> bool:
        name = "expert_g_idx" if expert else "g_idx"
        payload = self._compact3_group_caches.setdefault(name, {})
        payload["g_idx"] = getattr(self, name)
        return _get_prepared_compact3_k64_groups(payload)

    def prepare_compact3_k64_groups(self) -> bool:
        name = "expert_g_idx" if self._mode_host == 5 else "g_idx"
        payload = self._compact3_group_caches.setdefault(name, {})
        payload["g_idx"] = getattr(self, name)
        uniform = prepare_compact3_k64_groups(payload)
        if payload["g_idx"] is not getattr(self, name):
            setattr(self, name, payload["g_idx"])
        return uniform

    def _trim_runtime_state_for_mode(self):
        self._refresh_runtime_scalar_cache()
        device = self.fp_weight.device
        mode = self._mode_host

        for name in self._deployment_skip_state_names(mode):
            value = getattr(self, name, None)
            if value is None or getattr(value, "is_meta", False):
                continue
            if value.numel() == 0:
                continue
            replacement = self._empty_placeholder(name)
            if replacement is None:
                continue
            if hasattr(replacement, "to"):
                replacement = replacement.to(device=device)
            setattr(self, name, replacement)

        if mode == 4 and self.q_gmap.numel() > 0:
            self.q_gmap = self._empty_placeholder("q_gmap").to(device=device)
            self._clear_base_block_gr_buffers(device=device)
        if mode == 5 and self.expert_q_gmap.numel() > 0:
            self.expert_q_gmap = self._empty_placeholder("expert_q_gmap").to(device=device)
            self._clear_expert_block_gr_buffers(device=device)
    

    def bind_shared_matrix(self, shared_matrix_module):
        object.__setattr__(self, "_shared_matrix_module", shared_matrix_module)
        self.invalidate_cache()

    def prune_expert_runtime_state(self):
        device = self.fp_weight.device
        for name in self._expert_common_bitpacked_state_names() + self._expert_compact3_state_names():
            replacement = self._empty_placeholder(name)
            if hasattr(replacement, "to"):
                replacement = replacement.to(device=device)
            setattr(self, name, replacement)
        object.__setattr__(self, "_expert_runtime_pruned", True)
        self.invalidate_cache()

    def prune_ns_runtime_state(self):
        device = self.fp_weight.device
        self.ns_c1 = torch.zeros((0, 0), dtype=self.fp_weight.dtype, device=device)
        self.ns_c2 = torch.zeros((0, 0), dtype=self.fp_weight.dtype, device=device)
        self.ns_c1_quant.set_empty()
        self.ns_c2_quant.set_empty()
        object.__setattr__(self, "_ns_runtime_pruned", True)
        self.invalidate_cache()

    def _clone_linear_weight(self, linear):
        weight = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            weight = weight.t()
        return weight

    def _ensure_legacy_bitpacked_buffers(self, require_dense_qweight2: bool = False):
        if getattr(self.c_map, "dtype", None) == torch.bool:
            self.c_map = self._pack_bool_vector(self.c_map)
        if getattr(self.expert_c_map, "dtype", None) == torch.bool:
            self.expert_c_map = self._pack_bool_vector(self.expert_c_map)
        self._ensure_sparse_qweight2_prefix_buffers()
        self._ensure_block_gr_gmap_buffers()

    def _build_sparse_qweight2_prefix(self, c_map: torch.Tensor, ncols: int, device: torch.device) -> torch.Tensor:
        packed_rows = math.ceil(ncols / 32)
        prefix = torch.zeros((packed_rows + 1,), dtype=torch.int32, device=device)
        if packed_rows == 0 or c_map.numel() == 0:
            return prefix
        c_map_cpu = c_map.detach().to(device="cpu", dtype=torch.int64).contiguous()
        running = 0
        prefix_host = [0]
        for packed_row in range(packed_rows):
            valid_bits = min(32, ncols - packed_row * 32)
            word = int(c_map_cpu[packed_row].item())
            if valid_bits < 32:
                word &= (1 << valid_bits) - 1
            else:
                word &= 0xFFFFFFFF
            running += int(word).bit_count()
            prefix_host.append(running)
        return torch.tensor(prefix_host, dtype=torch.int32, device=device)

    def _ensure_sparse_qweight2_prefix_buffers(self):
        device = self.fp_weight.device
        base_ncols = int(self.g_idx.shape[0])
        if self.qweight2_sparse.numel() > 0 and base_ncols > 0 and self.qweight2_sparse_prefix.numel() == 0:
            self.qweight2_sparse_prefix = self._build_sparse_qweight2_prefix(self.c_map, base_ncols, device)
        expert_ncols = int(self.expert_g_idx.shape[0])
        if self.expert_qweight2_sparse.numel() > 0 and expert_ncols > 0 and self.expert_qweight2_sparse_prefix.numel() == 0:
            self.expert_qweight2_sparse_prefix = self._build_sparse_qweight2_prefix(self.expert_c_map, expert_ncols, device)

    def _clear_base_block_gr_buffers(self, device: torch.device | None = None):
        device = device or self.fp_weight.device
        self.q_gmap_gr_words = torch.zeros((0,), dtype=torch.int32, device=device)
        self.q_gmap_gr_offsets = torch.zeros((0,), dtype=torch.int32, device=device)
        self.q_gmap_gr_meta = torch.zeros((0,), dtype=torch.int32, device=device)

    def _clear_expert_block_gr_buffers(self, device: torch.device | None = None):
        device = device or self.fp_weight.device
        self.expert_q_gmap_gr_words = torch.zeros((0,), dtype=torch.int32, device=device)
        self.expert_q_gmap_gr_offsets = torch.zeros((0,), dtype=torch.int32, device=device)
        self.expert_q_gmap_gr_meta = torch.zeros((0,), dtype=torch.int32, device=device)

    def _clear_base_compact3_buffers(self, device: torch.device | None = None):
        device = device or self.fp_weight.device
        self.col_pack3 = torch.zeros((0,), dtype=torch.int64, device=device)
        self.row_pack3 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=device)
        self.q_gmap_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        self.q_gmap_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=device)
        self._clear_base_block_gr_buffers(device=device)

    def _clear_expert_compact3_buffers(self, device: torch.device | None = None):
        device = device or self.fp_weight.device
        self.expert_col_pack3 = torch.zeros((0,), dtype=torch.int64, device=device)
        self.expert_row_pack3 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=device)
        self.expert_q_gmap_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        self.expert_q_gmap_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=device)
        self._clear_expert_block_gr_buffers(device=device)

    def _restore_dense_qweight2_from_sparse(self, packed_sparse: torch.Tensor, c_map: torch.Tensor, ncols: int, device: torch.device) -> torch.Tensor:
        salient_cols = self._unpack_bool_vector_len(c_map, ncols, device)
        nsalient = int(salient_cols.sum().item())
        dense = torch.zeros((self.outfeatures, ncols), dtype=torch.bool, device=device)
        if nsalient > 0 and packed_sparse.numel() > 0:
            dense[:, salient_cols] = self._unpack_bool_matrix_ncols(packed_sparse, nsalient, device)
        return self._pack_bool_matrix(dense)

    def _maybe_restore_sparse_qweight2_buffers(self):
        self._ensure_sparse_qweight2_prefix_buffers()

    def _pack_bool_vector(self, values: torch.Tensor) -> torch.Tensor:
        device = values.device
        if device.type == "meta":
            device = self.fp_weight.device
        values = values.to(device=device, dtype=torch.bool).contiguous()
        packed = torch.zeros((math.ceil(values.shape[0] / 32),), dtype=torch.int32, device=device)
        for row in range(packed.shape[0]):
            start = row * 32
            end = min(start + 32, values.shape[0])
            chunk = values[start:end].to(torch.int64)
            shifts = torch.arange(end - start, dtype=torch.int64, device=device)
            packed[row] = torch.sum(chunk << shifts).to(torch.int32)
        return packed.contiguous()

    def _pack_bool_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        device = self.fp_weight.device
        matrix = matrix.to(device=device, dtype=torch.bool).t().contiguous()
        packed = torch.zeros((math.ceil(matrix.shape[0] / 32), matrix.shape[1]), dtype=torch.int32, device=device)
        for row in range(packed.shape[0]):
            start = row * 32
            end = min(start + 32, matrix.shape[0])
            chunk = matrix[start:end].to(torch.int64)
            shifts = torch.arange(end - start, dtype=torch.int64, device=device).unsqueeze(1)
            packed[row] = torch.sum(chunk << shifts, dim=0).to(torch.int32)
        return packed.contiguous()

    def _unpack_bool_matrix(self, packed: torch.Tensor, device: torch.device) -> torch.Tensor:
        packed = packed.to(device=device, dtype=torch.int64)
        shifts = torch.arange(32, device=device, dtype=torch.int64).view(1, 1, 32)
        unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).permute(0, 2, 1).reshape(-1, packed.shape[1])
        unpacked = unpacked[: self.infeatures].t().contiguous()
        return unpacked.bool()

    def _unpack_bool_vector_len(self, packed: torch.Tensor, ncols: int, device: torch.device) -> torch.Tensor:
        packed = packed.to(device=device, dtype=torch.int64)
        shifts = torch.arange(32, device=device, dtype=torch.int64).view(1, 32)
        unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)
        return unpacked[:ncols].contiguous().bool()

    def _unpack_bool_matrix_ncols(self, packed: torch.Tensor, ncols: int, device: torch.device) -> torch.Tensor:
        packed = packed.to(device=device, dtype=torch.int64)
        shifts = torch.arange(32, device=device, dtype=torch.int64).view(1, 1, 32)
        unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).permute(0, 2, 1).reshape(-1, packed.shape[1])
        unpacked = unpacked[:ncols].t().contiguous()
        return unpacked.bool()

    def _pack_sparse_bool_matrix(self, matrix: torch.Tensor, active_cols: torch.Tensor) -> torch.Tensor:
        device = self.fp_weight.device
        matrix = matrix.to(device=device, dtype=torch.bool).contiguous()
        active_cols = active_cols.to(device=device, dtype=torch.bool).contiguous()
        if active_cols.numel() == 0 or int(active_cols.sum().item()) == 0:
            return torch.zeros((0, matrix.shape[0]), dtype=torch.int32, device=device)
        return self._pack_bool_matrix(matrix[:, active_cols])

    def _build_sparse_prefix_from_mask(self, active_cols: torch.Tensor, device: torch.device) -> torch.Tensor:
        active_cols = active_cols.to(device=device, dtype=torch.bool).flatten()
        packed_rows = math.ceil(active_cols.numel() / 32)
        prefix = torch.zeros((packed_rows + 1,), dtype=torch.int32, device=device)
        if packed_rows == 0:
            return prefix
        running = 0
        values = [0]
        active_cpu = active_cols.detach().to(device='cpu')
        for packed_row in range(packed_rows):
            start = packed_row * 32
            end = min(start + 32, active_cpu.numel())
            running += int(active_cpu[start:end].sum().item())
            values.append(running)
        return torch.tensor(values, dtype=torch.int32, device=device)

    def _restore_dense_bool_matrix_from_sparse(
        self,
        packed_sparse: torch.Tensor,
        sparse_prefix: torch.Tensor,
        c_map: torch.Tensor,
        ncols: int,
        device: torch.device,
    ) -> torch.Tensor:
        if ncols <= 0:
            return torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        inactive_cols = ~self._unpack_bool_vector_len(c_map, ncols, device)
        dense = torch.zeros((self.outfeatures, ncols), dtype=torch.bool, device=device)
        if packed_sparse.numel() == 0 or inactive_cols.numel() == 0 or int(inactive_cols.sum().item()) == 0:
            return self._pack_bool_matrix(dense)
        sparse_ncols = int(sparse_prefix[-1].item()) if sparse_prefix.numel() > 0 else int(inactive_cols.sum().item())
        if sparse_ncols <= 0:
            return self._pack_bool_matrix(dense)
        dense[:, inactive_cols] = self._unpack_bool_matrix_ncols(packed_sparse, sparse_ncols, device)
        return self._pack_bool_matrix(dense)

    def _pack_half4_words(self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor, fourth: torch.Tensor) -> torch.Tensor:
        def half_bits(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.contiguous().to(dtype=self.fp_weight.dtype).view(dtype=torch.int16).to(dtype=torch.int64) & 0xFFFF

        return (
            half_bits(first)
            | (half_bits(second) << 16)
            | (half_bits(third) << 32)
            | (half_bits(fourth) << 48)
        ).contiguous()

    def _unpack_half4_words(self, packed: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = packed.to(device=device, dtype=torch.int64).contiguous()

        def unpack(shift: int) -> torch.Tensor:
            bits = torch.bitwise_and(torch.bitwise_right_shift(packed, shift), 0xFFFF).to(device=device, dtype=torch.int16)
            return bits.view(torch.float16).to(device=device, dtype=dtype).contiguous()

        return unpack(0), unpack(16), unpack(32), unpack(48)

    def _fold_expert_ns_row_into_payload(self, payload: dict, device: torch.device) -> dict:
        if payload.get("ns_mode") != "vector":
            return payload
        ns_vector_row = payload.get("ns_vector_row")
        if not isinstance(ns_vector_row, torch.Tensor) or ns_vector_row.numel() == 0:
            return payload

        payload = dict(payload)
        row_original = ns_vector_row
        row_scale = row_original.to(device=device, dtype=self.fp_weight.dtype).contiguous().unsqueeze(1)
        payload["ns_vector_row"] = torch.ones_like(row_original, dtype=self.fp_weight.dtype).contiguous()

        if payload.get("mode") == "expert_compact3":
            row_pack3 = payload.get("row_pack3")
            if isinstance(row_pack3, torch.Tensor) and row_pack3.numel() > 0:
                s_r1, s_r2, ns_d_r1, ns_nd_r1 = self._unpack_half4_words(row_pack3, device, self.fp_weight.dtype)
                payload["row_pack3"] = self._pack_half4_words(
                    s_r1 * row_scale,
                    s_r2 * row_scale,
                    ns_d_r1 * row_scale,
                    ns_nd_r1 * row_scale,
                )
            return payload

        return payload

    def _restore_dense_sparse_bool_matrix(self, packed_sparse: torch.Tensor, active_cols: torch.Tensor, ncols: int, device: torch.device) -> torch.Tensor:
        active_cols = active_cols.to(device=device, dtype=torch.bool).contiguous()
        dense = torch.zeros((self.outfeatures, ncols), dtype=torch.bool, device=device)
        nactive = int(active_cols.sum().item())
        if nactive > 0 and packed_sparse.numel() > 0:
            dense[:, active_cols] = self._unpack_bool_matrix_ncols(packed_sparse, nactive, device)
        return dense

    def _signed_int32_from_unsigned_words(self, words: list[int], device: torch.device) -> torch.Tensor:
        if not words:
            return torch.zeros((0,), dtype=torch.int32, device=device)
        tensor = torch.tensor(words, dtype=torch.int64)
        tensor = torch.where(tensor >= (1 << 31), tensor - (1 << 32), tensor)
        return tensor.to(device=device, dtype=torch.int32).contiguous()

    def _encode_block_gr_zero_runs(self, zero_runs: list[int], k: int) -> tuple[list[int], int]:
        words: list[int] = []
        current_word = 0
        bit_offset = 0
        total_bits = 0

        def push_bit(bit: int):
            nonlocal current_word, bit_offset, total_bits
            if bit:
                current_word |= 1 << bit_offset
            bit_offset += 1
            total_bits += 1
            if bit_offset == 32:
                words.append(current_word)
                current_word = 0
                bit_offset = 0

        for run in zero_runs:
            quotient = run >> k if k > 0 else run
            remainder = run & ((1 << k) - 1) if k > 0 else 0
            for _ in range(quotient):
                push_bit(0)
            push_bit(1)
            for shift in range(k):
                push_bit((remainder >> shift) & 1)

        if bit_offset != 0:
            words.append(current_word)
        return words, total_bits

    def _pack_block_gr_gmap_tensor(self, packed: torch.Tensor, force_raw: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = packed.device
        if packed.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.int32, device=device)
            return empty, empty, empty

        packed_cpu = packed.detach().to(device="cpu", dtype=torch.int64).contiguous()
        packed_rows = int(packed_cpu.shape[0])
        outfeatures = int(packed_cpu.shape[1])
        num_k_blocks = math.ceil(packed_rows / BLOCK_GR_PACKED_ROWS)
        num_out_blocks = math.ceil(outfeatures / BLOCK_GR_TILE_OUT)

        payload_words: list[int] = []
        offsets = [0]
        meta = []

        for k_block in range(num_k_blocks):
            packed_row_base = k_block * BLOCK_GR_PACKED_ROWS
            for out_block in range(num_out_blocks):
                out_base = out_block * BLOCK_GR_TILE_OUT
                block_words: list[int] = []
                one_positions: list[int] = []
                for local_packed_row in range(BLOCK_GR_PACKED_ROWS):
                    packed_row = packed_row_base + local_packed_row
                    for local_out in range(BLOCK_GR_TILE_OUT):
                        out_idx = out_base + local_out
                        if packed_row < packed_rows and out_idx < outfeatures:
                            word = int(packed_cpu[packed_row, out_idx].item()) & 0xFFFFFFFF
                        else:
                            word = 0
                        block_words.append(word)
                        word_base = len(block_words) - 1
                        scan = word
                        while scan != 0:
                            low_bit = scan & -scan
                            bit_idx = low_bit.bit_length() - 1
                            one_positions.append(word_base * 32 + bit_idx)
                            scan ^= low_bit

                best_words: list[int] | None = None
                best_bits = BLOCK_GR_WORDS_PER_BLOCK * 32 + 1
                best_k = 0
                zero_runs = []
                prev = -1
                for pos in one_positions:
                    zero_runs.append(pos - prev - 1)
                    prev = pos
                for k in range(BLOCK_GR_MAX_K + 1):
                    encoded_words, encoded_bits = self._encode_block_gr_zero_runs(zero_runs, k)
                    if encoded_bits < best_bits:
                        best_bits = encoded_bits
                        best_words = encoded_words
                        best_k = k
                use_raw = force_raw or best_words is None or len(best_words) >= BLOCK_GR_WORDS_PER_BLOCK
                if use_raw:
                    payload_words.extend(block_words)
                    meta.append(BLOCK_GR_META_RAW_FLAG)
                else:
                    payload_words.extend(best_words)
                    meta.append(best_k | (len(one_positions) << BLOCK_GR_META_ONE_COUNT_SHIFT))
                offsets.append(len(payload_words))

        return (
            self._signed_int32_from_unsigned_words(payload_words, device),
            torch.tensor(offsets, dtype=torch.int32, device=device).contiguous(),
            torch.tensor(meta, dtype=torch.int32, device=device).contiguous(),
        )

    def _ensure_block_gr_gmap_buffers(self):
        device = self.fp_weight.device
        if self.q_gmap_gr_meta.numel() > 0 and self.q_gmap_gr_offsets.numel() > 0 and self.q_gmap_gr_meta.device == device:
            pass
        elif (
            _compact3_block_gr_enabled()
            and self.mode.numel() > 0
            and int(self.mode.reshape(-1)[0].item()) == 4
            and self.q_gmap_sparse.numel() > 0
            and self.c_map.numel() > 0
            and self.g_idx.numel() > 0
            and (self.q_gmap_gr_meta.numel() == 0 or self.q_gmap_gr_offsets.numel() == 0 or self.q_gmap_gr_meta.device != device)
        ):
            packed = self._restore_dense_bool_matrix_from_sparse(
                self.q_gmap_sparse.to(device=device, dtype=torch.int32).contiguous(),
                self.q_gmap_sparse_prefix.to(device=device, dtype=torch.int32).contiguous(),
                self.c_map.to(device=device, dtype=torch.int32).contiguous(),
                int(self.g_idx.shape[0]),
                device,
            )
            self.q_gmap_gr_words, self.q_gmap_gr_offsets, self.q_gmap_gr_meta = self._pack_block_gr_gmap_tensor(packed)
        else:
            self._clear_base_block_gr_buffers(device=device)

        if self.expert_q_gmap_gr_meta.numel() > 0 and self.expert_q_gmap_gr_offsets.numel() > 0 and self.expert_q_gmap_gr_meta.device == device:
            pass
        elif (
            _compact3_block_gr_enabled()
            and self.mode.numel() > 0
            and int(self.mode.reshape(-1)[0].item()) == 5
            and self.expert_q_gmap_sparse.numel() > 0
            and self.expert_c_map.numel() > 0
            and self.expert_g_idx.numel() > 0
            and (self.expert_q_gmap_gr_meta.numel() == 0 or self.expert_q_gmap_gr_offsets.numel() == 0 or self.expert_q_gmap_gr_meta.device != device)
        ):
            packed = self._restore_dense_bool_matrix_from_sparse(
                self.expert_q_gmap_sparse.to(device=device, dtype=torch.int32).contiguous(),
                self.expert_q_gmap_sparse_prefix.to(device=device, dtype=torch.int32).contiguous(),
                self.expert_c_map.to(device=device, dtype=torch.int32).contiguous(),
                int(self.expert_g_idx.shape[0]),
                device,
            )
            self.expert_q_gmap_gr_words, self.expert_q_gmap_gr_offsets, self.expert_q_gmap_gr_meta = self._pack_block_gr_gmap_tensor(packed)
        else:
            self._clear_expert_block_gr_buffers(device=device)

    def _reset_ns_state(self):
        self.ns_mode.zero_()
        self.ns_vector_row.fill_(1)
        self.ns_vector_col.fill_(1)
        self.expert_input_scale.fill_(1)
        self.ns_c1 = torch.zeros((0, 0), dtype=self.fp_weight.dtype)
        self.ns_c2 = torch.zeros((0, 0), dtype=self.fp_weight.dtype)
        self.ns_c1_quant.set_empty()
        self.ns_c2_quant.set_empty()
        self._refresh_runtime_scalar_cache()

    def _effective_input_scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # For expert_compact3 (mode=5), vector-NS input scaling is persisted as
        # expert_input_scale. ns_vector_col remains the base/non-expert field and
        # only serves as a legacy load source for old expert checkpoints.
        source = self.expert_input_scale if self._mode_host == 5 and self._ns_mode_host == 1 else self.ns_vector_col
        return source.to(device=device, dtype=dtype).contiguous()

    def _pack_ns_payload(self, payload):
        self._reset_ns_state()
        device = self.fp_weight.device
        ns_mode = payload.get("ns_mode")
        if ns_mode == "vector":
            self.ns_mode.fill_(1)
            ns_vector_row = payload.get("ns_vector_row")
            ns_vector_col = payload.get("ns_vector_col")
            if isinstance(ns_vector_row, torch.Tensor) and ns_vector_row.numel() > 0:
                self.ns_vector_row = ns_vector_row.to(device=device, dtype=self.fp_weight.dtype).contiguous()
            else:
                self.ns_vector_row = torch.ones_like(self.ns_vector_row, device=device, dtype=self.fp_weight.dtype).contiguous()
            if isinstance(ns_vector_col, torch.Tensor) and ns_vector_col.numel() > 0:
                self.ns_vector_col = ns_vector_col.to(device=device, dtype=self.fp_weight.dtype).contiguous()
            else:
                self.ns_vector_col = torch.ones_like(self.ns_vector_col, device=device, dtype=self.fp_weight.dtype).contiguous()
            self._refresh_runtime_scalar_cache()
            return
        if ns_mode != "matrix" or payload.get("c_weight") is None:
            self._refresh_runtime_scalar_cache()
            return

        self.ns_mode.fill_(2)
        c_weight = payload["c_weight"]
        shared_c2 = isinstance(c_weight, dict) and c_weight.get("shared_c2")
        if shared_c2:
            c1_payload = c_weight.get("c1")
            c2_payload = None
        else:
            c1_payload, c2_payload = c_weight

        if isinstance(c1_payload, dict) and "qweight" in c1_payload:
            self.ns_c1_quant.set_from_payload(c1_payload, device=device)
        else:
            self.ns_c1 = c1_payload.to(device=device, dtype=self.fp_weight.dtype).contiguous()

        if c2_payload is not None:
            if isinstance(c2_payload, dict) and "qweight" in c2_payload:
                self.ns_c2_quant.set_from_payload(c2_payload, device=device)
            else:
                self.ns_c2 = c2_payload.to(device=device, dtype=self.fp_weight.dtype).contiguous()

        self._refresh_runtime_scalar_cache()

    def _get_effective_ns_c2_quant(self):
        if self.ns_c2_quant.is_quantized():
            return self.ns_c2_quant
        shared_module = self._shared_matrix_module
        if shared_module is not None and shared_module.shared_ns_c2_quant.is_quantized():
            return shared_module.shared_ns_c2_quant
        return None

    def _get_effective_ns_c2_source(self):
        if self.ns_c2_quant.is_quantized() or self.ns_c2.numel() > 0:
            return self
        shared_module = self._shared_matrix_module
        if shared_module is None:
            return None
        if shared_module.shared_ns_c2_quant.is_quantized() or shared_module.shared_ns_c2.numel() > 0:
            return shared_module
        return None

    def _get_effective_ns_c2_tensor(self, device: torch.device, dtype: torch.dtype):
        c2_quant = self._get_effective_ns_c2_quant()
        if c2_quant is not None:
            return c2_quant.dequantize(device, dtype)
        if self.ns_c2.numel() > 0:
            return self.ns_c2.to(device=device, dtype=dtype)
        shared_module = self._shared_matrix_module
        if shared_module is not None and shared_module.shared_ns_c2.numel() > 0:
            return shared_module.get_shared_ns_c2(device, dtype)
        return None

    def _get_effective_ns_quantized_pair(self):
        if not self.ns_c1_quant.is_quantized():
            return None, None
        c2_quant = self._get_effective_ns_c2_quant()
        if c2_quant is None:
            return None, None
        if int(c2_quant._rows_host) != int(self.ns_c1_quant._cols_host):
            return None, None
        return self.ns_c1_quant, c2_quant

    def _get_ns_kernel_args(self, device: torch.device, dtype: torch.dtype):
        empty, empty_i, empty_h, empty_g = _empty_precomputed_kernel_tensors(device, dtype)
        c1_quant, c2_quant = self._get_effective_ns_quantized_pair()
        if c1_quant is None or c2_quant is None:
            return empty_i, empty_i, empty_h, empty_g, empty_i, empty_i, empty_h, empty_g, False
        return (
            c1_quant.qweight.contiguous(),
            c1_quant.qzeros.contiguous(),
            c1_quant.scales.contiguous(),
            c1_quant.g_idx.contiguous(),
            c2_quant.qweight.contiguous(),
            c2_quant.qzeros.contiguous(),
            c2_quant.scales.contiguous(),
            c2_quant.g_idx.contiguous(),
            True,
        )

    def _get_ns_matrices(self, device: torch.device, dtype: torch.dtype):
        c1_quant, c2_quant = self._get_effective_ns_quantized_pair()
        if c1_quant is not None and c2_quant is not None:
            return c1_quant.dequantize(device, dtype), c2_quant.dequantize(device, dtype)
        c2 = self._get_effective_ns_c2_tensor(device, dtype)
        if self.ns_c1_quant.is_quantized() and c2 is not None:
            return self.ns_c1_quant.dequantize(device, dtype), c2
        if self.ns_c1.numel() > 0 and c2 is not None:
            return self.ns_c1.to(device=device, dtype=dtype), c2
        return None, None

    def pack_dense(self, linear, payload=None):
        self.mode.fill_(0)
        self._refresh_runtime_scalar_cache()
        object.__setattr__(self, "_shared_matrix_module", None)
        self.fp_weight = self._clone_linear_weight(linear).to(device=self.fp_weight.device, dtype=self.fp_weight.dtype).contiguous()
        self.qweight2_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.qweight2_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=self.fp_weight.device)
        self.q_gmap_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=self.fp_weight.device)
        self.q_gmap_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=self.fp_weight.device)
        self._clear_base_block_gr_buffers(device=self.fp_weight.device)
        self.row_pack1 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=self.fp_weight.device)
        self.row_pack2 = torch.zeros((self.outfeatures, 0), dtype=torch.int32, device=self.fp_weight.device)
        self._clear_base_compact3_buffers(device=self.fp_weight.device)
        self.expert_qweight2_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_qweight2_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=self.fp_weight.device)
        self.expert_q_gmap_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=self.fp_weight.device)
        self.expert_q_gmap_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=self.fp_weight.device)
        self._clear_expert_block_gr_buffers(device=self.fp_weight.device)
        self._reset_ns_state()
        self.expert_qweight1 = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_qweight2 = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_qweight2_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_q_gmap = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_c_map = torch.zeros((0,), dtype=torch.int32)
        self.expert_g_idx = torch.zeros((0,), dtype=torch.int32)
        self.expert_row_pack1 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=self.fp_weight.device)
        self.expert_row_pack2 = torch.zeros((self.outfeatures, 0), dtype=torch.int32, device=self.fp_weight.device)
        self._clear_expert_compact3_buffers(device=self.fp_weight.device)
        if linear.bias is not None and self.bias is not None:
            self.bias = linear.bias.data.clone().to(dtype=self.bias.dtype)
        self._trim_runtime_state_for_mode()
        self.invalidate_cache()

    def pack_structured(self, linear, payload):
        if payload.get("mode") != "compact3":
            raise RuntimeError("pack_structured only supports compact3 payloads; structured mode has been removed.")
        self.mode.fill_(4)
        self._refresh_runtime_scalar_cache()
        object.__setattr__(self, "_shared_matrix_module", None)
        device = self.fp_weight.device
        self.expert_qweight1 = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_qweight2 = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_qweight2_sparse = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_qweight2_sparse_prefix = torch.zeros((0,), dtype=torch.int32, device=device)
        self.expert_q_gmap = torch.zeros((0, self.outfeatures), dtype=torch.int32)
        self.expert_c_map = torch.zeros((0,), dtype=torch.int32)
        self.expert_g_idx = torch.zeros((0,), dtype=torch.int32)
        self.expert_row_pack1 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=device)
        self.expert_row_pack2 = torch.zeros((self.outfeatures, 0), dtype=torch.int32, device=device)
        self._clear_expert_compact3_buffers(device=device)

        qweight1 = payload.get("qweight1")
        c_map_packed = payload.get("c_map_packed")
        if qweight1 is None or c_map_packed is None or payload.get("g_idx") is None:
            raise RuntimeError("Packed-only compact3 payload requires qweight1, c_map_packed, and g_idx.")
        self.qweight1 = qweight1.to(device=device, dtype=torch.int32).contiguous()
        self.c_map = c_map_packed.to(device=device, dtype=torch.int32).contiguous()
        if payload.get("g_idx") is not None:
            self.g_idx = payload["g_idx"].to(device=device, dtype=torch.int32).contiguous()

        qweight2_sparse = payload.get("qweight2_sparse")
        qweight2_sparse_prefix = payload.get("qweight2_sparse_prefix")
        q_gmap_sparse = payload.get("q_gmap_sparse")
        q_gmap_sparse_prefix = payload.get("q_gmap_sparse_prefix")
        col_pack3 = payload.get("col_pack3")
        row_pack3 = payload.get("row_pack3")
        if (
            qweight2_sparse is None
            or qweight2_sparse_prefix is None
            or q_gmap_sparse is None
            or q_gmap_sparse_prefix is None
            or col_pack3 is None
            or row_pack3 is None
        ):
            raise RuntimeError("Packed-only compact3 payload requires sparse qweight/gmap prefixes and row/col pack3 tensors.")
        self.qweight2 = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        self.q_gmap = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        self.qweight2_sparse = qweight2_sparse.to(device=device, dtype=torch.int32).contiguous()
        self.qweight2_sparse_prefix = qweight2_sparse_prefix.to(device=device, dtype=torch.int32).contiguous()
        self.q_gmap_sparse = q_gmap_sparse.to(device=device, dtype=torch.int32).contiguous()
        self.q_gmap_sparse_prefix = q_gmap_sparse_prefix.to(device=device, dtype=torch.int32).contiguous()
        self.col_pack3 = col_pack3.to(device=device, dtype=torch.int64).contiguous()
        self.row_pack3 = row_pack3.to(device=device, dtype=torch.int64).contiguous()
        self.row_pack1 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=device)
        self.row_pack2 = torch.zeros((self.outfeatures, 0), dtype=torch.int32, device=device)
        self._clear_base_block_gr_buffers(device=device)

        self._pack_ns_payload(payload)
        if linear.bias is not None and self.bias is not None:
            self.bias = linear.bias.data.clone().to(dtype=self.bias.dtype)
        self._trim_runtime_state_for_mode()
        self.invalidate_cache()
        self.prepare_compact3_k64_groups()

    def pack_expert(self, linear, payload):
        payload = self._fold_expert_ns_row_into_payload(payload, self.fp_weight.device)
        if payload.get("mode") != "expert_compact3":
            raise RuntimeError("Expert payloads must use expert_compact3; legacy expert mode has been removed.")
        self.mode.fill_(5)
        self._refresh_runtime_scalar_cache()
        device = self.fp_weight.device
        self.row_pack1 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=device)
        self.row_pack2 = torch.zeros((self.outfeatures, 0), dtype=torch.int32, device=device)
        self._clear_base_compact3_buffers(device=device)
        qweight1 = payload.get("qweight1")
        c_map_packed = payload.get("c_map_packed")
        qweight2_sparse = payload.get("qweight2_sparse")
        qweight2_sparse_prefix = payload.get("qweight2_sparse_prefix")
        q_gmap_sparse = payload.get("q_gmap_sparse")
        q_gmap_sparse_prefix = payload.get("q_gmap_sparse_prefix")
        col_pack3 = payload.get("col_pack3")
        row_pack3 = payload.get("row_pack3")
        if (
            qweight1 is None
            or c_map_packed is None
            or payload.get("g_idx") is None
            or qweight2_sparse is None
            or qweight2_sparse_prefix is None
            or q_gmap_sparse is None
            or q_gmap_sparse_prefix is None
            or col_pack3 is None
            or row_pack3 is None
        ):
            raise RuntimeError("Packed-only expert_compact3 payload requires packed qweight/c_map/g_idx plus sparse qweight/gmap prefixes and row/col pack3.")
        self.expert_qweight1 = qweight1.to(device=device, dtype=torch.int32).contiguous()
        self.expert_qweight2 = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        self.expert_q_gmap = torch.zeros((0, self.outfeatures), dtype=torch.int32, device=device)
        self.expert_c_map = c_map_packed.to(device=device, dtype=torch.int32).contiguous()
        self.expert_g_idx = payload["g_idx"].to(device=device, dtype=torch.int32).contiguous()
        self.expert_qweight2_sparse = qweight2_sparse.to(device=device, dtype=torch.int32).contiguous()
        self.expert_qweight2_sparse_prefix = qweight2_sparse_prefix.to(device=device, dtype=torch.int32).contiguous()
        self.expert_q_gmap_sparse = q_gmap_sparse.to(device=device, dtype=torch.int32).contiguous()
        self.expert_q_gmap_sparse_prefix = q_gmap_sparse_prefix.to(device=device, dtype=torch.int32).contiguous()
        self.expert_col_pack3 = col_pack3.to(device=device, dtype=torch.int64).contiguous()
        self.expert_row_pack3 = row_pack3.to(device=device, dtype=torch.int64).contiguous()
        self.expert_scales_s_d_c1 = torch.zeros((0,), dtype=self.fp_weight.dtype, device=device)
        self.expert_scales_s_d_c2 = torch.zeros((0,), dtype=self.fp_weight.dtype, device=device)
        self.expert_scales_s_nd_c1 = torch.zeros((0,), dtype=self.fp_weight.dtype, device=device)
        self.expert_scales_s_nd_c2 = torch.zeros((0,), dtype=self.fp_weight.dtype, device=device)
        self.expert_scales_ns_d_c1 = torch.zeros((0,), dtype=self.fp_weight.dtype, device=device)
        self.expert_scales_ns_nd_c1 = torch.zeros((0,), dtype=self.fp_weight.dtype, device=device)
        self.expert_row_pack1 = torch.zeros((self.outfeatures, 0), dtype=torch.int64, device=device)
        self.expert_row_pack2 = torch.zeros((self.outfeatures, 0), dtype=torch.int32, device=device)
        self._clear_expert_block_gr_buffers(device=device)
        self._pack_ns_payload(payload)
        if self._ns_mode_host == 1:
            # New expert_compact3 format keeps the input-side vector scale in its
            # own buffer instead of persisting legacy ns_vector_col.
            self.expert_input_scale = self.ns_vector_col.to(device=device, dtype=self.fp_weight.dtype).contiguous()
            self.ns_vector_col = torch.ones_like(self.ns_vector_col, device=device, dtype=self.fp_weight.dtype).contiguous()
        if linear.bias is not None and self.bias is not None:
            self.bias = linear.bias.data.clone().to(dtype=self.bias.dtype)
        self._trim_runtime_state_for_mode()
        self.invalidate_cache()
        self.prepare_compact3_k64_groups()

    def _group_scales(self, row_scales: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        g_idx = self.g_idx.to(device=device, dtype=torch.long)
        return row_scales.to(device=device, dtype=dtype)[:, g_idx]

    def _group_scales_custom(self, row_scales: torch.Tensor, g_idx: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        g_idx = g_idx.to(device=device, dtype=torch.long)
        return row_scales.to(device=device, dtype=dtype)[:, g_idx]

    def _dequantize_compact3_matrix(
        self,
        qweight1: torch.Tensor,
        qweight2_sparse: torch.Tensor,
        q_gmap_sparse: torch.Tensor,
        c_map: torch.Tensor,
        g_idx: torch.Tensor,
        col_pack3: torch.Tensor,
        row_pack3: torch.Tensor,
        ncols: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        b1 = self._unpack_bool_matrix_ncols(qweight1, ncols, device).to(dtype=dtype).mul_(2).sub_(1)
        salient_cols = self._unpack_bool_vector_len(c_map, ncols, device)
        non_salient_cols = ~salient_cols
        b2_dense = self._restore_dense_sparse_bool_matrix(qweight2_sparse, salient_cols, ncols, device).to(dtype=dtype).mul_(2).sub_(1)
        gmap_dense = self._restore_dense_sparse_bool_matrix(q_gmap_sparse, non_salient_cols, ncols, device)

        s_c1, s_c2, ns_d_c1, ns_nd_c1 = self._unpack_half4_words(col_pack3, device, dtype)
        s_r1, s_r2, ns_d_r1, ns_nd_r1 = self._unpack_half4_words(row_pack3, device, dtype)
        g_idx_long = g_idx.to(device=device, dtype=torch.long)
        s_r1 = s_r1[:, g_idx_long]
        s_r2 = s_r2[:, g_idx_long]
        ns_d_r1 = ns_d_r1[:, g_idx_long]
        ns_nd_r1 = ns_nd_r1[:, g_idx_long]

        salient_mask = salient_cols.unsqueeze(0).expand(self.outfeatures, -1)
        ns_d_mask = non_salient_cols.unsqueeze(0).expand(self.outfeatures, -1) & gmap_dense
        ns_nd_mask = non_salient_cols.unsqueeze(0).expand(self.outfeatures, -1) & (~gmap_dense)

        weight = torch.zeros((self.outfeatures, ncols), device=device, dtype=dtype)
        weight += b1 * (s_r1 * s_c1.unsqueeze(0)) * salient_mask.to(dtype)
        weight += b2_dense * (s_r2 * s_c2.unsqueeze(0)) * salient_mask.to(dtype)
        weight += b1 * (ns_d_r1 * ns_d_c1.unsqueeze(0)) * ns_d_mask.to(dtype)
        weight += b1 * (ns_nd_r1 * ns_nd_c1.unsqueeze(0)) * ns_nd_mask.to(dtype)
        return weight

    def _run_compact3_cuda(self, x_base: torch.Tensor, x_original: torch.Tensor | None = None):
        ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
        if ext is None:
            return None
        if x_base.device.type != "cuda" or x_base.dtype != torch.float16:
            return None
        self._ensure_legacy_bitpacked_buffers(require_dense_qweight2=False)
        self._ensure_block_gr_gmap_buffers()
        kernel = getattr(ext, "compact3_linear_forward", None)
        if kernel is None:
            return None
        if self.col_pack3.numel() == 0 or self.row_pack3.numel() == 0:
            return None

        x_runtime = x_original if x_original is not None else x_base
        ns_mode = self._ns_mode_host
        has_input_scale = ns_mode == 1
        has_output_scale = ns_mode == 1
        has_ns_matrix = _can_fuse_packed_ns(self)
        input_scale = self.ns_vector_col.to(device=x_base.device, dtype=x_base.dtype).contiguous()
        output_scale = self.ns_vector_row.to(device=x_base.device, dtype=x_base.dtype).contiguous()
        bias = self.bias.to(device=x_base.device, dtype=x_base.dtype).contiguous() if self.bias is not None else x_base.new_empty((0,))
        (
            ns_c1_qweight,
            ns_c1_qzeros,
            ns_c1_scales,
            ns_c1_g_idx,
            ns_c2_qweight,
            ns_c2_qzeros,
            ns_c2_scales,
            ns_c2_g_idx,
            has_ns_matrix,
        ) = self._get_ns_kernel_args(x_base.device, x_base.dtype)
        batch = int(x_base.size(0))
        out_tiles = (self.outfeatures + 63) // 64
        decode_phase = _decode_phase_enabled(x_runtime)
        label = _module_debug_label(self, "compact3_linear_forward")
        use_block_gr = self.q_gmap_gr_offsets.numel() > 0 and self.q_gmap_gr_meta.numel() > 0
        fast_path_hit = (not has_input_scale) and (not has_output_scale) and (not has_ns_matrix) and (
            (self.bias is None) or use_block_gr
        )
        if use_block_gr and fast_path_hit:
            shared_expert_override = _resolve_shared_expert_compact3_block_gr_split_k_slices_py(
                rank=self.infeatures,
                batch=batch,
                out_tiles=out_tiles,
                device=x_base.device,
                decode_phase=decode_phase,
                label=label,
            )
            if shared_expert_override is not None:
                split_k_slices, split_k_source = shared_expert_override
            else:
                split_k_slices, split_k_source = _resolve_compact3_block_gr_split_k_slices_py(
                    self.infeatures,
                    batch,
                    out_tiles,
                    x_base.device,
                    decode_phase,
                )
        elif use_block_gr:
            split_k_slices, split_k_source = 1, "disabled:not-fast-path"
        else:
            split_k_slices, split_k_source = 1, "disabled:no-block-gr"
        prefill_tc_requested = (
            (not decode_phase)
            and batch >= 16
            and use_block_gr
            and (not has_input_scale)
            and (not has_output_scale)
            and (self.bias is None)
            and (not has_ns_matrix)
            and split_k_slices == 1
            and (not _env_flag_enabled("TRIS_DISABLE_COMPACT3_PREFILL_TC"))
        )
        prefill_tc_hit = prefill_tc_requested and _compact3_tensor_core_device_supported(x_base.device)
        _maybe_log_compact3_runtime(
            label,
            batch,
            out_tiles,
            decode_phase,
            use_block_gr,
            split_k_slices,
            split_k_source,
            fast_path_hit,
            prefill_tc_hit,
        )
        try:
            return kernel(
                x_base.contiguous(),
                self.qweight1.contiguous(),
                self.qweight2_sparse.contiguous(),
                self.qweight2_sparse_prefix.contiguous(),
                self.q_gmap_sparse.contiguous(),
                self.q_gmap_sparse_prefix.contiguous(),
                self.q_gmap_gr_words.contiguous(),
                self.q_gmap_gr_offsets.contiguous(),
                self.q_gmap_gr_meta.contiguous(),
                self.c_map.contiguous(),
                self.g_idx.contiguous(),
                self.col_pack3.contiguous(),
                self.row_pack3.contiguous(),
                input_scale,
                output_scale,
                bias,
                has_input_scale,
                has_output_scale,
                self.bias is not None,
                ns_c1_qweight,
                ns_c1_qzeros,
                ns_c1_scales,
                ns_c1_g_idx,
                ns_c2_qweight,
                ns_c2_qzeros,
                ns_c2_scales,
                ns_c2_g_idx,
                has_ns_matrix,
                split_k_slices,
                prefill_tc_requested,
                use_block_gr and self._compact3_k64_groups_uniform(),
            )
        except RuntimeError as exc:
            _print_cuda_fallback(_module_debug_label(self, "compact3_linear_forward"), exc)
            return None


    def _run_compact3_expert_precomputed_cuda(self, hidden: torch.Tensor, x_base: torch.Tensor, fuse_ns: bool = True, fuse_bias: bool = True):
        ext = autogptq_TriS_cuda if _autogptq_TriS_cuda_available else _load_TriS_extension()
        if ext is None:
            return None
        if hidden.device.type != "cuda" or hidden.dtype != torch.float16:
            return None
        if x_base.device.type != "cuda" or x_base.dtype != torch.float16:
            return None
        self._ensure_legacy_bitpacked_buffers(require_dense_qweight2=False)
        kernel = getattr(ext, "compact3_tris_precomputed_forward", None)
        if kernel is None:
            return None
        if self.expert_col_pack3.numel() == 0 or self.expert_row_pack3.numel() == 0:
            return None
        empty, empty_i, empty_h, empty_g = _empty_precomputed_kernel_tensors(hidden.device, hidden.dtype)
        output_scale = self.ns_vector_row.to(device=hidden.device, dtype=hidden.dtype).contiguous()
        has_bias = self.bias is not None and fuse_bias
        bias = self.bias.to(device=hidden.device, dtype=hidden.dtype).contiguous() if has_bias else empty
        (
            ns_c1_qweight,
            ns_c1_qzeros,
            ns_c1_scales,
            ns_c1_g_idx,
            ns_c2_qweight,
            ns_c2_qzeros,
            ns_c2_scales,
            ns_c2_g_idx,
            has_ns_matrix,
        ) = self._get_ns_kernel_args(x_base.device, x_base.dtype)
        has_ns_matrix = fuse_ns and has_ns_matrix
        try:
            return kernel(
                hidden.contiguous(),
                x_base.contiguous(),
                self.expert_qweight1.contiguous(),
                self.expert_qweight2_sparse.contiguous(),
                self.expert_qweight2_sparse_prefix.contiguous(),
                self.expert_q_gmap_sparse.contiguous(),
                self.expert_q_gmap_sparse_prefix.contiguous(),
                self.expert_c_map.contiguous(),
                self.expert_g_idx.contiguous(),
                self.expert_col_pack3.contiguous(),
                self.expert_row_pack3.contiguous(),
                output_scale,
                bias,
                has_bias,
                ns_c1_qweight if has_ns_matrix else empty_i,
                ns_c1_qzeros if has_ns_matrix else empty_i,
                ns_c1_scales if has_ns_matrix else empty_h,
                ns_c1_g_idx if has_ns_matrix else empty_g,
                ns_c2_qweight if has_ns_matrix else empty_i,
                ns_c2_qzeros if has_ns_matrix else empty_i,
                ns_c2_scales if has_ns_matrix else empty_h,
                ns_c2_g_idx if has_ns_matrix else empty_g,
            )
        except RuntimeError as exc:
            _print_cuda_fallback(_module_debug_label(self, "compact3_tris_precomputed_forward"), exc)
            return None

    def _run_ns_c1_only_cuda(self, hidden: torch.Tensor):
        if hidden.device.type != "cuda" or hidden.dtype != torch.float16:
            return None
        if self.ns_c1_quant.is_quantized():
            return _run_packed_4bit_linear_cuda(hidden, self.ns_c1_quant, _module_debug_label(self, "ns_c1_quant"))
        if self.ns_c1.numel() > 0:
            return F.linear(hidden, self.ns_c1.to(device=hidden.device, dtype=hidden.dtype))
        return None

    def _run_ns_compensation_cuda(self, x_base: torch.Tensor):
        if x_base.device.type != "cuda" or x_base.dtype != torch.float16:
            return None

        c2_quant = self._get_effective_ns_c2_quant()
        if c2_quant is not None:
            c2_source = self._get_effective_ns_c2_source()
            label_owner = c2_source if isinstance(c2_source, nn.Module) else self
            hidden = _run_packed_4bit_linear_cuda(x_base, c2_quant, _module_debug_label(label_owner, "ns_c2_quant"))
            if hidden is None:
                return None
        else:
            c2 = self._get_effective_ns_c2_tensor(x_base.device, x_base.dtype)
            if c2 is None:
                return None
            hidden = F.linear(x_base, c2)

        return self._run_ns_c1_only_cuda(hidden)

    def can_precompute_shared(self) -> bool:
        return self._mode_host == 5 and self._shared_matrix_module is not None

    def compute_shared_hidden(self, x: torch.Tensor):
        if not self.can_precompute_shared():
            return None
        if self._expert_runtime_pruned and not _deployment_only_enabled():
            raise RuntimeError("Expert runtime state was pruned outside deployment-only mode.")
        shared_module = self._shared_matrix_module
        x_shape = x.shape[:-1]
        x_base = x.reshape(-1, x.shape[-1])
        ns_mode = self._ns_mode_host
        use_fused_precompute = ns_mode == 2 and _shared_precompute_v1_enabled() and shared_module.shared_precompute_quant.is_quantized()
        if use_fused_precompute:
            shared_module.clear_cached_shared_ns_hidden()
        if x_base.shape[0] == 0:
            if use_fused_precompute:
                sv_rank = int(shared_module._shared_precompute_sv_rank_host)
                ns_rank = int(shared_module._shared_precompute_ns_rank_host)
                if ns_rank > 0:
                    shared_module.cache_shared_ns_hidden(x_base.new_empty((0, ns_rank)))
                return x_base.new_empty((0, sv_rank)).reshape(x_shape + (sv_rank,))
            shared_quant = shared_module.shared_sv_quant
            rank = shared_quant._rows_host if shared_quant.is_quantized() else shared_module.shared_sv.shape[0]
            return x_base.new_empty((0, rank)).reshape(x_shape + (rank,))

        x_shared = x_base
        if ns_mode == 1:
            ns_col = self._effective_input_scale(x_base.device, x_base.dtype).unsqueeze(0)
            x_shared = x_base * ns_col

        if use_fused_precompute:
            hidden_all = _run_packed_4bit_linear_cuda(x_shared, shared_module.shared_precompute_quant, _module_debug_label(shared_module, "shared_precompute_quant"))
            if hidden_all is not None:
                sv_rank = int(shared_module._shared_precompute_sv_rank_host)
                ns_rank = int(shared_module._shared_precompute_ns_rank_host)
                if ns_rank > 0:
                    shared_module.cache_shared_ns_hidden(hidden_all[:, sv_rank : sv_rank + ns_rank])
                return hidden_all[:, :sv_rank].reshape(x_shape + (sv_rank,))
            if _deployment_only_enabled():
                _raise_deployment_only_failure(_module_debug_label(self, "compute_shared_hidden"), "packed fused shared precompute kernel returned no result")

        shared_quant = shared_module.shared_sv_quant
        hidden = None
        if shared_quant.is_quantized():
            hidden = _run_packed_4bit_linear_cuda(x_shared, shared_quant, _module_debug_label(shared_module, "shared_sv_quant"))
        if hidden is None:
            if _deployment_only_enabled() and shared_quant.is_quantized():
                _raise_deployment_only_failure(_module_debug_label(self, "compute_shared_hidden"), "packed shared matrix kernel returned no result")
            shared_sv = shared_module.get_shared_sv(x_base.device, x_base.dtype)
            hidden = F.linear(x_shared, shared_sv)
        return hidden.reshape(x_shape + (hidden.shape[-1],))

    def forward_from_precomputed_hidden(self, hidden: torch.Tensor, x_base: torch.Tensor):
        mode = self._mode_host
        if mode != 5:
            raise RuntimeError(f"forward_from_precomputed_hidden only supports expert_compact3 mode, got mode={mode}")
        if self._expert_runtime_pruned:
            raise RuntimeError("Expert runtime state was pruned outside deployment-only mode.")

        out_shape = hidden.shape[:-1] + (self.outfeatures,)
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        x_base_flat = x_base.reshape(-1, x_base.shape[-1])
        if hidden_flat.shape[0] == 0:
            return hidden_flat.new_empty((0, self.outfeatures)).reshape(out_shape)

        ns_mode = self._ns_mode_host
        bias = self.bias.to(device=hidden_flat.device, dtype=hidden_flat.dtype) if self.bias is not None else None
        epilogue_fused = False
        ns_epilogue_fused = False
        fused_ns = _can_fuse_packed_ns(self)

        split_grouped_epilogue = fused_ns and _deployment_only_enabled() and _env_flag_enabled("OLMOE_GROUPED_EXPERTS")
        out = self._run_compact3_expert_precomputed_cuda(
            hidden_flat,
            x_base_flat,
            fuse_ns=not split_grouped_epilogue,
            fuse_bias=not split_grouped_epilogue,
        )
        ns_epilogue_fused = out is not None and fused_ns and not split_grouped_epilogue
        epilogue_fused = out is not None and not split_grouped_epilogue
        if out is None:
            if _deployment_only_enabled():
                _raise_deployment_only_failure(_module_debug_label(self, "forward_from_precomputed_hidden"), "compact3 expert kernel returned no result")
            expert_weight = self._dequantize_compact3_matrix(
                self.expert_qweight1,
                self.expert_qweight2_sparse,
                self.expert_q_gmap_sparse,
                self.expert_c_map,
                self.expert_g_idx,
                self.expert_col_pack3,
                self.expert_row_pack3,
                self.expert_g_idx.shape[0],
                hidden_flat.device,
                hidden_flat.dtype,
            )
            out = F.linear(hidden_flat, expert_weight)

        if not epilogue_fused and ns_mode == 1:
            ns_row = self.ns_vector_row.to(device=hidden_flat.device, dtype=hidden_flat.dtype).unsqueeze(0)
            out = out * ns_row

        if ns_mode == 2 and not ns_epilogue_fused:
            compensation = self._run_ns_compensation_cuda(x_base_flat)
            if compensation is not None:
                out = out + compensation
            else:
                if _deployment_only_enabled():
                    _raise_deployment_only_failure(_module_debug_label(self, "forward_from_precomputed_hidden"), "ns compensation kernel returned no result")
                c1, c2 = self._get_ns_matrices(x_base_flat.device, x_base_flat.dtype)
                if c1 is not None and c2 is not None:
                    out = out + F.linear(F.linear(x_base_flat, c2), c1)

        if bias is not None and not epilogue_fused:
            out = out + bias
        return out.reshape(out_shape)

    def forward_pair_from_precomputed_hidden(
        self,
        other: "QuantLinear",
        hidden: torch.Tensor,
        other_hidden: torch.Tensor,
        x_base: torch.Tensor,
    ):
        self_out_shape = hidden.shape[:-1] + (self.outfeatures,)
        other_out_shape = other_hidden.shape[:-1] + (other.outfeatures,)
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        other_hidden_flat = other_hidden.reshape(-1, other_hidden.shape[-1])
        x_base_flat = x_base.reshape(-1, x_base.shape[-1])

        if hidden_flat.shape[0] == 0:
            return (
                hidden_flat.new_empty((0, self.outfeatures)).reshape(self_out_shape),
                other_hidden_flat.new_empty((0, other.outfeatures)).reshape(other_out_shape),
            )

        return (
            self.forward_from_precomputed_hidden(hidden, x_base),
            other.forward_from_precomputed_hidden(other_hidden, x_base),
        )

    def _dequantize_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (device.type, device.index, dtype)
        if self._cached_weight is not None and self._cached_weight_key == cache_key:
            return self._cached_weight

        mode = self._mode_host
        ns_mode = self._ns_mode_host

        if _deployment_only_enabled() and mode in (4, 5):
            _raise_deployment_only_failure(_module_debug_label(self, "dequantize_weight"), "dense weight materialization is disabled")

        if mode == 0:
            weight = self.fp_weight.to(device=device, dtype=dtype)
        elif mode == 4:
            weight = self._dequantize_compact3_matrix(
                self.qweight1,
                self.qweight2_sparse,
                self.q_gmap_sparse,
                self.c_map,
                self.g_idx,
                self.col_pack3,
                self.row_pack3,
                self.infeatures,
                device,
                dtype,
            )
            if ns_mode == 1:
                weight = (
                    self.ns_vector_row.to(device=device, dtype=dtype).unsqueeze(1)
                    * weight
                    * self.ns_vector_col.to(device=device, dtype=dtype).unsqueeze(0)
                )
            c1, c2 = self._get_ns_matrices(device, dtype)
            if c1 is not None and c2 is not None:
                weight = weight + c1 @ c2
        elif mode == 5:
            if self._shared_matrix_module is None:
                raise RuntimeError("Expert QuantLinear is missing bound shared matrix module.")
            rank = self.expert_g_idx.shape[0]
            expert_factor = self._dequantize_compact3_matrix(
                self.expert_qweight1,
                self.expert_qweight2_sparse,
                self.expert_q_gmap_sparse,
                self.expert_c_map,
                self.expert_g_idx,
                self.expert_col_pack3,
                self.expert_row_pack3,
                rank,
                device,
                dtype,
            )
            shared_sv = self._shared_matrix_module.get_shared_sv(device, dtype)
            weight = expert_factor @ shared_sv
            if ns_mode == 1:
                weight = (
                    self.ns_vector_row.to(device=device, dtype=dtype).unsqueeze(1)
                    * weight
                    * self._effective_input_scale(device, dtype).unsqueeze(0)
                )
            c1, c2 = self._get_ns_matrices(device, dtype)
            if c1 is not None and c2 is not None:
                weight = weight + c1 @ c2
        else:
            raise RuntimeError(f"QuantLinear mode={mode} is not supported; structured modes have been removed.")

        self._cached_weight = weight
        self._cached_weight_key = cache_key
        return weight

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.shape[0] == 0:
            return x_flat.new_empty((0, self.outfeatures)).reshape(out_shape)
        bias = self.bias.to(device=x_flat.device, dtype=x_flat.dtype) if self.bias is not None else None
        mode = self._mode_host
        ns_mode = self._ns_mode_host
        if mode == 4:
            staging = getattr(self, "_qgmap_staging", None)
            if staging is not None:
                staging[0].stage(staging[1])
            x_base = x_flat
            fused_ns = _can_fuse_packed_ns(self)
            out = self._run_compact3_cuda(x_base, x)
            if out is None:
                if _deployment_only_enabled():
                    _raise_deployment_only_failure(_module_debug_label(self, "forward"), "compact3 linear kernel returned no result")
                weight = self._dequantize_weight(x_flat.device, x_flat.dtype)
                out = F.linear(x_flat, weight, bias)
            else:
                if ns_mode == 2 and not fused_ns:
                    compensation = self._run_ns_compensation_cuda(x_base)
                    if compensation is not None:
                        out = out + compensation
                    else:
                        if _deployment_only_enabled():
                            _raise_deployment_only_failure(_module_debug_label(self, "forward"), "ns compensation kernel returned no result")
                        c1, c2 = self._get_ns_matrices(x_base.device, x_base.dtype)
                        if c1 is not None and c2 is not None:
                            out = out + F.linear(F.linear(x_base, c2), c1)
        elif mode == 5:
            if self._shared_matrix_module is None:
                raise RuntimeError("Expert QuantLinear is missing bound shared matrix module.")
            x_base = x_flat
            ns_row = self.ns_vector_row.to(device=x_base.device, dtype=x_base.dtype).unsqueeze(0)
            ns_col = self._effective_input_scale(x_base.device, x_base.dtype).unsqueeze(0)

            hidden = None
            epilogue_fused = False
            ns_epilogue_fused = False
            fused_ns = _can_fuse_packed_ns(self)
            shared_quant = self._shared_matrix_module.shared_sv_quant
            x_shared = x_base * ns_col if ns_mode == 1 else x_base
            if shared_quant.is_quantized():
                hidden = _run_packed_4bit_linear_cuda(x_shared, shared_quant, _module_debug_label(self._shared_matrix_module, "shared_sv_quant"))
            if hidden is None:
                if _deployment_only_enabled() and shared_quant.is_quantized():
                    _raise_deployment_only_failure(_module_debug_label(self, "forward"), "packed shared matrix kernel returned no result")
                shared_sv = self._shared_matrix_module.get_shared_sv(x_base.device, x_base.dtype)
                hidden = F.linear(x_shared, shared_sv)

            if self._expert_runtime_pruned:
                _raise_deployment_only_failure(_module_debug_label(self, "forward"), "expert runtime state was pruned")
            out = self._run_compact3_expert_precomputed_cuda(hidden, x_base)
            ns_epilogue_fused = out is not None and fused_ns
            epilogue_fused = out is not None
            if out is None:
                if _deployment_only_enabled():
                    _raise_deployment_only_failure(_module_debug_label(self, "forward"), "compact3 expert kernel returned no result")
                expert_weight = self._dequantize_compact3_matrix(
                    self.expert_qweight1,
                    self.expert_qweight2_sparse,
                    self.expert_q_gmap_sparse,
                    self.expert_c_map,
                    self.expert_g_idx,
                    self.expert_col_pack3,
                    self.expert_row_pack3,
                    self.expert_g_idx.shape[0],
                    x_base.device,
                    x_base.dtype,
                )
                out = F.linear(hidden, expert_weight)

            if not epilogue_fused and ns_mode == 1:
                out = out * ns_row

            if ns_mode == 2 and not ns_epilogue_fused:
                compensation = self._run_ns_compensation_cuda(x_base)
                if compensation is not None:
                    out = out + compensation
                else:
                    if _deployment_only_enabled():
                        _raise_deployment_only_failure(_module_debug_label(self, "forward"), "ns compensation kernel returned no result")
                    c1, c2 = self._get_ns_matrices(x_base.device, x_base.dtype)
                    if c1 is not None and c2 is not None:
                        out = out + F.linear(F.linear(x_base, c2), c1)
            if bias is not None and not epilogue_fused:
                out = out + bias
        else:
            weight = self._dequantize_weight(x_flat.device, x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
        return out.reshape(out_shape)


def fast_load_TriS_runtime_modules(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    loader_modules = {}
    selected_prefixes = []
    for name, submodule in module.named_modules():
        if not name:
            continue
        loader = getattr(submodule, "_fast_load_from_local_state_dict", None)
        if not callable(loader):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in selected_prefixes):
            continue
        loader_modules[name] = submodule
        selected_prefixes.append(name)

    if not loader_modules:
        return state_dict

    grouped_state = {name: {} for name in loader_modules}
    for full_key, value in state_dict.items():
        prefix = full_key
        matched = None
        while "." in prefix:
            prefix = prefix.rsplit(".", 1)[0]
            if prefix in loader_modules:
                matched = prefix
                break
        if matched is None:
            continue
        grouped_state[matched][full_key[len(matched) + 1 :]] = value

    handled_keys = set()
    for name, sub_state in grouped_state.items():
        if not sub_state:
            continue
        if loader_modules[name]._fast_load_from_local_state_dict(sub_state):
            handled_keys.update(f"{name}.{local_key}" for local_key in sub_state)

    if not handled_keys:
        return state_dict
    return {key: value for key, value in state_dict.items() if key not in handled_keys}


def bind_packed4_debug_names(root_module: nn.Module):
    for name, module in root_module.named_modules():
        if isinstance(module, (QuantLinear, ExpertSharedMatrix)):
            module._TriS_debug_name = name
    return root_module


__all__ = [
    "ExpertSharedMatrix",
    "PackedQuantMatrix4bit",
    "QuantLinear",
    "bind_packed4_debug_names",
    "build_grouped_precomputed_payload",
    "build_grouped_ns_compensation_payload",
    "build_grouped_route",
    "fast_load_TriS_runtime_modules",
    "prepare_compact3_k64_groups",
    "run_grouped_ns_c1_forward",
    "run_grouped_ns_compensation_forward",
    "run_grouped_structured_expert_forward",
    "run_grouped_structured_expert_pair_forward",
]
