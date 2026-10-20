import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOGPTQ_ROOT = REPO_ROOT / "AutoGPTQ"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AUTOGPTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOGPTQ_ROOT))

from auto_gptq.modeling._utils import check_and_get_model_type, get_checkpoints  # noqa: E402
from auto_gptq.quantization.config import BaseQuantizeConfig  # noqa: E402


CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".pt")
DEFAULT_GROUPED_CHECKPOINT_NAME = "grouped_runtime_payload.safetensors"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def detect_checkpoint_model_type(model_dir: str) -> str:
    model_type = check_and_get_model_type(model_dir, trust_remote_code=True)
    if model_type in {"mixtral", "olmoe", "qwen2_moe", "qwen3_5_moe_text"}:
        return model_type
    raise ValueError(
        f"Unsupported checkpoint model_type for grouped export: {model_type}. "
        "Expected one of: mixtral, olmoe, qwen2_moe, qwen3_5_moe_text."
    )


def find_checkpoint_mode(model_dir: str) -> bool:
    return any(name.endswith(".safetensors") for name in os.listdir(model_dir))


def default_model_basenames(quantize_config: BaseQuantizeConfig) -> list[str]:
    if quantize_config.model_file_base_name:
        return [quantize_config.model_file_base_name]
    return [
        f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
        "model",
    ]


def resolve_main_checkpoint_path(
    model_dir: str,
    use_safetensors: bool,
    excluded_paths: set[str] | None = None,
    allow_sharded: bool = False,
) -> tuple[str, str]:
    excluded = {os.path.abspath(path) for path in (excluded_paths or set())}
    quantize_config = BaseQuantizeConfig.from_pretrained(model_dir)
    extensions = [".safetensors"] if use_safetensors else [".bin", ".pt"]
    possible_model_basenames = default_model_basenames(quantize_config)
    is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
        model_name_or_path=model_dir,
        extensions=extensions,
        possible_model_basenames=possible_model_basenames,
    )
    if is_sharded:
        if not allow_sharded:
            raise ValueError("Sharded checkpoints are not supported by grouped checkpoint export scripts.")
        resolved_archive_file = os.path.abspath(resolved_archive_file)
        index_suffix = extensions[0] + ".index.json"
        if os.path.basename(resolved_archive_file).endswith(index_suffix):
            true_model_basename = os.path.basename(resolved_archive_file)[: -len(index_suffix)]
        return resolved_archive_file, true_model_basename

    resolved_archive_file = os.path.abspath(resolved_archive_file)
    if resolved_archive_file not in excluded:
        return resolved_archive_file, true_model_basename

    suffixes = tuple(extensions)
    candidates = []
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".index.json") or not name.endswith(suffixes):
            continue
        candidate_path = os.path.abspath(os.path.join(model_dir, name))
        if candidate_path in excluded:
            continue
        candidates.append(candidate_path)

    if len(candidates) == 1:
        candidate = candidates[0]
        return candidate, os.path.splitext(os.path.basename(candidate))[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a non-sidecar checkpoint in {model_dir} after excluding: {sorted(excluded)}"
        )
    raise ValueError(
        f"Found multiple checkpoint candidates after excluding sidecar paths: {candidates}. "
        "Please remove ambiguous files or export to a clean directory."
    )


def _is_checkpoint_artifact(path: Path) -> bool:
    if path.is_dir():
        return False
    return path.name.endswith(CHECKPOINT_SUFFIXES) or path.name.endswith(".index.json")


def copy_non_checkpoint_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        if _is_checkpoint_artifact(child):
            continue
        target = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def update_model_file_base_name(model_dir: Path, checkpoint_name: str) -> None:
    model_file_base_name = Path(checkpoint_name).stem

    quantize_config_path = model_dir / "quantize_config.json"
    if quantize_config_path.exists():
        quantize_config = json.loads(quantize_config_path.read_text(encoding="utf-8"))
        quantize_config["model_file_base_name"] = model_file_base_name
        _write_json(quantize_config_path, quantize_config)

    config_path = model_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization_config = config.get("quantization_config")
        if isinstance(quantization_config, dict):
            quantization_config["model_file_base_name"] = model_file_base_name
        _write_json(config_path, config)


def prepare_output_dir(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is not empty: {output_dir}")
    copy_non_checkpoint_files(source_dir, output_dir)
