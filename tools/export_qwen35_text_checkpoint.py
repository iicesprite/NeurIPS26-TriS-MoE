#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


GATE_UP_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
DOWN_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_proj$")
CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".pt", ".index.json")
COPY_FILENAMES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_size_bytes(value: str) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGTP]?B)?", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid size string: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multipliers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported size unit in {value!r}")
    return int(amount * multipliers[unit])


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def build_text_config(full_config: dict) -> dict:
    text_config = full_config.get("text_config")
    if not isinstance(text_config, dict):
        if full_config.get("model_type") == "qwen3_5_moe_text":
            text_config = dict(full_config)
        else:
            raise ValueError("Qwen3.5 full config must contain a text_config object.")
    result = dict(text_config)
    result["model_type"] = "qwen3_5_moe_text"
    result["architectures"] = ["Qwen3_5MoeForCausalLM"]
    result.pop("text_config", None)
    result.pop("vision_config", None)
    return result


def map_non_expert_key(key: str) -> str | None:
    if key.startswith("model.visual.") or key.startswith("mtp."):
        return None
    if GATE_UP_RE.match(key) or DOWN_RE.match(key):
        return None
    prefix = "model.language_model."
    if key.startswith(prefix):
        return "model." + key[len(prefix) :]
    if key == "lm_head.weight":
        return key
    return None


def expand_expert_tensor(key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    gate_up_match = GATE_UP_RE.match(key)
    if gate_up_match is not None:
        if tensor.dim() != 3:
            raise ValueError(f"{key} must be a 3D tensor, got shape={tuple(tensor.shape)}")
        num_experts, doubled_intermediate, _ = tensor.shape
        if doubled_intermediate % 2 != 0:
            raise ValueError(f"{key} second dimension must be even, got {doubled_intermediate}")
        layer_idx = gate_up_match.group(1)
        intermediate = doubled_intermediate // 2
        expanded = {}
        for expert_idx in range(int(num_experts)):
            expanded[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = (
                tensor[expert_idx, :intermediate, :].contiguous()
            )
            expanded[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = (
                tensor[expert_idx, intermediate:, :].contiguous()
            )
        return expanded

    down_match = DOWN_RE.match(key)
    if down_match is not None:
        if tensor.dim() != 3:
            raise ValueError(f"{key} must be a 3D tensor, got shape={tuple(tensor.shape)}")
        layer_idx = down_match.group(1)
        expanded = {}
        for expert_idx in range(int(tensor.shape[0])):
            expanded[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = (
                tensor[expert_idx].contiguous()
            )
        return expanded

    raise ValueError(f"{key} is not a supported Qwen3.5 expert tensor key.")


def _source_safetensors_files(source_dir: Path) -> list[Path]:
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        names = sorted(set(weight_map.values()))
        return [source_dir / name for name in names]
    files = sorted(path for path in source_dir.glob("*.safetensors") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {source_dir}")
    return files


def _copy_side_files(source_dir: Path, output_dir: Path) -> None:
    for child in source_dir.iterdir():
        if child.is_dir():
            continue
        if child.name in COPY_FILENAMES:
            shutil.copy2(child, output_dir / child.name)
            continue
        if child.name.endswith(CHECKPOINT_SUFFIXES):
            continue


class _ShardWriter:
    def __init__(self, output_dir: Path, max_shard_size: str):
        self.output_dir = output_dir
        self.max_bytes = _parse_size_bytes(max_shard_size)
        self.current: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_bytes = 0
        self.shards: list[tuple[str, list[str], int]] = []

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().cpu().contiguous()
        nbytes = _tensor_nbytes(tensor)
        if self.current and self.current_bytes + nbytes > self.max_bytes:
            self.flush()
        self.current[name] = tensor
        self.current_bytes += nbytes
        if nbytes > self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.current:
            return
        shard_idx = len(self.shards) + 1
        filename = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_path = self.output_dir / filename
        save_file(dict(self.current), save_path, metadata={"format": "pt"})
        self.shards.append((filename, list(self.current.keys()), self.current_bytes))
        self.current.clear()
        self.current_bytes = 0

    def finalize(self) -> dict:
        self.flush()
        if not self.shards:
            raise RuntimeError("No tensors were written to the output checkpoint.")

        if len(self.shards) == 1:
            old_name, tensor_names, total_bytes = self.shards[0]
            old_path = self.output_dir / old_name
            final_name = "model.safetensors"
            old_path.rename(self.output_dir / final_name)
            self.shards = [(final_name, tensor_names, total_bytes)]
            return {
                "num_shards": 1,
                "total_size": total_bytes,
                "weight_map": {name: final_name for name in tensor_names},
            }

        total = len(self.shards)
        renamed = []
        weight_map = {}
        total_size = 0
        for idx, (old_name, tensor_names, shard_bytes) in enumerate(self.shards, start=1):
            final_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
            (self.output_dir / old_name).rename(self.output_dir / final_name)
            renamed.append((final_name, tensor_names, shard_bytes))
            total_size += shard_bytes
            for tensor_name in tensor_names:
                weight_map[tensor_name] = final_name
        self.shards = renamed
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        write_json(self.output_dir / "model.safetensors.index.json", index)
        return {"num_shards": total, "total_size": total_size, "weight_map": weight_map}


def export_qwen35_text_checkpoint(
    source_dir: Path | str,
    output_dir: Path | str,
    max_shard_size: str = "5GB",
    overwrite: bool = False,
) -> dict:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_config = load_json(source_dir / "config.json")
    text_config = build_text_config(full_config)
    write_json(output_dir / "config.json", text_config)
    _copy_side_files(source_dir, output_dir)

    writer = _ShardWriter(output_dir, max_shard_size=max_shard_size)
    stats = {
        "input_files": 0,
        "kept_tensors": 0,
        "expanded_source_tensors": 0,
        "expanded_expert_tensors": 0,
        "dropped_tensors": 0,
    }

    for shard_path in _source_safetensors_files(source_dir):
        stats["input_files"] += 1
        tensors = load_file(shard_path, device="cpu")
        for key in sorted(tensors.keys()):
            tensor = tensors[key]
            if GATE_UP_RE.match(key) or DOWN_RE.match(key):
                expanded = expand_expert_tensor(key, tensor)
                stats["expanded_source_tensors"] += 1
                stats["expanded_expert_tensors"] += len(expanded)
                for out_key, out_tensor in expanded.items():
                    writer.add_tensor(out_key, out_tensor)
                continue

            mapped_key = map_non_expert_key(key)
            if mapped_key is None:
                stats["dropped_tensors"] += 1
                continue
            stats["kept_tensors"] += 1
            writer.add_tensor(mapped_key, tensor)
        del tensors

    shard_info = writer.finalize()
    stats.update(
        {
            "num_shards": shard_info["num_shards"],
            "total_size": shard_info["total_size"],
            "model_type": text_config["model_type"],
        }
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a text-only expanded Qwen3.5-MoE checkpoint.")
    parser.add_argument("--source-dir", required=True, help="Original Qwen3.5 multimodal checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination text-only checkpoint directory.")
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum safetensors shard size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output-dir if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_qwen35_text_checkpoint(
        Path(args.source_dir),
        Path(args.output_dir),
        max_shard_size=args.max_shard_size,
        overwrite=args.overwrite,
    )
    for key in sorted(stats.keys()):
        print(key, stats[key])


if __name__ == "__main__":
    main()
