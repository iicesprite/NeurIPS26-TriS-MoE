#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT_VERSION = "olmoe-layer-merged-v1"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
SAFE_DTYPE_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline-merge an OLMoE GPTQ safetensors checkpoint into per-group bank tensors.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(os.environ.get("TRIS_MERGE_SRC", "checkpoints/grouped_sparse")),
        help="Source checkpoint directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination directory for the merged checkpoint. Defaults to <src>_layer_merged.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Checkpoint safetensors filename inside --src. Defaults to the single .safetensors file in the directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the merge plan and print a summary without writing files.",
    )
    return parser.parse_args()


def detect_checkpoint_file(src_dir: Path, checkpoint_file: str | None) -> Path:
    if checkpoint_file is not None:
        checkpoint_path = src_dir / checkpoint_file
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path

    candidates = sorted(src_dir.glob('*.safetensors'))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors checkpoint found under {src_dir}")
    if len(candidates) > 1:
        names = ', '.join(path.name for path in candidates)
        raise RuntimeError(f"Multiple .safetensors files found under {src_dir}: {names}. Please pass --checkpoint-file.")
    return candidates[0]


def infer_group(key: str) -> str:
    if key.startswith('model.embed_tokens.'):
        return 'model.embed_tokens'
    match = LAYER_RE.match(key)
    if match is not None:
        return f"model.layers.{int(match.group(1))}"
    if key.startswith('model.norm.'):
        return 'model.norm'
    if key.startswith('lm_head.'):
        return 'lm_head'
    return '__other__'


def group_sort_key(group: str):
    if group == 'model.embed_tokens':
        return (0, -1)
    match = re.match(r'^model\.layers\.(\d+)$', group)
    if match is not None:
        return (1, int(match.group(1)))
    if group == 'model.norm':
        return (2, 0)
    if group == 'lm_head':
        return (3, 0)
    return (4, group)


def sanitize_group_name(group: str) -> str:
    if group == '__other__':
        return 'other'
    return group.replace('.', '_')


def bank_tensor_key(dtype_code: str) -> str:
    return f"bank.{dtype_code.lower()}"


def build_merge_plan(checkpoint_path: Path):
    plan = {
        'format': FORMAT_VERSION,
        'source_checkpoint': checkpoint_path.name,
        'groups': {},
    }

    with safe_open(checkpoint_path, framework='pt', device='cpu') as reader:
        source_metadata = reader.metadata()
        if source_metadata is not None:
            plan['source_metadata'] = source_metadata

        for key in sorted(reader.keys(), key=lambda name: (group_sort_key(infer_group(name)), name)):
            tensor_slice = reader.get_slice(key)
            dtype_code = str(tensor_slice.get_dtype())
            if dtype_code not in SAFE_DTYPE_TO_TORCH:
                raise RuntimeError(f"Unsupported safetensors dtype {dtype_code} for key {key}")
            shape = [int(dim) for dim in tensor_slice.get_shape()]
            numel = int(math.prod(shape)) if shape else 1
            group_name = infer_group(key)
            group_plan = plan['groups'].setdefault(
                group_name,
                {
                    'banks': {},
                    'entries': [],
                },
            )

            entry = {
                'name': key,
                'dtype': dtype_code,
                'shape': shape,
                'numel': numel,
            }
            if numel == 0:
                entry['bank'] = None
                entry['offset'] = 0
            else:
                bank_key = bank_tensor_key(dtype_code)
                bank_plan = group_plan['banks'].setdefault(
                    bank_key,
                    {
                        'dtype': dtype_code,
                        'numel': 0,
                        'tensor_count': 0,
                    },
                )
                entry['bank'] = bank_key
                entry['offset'] = int(bank_plan['numel'])
                bank_plan['numel'] += numel
                bank_plan['tensor_count'] += 1
            group_plan['entries'].append(entry)

    return plan


def ensure_destination(dst_dir: Path, overwrite: bool):
    if dst_dir.exists() and any(dst_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Destination directory {dst_dir} already exists and is not empty. Pass --overwrite to reuse it."
        )
    dst_dir.mkdir(parents=True, exist_ok=True)


def copy_sidecars(src_dir: Path, dst_dir: Path, checkpoint_name: str):
    copied = []
    for path in sorted(src_dir.iterdir()):
        if path.name == checkpoint_name:
            continue
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)
            copied.append(path.name)
    return copied


def write_group_banks(checkpoint_path: Path, dst_dir: Path, checkpoint_stem: str, plan):
    outputs = {}
    with safe_open(checkpoint_path, framework='pt', device='cpu') as reader:
        for group_name in sorted(plan['groups'].keys(), key=group_sort_key):
            group_plan = plan['groups'][group_name]
            if not group_plan['banks']:
                group_plan['file'] = None
                continue

            bank_tensors = {
                bank_key: torch.empty((bank_plan['numel'],), dtype=SAFE_DTYPE_TO_TORCH[bank_plan['dtype']])
                for bank_key, bank_plan in group_plan['banks'].items()
            }

            for entry in group_plan['entries']:
                bank_key = entry['bank']
                if bank_key is None:
                    continue
                tensor = reader.get_tensor(entry['name']).contiguous().reshape(-1)
                start = int(entry['offset'])
                end = start + int(entry['numel'])
                bank_tensors[bank_key][start:end].copy_(tensor)

            file_name = f"{checkpoint_stem}--{sanitize_group_name(group_name)}.safetensors"
            save_file(
                bank_tensors,
                str(dst_dir / file_name),
                metadata={
                    'format': FORMAT_VERSION,
                    'group': group_name,
                    'source_checkpoint': checkpoint_path.name,
                },
            )
            group_plan['file'] = file_name
            outputs[group_name] = file_name
            print(f"[write] {group_name}: {file_name} ({len(bank_tensors)} banks)")
    return outputs


def summarize_plan(plan):
    group_names = sorted(plan['groups'].keys(), key=group_sort_key)
    total_tensors = 0
    total_non_empty = 0
    total_elems = 0
    lines = []
    for group_name in group_names:
        group_plan = plan['groups'][group_name]
        tensor_count = len(group_plan['entries'])
        non_empty_count = sum(1 for entry in group_plan['entries'] if entry['numel'] > 0)
        group_elems = sum(bank['numel'] for bank in group_plan['banks'].values())
        bank_summary = ', '.join(
            f"{bank_key}:{bank['dtype']}:{bank['numel']}"
            for bank_key, bank in sorted(group_plan['banks'].items())
        ) or 'no-data'
        lines.append(
            f"{group_name}: tensors={tensor_count}, non_empty={non_empty_count}, banks={bank_summary}"
        )
        total_tensors += tensor_count
        total_non_empty += non_empty_count
        total_elems += group_elems
    return {
        'group_count': len(group_names),
        'tensor_count': total_tensors,
        'non_empty_tensor_count': total_non_empty,
        'total_numel_in_banks': total_elems,
        'lines': lines,
    }


def write_manifest(dst_dir: Path, checkpoint_stem: str, src_dir: Path, copied_sidecars, plan):
    manifest = {
        'format': FORMAT_VERSION,
        'source_dir': str(src_dir),
        'source_checkpoint': plan['source_checkpoint'],
        'copied_sidecars': copied_sidecars,
        'groups': plan['groups'],
    }
    if 'source_metadata' in plan:
        manifest['source_metadata'] = plan['source_metadata']
    manifest_path = dst_dir / f"{checkpoint_stem}.layer_merged.index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] manifest: {manifest_path.name}")
    return manifest_path


def main():
    args = parse_args()
    src_dir = args.src.resolve()
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {src_dir}")
    dst_dir = args.dst.resolve() if args.dst is not None else src_dir.with_name(src_dir.name + '_layer_merged')
    checkpoint_path = detect_checkpoint_file(src_dir, args.checkpoint_file)
    checkpoint_stem = checkpoint_path.stem + '.layer_merged'

    print(f"[plan] source checkpoint: {checkpoint_path}")
    plan = build_merge_plan(checkpoint_path)
    summary = summarize_plan(plan)
    print(f"[plan] groups={summary['group_count']} tensors={summary['tensor_count']} non_empty={summary['non_empty_tensor_count']} total_numel={summary['total_numel_in_banks']}")
    for line in summary['lines']:
        print(f"[plan] {line}")

    if args.dry_run:
        print('[done] dry-run only, no files written.')
        return

    ensure_destination(dst_dir, overwrite=args.overwrite)
    copied_sidecars = copy_sidecars(src_dir, dst_dir, checkpoint_path.name)
    print(f"[write] copied sidecars: {len(copied_sidecars)}")
    write_group_banks(checkpoint_path, dst_dir, checkpoint_stem, plan)
    write_manifest(dst_dir, checkpoint_stem, src_dir, copied_sidecars, plan)
    print(f"[done] merged checkpoint written to {dst_dir}")


if __name__ == '__main__':
    main()
