#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOGPTQ_ROOT = REPO_ROOT / "AutoGPTQ"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AUTOGPTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOGPTQ_ROOT))

from auto_gptq.modeling_tris_moe.grouped_payload_sidecar import _load_safetensors_to_cpu
from rewrite_runtime_min_checkpoint import (
    BLOCK_GR_MAX_K,
    copy_non_checkpoint_files,
    pack_block_gr_gmap_tensor,
    resolve_block_gr_search_bounds,
    restore_dense_packed_gmap_from_sparse,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a grouped-runtime checkpoint with prepacked block-GR q_gmap payloads for non-grouped mode=4 compact3 modules."
    )
    parser.add_argument("--source-dir", required=True, help="Source checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination checkpoint directory.")
    parser.add_argument(
        "--checkpoint-name",
        default="grouped_runtime_payload.safetensors",
        help="Checkpoint safetensors basename inside the source/output directories.",
    )
    parser.add_argument(
        "--blockgr-mode",
        choices=("compressed", "raw"),
        default="compressed",
        help="Whether to emit compressed or raw block-GR payloads.",
    )
    parser.add_argument(
        "--blockgr-k",
        type=int,
        default=None,
        help="Optional fixed Golomb-Rice k for compressed block-GR. If unset, search k in [0, BLOCK_GR_MAX_K].",
    )
    parser.add_argument(
        "--blockgr-k-min",
        type=int,
        default=None,
        help="Optional minimum Golomb-Rice k to search for compressed block-GR.",
    )
    parser.add_argument(
        "--blockgr-k-max",
        type=int,
        default=None,
        help="Optional maximum Golomb-Rice k to search for compressed block-GR.",
    )
    parser.add_argument(
        "--drop-sparse",
        action="store_true",
        help="Drop non-grouped compact3 q_gmap_sparse/q_gmap_sparse_prefix after block-GR export. Disabled by default for compatibility.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


def _is_supported_nongrouped_compact3_prefix(prefix: str) -> bool:
    if ".self_attn." in prefix:
        return True
    if ".mlp.shared_expert." in prefix:
        return True
    return False


def _nongrouped_compact3_module_prefixes(tensors: dict[str, torch.Tensor]) -> list[str]:
    prefixes: list[str] = []
    for key, value in tensors.items():
        if not key.endswith(".mode"):
            continue
        prefix = key[:-5]
        if not _is_supported_nongrouped_compact3_prefix(prefix):
            continue
        if int(value.reshape(-1)[0].item()) != 4:
            continue
        required = (
            f"{prefix}.q_gmap_sparse",
            f"{prefix}.q_gmap_sparse_prefix",
            f"{prefix}.c_map",
            f"{prefix}.qweight1",
            f"{prefix}.g_idx",
        )
        if all(name in tensors for name in required):
            prefixes.append(prefix)
    return sorted(prefixes)


def build_attn_blockgr_payloads(
    tensors: dict[str, torch.Tensor],
    blockgr_mode: str,
    fixed_k: int | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
    drop_sparse: bool = False,
    progress_callback=None,
) -> tuple[int, int, int]:
    force_raw = blockgr_mode == "raw"
    converted = 0
    total_words = 0
    removed_sparse = 0

    prefixes = _nongrouped_compact3_module_prefixes(tensors)

    for prefix in prefixes:
        q_gmap_sparse = tensors[f"{prefix}.q_gmap_sparse"]
        q_gmap_sparse_prefix = tensors[f"{prefix}.q_gmap_sparse_prefix"]
        c_map = tensors[f"{prefix}.c_map"]
        qweight1 = tensors[f"{prefix}.qweight1"]
        g_idx = tensors[f"{prefix}.g_idx"]

        outfeatures = int(qweight1.shape[1])
        ncols = int(g_idx.numel())
        packed = restore_dense_packed_gmap_from_sparse(
            q_gmap_sparse,
            q_gmap_sparse_prefix,
            c_map,
            ncols,
            outfeatures,
        )
        words, offsets, meta = pack_block_gr_gmap_tensor(
            packed,
            force_raw=force_raw,
            fixed_k=fixed_k,
            k_min=k_min,
            k_max=k_max,
        )

        tensors[f"{prefix}.q_gmap_gr_words"] = words
        tensors[f"{prefix}.q_gmap_gr_offsets"] = offsets
        tensors[f"{prefix}.q_gmap_gr_meta"] = meta
        converted += 1
        total_words += int(words.numel())
        if progress_callback is not None:
            progress_callback(
                phase="attn",
                label=prefix,
                completed=converted,
                total=len(prefixes),
                payload_words=int(words.numel()),
                total_words=total_words,
            )

        if drop_sparse:
            del tensors[f"{prefix}.q_gmap_sparse"]
            del tensors[f"{prefix}.q_gmap_sparse_prefix"]
            removed_sparse += 2

    return converted, total_words, removed_sparse


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_name = args.checkpoint_name

    if not source_dir.is_dir():
        raise SystemExit(f"missing source directory: {source_dir}")

    source_checkpoint = source_dir / checkpoint_name
    if not source_checkpoint.is_file():
        raise SystemExit(f"missing source checkpoint: {source_checkpoint}")

    try:
        search_k_min, search_k_max = resolve_block_gr_search_bounds(
            fixed_k=args.blockgr_k,
            k_min=args.blockgr_k_min,
            k_max=args.blockgr_k_max,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is not empty: {output_dir}")

    copy_non_checkpoint_files(source_dir, output_dir, checkpoint_name)

    tensors, metadata = _load_safetensors_to_cpu(str(source_checkpoint))
    original_tensor_count = len(tensors)
    converted, total_words, removed_sparse = build_attn_blockgr_payloads(
        tensors,
        args.blockgr_mode,
        fixed_k=args.blockgr_k,
        k_min=search_k_min,
        k_max=search_k_max,
        drop_sparse=args.drop_sparse,
    )

    metadata = dict(metadata)
    metadata["olmoe_attn_block_gr_export"] = "v1"
    metadata["olmoe_attn_block_gr_mode"] = args.blockgr_mode
    metadata["olmoe_attn_block_gr_drop_sparse"] = "1" if args.drop_sparse else "0"
    if args.blockgr_k is not None:
        metadata["olmoe_attn_block_gr_k"] = str(args.blockgr_k)

    output_checkpoint = output_dir / checkpoint_name
    save_file(tensors, str(output_checkpoint), metadata=metadata)

    print("source_dir", source_dir)
    print("output_dir", output_dir)
    print("source_checkpoint", source_checkpoint)
    print("output_checkpoint", output_checkpoint)
    print("original_tensor_count", original_tensor_count)
    print("output_tensor_count", len(tensors))
    print("blockgr_mode", args.blockgr_mode)
    print("blockgr_k", args.blockgr_k)
    print("blockgr_k_search_min", search_k_min)
    print("blockgr_k_search_max", search_k_max)
    print("converted_nongrouped_modules", converted)
    print("total_blockgr_words", total_words)
    print("drop_sparse", bool(args.drop_sparse))
    print("removed_sparse_keys", removed_sparse)


if __name__ == "__main__":
    main()
