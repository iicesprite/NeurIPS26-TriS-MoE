#!/usr/bin/env python3
import argparse
import math
import os
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

from auto_gptq.modeling_tris_moe.grouped_payload_sidecar import (
    GROUPED_PRECOMPUTED_KEYS,
    _embedded_checkpoint_metadata,
    _load_safetensors_to_cpu,
    _payload_tensor_name,
    _split_grouped_payload_tensor_name,
    build_deployment_only_checkpoint_tensors,
)
from auto_gptq.nn_modules.qlinear.qlinear_cuda import (
    BLOCK_GR_MAX_K,
    BLOCK_GR_META_ONE_COUNT_SHIFT,
    BLOCK_GR_META_RAW_FLAG,
    BLOCK_GR_PACKED_ROWS,
    BLOCK_GR_TILE_OUT,
    BLOCK_GR_WORDS_PER_BLOCK,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite a grouped-runtime OLMoE checkpoint into a runtime-min deployment-only checkpoint."
    )
    parser.add_argument("--source-dir", required=True, help="Source checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination checkpoint directory.")
    parser.add_argument(
        "--checkpoint-name",
        default="grouped_runtime_payload.safetensors",
        help="Checkpoint safetensors basename inside the source/output directories.",
    )
    parser.add_argument(
        "--drop-blockgr-sparse",
        action="store_true",
        help="Drop q_gmap_sparse/q_gmap_sparse_prefix from grouped compact3 payloads that already use block-GR.",
    )
    parser.add_argument(
        "--build-blockgr",
        choices=("none", "compressed", "raw"),
        default="none",
        help="Offline-build grouped compact3 q_gmap_gr_* payloads from q_gmap_sparse tensors.",
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


def copy_non_checkpoint_files(source_dir: Path, output_dir: Path, checkpoint_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        if child.name == checkpoint_name:
            continue
        target = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def drop_blockgr_sparse_payloads(tensors: dict[str, object]) -> tuple[int, int]:
    blockgr_payloads: set[tuple[str, str]] = set()
    removed = 0

    for name, tensor in tensors.items():
        split_name = _split_grouped_payload_tensor_name(name)
        if split_name is None:
            continue
        layer_prefix, cache_key, payload_name = split_name
        if cache_key not in GROUPED_PRECOMPUTED_KEYS or payload_name != "compact3_use_block_gr":
            continue
        if bool(tensor.reshape(-1)[0].item()):
            blockgr_payloads.add((layer_prefix, cache_key))

    for layer_prefix, cache_key in blockgr_payloads:
        for payload_name in ("q_gmap_sparse", "q_gmap_sparse_prefix"):
            key = _payload_tensor_name(layer_prefix, cache_key, payload_name)
            if key in tensors:
                del tensors[key]
                removed += 1

    return len(blockgr_payloads), removed


def unpack_bool_vector_len(packed: torch.Tensor, ncols: int) -> torch.Tensor:
    packed = packed.to(dtype=torch.int64)
    shifts = torch.arange(32, dtype=torch.int64).view(1, 32)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)
    return unpacked[:ncols].contiguous().bool()


def unpack_bool_matrix_ncols(packed: torch.Tensor, ncols: int) -> torch.Tensor:
    packed = packed.to(dtype=torch.int64)
    shifts = torch.arange(32, dtype=torch.int64).view(1, 1, 32)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).permute(0, 2, 1).reshape(-1, packed.shape[1])
    unpacked = unpacked[:ncols].t().contiguous()
    return unpacked.bool()


def pack_bool_matrix(matrix: torch.Tensor) -> torch.Tensor:
    matrix_t = matrix.to(dtype=torch.bool).t().contiguous()
    packed_rows = math.ceil(matrix_t.shape[0] / 32)
    packed = torch.zeros((packed_rows, matrix_t.shape[1]), dtype=torch.int64)
    for row in range(packed_rows):
        start = row * 32
        end = min(start + 32, matrix_t.shape[0])
        chunk = matrix_t[start:end].to(torch.int64)
        shifts = torch.arange(end - start, dtype=torch.int64).unsqueeze(1)
        packed[row] = torch.sum(chunk << shifts, dim=0)
    packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return packed.to(torch.int32).contiguous()


def restore_dense_packed_gmap_from_sparse(
    packed_sparse: torch.Tensor,
    sparse_prefix: torch.Tensor,
    c_map: torch.Tensor,
    ncols: int,
    outfeatures: int,
) -> torch.Tensor:
    inactive = torch.logical_not(unpack_bool_vector_len(c_map, ncols))
    dense = torch.zeros((outfeatures, ncols), dtype=torch.bool)
    if packed_sparse.numel() > 0 and int(inactive.sum().item()) > 0:
        sparse_ncols = int(sparse_prefix[-1].item()) if sparse_prefix.numel() > 0 else int(inactive.sum().item())
        dense[:, inactive] = unpack_bool_matrix_ncols(packed_sparse, sparse_ncols)
    return pack_bool_matrix(dense)


def signed_int32_from_unsigned_words(words: list[int]) -> torch.Tensor:
    if not words:
        return torch.zeros((0,), dtype=torch.int32)
    tensor = torch.tensor(words, dtype=torch.int64)
    tensor = torch.where(tensor >= (1 << 31), tensor - (1 << 32), tensor)
    return tensor.to(dtype=torch.int32).contiguous()


def encode_block_gr_zero_runs(zero_runs: list[int], k: int) -> tuple[list[int], int]:
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

    def push_zero_bits(count: int):
        nonlocal current_word, bit_offset, total_bits
        if count <= 0:
            return
        total_bits += count
        if bit_offset:
            fill = min(count, 32 - bit_offset)
            bit_offset += fill
            count -= fill
            if bit_offset == 32:
                words.append(current_word)
                current_word = 0
                bit_offset = 0
            if count == 0:
                return
        if count >= 32:
            whole_words = count // 32
            words.extend([0] * whole_words)
            count -= whole_words * 32
        bit_offset = count

    for run in zero_runs:
        quotient = run >> k if k > 0 else run
        remainder = run & ((1 << k) - 1) if k > 0 else 0
        push_zero_bits(quotient)
        push_bit(1)
        for shift in range(k):
            push_bit((remainder >> shift) & 1)

    if bit_offset != 0:
        words.append(current_word)
    return words, total_bits


def count_block_gr_zero_runs_bits(zero_runs: list[int], k: int) -> int:
    total_bits = 0
    for run in zero_runs:
        total_bits += (run >> k if k > 0 else run) + 1 + k
    return total_bits


def resolve_block_gr_search_bounds(
    fixed_k: int | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
) -> tuple[int, int]:
    if fixed_k is not None:
        if k_min is not None or k_max is not None:
            raise ValueError("--blockgr-k cannot be combined with --blockgr-k-min/--blockgr-k-max")
        if not (0 <= fixed_k <= BLOCK_GR_MAX_K):
            raise ValueError(f"--blockgr-k must be in [0, {BLOCK_GR_MAX_K}], got {fixed_k}")
        return fixed_k, fixed_k

    if k_min is None and k_max is None:
        return 0, BLOCK_GR_MAX_K
    if k_min is None or k_max is None:
        raise ValueError("--blockgr-k-min and --blockgr-k-max must be provided together")
    if not (0 <= k_min <= k_max <= BLOCK_GR_MAX_K):
        raise ValueError(
            f"--blockgr-k-min/--blockgr-k-max must satisfy 0 <= min <= max <= {BLOCK_GR_MAX_K}, "
            f"got min={k_min}, max={k_max}"
        )
    return k_min, k_max


def pack_block_gr_gmap_tensor(
    packed: torch.Tensor,
    force_raw: bool,
    fixed_k: int | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.int32)
        return empty, empty, empty

    search_k_min, search_k_max = resolve_block_gr_search_bounds(fixed_k=fixed_k, k_min=k_min, k_max=k_max)
    packed_cpu = packed.to(device="cpu", dtype=torch.int32).contiguous()
    packed_rows = int(packed_cpu.shape[0])
    outfeatures = int(packed_cpu.shape[1])
    num_k_blocks = math.ceil(packed_rows / BLOCK_GR_PACKED_ROWS)
    num_out_blocks = math.ceil(outfeatures / BLOCK_GR_TILE_OUT)

    if force_raw:
        padded_rows = num_k_blocks * BLOCK_GR_PACKED_ROWS
        padded_outfeatures = num_out_blocks * BLOCK_GR_TILE_OUT
        if padded_rows != packed_rows or padded_outfeatures != outfeatures:
            packed_cpu = torch.nn.functional.pad(
                packed_cpu,
                (0, padded_outfeatures - outfeatures, 0, padded_rows - packed_rows),
            )
        words = (
            packed_cpu.view(num_k_blocks, BLOCK_GR_PACKED_ROWS, num_out_blocks, BLOCK_GR_TILE_OUT)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(-1)
        )
        num_blocks = num_k_blocks * num_out_blocks
        offsets = torch.arange(num_blocks + 1, dtype=torch.int32) * BLOCK_GR_WORDS_PER_BLOCK
        meta = torch.full((num_blocks,), BLOCK_GR_META_RAW_FLAG, dtype=torch.int32)
        return words.contiguous(), offsets.contiguous(), meta.contiguous()

    packed_cpu = packed_cpu.to(dtype=torch.int64)

    payload_words: list[int] = []
    offsets = [0]
    meta = []

    for k_block in range(num_k_blocks):
        packed_row_base = k_block * BLOCK_GR_PACKED_ROWS
        for out_block in range(num_out_blocks):
            out_base = out_block * BLOCK_GR_TILE_OUT
            block_words: list[int] = []
            zero_runs: list[int] = []
            one_count = 0
            prev_one_position = -1

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
                    while bool(scan):
                        low_bit = scan & (-scan)
                        bit_idx = low_bit.bit_length() - 1
                        one_position = word_base * 32 + bit_idx
                        zero_runs.append(one_position - prev_one_position - 1)
                        prev_one_position = one_position
                        one_count += 1
                        scan -= low_bit

            best_bits = BLOCK_GR_WORDS_PER_BLOCK * 32 + 1
            best_k = search_k_min
            for k in range(search_k_min, search_k_max + 1):
                encoded_bits = count_block_gr_zero_runs_bits(zero_runs, k)
                if encoded_bits < best_bits:
                    best_bits = encoded_bits
                    best_k = k

            encoded_word_count = (best_bits + 31) // 32
            use_raw = force_raw or encoded_word_count >= BLOCK_GR_WORDS_PER_BLOCK
            if use_raw:
                payload_words.extend(block_words)
                meta.append(BLOCK_GR_META_RAW_FLAG)
            else:
                best_words, _ = encode_block_gr_zero_runs(zero_runs, best_k)
                payload_words.extend(best_words)
                meta.append(best_k | (one_count << BLOCK_GR_META_ONE_COUNT_SHIFT))
            offsets.append(len(payload_words))

    return (
        signed_int32_from_unsigned_words(payload_words),
        torch.tensor(offsets, dtype=torch.int32).contiguous(),
        torch.tensor(meta, dtype=torch.int32).contiguous(),
    )


def collect_grouped_payloads(tensors: dict[str, torch.Tensor]) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    payloads: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        split_name = _split_grouped_payload_tensor_name(name)
        if split_name is None:
            continue
        layer_prefix, cache_key, payload_name = split_name
        if cache_key not in GROUPED_PRECOMPUTED_KEYS:
            continue
        payloads.setdefault((layer_prefix, cache_key), {})[payload_name] = tensor
    return payloads


def build_grouped_blockgr_payloads(
    tensors: dict[str, torch.Tensor],
    mode: str,
    fixed_k: int | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
    progress_callback=None,
) -> tuple[int, int]:
    force_raw = mode == "raw"
    updated_payloads = 0
    total_words = 0
    payloads = collect_grouped_payloads(tensors)
    compact3_payloads = [
        ((layer_prefix, cache_key), payload)
        for (layer_prefix, cache_key), payload in sorted(payloads.items())
        if bool(payload.get("is_compact3", torch.tensor([False])).reshape(-1)[0].item())
    ]

    for (layer_prefix, cache_key), payload in compact3_payloads:
        q_gmap_sparse = payload.get("q_gmap_sparse")
        q_gmap_sparse_prefix = payload.get("q_gmap_sparse_prefix")
        c_map = payload.get("c_map")
        qweight1 = payload.get("qweight1")
        g_idx = payload.get("g_idx")
        if not all(isinstance(x, torch.Tensor) for x in (q_gmap_sparse, q_gmap_sparse_prefix, c_map, qweight1, g_idx)):
            raise RuntimeError(f"Missing compact3 payload tensors for {layer_prefix}:{cache_key}")

        num_experts = int(q_gmap_sparse.shape[0])
        outfeatures = int(qweight1.shape[2])
        ncols = int(g_idx.shape[1])
        words_list = []
        offsets_list = []
        meta_list = []
        max_words = 0

        for expert_idx in range(num_experts):
            packed = restore_dense_packed_gmap_from_sparse(
                q_gmap_sparse[expert_idx],
                q_gmap_sparse_prefix[expert_idx],
                c_map[expert_idx],
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
            words_list.append(words)
            offsets_list.append(offsets)
            meta_list.append(meta)
            max_words = max(max_words, int(words.numel()))

        words_stacked = torch.stack(
            [
                torch.nn.functional.pad(words.to(torch.int32), (0, max_words - int(words.numel())))
                for words in words_list
            ],
            dim=0,
        ).contiguous()
        offsets_stacked = torch.stack([offsets.to(torch.int32) for offsets in offsets_list], dim=0).contiguous()
        meta_stacked = torch.stack([meta.to(torch.int32) for meta in meta_list], dim=0).contiguous()

        tensors[_payload_tensor_name(layer_prefix, cache_key, "compact3_use_block_gr")] = torch.tensor([True], dtype=torch.bool)
        tensors[_payload_tensor_name(layer_prefix, cache_key, "q_gmap_gr_words")] = words_stacked
        tensors[_payload_tensor_name(layer_prefix, cache_key, "q_gmap_gr_offsets")] = offsets_stacked
        tensors[_payload_tensor_name(layer_prefix, cache_key, "q_gmap_gr_meta")] = meta_stacked
        updated_payloads += 1
        total_words += int(words_stacked.numel())
        if progress_callback is not None:
            progress_callback(
                phase="grouped",
                label=f"{layer_prefix}:{cache_key}",
                completed=updated_payloads,
                total=len(compact3_payloads),
                payload_words=int(words_stacked.numel()),
                total_words=total_words,
            )

    return updated_payloads, total_words


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

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is not empty: {output_dir}")

    copy_non_checkpoint_files(source_dir, output_dir, checkpoint_name)

    tensors, metadata = _load_safetensors_to_cpu(str(source_checkpoint))
    original_tensor_count = len(tensors)
    deployment_tensors = build_deployment_only_checkpoint_tensors(tensors)
    built_blockgr_payloads = 0
    built_blockgr_words = 0
    try:
        search_k_min, search_k_max = resolve_block_gr_search_bounds(
            fixed_k=args.blockgr_k,
            k_min=args.blockgr_k_min,
            k_max=args.blockgr_k_max,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.build_blockgr != "none":
        built_blockgr_payloads, built_blockgr_words = build_grouped_blockgr_payloads(
            deployment_tensors,
            args.build_blockgr,
            fixed_k=args.blockgr_k,
            k_min=search_k_min,
            k_max=search_k_max,
        )
    blockgr_payload_count = 0
    removed_sparse_keys = 0
    if args.drop_blockgr_sparse:
        blockgr_payload_count, removed_sparse_keys = drop_blockgr_sparse_payloads(deployment_tensors)

    output_checkpoint = output_dir / checkpoint_name
    save_file(
        deployment_tensors,
        str(output_checkpoint),
        metadata=_embedded_checkpoint_metadata(metadata, deployment_only=True),
    )

    print("source_dir", source_dir)
    print("output_dir", output_dir)
    print("source_checkpoint", source_checkpoint)
    print("output_checkpoint", output_checkpoint)
    print("original_tensor_count", original_tensor_count)
    print("deployment_tensor_count", len(deployment_tensors))
    print("build_blockgr", args.build_blockgr)
    print("blockgr_k", args.blockgr_k)
    print("blockgr_k_search_min", search_k_min)
    print("blockgr_k_search_max", search_k_max)
    print("built_blockgr_payloads", built_blockgr_payloads)
    print("built_blockgr_words", built_blockgr_words)
    print("drop_blockgr_sparse", bool(args.drop_blockgr_sparse))
    print("blockgr_payload_count", blockgr_payload_count)
    print("removed_sparse_keys", removed_sparse_keys)


if __name__ == "__main__":
    main()
