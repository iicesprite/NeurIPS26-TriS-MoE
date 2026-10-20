#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from grouped_checkpoint_utils import (
    DEFAULT_GROUPED_CHECKPOINT_NAME,
    prepare_output_dir,
    update_model_file_base_name,
)

from auto_gptq.modeling_tris_moe.grouped_payload_sidecar import (
    _embedded_checkpoint_metadata,
    _load_safetensors_to_cpu,
    build_deployment_only_checkpoint_tensors,
    repair_grouped_payload_tensors,
)
from export_attn_blockgr_checkpoint import (
    _nongrouped_compact3_module_prefixes,
    build_attn_blockgr_payloads,
)
from rewrite_runtime_min_checkpoint import (
    BLOCK_GR_MAX_K,
    build_grouped_blockgr_payloads,
    collect_grouped_payloads,
    drop_blockgr_sparse_payloads,
    resolve_block_gr_search_bounds,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm should generally exist, but keep script usable without it.
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a single-file grouped block-GR checkpoint for grouped-MoE experts and non-grouped compact3 modules."
    )
    parser.add_argument("--source-dir", required=True, help="Source grouped sparse checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination checkpoint directory.")
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_GROUPED_CHECKPOINT_NAME,
        help="Checkpoint filename inside source/output directories.",
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
        help=f"Optional fixed Golomb-Rice k in [0, {BLOCK_GR_MAX_K}].",
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
        "--keep-sparse",
        action="store_true",
        help="Keep sparse qgmap tensors after exporting block-GR payloads.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _BlockGRProgressReporter:
    def __init__(self, grouped_total: int, attn_total: int):
        self._bars = {}
        self._phase_start_times = {}
        self._phase_totals = {
            "grouped": grouped_total,
            "attn": attn_total,
        }
        self._descriptions = {
            "grouped": "block-gr grouped",
            "attn": "block-gr attn",
        }

    def _ensure_bar(self, phase: str, total: int):
        if total <= 0 or phase in self._bars:
            return
        if tqdm is None:
            return
        self._bars[phase] = tqdm(total=total, desc=self._descriptions.get(phase, phase), unit="payload")

    def __call__(self, *, phase: str, label: str, completed: int, total: int, payload_words: int, total_words: int):
        total = max(total, self._phase_totals.get(phase, total))
        self._ensure_bar(phase, total)
        start_time = self._phase_start_times.setdefault(phase, time.monotonic())
        elapsed = time.monotonic() - start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = max(total - completed, 0)
        eta_seconds = remaining / rate if rate > 0 else 0.0
        progress_message = (
            f"[blockgr-export] phase={phase} progress={completed}/{total} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)} "
            f"payload_words={payload_words} total_words={total_words} label={label}"
        )

        bar = self._bars.get(phase)
        if bar is not None:
            bar.n = completed
            bar.set_postfix_str(f"eta={_format_duration(eta_seconds)} words={payload_words} label={label}")
            bar.refresh()
            bar.write(progress_message)
        else:
            print(progress_message, flush=True)

    def close(self):
        for bar in self._bars.values():
            bar.close()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_checkpoint = source_dir / args.checkpoint_name
    output_checkpoint = output_dir / args.checkpoint_name

    if not source_dir.is_dir():
        raise SystemExit(f"missing source directory: {source_dir}")
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

    prepare_output_dir(source_dir, output_dir, overwrite=args.overwrite)

    tensors, metadata = _load_safetensors_to_cpu(str(source_checkpoint))
    tensors = build_deployment_only_checkpoint_tensors(tensors)
    repaired_output_scales = repair_grouped_payload_tensors(tensors)
    grouped_total = sum(
        1
        for payload in collect_grouped_payloads(tensors).values()
        if bool(payload.get("is_compact3", torch.tensor([False])).reshape(-1)[0].item())
    )
    attn_total = len(_nongrouped_compact3_module_prefixes(tensors))
    reporter = _BlockGRProgressReporter(grouped_total=grouped_total, attn_total=attn_total)

    print(
        f"[blockgr-export] start grouped_total={grouped_total} attn_total={attn_total} "
        f"mode={args.blockgr_mode} fixed_k={args.blockgr_k} "
        f"search_k_min={search_k_min} search_k_max={search_k_max}",
        flush=True,
    )

    try:
        built_grouped_payloads, built_grouped_words = build_grouped_blockgr_payloads(
            tensors,
            args.blockgr_mode,
            fixed_k=args.blockgr_k,
            k_min=search_k_min,
            k_max=search_k_max,
            progress_callback=reporter,
        )
        converted_attn_modules, attn_words, removed_attn_sparse = build_attn_blockgr_payloads(
            tensors,
            args.blockgr_mode,
            fixed_k=args.blockgr_k,
            k_min=search_k_min,
            k_max=search_k_max,
            drop_sparse=not args.keep_sparse,
            progress_callback=reporter,
        )
    finally:
        reporter.close()

    removed_grouped_sparse = 0
    grouped_blockgr_payload_count = 0
    if not args.keep_sparse:
        grouped_blockgr_payload_count, removed_grouped_sparse = drop_blockgr_sparse_payloads(tensors)

    metadata = _embedded_checkpoint_metadata(metadata, deployment_only=True)
    metadata["olmoe_grouped_block_gr_export"] = "v1"
    metadata["olmoe_grouped_block_gr_mode"] = args.blockgr_mode
    metadata["olmoe_grouped_block_gr_keep_sparse"] = "1" if args.keep_sparse else "0"
    if args.blockgr_k is not None:
        metadata["olmoe_grouped_block_gr_k"] = str(args.blockgr_k)
    else:
        metadata["olmoe_grouped_block_gr_k_min"] = str(search_k_min)
        metadata["olmoe_grouped_block_gr_k_max"] = str(search_k_max)

    print("[blockgr-export] writing checkpoint", flush=True)
    save_file(tensors, str(output_checkpoint), metadata=metadata)
    update_model_file_base_name(output_dir, args.checkpoint_name)

    print("source_dir", source_dir)
    print("output_dir", output_dir)
    print("output_checkpoint", output_checkpoint)
    print("built_grouped_payloads", built_grouped_payloads)
    print("built_grouped_words", built_grouped_words)
    print("repaired_grouped_output_scales", repaired_output_scales)
    print("converted_nongrouped_modules", converted_attn_modules)
    print("attn_blockgr_words", attn_words)
    print("blockgr_k_search_min", search_k_min)
    print("blockgr_k_search_max", search_k_max)
    print("keep_sparse", bool(args.keep_sparse))
    print("grouped_blockgr_payload_count", grouped_blockgr_payload_count)
    print("removed_grouped_sparse_keys", removed_grouped_sparse)
    print("removed_attn_sparse_keys", removed_attn_sparse)


if __name__ == "__main__":
    main()
