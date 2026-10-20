#!/usr/bin/env python3
import argparse
from pathlib import Path

from safetensors.torch import save_file

from grouped_checkpoint_utils import (
    DEFAULT_GROUPED_CHECKPOINT_NAME,
    prepare_output_dir,
    update_model_file_base_name,
)

from auto_gptq.modeling_tris_moe.grouped_payload_sidecar import (
    _embedded_checkpoint_metadata,
    _load_safetensors_to_cpu,
    _split_grouped_payload_tensor_name,
    build_deployment_only_checkpoint_tensors,
    repair_grouped_payload_tensors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a single-file grouped sparse checkpoint for grouped-MoE qgmap CPU staging / streaming load."
    )
    parser.add_argument("--source-dir", required=True, help="Source grouped sparse checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination checkpoint directory.")
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_GROUPED_CHECKPOINT_NAME,
        help="Checkpoint filename inside source/output directories.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


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

    prepare_output_dir(source_dir, output_dir, overwrite=args.overwrite)

    tensors, metadata = _load_safetensors_to_cpu(str(source_checkpoint))
    tensors = build_deployment_only_checkpoint_tensors(tensors)
    repaired_output_scales = repair_grouped_payload_tensors(tensors)

    has_grouped_sparse = False
    has_attn_sparse = False
    for name in tensors:
        split_name = _split_grouped_payload_tensor_name(name)
        if split_name is not None:
            _, cache_key, payload_name = split_name
            if cache_key in {"gate", "up", "down"} and payload_name == "q_gmap_sparse":
                has_grouped_sparse = True
        elif ".self_attn." in name and name.endswith(".q_gmap_sparse"):
            has_attn_sparse = True
        if has_grouped_sparse or has_attn_sparse:
            break

    if not has_grouped_sparse and not has_attn_sparse:
        raise SystemExit(
            "source checkpoint does not contain sparse qgmap payloads; "
            "export_grouped_streaming_sparse_checkpoint.py expects a sparse source checkpoint."
        )

    metadata = _embedded_checkpoint_metadata(metadata, deployment_only=True)
    metadata["olmoe_grouped_qgmap_cpu_staging_ready"] = "v1"
    metadata["olmoe_grouped_qgmap_cpu_staging_single_file"] = "1"

    save_file(tensors, str(output_checkpoint), metadata=metadata)
    update_model_file_base_name(output_dir, args.checkpoint_name)

    print("source_dir", source_dir)
    print("output_dir", output_dir)
    print("output_checkpoint", output_checkpoint)
    print("streaming_sparse", True)
    print("single_file", True)
    print("repaired_grouped_output_scales", repaired_output_scales)


if __name__ == "__main__":
    main()
