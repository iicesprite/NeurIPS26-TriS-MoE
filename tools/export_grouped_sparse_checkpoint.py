#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch

from grouped_checkpoint_utils import (
    DEFAULT_GROUPED_CHECKPOINT_NAME,
    detect_checkpoint_model_type,
    prepare_output_dir,
    resolve_main_checkpoint_path,
    update_model_file_base_name,
    find_checkpoint_mode,
)

from auto_gptq.modeling.mixtral import MixtralGPTQForCausalLM
from auto_gptq.modeling.olmoe import OlmoeGPTQForCausalLM
from auto_gptq.modeling.qwen2_moe import Qwen2MoeGPTQForCausalLM
from auto_gptq.modeling.qwen3_5_moe import Qwen3_5MoeGPTQForCausalLM
from auto_gptq.modeling_tris_moe.grouped_payload_sidecar import save_grouped_payload_sidecar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a single-file grouped sparse deployment checkpoint from a quantized grouped-MoE checkpoint."
    )
    parser.add_argument("--source-dir", required=True, help="Source quantized checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination checkpoint directory.")
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_GROUPED_CHECKPOINT_NAME,
        help="Output checkpoint filename inside output-dir.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "mixtral", "olmoe", "qwen2_moe", "qwen3_5_moe_text"),
        default="auto",
        help="Checkpoint model type. Defaults to auto-detect from config.json.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used to build grouped payloads.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


def get_model_wrapper_cls(model_type: str):
    if model_type == "mixtral":
        return MixtralGPTQForCausalLM
    if model_type == "olmoe":
        return OlmoeGPTQForCausalLM
    if model_type == "qwen2_moe":
        return Qwen2MoeGPTQForCausalLM
    if model_type == "qwen3_5_moe_text":
        return Qwen3_5MoeGPTQForCausalLM
    raise ValueError(f"unsupported model_type: {model_type}")


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_checkpoint = output_dir / args.checkpoint_name

    if not source_dir.is_dir():
        raise SystemExit(f"missing source directory: {source_dir}")

    prepare_output_dir(source_dir, output_dir, overwrite=args.overwrite)

    os.environ.pop("OLMOE_DEPLOYMENT_ONLY", None)
    os.environ.pop("TRIS_DEPLOYMENT_ONLY", None)

    use_safetensors = find_checkpoint_mode(str(source_dir))
    _, checkpoint_model_basename = resolve_main_checkpoint_path(
        str(source_dir),
        use_safetensors=use_safetensors,
        allow_sharded=True,
    )

    model_type = args.model_type
    if model_type == "auto":
        model_type = detect_checkpoint_model_type(str(source_dir))

    model_wrapper_cls = get_model_wrapper_cls(model_type)

    model = model_wrapper_cls.from_quantized(
        str(source_dir),
        model_basename=checkpoint_model_basename,
        use_safetensors=use_safetensors,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device=args.device,
        use_triton=False,
        inject_fused_attention=False,
        inject_fused_mlp=False,
        disable_exllama=True,
        disable_exllamav2=True,
    )
    model = model.eval()

    save_grouped_payload_sidecar(
        model.model,
        str(output_checkpoint),
        include_model_state=True,
        deployment_only=True,
    )

    del model
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    update_model_file_base_name(output_dir, args.checkpoint_name)

    print("source_dir", source_dir)
    print("output_dir", output_dir)
    print("output_checkpoint", output_checkpoint)
    print("model_type", model_type)
    print("grouped_payload", "sparse")
    print("deployment_only", True)


if __name__ == "__main__":
    main()
