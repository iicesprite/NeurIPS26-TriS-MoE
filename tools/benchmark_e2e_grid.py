#!/usr/bin/env python3
"""Single-GPU OLMoE latency, throughput, and peak-memory grid benchmark.

The default grid covers:

* input lengths: 128, 256, 512, 1024, 2048
* output lengths: 128, 256, 512, 1024, 2048, 4096
* batch sizes: 1, 2, 4
* models: FP16 and the current grouped compact3 quantized checkpoint

Each configuration is measured in two independent phases:

1. latency: one untimed warmup followed by two timed repeats;
2. memory: one untimed warmup followed by two untimed repeats whose CUDA
   allocated/reserved peaks are recorded.

Raw records are appended to JSONL after every repeat. Detail, summary, and
FP16-vs-quant comparison CSV files are regenerated as progress is made, so a
long run can be resumed safely.

The default quantized path is checkpoints/olmoe-block-gr, as exported in README.md.
Use --fp16-path and --quant-path to select other OLMoE checkpoints. For the
streaming export, set TRIS_COMPACT3_BLOCK_GR=0 and
TRIS_GROUPED_QGMAP_CPU_STAGING=1, and pass
--quant-path checkpoints/olmoe-streaming. Runtime defaults never override
explicit environment settings; router quantization follows the checkpoint config.

Use --dry-run to inspect paths, runtime settings, and the grid without loading
model weights or initializing CUDA. Model configuration files must be available.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any, Iterable

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM as HFOlmoeForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOGPTQ_ROOT = REPO_ROOT / "AutoGPTQ"
if str(AUTOGPTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOGPTQ_ROOT))

from auto_gptq import OlmoeGPTQForCausalLM  # noqa: E402
from auto_gptq.nn_modules.qlinear.qlinear_cuda import reset_grouped_runtime_logs  # noqa: E402


DEFAULT_FP16_PATH = os.environ.get("TRIS_FP16_MODEL_PATH", "allenai/OLMoE-1B-7B-0125")
DEFAULT_QUANT_PATH = os.environ.get(
    "TRIS_QUANT_MODEL_PATH", str(REPO_ROOT / "checkpoints" / "olmoe-block-gr")
)
DEFAULT_PROMPT = "Explain what mixture-of-experts means."
DEFAULT_INPUT_LENGTHS = "128,256,512,1024,2048"
DEFAULT_OUTPUT_LENGTHS = "128,256,512,1024,2048,4096"
DEFAULT_BATCH_SIZES = "1,2,4"
DEFAULT_MODELS = "fp16,quant"
DEFAULT_PHASES = "latency,memory"

QUANT_RUNTIME_DEFAULTS = {
    "TRIS_GROUPED_EXPERTS": "1",
    "TRIS_DEPLOYMENT_ONLY": "1",
    "TRIS_COMPACT3_BLOCK_GR": "1",
    "TRIS_GROUPED_QGMAP_CPU_STAGING": "0",
}

DETAIL_FIELDS = [
    "timestamp_utc",
    "record_type",
    "status",
    "phase",
    "model",
    "batch_size",
    "input_length",
    "output_length",
    "total_context",
    "context_limit",
    "context_overflow",
    "repeat_index",
    "decode_steps",
    "prefill_ms",
    "first_token_select_ms",
    "decode_ms",
    "e2e_ms",
    "avg_decode_step_ms",
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "e2e_output_tokens_per_s",
    "e2e_total_tokens_per_s",
    "baseline_allocated_bytes",
    "baseline_reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "incremental_peak_allocated_bytes",
    "incremental_peak_reserved_bytes",
    "baseline_allocated_mib",
    "baseline_reserved_mib",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "incremental_peak_allocated_mib",
    "incremental_peak_reserved_mib",
    "error_type",
    "error_message",
]

SUMMARY_FIELDS = [
    "status",
    "model",
    "batch_size",
    "input_length",
    "output_length",
    "total_context",
    "context_limit",
    "context_overflow",
    "latency_repeats",
    "memory_repeats",
    "prefill_ms_mean",
    "prefill_ms_std",
    "prefill_ms_min",
    "prefill_ms_max",
    "first_token_select_ms_mean",
    "decode_ms_mean",
    "decode_ms_std",
    "decode_ms_min",
    "decode_ms_max",
    "e2e_ms_mean",
    "e2e_ms_std",
    "e2e_ms_min",
    "e2e_ms_max",
    "avg_decode_step_ms",
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "e2e_output_tokens_per_s",
    "e2e_total_tokens_per_s",
    "baseline_allocated_mib_mean",
    "baseline_reserved_mib_mean",
    "peak_allocated_mib_mean",
    "peak_allocated_mib_max",
    "peak_reserved_mib_mean",
    "peak_reserved_mib_max",
    "incremental_peak_allocated_mib_mean",
    "incremental_peak_allocated_mib_max",
    "incremental_peak_reserved_mib_mean",
    "incremental_peak_reserved_mib_max",
    "terminal_status",
    "error_message",
]

COMPARISON_FIELDS = [
    "batch_size",
    "input_length",
    "output_length",
    "total_context",
    "context_overflow",
    "fp16_status",
    "quant_status",
    "fp16_prefill_ms",
    "quant_prefill_ms",
    "prefill_time_ratio_quant_over_fp16",
    "fp16_decode_ms",
    "quant_decode_ms",
    "decode_speedup_fp16_over_quant",
    "fp16_e2e_ms",
    "quant_e2e_ms",
    "e2e_speedup_fp16_over_quant",
    "fp16_output_tokens_per_s",
    "quant_output_tokens_per_s",
    "throughput_speedup_quant_over_fp16",
    "fp16_peak_allocated_mib",
    "quant_peak_allocated_mib",
    "peak_allocated_ratio_quant_over_fp16",
    "fp16_peak_reserved_mib",
    "quant_peak_reserved_mib",
    "peak_reserved_ratio_quant_over_fp16",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_values(raw: str, name: str, allowed: set[str] | None = None) -> list[Any]:
    values: list[Any] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value: Any = part if allowed is not None else int(part)
        if allowed is not None:
            if value not in allowed:
                raise ValueError(f"--{name} contains unsupported value {value!r}; allowed={sorted(allowed)}")
        elif value <= 0:
            raise ValueError(f"--{name} expects positive integers, got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"--{name} cannot be empty")
    return values


def sync_cuda(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def safe_mean(values: Iterable[float]) -> float | str:
    values = list(values)
    return statistics.mean(values) if values else ""


def safe_pstdev(values: Iterable[float]) -> float | str:
    values = list(values)
    return statistics.pstdev(values) if len(values) >= 2 else (0.0 if values else "")


def safe_min(values: Iterable[float]) -> float | str:
    values = list(values)
    return min(values) if values else ""


def safe_max(values: Iterable[float]) -> float | str:
    values = list(values)
    return max(values) if values else ""


def safe_ratio(numerator: Any, denominator: Any) -> float | str:
    if numerator in (None, "") or denominator in (None, ""):
        return ""
    numerator = float(numerator)
    denominator = float(denominator)
    return numerator / denominator if denominator != 0.0 else ""


def configure_quant_runtime_defaults() -> None:
    for name, value in QUANT_RUNTIME_DEFAULTS.items():
        os.environ.setdefault(name, value)


def effective_runtime_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in QUANT_RUNTIME_DEFAULTS}


def load_model(model_kind: str, fp16_path: str, quant_path: str, device: torch.device):
    torch.cuda.set_device(device)
    if model_kind == "fp16":
        # Quantized loading registers TriSMoE classes with the AutoModel APIs.
        # Keep every FP16 phase on the upstream class, even after a quant run.
        model = HFOlmoeForCausalLM.from_pretrained(
            fp16_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
    elif model_kind == "quant":
        model = OlmoeGPTQForCausalLM.from_quantized(
            quant_path,
            use_safetensors=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device=str(device),
            use_triton=False,
            inject_fused_attention=False,
            inject_fused_mlp=False,
            disable_exllama=True,
            disable_exllamav2=True,
        )
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    return model.eval()


def cleanup_cuda(device: torch.device) -> None:
    # The caller must release its model reference before clearing the allocator.
    gc.collect()
    sync_cuda(device)
    torch.cuda.empty_cache()
    gc.collect()


def build_inputs(
    tokenizer,
    prompt: str,
    batch_size: int,
    input_length: int,
    output_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    base_ids = tokenized["input_ids"][0]
    if base_ids.numel() == 0:
        raise ValueError("prompt produced zero tokens")
    repeat_factor = math.ceil(input_length / int(base_ids.numel()))
    input_ids = base_ids.repeat(repeat_factor)[:input_length].unsqueeze(0)
    input_ids = input_ids.repeat(batch_size, 1).to(device)

    # Preallocate the longest attention mask. Decode uses inexpensive views
    # instead of allocating and concatenating a new mask at every step.
    maximum_mask_length = input_length + max(output_length - 1, 0)
    attention_mask = torch.ones(
        (batch_size, maximum_mask_length),
        dtype=torch.long,
        device=device,
    )
    return input_ids, attention_mask


@torch.inference_mode()
def run_exact_generation(
    model,
    input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    output_length: int,
    device: torch.device,
    *,
    measure_latency: bool,
) -> dict[str, float | int]:
    """Generate exactly output_length tokens, ignoring EOS.

    Timed runs synchronize only at phase boundaries. The prefill timer covers
    the prefill model forward. The decode timer covers all output_length - 1
    cached forwards plus their in-loop greedy token selection. E2E also covers
    the first greedy token selection between the two phases.
    """

    batch_size, input_length = map(int, input_ids.shape)
    decode_steps = max(output_length - 1, 0)
    prefill_ms = 0.0
    first_token_select_ms = 0.0
    decode_ms = 0.0
    e2e_ms = 0.0

    if measure_latency:
        sync_cuda(device)
        e2e_start = time.perf_counter()
        prefill_start = e2e_start

    outputs = model(
        input_ids=input_ids,
        attention_mask=full_attention_mask[:, :input_length],
        use_cache=True,
        return_dict=True,
    )

    if measure_latency:
        sync_cuda(device)
        prefill_end = time.perf_counter()
        prefill_ms = (prefill_end - prefill_start) * 1000.0

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    del outputs

    if measure_latency:
        sync_cuda(device)
        decode_start = time.perf_counter()
        first_token_select_ms = (decode_start - prefill_end) * 1000.0

    for step in range(1, output_length):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=full_attention_mask[:, : input_length + step],
            use_cache=True,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        del outputs

    sync_cuda(device)
    if measure_latency:
        end = time.perf_counter()
        decode_ms = (end - decode_start) * 1000.0 if decode_steps else 0.0
        e2e_ms = (end - e2e_start) * 1000.0

    # Explicitly release the dynamic KV cache before returning. CUDA peak
    # counters retain the peak even after these tensors are freed.
    del next_token
    del past_key_values

    result: dict[str, float | int] = {
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "decode_steps": decode_steps,
    }
    if measure_latency:
        result.update(
            {
                "prefill_ms": prefill_ms,
                "first_token_select_ms": first_token_select_ms,
                "decode_ms": decode_ms,
                "e2e_ms": e2e_ms,
                "avg_decode_step_ms": decode_ms / decode_steps if decode_steps else 0.0,
                "prefill_tokens_per_s": batch_size * input_length * 1000.0 / prefill_ms,
                "decode_tokens_per_s": (
                    batch_size * decode_steps * 1000.0 / decode_ms if decode_steps and decode_ms > 0.0 else 0.0
                ),
                "e2e_output_tokens_per_s": batch_size * output_length * 1000.0 / e2e_ms,
                "e2e_total_tokens_per_s": batch_size * (input_length + output_length) * 1000.0 / e2e_ms,
            }
        )
    return result


def release_iteration_memory(device: torch.device, *, empty_cache: bool) -> None:
    gc.collect()
    sync_cuda(device)
    if empty_cache:
        torch.cuda.empty_cache()
        sync_cuda(device)


def base_record(
    phase: str,
    model_kind: str,
    batch_size: int,
    input_length: int,
    output_length: int,
    context_limit: int | None,
) -> dict[str, Any]:
    total_context = input_length + output_length
    return {
        "timestamp_utc": utc_now(),
        "phase": phase,
        "model": model_kind,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "total_context": total_context,
        "context_limit": context_limit if context_limit is not None else "",
        "context_overflow": bool(context_limit is not None and total_context > context_limit),
    }


def append_jsonl(path: Path, records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    records.append(record)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING - ignoring invalid JSONL line {line_number}: {exc}", flush=True)
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def phase_key(record: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(record.get("phase", "")),
        str(record.get("model", "")),
        int(record.get("batch_size", 0)),
        int(record.get("input_length", 0)),
        int(record.get("output_length", 0)),
    )


def completed_repeats(records: list[dict[str, Any]], key: tuple[str, str, int, int, int]) -> set[int]:
    return {
        int(record["repeat_index"])
        for record in records
        if record.get("record_type") == "measurement"
        and record.get("status") == "ok"
        and phase_key(record) == key
        and isinstance(record.get("repeat_index"), int)
    }


def terminal_record(
    records: list[dict[str, Any]], key: tuple[str, str, int, int, int]
) -> dict[str, Any] | None:
    matches = [
        record
        for record in records
        if record.get("record_type") == "terminal_status" and phase_key(record) == key
    ]
    return matches[-1] if matches else None


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def detail_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("record_type") not in {"measurement", "terminal_status"}:
            continue
        row = dict(record)
        for name in (
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "incremental_peak_allocated_bytes",
            "incremental_peak_reserved_bytes",
        ):
            value = row.get(name)
            if isinstance(value, (int, float)):
                row[name.replace("_bytes", "_mib")] = mib(value)
        rows.append(row)
    return rows


def make_summary_rows(
    records: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    expected_repeat: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in planned:
        model = config["model"]
        batch_size = config["batch_size"]
        input_length = config["input_length"]
        output_length = config["output_length"]

        def matching(phase: str) -> list[dict[str, Any]]:
            return [
                record
                for record in records
                if record.get("record_type") == "measurement"
                and record.get("status") == "ok"
                and record.get("phase") == phase
                and record.get("model") == model
                and record.get("batch_size") == batch_size
                and record.get("input_length") == input_length
                and record.get("output_length") == output_length
            ]

        latency = matching("latency")
        memory = matching("memory")
        terminal = [
            record
            for record in records
            if record.get("record_type") == "terminal_status"
            and record.get("model") == model
            and record.get("batch_size") == batch_size
            and record.get("input_length") == input_length
            and record.get("output_length") == output_length
        ]

        prefill_values = [float(record["prefill_ms"]) for record in latency]
        first_select_values = [float(record["first_token_select_ms"]) for record in latency]
        decode_values = [float(record["decode_ms"]) for record in latency]
        e2e_values = [float(record["e2e_ms"]) for record in latency]
        baseline_allocated = [mib(record["baseline_allocated_bytes"]) for record in memory]
        baseline_reserved = [mib(record["baseline_reserved_bytes"]) for record in memory]
        peak_allocated = [mib(record["peak_allocated_bytes"]) for record in memory]
        peak_reserved = [mib(record["peak_reserved_bytes"]) for record in memory]
        incremental_allocated = [mib(record["incremental_peak_allocated_bytes"]) for record in memory]
        incremental_reserved = [mib(record["incremental_peak_reserved_bytes"]) for record in memory]

        if terminal:
            status = "terminal_error"
        elif len(latency) >= expected_repeat and len(memory) >= expected_repeat:
            status = "complete"
        elif latency or memory:
            status = "partial"
        else:
            status = "pending"

        mean_prefill = safe_mean(prefill_values)
        mean_decode = safe_mean(decode_values)
        mean_e2e = safe_mean(e2e_values)
        decode_steps = max(output_length - 1, 0)
        row = {
            "status": status,
            "model": model,
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": output_length,
            "total_context": config["total_context"],
            "context_limit": config["context_limit"],
            "context_overflow": config["context_overflow"],
            "latency_repeats": len(latency),
            "memory_repeats": len(memory),
            "prefill_ms_mean": mean_prefill,
            "prefill_ms_std": safe_pstdev(prefill_values),
            "prefill_ms_min": safe_min(prefill_values),
            "prefill_ms_max": safe_max(prefill_values),
            "first_token_select_ms_mean": safe_mean(first_select_values),
            "decode_ms_mean": mean_decode,
            "decode_ms_std": safe_pstdev(decode_values),
            "decode_ms_min": safe_min(decode_values),
            "decode_ms_max": safe_max(decode_values),
            "e2e_ms_mean": mean_e2e,
            "e2e_ms_std": safe_pstdev(e2e_values),
            "e2e_ms_min": safe_min(e2e_values),
            "e2e_ms_max": safe_max(e2e_values),
            "avg_decode_step_ms": (
                float(mean_decode) / decode_steps if mean_decode != "" and decode_steps else 0.0
            ),
            "prefill_tokens_per_s": (
                batch_size * input_length * 1000.0 / float(mean_prefill) if mean_prefill != "" else ""
            ),
            "decode_tokens_per_s": (
                batch_size * decode_steps * 1000.0 / float(mean_decode)
                if mean_decode != "" and decode_steps
                else (0.0 if mean_decode != "" else "")
            ),
            "e2e_output_tokens_per_s": (
                batch_size * output_length * 1000.0 / float(mean_e2e) if mean_e2e != "" else ""
            ),
            "e2e_total_tokens_per_s": (
                batch_size * (input_length + output_length) * 1000.0 / float(mean_e2e)
                if mean_e2e != ""
                else ""
            ),
            "baseline_allocated_mib_mean": safe_mean(baseline_allocated),
            "baseline_reserved_mib_mean": safe_mean(baseline_reserved),
            "peak_allocated_mib_mean": safe_mean(peak_allocated),
            "peak_allocated_mib_max": safe_max(peak_allocated),
            "peak_reserved_mib_mean": safe_mean(peak_reserved),
            "peak_reserved_mib_max": safe_max(peak_reserved),
            "incremental_peak_allocated_mib_mean": safe_mean(incremental_allocated),
            "incremental_peak_allocated_mib_max": safe_max(incremental_allocated),
            "incremental_peak_reserved_mib_mean": safe_mean(incremental_reserved),
            "incremental_peak_reserved_mib_max": safe_max(incremental_reserved),
            "terminal_status": ";".join(
                f"{record.get('phase')}:{record.get('status')}" for record in terminal
            ),
            "error_message": "; ".join(str(record.get("error_message", "")) for record in terminal),
        }
        rows.append(row)
    return rows


def make_comparison_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["model"],
            row["batch_size"],
            row["input_length"],
            row["output_length"],
        ): row
        for row in summary_rows
    }
    configs = sorted(
        {
            (row["batch_size"], row["input_length"], row["output_length"])
            for row in summary_rows
        },
        key=lambda item: (item[1], item[2], item[0]),
    )
    rows: list[dict[str, Any]] = []
    for batch_size, input_length, output_length in configs:
        fp16 = by_key.get(("fp16", batch_size, input_length, output_length), {})
        quant = by_key.get(("quant", batch_size, input_length, output_length), {})
        fp16_prefill = fp16.get("prefill_ms_mean", "")
        quant_prefill = quant.get("prefill_ms_mean", "")
        fp16_decode = fp16.get("decode_ms_mean", "")
        quant_decode = quant.get("decode_ms_mean", "")
        fp16_e2e = fp16.get("e2e_ms_mean", "")
        quant_e2e = quant.get("e2e_ms_mean", "")
        fp16_tps = fp16.get("e2e_output_tokens_per_s", "")
        quant_tps = quant.get("e2e_output_tokens_per_s", "")
        fp16_peak_allocated = fp16.get("peak_allocated_mib_max", "")
        quant_peak_allocated = quant.get("peak_allocated_mib_max", "")
        fp16_peak_reserved = fp16.get("peak_reserved_mib_max", "")
        quant_peak_reserved = quant.get("peak_reserved_mib_max", "")
        rows.append(
            {
                "batch_size": batch_size,
                "input_length": input_length,
                "output_length": output_length,
                "total_context": input_length + output_length,
                "context_overflow": fp16.get("context_overflow", quant.get("context_overflow", "")),
                "fp16_status": fp16.get("status", "missing"),
                "quant_status": quant.get("status", "missing"),
                "fp16_prefill_ms": fp16_prefill,
                "quant_prefill_ms": quant_prefill,
                "prefill_time_ratio_quant_over_fp16": safe_ratio(quant_prefill, fp16_prefill),
                "fp16_decode_ms": fp16_decode,
                "quant_decode_ms": quant_decode,
                "decode_speedup_fp16_over_quant": safe_ratio(fp16_decode, quant_decode),
                "fp16_e2e_ms": fp16_e2e,
                "quant_e2e_ms": quant_e2e,
                "e2e_speedup_fp16_over_quant": safe_ratio(fp16_e2e, quant_e2e),
                "fp16_output_tokens_per_s": fp16_tps,
                "quant_output_tokens_per_s": quant_tps,
                "throughput_speedup_quant_over_fp16": safe_ratio(quant_tps, fp16_tps),
                "fp16_peak_allocated_mib": fp16_peak_allocated,
                "quant_peak_allocated_mib": quant_peak_allocated,
                "peak_allocated_ratio_quant_over_fp16": safe_ratio(
                    quant_peak_allocated, fp16_peak_allocated
                ),
                "fp16_peak_reserved_mib": fp16_peak_reserved,
                "quant_peak_reserved_mib": quant_peak_reserved,
                "peak_reserved_ratio_quant_over_fp16": safe_ratio(
                    quant_peak_reserved, fp16_peak_reserved
                ),
            }
        )
    return rows


def write_tables(
    output_dir: Path,
    records: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    expected_repeat: int,
) -> None:
    details = detail_rows(records)
    summaries = make_summary_rows(records, planned, expected_repeat)
    comparisons = make_comparison_rows(summaries)
    atomic_write_csv(output_dir / "details.csv", DETAIL_FIELDS, details)
    atomic_write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    atomic_write_csv(output_dir / "fp16_vs_quant.csv", COMPARISON_FIELDS, comparisons)


def record_terminal_error(
    raw_path: Path,
    records: list[dict[str, Any]],
    base: dict[str, Any],
    status: str,
    exc: BaseException | None = None,
) -> None:
    record = {
        **base,
        "record_type": "terminal_status",
        "status": status,
        "repeat_index": -1,
    }
    if exc is not None:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc).replace("\n", " ")
        record["traceback"] = traceback.format_exc(limit=5)
    append_jsonl(raw_path, records, record)


def run_latency_phase(
    model,
    model_kind: str,
    tokenizer,
    config: dict[str, Any],
    args,
    device: torch.device,
    raw_path: Path,
    records: list[dict[str, Any]],
) -> None:
    base = base_record(
        "latency",
        model_kind,
        config["batch_size"],
        config["input_length"],
        config["output_length"],
        config["context_limit"],
    )
    key = phase_key(base)
    done = completed_repeats(records, key)
    if len({index for index in done if index < args.repeat}) >= args.repeat:
        print(f"[resume] latency already complete: {key[1:]}", flush=True)
        return
    terminal = terminal_record(records, key)
    if terminal is not None and not args.retry_errors:
        print(f"[resume] latency terminal status exists ({terminal.get('status')}): {key[1:]}", flush=True)
        return

    input_ids, attention_mask = build_inputs(
        tokenizer,
        args.prompt,
        config["batch_size"],
        config["input_length"],
        config["output_length"],
        device,
    )
    try:
        for _ in range(args.warmup):
            run_exact_generation(
                model,
                input_ids,
                attention_mask,
                config["output_length"],
                device,
                measure_latency=False,
            )
            release_iteration_memory(device, empty_cache=False)

        for repeat_index in range(args.repeat):
            if repeat_index in done:
                continue
            result = run_exact_generation(
                model,
                input_ids,
                attention_mask,
                config["output_length"],
                device,
                measure_latency=True,
            )
            record = {
                **base,
                **result,
                "timestamp_utc": utc_now(),
                "record_type": "measurement",
                "status": "ok",
                "repeat_index": repeat_index,
            }
            append_jsonl(raw_path, records, record)
            print(
                f"[latency] model={model_kind} batch={config['batch_size']} "
                f"input={config['input_length']} output={config['output_length']} "
                f"repeat={repeat_index + 1}/{args.repeat} prefill={result['prefill_ms']:.3f}ms "
                f"decode={result['decode_ms']:.3f}ms e2e={result['e2e_ms']:.3f}ms "
                f"output_tps={result['e2e_output_tokens_per_s']:.3f}",
                flush=True,
            )
            release_iteration_memory(device, empty_cache=False)
    except torch.OutOfMemoryError as exc:
        record_terminal_error(raw_path, records, base, "oom", exc)
        print(f"ERROR - CUDA OOM in latency phase: {key[1:]}: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001 - long grid must preserve the error and continue
        record_terminal_error(raw_path, records, base, "error", exc)
        print(f"ERROR - latency phase failed: {key[1:]}: {exc}", flush=True)
        if not args.continue_on_error:
            raise
    finally:
        del input_ids
        del attention_mask
        release_iteration_memory(device, empty_cache=True)


def run_memory_phase(
    model,
    model_kind: str,
    tokenizer,
    config: dict[str, Any],
    args,
    device: torch.device,
    raw_path: Path,
    records: list[dict[str, Any]],
) -> None:
    base = base_record(
        "memory",
        model_kind,
        config["batch_size"],
        config["input_length"],
        config["output_length"],
        config["context_limit"],
    )
    key = phase_key(base)
    done = completed_repeats(records, key)
    if len({index for index in done if index < args.repeat}) >= args.repeat:
        print(f"[resume] memory already complete: {key[1:]}", flush=True)
        return
    terminal = terminal_record(records, key)
    if terminal is not None and not args.retry_errors:
        print(f"[resume] memory terminal status exists ({terminal.get('status')}): {key[1:]}", flush=True)
        return

    input_ids, attention_mask = build_inputs(
        tokenizer,
        args.prompt,
        config["batch_size"],
        config["input_length"],
        config["output_length"],
        device,
    )
    try:
        for _ in range(args.warmup):
            run_exact_generation(
                model,
                input_ids,
                attention_mask,
                config["output_length"],
                device,
                measure_latency=False,
            )
            release_iteration_memory(device, empty_cache=True)

        for repeat_index in range(args.repeat):
            if repeat_index in done:
                continue
            release_iteration_memory(device, empty_cache=True)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)

            # Deliberately no wall-clock or CUDA-event timing in this phase.
            result = run_exact_generation(
                model,
                input_ids,
                attention_mask,
                config["output_length"],
                device,
                measure_latency=False,
            )
            sync_cuda(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            record = {
                **base,
                **result,
                "timestamp_utc": utc_now(),
                "record_type": "measurement",
                "status": "ok",
                "repeat_index": repeat_index,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
                "incremental_peak_reserved_bytes": max(0, peak_reserved - baseline_reserved),
            }
            append_jsonl(raw_path, records, record)
            print(
                f"[memory] model={model_kind} batch={config['batch_size']} "
                f"input={config['input_length']} output={config['output_length']} "
                f"repeat={repeat_index + 1}/{args.repeat} "
                f"peak_allocated={mib(peak_allocated):.3f}MiB "
                f"incremental_allocated={mib(max(0, peak_allocated - baseline_allocated)):.3f}MiB "
                f"peak_reserved={mib(peak_reserved):.3f}MiB",
                flush=True,
            )
            release_iteration_memory(device, empty_cache=True)
    except torch.OutOfMemoryError as exc:
        record_terminal_error(raw_path, records, base, "oom", exc)
        print(f"ERROR - CUDA OOM in memory phase: {key[1:]}: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001 - long grid must preserve the error and continue
        record_terminal_error(raw_path, records, base, "error", exc)
        print(f"ERROR - memory phase failed: {key[1:]}: {exc}", flush=True)
        if not args.continue_on_error:
            raise
    finally:
        del input_ids
        del attention_mask
        release_iteration_memory(device, empty_cache=True)


def config_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU FP16/quant OLMoE latency, throughput, and peak-memory grid benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "For streaming, set TRIS_COMPACT3_BLOCK_GR=0 and "
            "TRIS_GROUPED_QGMAP_CPU_STAGING=1, and pass "
            "--quant-path checkpoints/olmoe-streaming. Use separate output directories "
            "for different checkpoints or runtime settings."
        ),
    )
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated subset of fp16,quant")
    parser.add_argument("--phases", default=DEFAULT_PHASES, help="Comma-separated subset of latency,memory")
    parser.add_argument(
        "--fp16-path", default=DEFAULT_FP16_PATH,
        help="FP16 OLMoE directory or model ID; also configurable via TRIS_FP16_MODEL_PATH.",
    )
    parser.add_argument(
        "--quant-path", default=DEFAULT_QUANT_PATH,
        help="Exported grouped checkpoint directory; also configurable via TRIS_QUANT_MODEL_PATH.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--input-lengths", default=DEFAULT_INPUT_LENGTHS)
    parser.add_argument("--output-lengths", default=DEFAULT_OUTPUT_LENGTHS)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device", default="cuda:0", help="Exactly one CUDA device, e.g. cuda:0")
    parser.add_argument(
        "--context-policy",
        choices=("allow", "skip", "error"),
        default="allow",
        help=(
            "How to handle input_length + output_length above max_position_embeddings. "
            "Default allow runs and marks those rows as context_overflow."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Result directory. Default: results/olmoe_e2e_grid_<timestamp>.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Split the config grid for resumable chunks.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based grid shard index.")
    parser.add_argument("--max-configs", type=int, default=0, help="Optional per-model config limit for smoke tests.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read model configs and print the plan without loading weights or initializing CUDA.",
    )
    parser.add_argument("--retry-errors", action="store_true", help="Retry phases with a prior terminal error.")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record errors/OOM and continue to the next configuration (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if not args.device.startswith("cuda:") or "," in args.device:
        raise ValueError("--device must name exactly one CUDA device, e.g. cuda:0")
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    models = parse_csv_values(args.models, "models", {"fp16", "quant"})
    phases = parse_csv_values(args.phases, "phases", {"latency", "memory"})
    input_lengths = parse_csv_values(args.input_lengths, "input-lengths")
    output_lengths = parse_csv_values(args.output_lengths, "output-lengths")
    batch_sizes = parse_csv_values(args.batch_sizes, "batch-sizes")
    device = torch.device(args.device)
    if not args.dry_run:
        torch.cuda.set_device(device)
    configure_quant_runtime_defaults()

    model_paths = {"fp16": args.fp16_path, "quant": args.quant_path}
    context_limits: dict[str, int | None] = {}
    for model_kind in models:
        config = AutoConfig.from_pretrained(model_paths[model_kind], trust_remote_code=True)
        value = getattr(config, "max_position_embeddings", None)
        context_limits[model_kind] = int(value) if value is not None else None

    grid = list(itertools.product(input_lengths, output_lengths, batch_sizes))
    grid = [config for index, config in enumerate(grid) if index % args.num_shards == args.shard_index]
    if args.max_configs > 0:
        grid = grid[: args.max_configs]

    planned: list[dict[str, Any]] = []
    for model_kind in models:
        limit = context_limits[model_kind]
        for input_length, output_length, batch_size in grid:
            total_context = input_length + output_length
            planned.append(
                {
                    "model": model_kind,
                    "batch_size": batch_size,
                    "input_length": input_length,
                    "output_length": output_length,
                    "total_context": total_context,
                    "context_limit": limit if limit is not None else "",
                    "context_overflow": bool(limit is not None and total_context > limit),
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / f"olmoe_e2e_grid_{timestamp}"
    output_dir = output_dir.resolve()
    raw_path = output_dir / "raw_records.jsonl"

    metadata = {
        "created_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "device": str(device),
        "gpu_name": None if args.dry_run else torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": (
            None if args.dry_run else torch.cuda.get_device_properties(device).total_memory
        ),
        "models": models,
        "phases": phases,
        "model_paths": model_paths,
        "context_limits": context_limits,
        "context_policy": args.context_policy,
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
        "batch_sizes": batch_sizes,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "runtime_environment": effective_runtime_environment(),
        "timing_scope": {
            "prefill": "prefill model forward with CUDA synchronization at phase boundary",
            "decode": "all cached forwards and in-loop greedy argmax operations",
            "e2e": "prefill + first token argmax + decode; excludes tokenizer and model loading",
        },
        "memory_scope": {
            "peak_allocated": "torch.cuda.max_memory_allocated after one warmup",
            "peak_reserved": "torch.cuda.max_memory_reserved after one warmup",
            "baseline": "model, persistent runtime workspaces, and benchmark input tensors",
        },
    }
    # Creation time identifies a particular launch but must not participate in
    # the fingerprint; identical resume commands should produce the same ID.
    metadata["fingerprint"] = config_fingerprint(
        {key: value for key, value in metadata.items() if key != "created_at_utc"}
    )

    overflow_count = sum(1 for config in planned if config["context_overflow"])
    inference_passes = len(planned) * len(phases) * (args.warmup + args.repeat)
    print("============== OLMoE E2E Grid Benchmark ==============", flush=True)
    print(f"device={device} gpu={metadata['gpu_name']}", flush=True)
    print(f"models={models} phases={phases}", flush=True)
    print(f"model_paths={model_paths}", flush=True)
    print(f"runtime_environment={metadata['runtime_environment']}", flush=True)
    print(f"input_lengths={input_lengths}", flush=True)
    print(f"output_lengths={output_lengths}", flush=True)
    print(f"batch_sizes={batch_sizes}", flush=True)
    print(f"warmup={args.warmup} repeat={args.repeat}", flush=True)
    print(f"planned_model_configs={len(planned)} planned_inference_passes={inference_passes}", flush=True)
    print(f"context_policy={args.context_policy} context_overflow_rows={overflow_count}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"fingerprint={metadata['fingerprint']}", flush=True)

    if args.dry_run:
        print("[dry-run] no model was loaded and no result files were written", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(f"[resume] preserving existing metadata: {metadata_path}", flush=True)

    records = read_jsonl(raw_path)
    write_tables(output_dir, records, planned, args.repeat)
    tokenizer = AutoTokenizer.from_pretrained(args.fp16_path, trust_remote_code=True)

    # Phase-major execution makes the requested first timing test and second
    # memory-only test genuinely separate. Models are reloaded between phases,
    # so the memory phase does not inherit CUDA allocator state from timing.
    for phase in phases:
        print(f"========== phase={phase} ==========", flush=True)
        for model_kind in models:
            model_configs = [config for config in planned if config["model"] == model_kind]
            runnable: list[dict[str, Any]] = []
            for config in model_configs:
                if not config["context_overflow"]:
                    runnable.append(config)
                    continue
                message = (
                    f"model={model_kind} batch={config['batch_size']} input={config['input_length']} "
                    f"output={config['output_length']} total={config['total_context']} "
                    f"> context_limit={config['context_limit']}"
                )
                if args.context_policy == "allow":
                    print(f"WARNING - allowing context overflow: {message}", flush=True)
                    runnable.append(config)
                elif args.context_policy == "error":
                    raise ValueError(f"context limit exceeded: {message}")
                else:
                    base = base_record(
                        phase,
                        model_kind,
                        config["batch_size"],
                        config["input_length"],
                        config["output_length"],
                        int(config["context_limit"]),
                    )
                    key = phase_key(base)
                    if terminal_record(records, key) is None:
                        record_terminal_error(raw_path, records, base, "skipped_context_limit")
                    print(f"[skip] context limit exceeded: {message}", flush=True)

            pending = []
            for config in runnable:
                key = (
                    phase,
                    model_kind,
                    config["batch_size"],
                    config["input_length"],
                    config["output_length"],
                )
                done = completed_repeats(records, key)
                terminal = terminal_record(records, key)
                complete = len({index for index in done if index < args.repeat}) >= args.repeat
                blocked = terminal is not None and not args.retry_errors
                if not complete and not blocked:
                    pending.append(config)

            write_tables(output_dir, records, planned, args.repeat)
            if not pending:
                print(f"[resume] no pending {phase} configs for model={model_kind}", flush=True)
                continue

            print(f"----- loading model={model_kind} for phase={phase} -----", flush=True)
            model = load_model(model_kind, args.fp16_path, args.quant_path, device)
            reset_grouped_runtime_logs()
            try:
                for config_index, config in enumerate(runnable, start=1):
                    print(
                        f"[config {config_index}/{len(runnable)}] phase={phase} model={model_kind} "
                        f"batch={config['batch_size']} input={config['input_length']} "
                        f"output={config['output_length']} total_context={config['total_context']} "
                        f"overflow={config['context_overflow']}",
                        flush=True,
                    )
                    if phase == "latency":
                        run_latency_phase(
                            model, model_kind, tokenizer, config, args, device, raw_path, records
                        )
                    else:
                        run_memory_phase(
                            model, model_kind, tokenizer, config, args, device, raw_path, records
                        )
                    write_tables(output_dir, records, planned, args.repeat)
            finally:
                del model
                cleanup_cuda(device)

    write_tables(output_dir, records, planned, args.repeat)
    print("============== Benchmark Complete ==============", flush=True)
    print(f"raw_jsonl={raw_path}", flush=True)
    print(f"details_csv={output_dir / 'details.csv'}", flush=True)
    print(f"summary_csv={output_dir / 'summary.csv'}", flush=True)
    print(f"comparison_csv={output_dir / 'fp16_vs_quant.csv'}", flush=True)


if __name__ == "__main__":
    main()
