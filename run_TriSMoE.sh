#!/bin/bash
# Example:
#   TRIS_MODEL_PATH=allenai/OLMoE-1B-7B-0125 bash run_TriSMoE.sh
MODEL_PATH="${TRIS_MODEL_PATH:-allenai/OLMoE-1B-7B-0125}"
OUTPUT_DIR="${TRIS_OUTPUT_DIR:-TriSMoE_quantized}"
DEVICE="${TRIS_DEVICE:-cuda:0}"

python tris_main.py "${MODEL_PATH}" wikitext2 --blocksize 128 --device "${DEVICE}" --save "${OUTPUT_DIR}" --idx_rule fit_nohead --ns_rule vector --quant_router false --real_quant false
