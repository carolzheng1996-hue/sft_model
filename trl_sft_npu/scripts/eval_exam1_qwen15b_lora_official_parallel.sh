#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NUM_PROCESSES:=2}"

accelerate launch --num_processes "${NUM_PROCESSES}" scripts/eval_exam1_qwen15b_lora_official_parallel.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 50
