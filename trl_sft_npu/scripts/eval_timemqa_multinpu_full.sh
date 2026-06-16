#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NUM_PROCESSES:=8}"
: "${MODEL_NAME_OR_PATH:=../models/Qwen2.5-1.5B}"
: "${ADAPTER_NAME_OR_PATH:=outputs/qwen2.5-1.5b-timemqa-local-multigpu-lora-npu}"
: "${TIMEMQA_CSV:=../timemqa/open_ended_QA.csv}"
: "${DATA_FILE:=data/processed/timemqa_local_eval.json}"
: "${OUTPUT_FILE:=reports/timemqa_local_full_predictions_parallel_npu.jsonl}"
: "${MAX_NEW_TOKENS:=256}"

python scripts/prepare_timemqa_local_data.py \
  --input "${TIMEMQA_CSV}" \
  --output "${DATA_FILE}"

accelerate launch --num_processes "${NUM_PROCESSES}" scripts/eval_timemqa_parallel.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --adapter_name_or_path "${ADAPTER_NAME_OR_PATH}" \
  --data_file "${DATA_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --max_samples 0 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  "$@"
