#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL_NAME_OR_PATH:=../models/Qwen2.5-1.5B}"
: "${DATA_FILE:=../datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json}"
: "${OUTPUT_FILE:=reports/timeseries_exam1_qwen15b_base_official_predictions.jsonl}"
: "${MAX_SAMPLES:=50}"
: "${MAX_NEW_TOKENS:=1024}"

python scripts/eval_exam1_qwen15b_base_official.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_file "${DATA_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  "$@"

