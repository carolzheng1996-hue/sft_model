#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-../datasets/.hf_cache}" \
HF_HOME="${HF_HOME:-../datasets/.hf_home}" \
python prepare_timemqa_data.py \
  --dataset_name local \
  --data_files ../datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json \
  --output data/timeseries_exam1_alpaca.json
