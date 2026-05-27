#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python prepare_timemqa_data.py \
  --dataset_name Time-MQA/TSQA \
  --output data/timemqa_tsqa_alpaca.json
