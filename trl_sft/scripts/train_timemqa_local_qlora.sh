#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python train_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json \
  --use_4bit \
  --bf16 \
  --output_dir outputs/qwen2.5-1.5b-timemqa-local-lora
