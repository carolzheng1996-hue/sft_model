#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NUM_PROCESSES:=4}"

accelerate launch --num_processes "${NUM_PROCESSES}" train_sft_multigpu_qlora_assistant_only.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/train_cot_messages.jsonl \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --num_train_epochs 2 \
  --output_dir outputs/qwen2.5-1.5b-train-cot-assistant-only-multigpu-qlora
