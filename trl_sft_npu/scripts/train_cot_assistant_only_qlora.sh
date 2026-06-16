#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL_NAME_OR_PATH:=../models/Qwen2.5-1.5B}"
: "${DATA_FILES:=data/processed/train_cot_messages.jsonl}"
: "${OUTPUT_DIR:=outputs/qwen2.5-1.5b-train-cot-assistant-only-lora}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${MAX_SEQ_LENGTH:=2048}"
: "${NUM_TRAIN_EPOCHS:=2}"

python train_sft_assistant_only.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_name local \
  --data_files "${DATA_FILES}" \
  --bf16 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --output_dir "${OUTPUT_DIR}"
