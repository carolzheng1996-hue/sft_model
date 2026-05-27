#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

llamafactory-cli train configs/qwen25_15b_timemqa_qlora_sft.yaml \
  max_samples=128 \
  save_steps=50 \
  eval_steps=50 \
  output_dir=saves/qwen2.5-1.5b/timemqa/smoke-qlora-sft
