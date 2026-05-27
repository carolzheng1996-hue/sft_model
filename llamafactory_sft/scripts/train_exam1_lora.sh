#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

llamafactory-cli train configs/qwen25_15b_timeseries_exam1_lora_sft.yaml
