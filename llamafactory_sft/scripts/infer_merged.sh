#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/infer.py \
  --model_name_or_path saves/qwen2.5-1.5b/timemqa/merged \
  --no-use_adapter \
  "$@"
