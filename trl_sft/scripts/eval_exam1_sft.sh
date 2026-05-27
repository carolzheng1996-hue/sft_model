#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/eval_exam1_sft.py "$@"
