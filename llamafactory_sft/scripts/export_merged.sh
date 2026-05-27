#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

llamafactory-cli export configs/qwen25_15b_timemqa_export.yaml
