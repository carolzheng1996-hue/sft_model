#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

llamafactory-cli chat configs/qwen25_15b_timemqa_merged_chat.yaml
