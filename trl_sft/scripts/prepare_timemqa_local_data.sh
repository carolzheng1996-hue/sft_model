#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_timemqa_local_data.py "$@"
