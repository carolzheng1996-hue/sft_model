#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "train_sft.py now only accepts TRL conversational messages data."
echo "Convert TimeSeriesExam1 to a JSON/JSONL file with a 'messages' column before training."
exit 1
