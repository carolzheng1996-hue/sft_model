#!/usr/bin/env python3
"""Evaluate a local Qwen2.5-1.5B LoRA adapter with official TimeSeriesExam scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exam1_official_eval import add_common_args, run_official_evaluation


DEFAULT_ADAPTER_PATH = "outputs/qwen2.5-1.5b-timemqa-local-lora"
DEFAULT_OUTPUT_FILE = "reports/timeseries_exam1_qwen15b_lora_official_predictions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style TimeSeriesExam1 evaluation for Qwen2.5-1.5B + LoRA.")
    add_common_args(parser)
    parser.add_argument("--adapter_name_or_path", default=DEFAULT_ADAPTER_PATH, help="Local LoRA adapter directory.")
    parser.set_defaults(output_file=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    run_official_evaluation(parse_args(), use_adapter=True)


if __name__ == "__main__":
    main()
