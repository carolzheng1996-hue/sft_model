#!/usr/bin/env python3
"""Inspect Hugging Face or local QA datasets before SFT."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_CACHE = PROJECT_ROOT / "datasets" / ".hf_cache"
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_DATASETS_CACHE))

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="Time-MQA/TSQA")
    parser.add_argument(
        "--data_files",
        nargs="*",
        default=[
            "Forecasting+Imputation/*.csv",
            "Anomaly_Detection/*.csv",
            "Classification/*.csv",
            "Open_Ended_QA/*.csv",
        ],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_examples", type=int, default=2)
    return parser.parse_args()


def maybe_login() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def configure_local_dataset_cache() -> None:
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_DATASETS_CACHE))
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


def expand_local_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    return files


def local_dataset_loader(files: list[str]) -> str:
    extension = os.path.splitext(files[0])[1].lower().lstrip(".")
    if extension == "jsonl":
        return "json"
    if extension in {"csv", "json", "parquet"}:
        return extension
    raise ValueError(f"Unsupported local file extension: {extension!r}. Use csv, json, jsonl, or parquet.")


def load_any(args: argparse.Namespace) -> DatasetDict:
    local_files = expand_local_files(args.data_files) if args.data_files else []
    all_local = bool(local_files) and all(os.path.exists(path) for path in local_files)
    if all_local:
        data_files = {args.split: local_files}
        ext = local_dataset_loader(local_files)
        ds = load_dataset(ext, data_files=data_files)
    else:
        data_files = {args.split: args.data_files} if args.data_files else None
        ds = load_dataset(args.dataset_name, data_files=data_files)
    if isinstance(ds, Dataset):
        return DatasetDict({args.split: ds})
    return ds


def main() -> None:
    args = parse_args()
    configure_local_dataset_cache()
    maybe_login()
    ds = load_any(args)
    print("Splits:")
    for name, split in ds.items():
        print(f"  {name}: {len(split):,} rows")
        print(f"  columns: {split.column_names}")
        print("  features:")
        pprint(split.features)
    target = ds[args.split]
    for idx in range(min(args.num_examples, len(target))):
        print(f"\nExample {idx}:")
        print(json.dumps(target[idx], ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
