#!/usr/bin/env python3
"""Convert Time-MQA/TSQA to LLaMA-Factory Alpaca format.

The Hugging Face dataset is gated. Accept access on the Hub and provide HF_TOKEN
in the shell environment before running this script.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login


QUESTION_KEYS = (
    "question",
    "query",
    "prompt",
    "instruction",
    "problem",
    "input",
    "user",
)
ANSWER_KEYS = (
    "answer",
    "answers",
    "output",
    "response",
    "completion",
    "target",
    "label",
    "assistant",
)
CONTEXT_KEYS = (
    "context",
    "background",
    "description",
    "time_series",
    "timeseries",
    "ts",
    "ts1",
    "ts2",
    "series",
    "data",
    "values",
    "history",
    "options",
    "choices",
    "metadata",
    "difficulty",
    "format_hint",
    "relevant_concepts",
    "question_hint",
    "category",
    "subcategory",
    "domain",
    "task",
    "task_type",
    "question_type",
)
SYSTEM_PROMPT = (
    "You are a time-series question answering expert. "
    "Answer the user's question from the provided time-series information. "
    "Be concise, numerical when appropriate, and do not invent missing facts."
)


@dataclass(frozen=True)
class FieldMapping:
    question: str
    answer: str
    contexts: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Time-MQA/TSQA for LLaMA-Factory SFT")
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
        help="HF repo globs or local CSV/JSON/JSONL files.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/timemqa_tsqa_alpaca.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=2)
    return parser.parse_args()


def maybe_login() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def expand_local_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    return files


def local_dataset_loader(files: list[str]) -> str:
    extension = Path(files[0]).suffix.lower().lstrip(".")
    if extension == "jsonl":
        return "json"
    if extension in {"csv", "json", "parquet"}:
        return extension
    raise ValueError(f"Unsupported local file extension: {extension!r}")


def load_raw_dataset(args: argparse.Namespace) -> Dataset:
    local_files = expand_local_files(args.data_files)
    all_local = bool(local_files) and all(Path(path).exists() for path in local_files)
    data_files = {args.split: local_files if all_local else args.data_files}

    if all_local:
        loaded = load_dataset(local_dataset_loader(local_files), data_files=data_files)
    else:
        loaded = load_dataset(args.dataset_name, data_files=data_files)

    if isinstance(loaded, Dataset):
        dataset = loaded
    elif isinstance(loaded, DatasetDict):
        if args.split not in loaded:
            available = ", ".join(loaded.keys())
            raise KeyError(f"Split {args.split!r} not found. Available splits: {available}")
        dataset = loaded[args.split]
    else:
        raise TypeError(f"Unexpected dataset type: {type(loaded)}")

    if args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.max_samples, len(dataset))))
    return dataset


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def choose_first(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for key in candidates:
        if key in lower_to_original:
            return lower_to_original[key]
    for col in columns:
        low = col.lower()
        if any(key in low for key in candidates):
            return col
    return None


def infer_mapping(dataset: Dataset) -> FieldMapping:
    columns = list(dataset.column_names)
    question = choose_first(columns, QUESTION_KEYS)
    answer = choose_first(columns, ANSWER_KEYS)
    if question is None or answer is None:
        raise ValueError(
            "Could not infer question/answer columns. "
            f"Columns found: {columns}. Edit QUESTION_KEYS/ANSWER_KEYS in prepare_timemqa_data.py."
        )

    contexts = []
    for col in columns:
        if col in {question, answer}:
            continue
        low = col.lower()
        if any(key in low for key in CONTEXT_KEYS):
            contexts.append(col)
    return FieldMapping(question=question, answer=answer, contexts=tuple(contexts))


def to_alpaca(example: dict[str, Any], mapping: FieldMapping) -> dict[str, Any]:
    instruction = stringify(example.get(mapping.question))
    output = stringify(example.get(mapping.answer))
    context_lines = []
    for key in mapping.contexts:
        value = stringify(example.get(key))
        if value:
            context_lines.append(f"{key}: {value}")
    return {
        "instruction": instruction,
        "input": "\n".join(context_lines),
        "output": output,
        "system": SYSTEM_PROMPT,
        "history": [],
    }


def main() -> None:
    args = parse_args()
    maybe_login()
    dataset = load_raw_dataset(args)
    mapping = infer_mapping(dataset)
    print(f"Loaded {len(dataset):,} rows")
    print(f"Using columns: question={mapping.question!r}, answer={mapping.answer!r}, contexts={mapping.contexts}")

    rows = [to_alpaca(example, mapping) for example in dataset]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows):,} Alpaca records to {output_path}")

    for idx, row in enumerate(rows[: args.preview]):
        print(f"\nPreview {idx}:")
        print(json.dumps(row, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
