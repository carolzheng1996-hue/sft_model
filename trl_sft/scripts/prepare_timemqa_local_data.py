#!/usr/bin/env python3
"""Convert local Time-MQA CSV examples into TRL conversational SFT JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = "../timemqa/open_ended_QA.csv"
DEFAULT_OUTPUT = "data/processed/timemqa_local_train.json"
SYSTEM_PROMPT = (
    "You are a time-series question answering expert. "
    "Answer the user's question from the provided time-series information. "
    "Be concise, numerical when appropriate, and do not invent missing facts."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local Time-MQA CSV for TRL conversational SFT.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Local Time-MQA CSV with QA_list column.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file with a messages column.")
    parser.add_argument("--qa_field", default="QA_list")
    parser.add_argument("--no_system", action="store_true", help="Omit the system message from each example.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preview", type=int, default=2)
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def parse_qa_payload(value: Any) -> dict[str, str]:
    text = stringify(value)
    if not text:
        return {}

    candidates = [text]
    if not text.startswith("{"):
        candidates.append("{" + text + "}")

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return {str(key): stringify(val) for key, val in parsed.items()}

    result: dict[str, str] = {}
    for key in ("question", "answer"):
        match = re.search(rf'"{key}"\s*:\s*"(.*?)"(?=\s*,\s*"\w+"\s*:|\s*$|\s*}})', text, flags=re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
    return result


def build_user_content(row: dict[str, Any], question: str) -> str:
    context_lines = []
    for key in ("application_domain", "task_type", "question_format"):
        value = stringify(row.get(key))
        if value:
            context_lines.append(f"{key}: {value}")

    parts = []
    if context_lines:
        parts.append("Context:\n" + "\n".join(context_lines))
    parts.append(f"Question:\n{question}")
    return "\n\n".join(parts)


def convert_row(row: dict[str, Any], qa_field: str, include_system: bool) -> dict[str, Any] | None:
    qa = parse_qa_payload(row.get(qa_field))
    question = stringify(qa.get("question"))
    answer = stringify(qa.get("answer"))
    if not question or not answer:
        return None

    messages = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(
        [
            {"role": "user", "content": build_user_content(row, question)},
            {"role": "assistant", "content": answer},
        ]
    )
    return {
        "messages": messages,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.qa_field not in df.columns:
        raise ValueError(f"QA field {args.qa_field!r} not found. Columns: {list(df.columns)}")
    if args.max_samples:
        df = df.head(args.max_samples)

    rows = []
    skipped = 0
    for record in df.to_dict(orient="records"):
        converted = convert_row(record, args.qa_field, include_system=not args.no_system)
        if converted is None:
            skipped += 1
            continue
        rows.append(converted)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded {len(df):,} rows from {input_path}")
    print(f"Wrote {len(rows):,} conversational SFT rows to {output_path}")
    print(f"Skipped {skipped:,} rows with missing/unparseable question or answer")
    for idx, row in enumerate(rows[: args.preview]):
        print(f"\nPreview {idx}:")
        print(json.dumps(row, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
