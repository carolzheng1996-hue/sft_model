#!/usr/bin/env python3
"""Convert train_cot.jsonl into TRL conversational SFT messages JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "../train_cot.jsonl"
DEFAULT_OUTPUT = "data/processed/train_cot_messages.jsonl"
SYSTEM_PROMPT = (
    "You are a time-series reasoning assistant. "
    "Answer the user's question using the provided time-series data and context."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train_cot.jsonl for TRL conversational SFT.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL with problem, answer, and timeseries.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL with a messages column.")
    parser.add_argument("--no_system", action="store_true", help="Omit the system message from each example.")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping invalid JSONL records.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preview", type=int, default=2)
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def format_timeseries(timeseries: Any) -> str:
    if timeseries is None:
        return ""
    return json.dumps(timeseries, ensure_ascii=False, separators=(",", ":"))


def build_user_content(record: dict[str, Any]) -> str:
    problem = stringify(record.get("problem"))
    timeseries = format_timeseries(record.get("timeseries"))

    if timeseries:
        if "<ts>" in problem:
            problem = problem.replace("<ts>", timeseries)
        else:
            problem = f"{problem}\n\nTime series:\n{timeseries}"

    question_type = stringify(record.get("question_type"))
    if question_type:
        return f"Question type: {question_type}\n\n{problem}"
    return problem


def convert_record(record: dict[str, Any], include_system: bool) -> dict[str, Any] | None:
    user_content = build_user_content(record)
    assistant_content = stringify(record.get("answer"))
    if not user_content or not assistant_content:
        return None

    messages = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    )
    return {"messages": messages}


def iter_jsonl(path: Path, strict: bool):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"Skipping invalid JSON on line {line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                yield None, message
                continue
            if not isinstance(parsed, dict):
                message = f"Skipping line {line_number}: expected a JSON object."
                if strict:
                    raise ValueError(message)
                yield None, message
                continue
            yield parsed, None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    write_count = 0
    skipped_count = 0
    invalid_count = 0
    previews = []

    with output_path.open("w", encoding="utf-8") as output:
        for record, invalid_message in iter_jsonl(input_path, strict=args.strict):
            if invalid_message:
                print(invalid_message)
                invalid_count += 1
                continue
            if args.max_samples is not None and read_count >= args.max_samples:
                break
            read_count += 1
            converted = convert_record(record, include_system=not args.no_system)
            if converted is None:
                skipped_count += 1
                continue
            if len(previews) < args.preview:
                previews.append(converted)
            output.write(json.dumps(converted, ensure_ascii=False) + "\n")
            write_count += 1

    print(f"Loaded {read_count:,} rows from {input_path}")
    print(f"Wrote {write_count:,} conversational SFT rows to {output_path}")
    print(f"Skipped {skipped_count:,} rows with missing problem or answer")
    print(f"Skipped {invalid_count:,} invalid JSONL rows")
    for index, row in enumerate(previews):
        print(f"\nPreview {index}:")
        print(json.dumps(row, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
