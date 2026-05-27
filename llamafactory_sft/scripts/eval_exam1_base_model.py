#!/usr/bin/env python3
"""Evaluate a local base model on TimeSeriesExam1 examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "../models/Qwen2.5-1.5B"
DEFAULT_DATA_FILE = "data/timeseries_exam1_alpaca.json"
DEFAULT_OUTPUT_FILE = "reports/timeseries_exam1_base_predictions.jsonl"
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local base model on TimeSeriesExam1.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH, help="Local base model directory.")
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE, help="Alpaca JSON data file.")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="JSONL file for predictions.")
    parser.add_argument("--max_samples", type=int, default=20, help="Number of examples to evaluate. Use 0 for all.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit for lower VRAM usage.")
    parser.add_argument("--chat_template", choices=("auto", "qwen"), default="auto")
    return parser.parse_args()


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_user_content(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if input_text:
        return f"Context:\n{input_text}\n\nQuestion:\n{instruction}"
    return instruction


def configure_chat_template(tokenizer: Any, model_path: str, mode: str) -> None:
    if mode == "qwen":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    elif not tokenizer.chat_template and "qwen" in model_path.lower():
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template. Use --chat_template qwen for Qwen-style models.")


def main() -> None:
    args = parse_args()
    require_dir(args.model_name_or_path, "Model")
    require_file(args.data_file, "Data file")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--data_file must contain a JSON list in Alpaca format.")

    end = None if args.max_samples == 0 else args.start_index + args.max_samples
    selected = rows[args.start_index : end]
    if not selected:
        raise ValueError("No examples selected. Check --start_index and --max_samples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
    configure_chat_template(tokenizer, args.model_name_or_path, args.chat_template)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    with output_path.open("w", encoding="utf-8") as out:
        for offset, row in enumerate(selected):
            index = args.start_index + offset
            messages = [
                {"role": "system", "content": str(row.get("system", ""))},
                {"role": "user", "content": build_user_content(row)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)

            generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            expected = str(row.get("output", "")).strip()
            is_exact_match = normalize_answer(prediction) == normalize_answer(expected)
            correct += int(is_exact_match)
            record = {
                "index": index,
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "expected": expected,
                "prediction": prediction,
                "exact_match": is_exact_match,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}] exact_match={is_exact_match}")
            print(f"expected:   {expected}")
            print(f"prediction: {prediction}\n")

    total = len(selected)
    print(f"Wrote predictions to {output_path}")
    print(f"Exact match: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    main()
