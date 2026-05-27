#!/usr/bin/env python3
"""Evaluate a local TRL SFT adapter or merged model on TimeSeriesExam1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_PATH = "../models/Qwen2.5-1.5B"
DEFAULT_ADAPTER_PATH = "outputs/qwen2.5-1.5b-timeseries-exam1-lora"
DEFAULT_DATA_FILE = "../datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json"
DEFAULT_OUTPUT_FILE = "reports/timeseries_exam1_sft_predictions.jsonl"
SYSTEM_PROMPT = (
    "You are a time-series question answering expert. "
    "Answer the user's question from the provided time-series information. "
    "Be concise, numerical when appropriate, and do not invent missing facts."
)
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
DEFAULT_CONTEXT_FIELDS = (
    "options",
    "question_type",
    "ts1",
    "ts2",
    "difficulty",
    "format_hint",
    "relevant_concepts",
    "question_hint",
    "category",
    "subcategory",
    "ts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TRL SFT model on TimeSeriesExam1.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_BASE_MODEL_PATH, help="Local base or merged model directory.")
    parser.add_argument("--adapter_name_or_path", default=DEFAULT_ADAPTER_PATH, help="Local LoRA adapter directory.")
    parser.add_argument("--use_adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--context_fields", nargs="*", default=list(DEFAULT_CONTEXT_FIELDS))
    parser.add_argument("--max_samples", type=int, default=50, help="Number of examples to evaluate. Use 0 for all.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_4bit", action="store_true", help="Load the base model in 4-bit.")
    parser.add_argument("--chat_template", choices=("auto", "qwen"), default="auto")
    return parser.parse_args()


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_user_content(row: dict[str, Any], question_field: str, context_fields: list[str]) -> str:
    question = stringify(row.get(question_field))
    context_lines = []
    for key in context_fields:
        value = stringify(row.get(key))
        if value:
            context_lines.append(f"{key}: {value}")
    if context_lines:
        return "Context:\n" + "\n".join(context_lines) + f"\n\nQuestion:\n{question}"
    return f"Question:\n{question}"


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
    if args.use_adapter:
        require_dir(args.adapter_name_or_path, "Adapter")
    require_file(args.data_file, "Data file")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--data_file must contain a JSON list.")

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
    if args.use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, local_files_only=True)
    model.eval()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exact = 0
    with output_path.open("w", encoding="utf-8") as out:
        for offset, row in enumerate(selected):
            index = args.start_index + offset
            expected = stringify(row.get(args.answer_field))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(row, args.question_field, args.context_fields)},
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
            is_exact = normalize_answer(prediction) == normalize_answer(expected)
            exact += int(is_exact)
            record = {
                "index": index,
                "question": row.get(args.question_field, ""),
                "expected": expected,
                "prediction": prediction,
                "exact_match": is_exact,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}] exact_match={is_exact}")
            print(f"expected:   {expected}")
            print(f"prediction: {prediction}\n")

    total = len(selected)
    print(f"Wrote predictions to {output_path}")
    print(f"Exact match: {exact}/{total} = {exact / total:.4f}")


if __name__ == "__main__":
    main()
