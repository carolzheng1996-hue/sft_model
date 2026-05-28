#!/usr/bin/env python3
"""Official-style TimeSeriesExam1 evaluation helpers.

This adapts the public TimeSeriesExam evaluation logic for local Qwen models.
Official flexible scoring marks a response correct when it contains:

    "<correct option letter>) <answer text>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_PATH = "../models/Qwen2.5-1.5B"
DEFAULT_DATA_FILE = "../datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json"
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
EXAMPLE_PROMPT = (
    "Here is an example question answer pair to help you understand the format better: "
    "EXAMPLE QUESTION: What is the most likely autocorrelation at lag 1 for the given time series?\n \n "
    "Choose From Following Options: \n \n "
    "A) High positive autocorrelation\n"
    "B) No autocorrelation\n"
    "C) Negative autocorrelation\n"
    "Now, answer the question. "
    "EXAMPLE RESPONSE: Based on the given time series, the data points appear to fluctuate randomly around the mean "
    "with no clear pattern of persistence or trend.\n"
    "This suggests that the time series does not exhibit a strong relationship between consecutive data points.\n\n"
    "Given the options:\n\n"
    "A) High positive autocorrelation\n"
    "B) No autocorrelation\n"
    "C) Negative autocorrelation\n\n"
    "The most likely autocorrelation at lag 1 for the given time series is:\n\n"
    "B) No autocorrelation Now, answer the given question and also explain your thought process: "
)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_name_or_path", default=DEFAULT_BASE_MODEL_PATH, help="Local base model directory.")
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_samples", type=int, default=50, help="Number of examples to evaluate. Use 0 for all.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_4bit", action="store_true", help="Load the base model in 4-bit.")
    parser.add_argument("--chat_template", choices=("auto", "qwen"), default="auto")
    return parser


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


def tokenize_timeseries(row: dict[str, Any]) -> str:
    if "ts" in row and row["ts"] is not None:
        return "Timeseries: \n" + ",".join([str(round(float(x), 2)) for x in row["ts"]])
    if "ts1" in row and "ts2" in row:
        ts1 = ",".join([str(round(float(x), 1)) for x in row["ts1"]])
        ts2 = ",".join([str(round(float(x), 1)) for x in row["ts2"]])
        return f"Timeseries1: \n{ts1}\nTimeseries2: \n{ts2}"
    raise ValueError("Each row must contain either 'ts' or both 'ts1' and 'ts2'.")


def build_official_query(row: dict[str, Any]) -> str:
    options = row.get("options") or []
    options_string = "\n".join([f"{chr(65 + index)}) {stringify(option)}" for index, option in enumerate(options)])
    prompt = f"{stringify(row.get('question'))} Choose From Following Options: {options_string}\n"
    format_hint = stringify(row.get("format_hint"))
    if format_hint:
        prompt += f"{format_hint}.\n"
    prompt += EXAMPLE_PROMPT
    return prompt


def build_user_content(row: dict[str, Any]) -> str:
    ts_text = tokenize_timeseries(row)
    query = build_official_query(row)
    if "ts" in row and row["ts"] is not None:
        prefix = (
            "You are given one time series, where each step is separated by a comma.\n"
            f"{ts_text}\n"
            "Answer the following question based on the time series. In your analysis, try not to repeat large chunk "
            "of values in the time series to save space. Question: \n"
        )
    else:
        prefix = (
            "You are given two time series here, where each step is separated by a comma. "
            f"{ts_text}\n"
            "Answer the following question based on the time series. In your analysis, try not to repeat large chunk "
            "of values in the time series to save space.\nQuestion: \n"
        )
    return prefix + query


def correct_option_letter(row: dict[str, Any]) -> str:
    options = [stringify(option) for option in row.get("options", [])]
    answer = stringify(row.get("answer"))
    if answer not in options:
        raise ValueError(f"Answer does not match any option: {answer!r}")
    return chr(options.index(answer) + 65)


def official_flexible_correct(row: dict[str, Any], response: str) -> bool:
    answer = stringify(row.get("answer"))
    letter = correct_option_letter(row)
    return f"{letter}) {answer}".lower() in response.lower()


def official_strict_correct(row: dict[str, Any], response: str) -> bool:
    answer = stringify(row.get("answer"))
    last_line = response.split("\n")[-1].lower()
    return answer.lower() in last_line


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    require_file(args.data_file, "Data file")
    rows = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--data_file must contain a JSON list.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        if not isinstance(row.get("options"), list) or not row["options"]:
            raise ValueError(f"Row {index} must contain a non-empty options list.")
        correct_option_letter(row)
    return rows


def configure_chat_template(tokenizer: Any, model_path: str, mode: str) -> None:
    if mode == "qwen":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    elif not tokenizer.chat_template and "qwen" in model_path.lower():
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template. Use --chat_template qwen for Qwen-style models.")


def load_model_and_tokenizer(args: argparse.Namespace, use_adapter: bool):
    require_dir(args.model_name_or_path, "Model")
    if use_adapter:
        require_dir(args.adapter_name_or_path, "Adapter")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    if use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, local_files_only=True)
    model.eval()
    return model, tokenizer


def run_official_evaluation(args: argparse.Namespace, use_adapter: bool) -> None:
    import torch

    rows = load_rows(args)
    end = None if args.max_samples == 0 else args.start_index + args.max_samples
    selected = rows[args.start_index : end]
    if not selected:
        raise ValueError("No examples selected. Check --start_index and --max_samples.")

    model, tokenizer = load_model_and_tokenizer(args, use_adapter=use_adapter)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flexible_correct = 0
    strict_correct = 0
    results = []
    with output_path.open("w", encoding="utf-8") as out:
        for offset, row in enumerate(selected):
            index = args.start_index + offset
            messages = [{"role": "user", "content": build_user_content(row)}]
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
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            flexible = official_flexible_correct(row, response)
            strict = official_strict_correct(row, response)
            flexible_correct += int(flexible)
            strict_correct += int(strict)

            record = {
                "index": index,
                "question": row.get("question", ""),
                "options": row.get("options", []),
                "answer": row.get("answer", ""),
                "answer_option_letter": correct_option_letter(row),
                "response": response,
                "correct": flexible,
                "official_flexible_correct": flexible,
                "official_strict_correct": strict,
            }
            results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}] official_flexible_correct={flexible} official_strict_correct={strict}")
            print(f"expected: {record['answer_option_letter']}) {record['answer']}")
            print(f"response: {response}\n")

    total = len(results)
    official_accuracy = flexible_correct / total
    strict_accuracy = strict_correct / total
    print(f"Wrote predictions to {output_path}")
    print(f"Official flexible accuracy: {flexible_correct}/{total} = {official_accuracy:.4f}")
    print(f"Official strict accuracy: {strict_correct}/{total} = {strict_accuracy:.4f}")
