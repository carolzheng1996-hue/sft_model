#!/usr/bin/env python3
"""Run local inference for a LLaMA-Factory LoRA adapter or merged model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_PATH = "../models/Qwen2.5-1.5B"
DEFAULT_ADAPTER_PATH = "saves/qwen2.5-1.5b/timemqa/qlora-sft"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a local SFT model")
    parser.add_argument("--model_name_or_path", default=DEFAULT_BASE_MODEL_PATH, help="Local base or merged model directory.")
    parser.add_argument("--adapter_name_or_path", default=DEFAULT_ADAPTER_PATH, help="Local LoRA adapter directory.")
    parser.add_argument("--use_adapter", action=argparse.BooleanOptionalAction, default=True, help="Load LoRA adapter on top of the base model.")
    parser.add_argument("--instruction", default=None, help="Question/instruction text.")
    parser.add_argument("--input", default="", help="Optional context text.")
    parser.add_argument("--example_file", default=None, help="Alpaca JSON file, for example data/timemqa_tsqa_alpaca.json.")
    parser.add_argument("--example_index", type=int, default=0, help="Example index when --example_file is used.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit for lower VRAM inference.")
    parser.add_argument("--chat_template", choices=("auto", "qwen"), default="auto")
    return parser.parse_args()


def require_dir(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def read_example(path: str, index: int) -> tuple[str, str, str | None]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--example_file must contain a JSON list in Alpaca format.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"--example_index {index} out of range for {len(rows)} examples.")
    row: dict[str, Any] = rows[index]
    return str(row.get("instruction", "")), str(row.get("input", "")), str(row.get("output", "")) or None


def build_user_content(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return f"Context:\n{input_text.strip()}\n\nQuestion:\n{instruction.strip()}"
    return instruction.strip()


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

    expected_output = None
    instruction = args.instruction
    input_text = args.input
    if args.example_file:
        instruction, input_text, expected_output = read_example(args.example_file, args.example_index)
    if not instruction:
        raise ValueError("Provide --instruction, or use --example_file with --example_index.")

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
    if args.use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, local_files_only=True)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(instruction, input_text)},
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
    print("Question:")
    print(instruction.strip())
    if input_text.strip():
        print("\nContext:")
        print(input_text.strip()[:4000])
    if expected_output:
        print("\nExpected:")
        print(expected_output.strip())
    print("\nPrediction:")
    print(prediction)


if __name__ == "__main__":
    main()
