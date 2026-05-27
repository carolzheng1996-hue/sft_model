#!/usr/bin/env python3
"""Check whether a local causal-LM model directory can be loaded and generate text."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "models/Qwen2.5-1.5B"
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
    parser = argparse.ArgumentParser(description="Check a locally downloaded Hugging Face causal-LM model.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH, help="Local model directory.")
    parser.add_argument("--prompt", default="Answer briefly: what is a time series?", help="Prompt used for test generation.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--load_4bit", action="store_true", help="Load in 4-bit with bitsandbytes.")
    parser.add_argument("--chat_template", choices=("auto", "qwen", "none"), default="auto")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_model_files(model_path: Path) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    required_any = [
        ("config", ["config.json"]),
        ("tokenizer", ["tokenizer.json", "tokenizer.model", "vocab.json"]),
        ("weights", ["*.safetensors", "*.bin", "*.pt"]),
    ]
    for label, patterns in required_any:
        if not any(match for pattern in patterns for match in model_path.glob(pattern)):
            raise FileNotFoundError(f"No {label} file found in {model_path}. Checked: {patterns}")


def configure_chat_template(tokenizer: Any, model_path: str, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "qwen":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        return True
    if tokenizer.chat_template:
        return True
    if "qwen" in model_path.lower():
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        return True
    return False


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_name_or_path).expanduser().resolve()
    require_model_files(model_path)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Checking local model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    has_chat_template = configure_chat_template(tokenizer, str(model_path), args.chat_template)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. vocab_size={len(tokenizer)}, chat_template={bool(has_chat_template)}")

    quantization_config = None
    if args.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()
    param = next(model.parameters())
    print(f"Model loaded. dtype={param.dtype}, device={param.device}")

    if has_chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = args.prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\nPrompt:")
    print(args.prompt)
    print("\nGenerated:")
    print(output)
    print("\nOK: local model can be loaded and used for generation.")


if __name__ == "__main__":
    main()
