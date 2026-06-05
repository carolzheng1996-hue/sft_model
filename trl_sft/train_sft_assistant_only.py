#!/usr/bin/env python3
"""SFT a local Qwen2.5 model with assistant-only loss on TRL messages data."""

from __future__ import annotations

import argparse
import glob
import inspect
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS_CACHE = PROJECT_ROOT / "datasets" / ".hf_cache"
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_DATASETS_CACHE))

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_LOCAL_MODEL_PATH = "../models/Qwen2.5-1.5B"

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endgeneration %}"
    "{% else %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL SFT for Qwen2.5 on conversational messages data.")
    parser.add_argument(
        "--model_name_or_path",
        default=DEFAULT_LOCAL_MODEL_PATH,
        help=f"Local model directory. Defaults to {DEFAULT_LOCAL_MODEL_PATH!r}.",
    )
    parser.add_argument(
        "--chat_template",
        choices=("auto", "qwen", "none"),
        default="auto",
        help=(
            "Tokenizer chat-template handling. 'auto' uses the tokenizer template and falls back to Qwen "
            "for Qwen models without one."
        ),
    )
    parser.add_argument("--dataset_name", default="local")
    parser.add_argument(
        "--data_files",
        nargs="*",
        default=["data/processed/timemqa_local_train.json"],
        help="Local or HF JSON/JSONL/Parquet files containing a TRL conversational 'messages' column.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval_split", default=None)
    parser.add_argument("--test_size", type=float, default=0.02)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--output_dir", default="outputs/qwen2.5-1.5b-tsqa-lora")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument(
        "--no_assistant_only_loss",
        action="store_false",
        dest="assistant_only_loss",
        help="Disable assistant-only loss and train on the full rendered chat sequence.",
    )
    parser.add_argument("--bf16", action="store_true", help="Use bf16. Recommended on A100/H100/RTX 40xx.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16. Use when bf16 is unsupported.")
    parser.add_argument("--use_4bit", action="store_true", help="Enable QLoRA 4-bit loading via bitsandbytes.")
    parser.add_argument("--no_lora", action="store_true", help="Full fine-tuning instead of LoRA.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.set_defaults(assistant_only_loss=True)
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


def missing_local_files(files: list[str]) -> list[str]:
    return [path for path in files if not os.path.exists(path)]


def local_dataset_loader(files: list[str]) -> str:
    extension = os.path.splitext(files[0])[1].lower().lstrip(".")
    if extension == "jsonl":
        return "json"
    if extension in {"csv", "json", "parquet"}:
        return extension
    raise ValueError(f"Unsupported local file extension: {extension!r}. Use csv, json, jsonl, or parquet.")


def load_raw_dataset(args: argparse.Namespace) -> DatasetDict:
    local_files = expand_local_files(args.data_files) if args.data_files else []
    all_local = bool(local_files) and all(os.path.exists(path) for path in local_files)
    data_files: dict[str, list[str]] | None = None

    if all_local:
        data_files = {args.split: local_files}
        extension = local_dataset_loader(local_files)
        loaded = load_dataset(extension, data_files=data_files)
    else:
        missing_files = missing_local_files(local_files)
        if args.dataset_name == "local":
            raise FileNotFoundError(
                "Local dataset files were not found, so the trainer cannot load --dataset_name local.\n"
                f"Current working directory: {os.getcwd()}\n"
                f"--data_files resolved to: {local_files}\n"
                f"Missing files: {missing_files}\n"
                "Run from the trl_sft directory or pass an absolute path, for example:\n"
                "  --data_files data/processed/timemqa_local_train.json"
            )
        data_files = {args.split: args.data_files} if args.data_files else None
        loaded = load_dataset(args.dataset_name, data_files=data_files)

    if isinstance(loaded, Dataset):
        loaded = DatasetDict({args.split: loaded})

    if args.eval_split and args.eval_split in loaded:
        return DatasetDict(train=loaded[args.split], eval=loaded[args.eval_split])

    split_ds = loaded[args.split].train_test_split(test_size=args.test_size, seed=args.seed)
    return DatasetDict(train=split_ds["train"], eval=split_ds["test"])


def validate_message(message: Any, split_name: str, index: int, message_index: int) -> None:
    if not isinstance(message, dict):
        raise ValueError(f"{split_name}[{index}].messages[{message_index}] must be an object.")
    role = message.get("role")
    content = message.get("content")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(
            f"{split_name}[{index}].messages[{message_index}].role must be one of "
            f"'system', 'user', or 'assistant'; got {role!r}."
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{split_name}[{index}].messages[{message_index}].content must be a non-empty string.")


def validate_messages_example(example: dict[str, Any], split_name: str, index: int) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{split_name}[{index}].messages must be a non-empty list.")
    roles = []
    for message_index, message in enumerate(messages):
        validate_message(message, split_name, index, message_index)
        roles.append(message["role"])
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"{split_name}[{index}].messages must include at least one user and one assistant message.")


def prepare_messages_dataset(raw: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    prepared = raw
    if args.max_train_samples:
        prepared["train"] = prepared["train"].select(range(min(args.max_train_samples, len(prepared["train"]))))
    if args.max_eval_samples and "eval" in prepared:
        prepared["eval"] = prepared["eval"].select(range(min(args.max_eval_samples, len(prepared["eval"]))))

    for split_name, dataset in prepared.items():
        if "messages" not in dataset.column_names:
            raise ValueError(
                f"Split {split_name!r} does not contain a 'messages' column. "
                "Preprocess raw datasets into TRL conversational language modeling format first."
            )
        for index, example in enumerate(dataset):
            validate_messages_example(example, split_name, index)

    print("Using TRL conversational messages format.")
    return prepared


def configure_chat_template(tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    if args.chat_template == "none":
        return
    if args.chat_template == "qwen":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        return
    if args.assistant_only_loss and "qwen" in args.model_name_or_path.lower():
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        return
    if tokenizer.chat_template:
        return
    if "qwen" in args.model_name_or_path.lower():
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        return
    raise ValueError(
        "Tokenizer has no chat_template. Set --chat_template qwen for Qwen-style chat formatting, "
        "or --chat_template none and customize build_formatter for a plain prompt format."
    )


def validate_local_model_dir(model_name_or_path: str) -> None:
    if not os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"Local model directory not found: {model_name_or_path!r}. "
            "Download the base model first and pass its directory with --model_name_or_path."
        )


def build_model_and_tokenizer(args: argparse.Namespace):
    validate_local_model_dir(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    configure_chat_template(tokenizer, args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto"),
        quantization_config=quantization_config,
        device_map="auto" if args.use_4bit else None,
        local_files_only=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def make_sft_config(args: argparse.Namespace) -> SFTConfig:
    """Build SFTConfig across TRL/Transformers minor API differences."""
    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        packing=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    if "max_seq_length" in supported:
        kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in supported:
        kwargs["max_length"] = args.max_seq_length

    if "assistant_only_loss" in supported:
        kwargs["assistant_only_loss"] = args.assistant_only_loss
    elif args.assistant_only_loss:
        print("WARNING: This TRL version does not support SFTConfig.assistant_only_loss; full-sequence loss will be used.")

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"

    return SFTConfig(**{key: value for key, value in kwargs.items() if key in supported})


def make_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: SFTConfig,
    dataset: DatasetDict,
    peft_config: LoraConfig | None,
) -> SFTTrainer:
    kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        peft_config=peft_config,
    )
    supported = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**{key: value for key, value in kwargs.items() if key in supported})


def main() -> None:
    args = parse_args()
    configure_local_dataset_cache()
    maybe_login()

    raw = load_raw_dataset(args)
    dataset = prepare_messages_dataset(raw, args)
    model, tokenizer = build_model_and_tokenizer(args)

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    training_args = make_sft_config(args)
    trainer = make_trainer(model, tokenizer, training_args, dataset, peft_config)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
