#!/usr/bin/env python3
"""Multi-NPU LoRA SFT for a local Qwen2.5 model with assistant-only loss.

Launch with accelerate or torchrun, for example:

  accelerate launch --num_processes 4 train_sft_multigpu_qlora.py ...

The number of NPUs is controlled by the launcher, not by SFTTrainer args.
Each distributed process loads the model onto its own LOCAL_RANK device,
then TRL/HF Trainer handles the distributed training loop.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS_CACHE = PROJECT_ROOT / "datasets" / ".hf_cache"
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_DATASETS_CACHE))

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from npu_utils import current_device_map, get_torch_dtype, setup_npu, training_optim, validate_quantization_args


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
        "--min_assistant_tokens",
        type=int,
        default=1,
        help=(
            "Minimum number of assistant tokens that must remain after truncation when using "
            "assistant-only loss. Samples below this threshold are dropped before training/eval."
        ),
    )
    parser.add_argument(
        "--no_filter_empty_assistant_labels",
        action="store_false",
        dest="filter_empty_assistant_labels",
        help="Disable pre-filtering examples whose assistant labels disappear after max_seq_length truncation.",
    )
    parser.add_argument(
        "--no_assistant_only_loss",
        action="store_false",
        dest="assistant_only_loss",
        help="Disable assistant-only loss and train on the full rendered chat sequence.",
    )
    parser.add_argument("--bf16", action="store_true", help="Use bf16. Recommended on A100/H100/RTX 40xx.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16. Use when bf16 is unsupported.")
    parser.add_argument("--use_4bit", action="store_true", help="Unsupported on Ascend NPU; kept only for CLI compatibility and will raise if set.")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit", help="Disable 4-bit loading (unsupported on Ascend NPU).")
    parser.add_argument("--no_lora", action="store_true", help="Full fine-tuning instead of LoRA.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.set_defaults(use_4bit=False, assistant_only_loss=True, filter_empty_assistant_labels=True)
    return parser.parse_args()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def get_global_rank() -> int:
    return int(os.environ.get("RANK", "-1"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    rank = get_global_rank()
    return rank in {-1, 0}


def qlora_device_map() -> None:
    return current_device_map()


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


def load_local_json_records(files: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in files:
        extension = os.path.splitext(file_path)[1].lower()
        with open(file_path, "r", encoding="utf-8") as handle:
            if extension == ".jsonl":
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise ValueError(f"{file_path}:{line_number} must be a JSON object.")
                    rows.append(parsed)
                continue

            parsed = json.load(handle)
            if isinstance(parsed, list):
                for index, row in enumerate(parsed):
                    if not isinstance(row, dict):
                        raise ValueError(f"{file_path}[{index}] must be a JSON object.")
                    rows.append(row)
            elif isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
                rows.append(parsed)
            else:
                raise ValueError(f"{file_path} must contain a JSON list of objects or one messages object.")
    return rows


def split_records(rows: list[dict[str, Any]], test_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Dataset is empty after loading local JSON/JSONL files.")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(len(indices) * test_size))
    eval_indices = set(indices[:eval_size])
    train_rows = [row for index, row in enumerate(rows) if index not in eval_indices]
    eval_rows = [row for index, row in enumerate(rows) if index in eval_indices]
    return train_rows, eval_rows


def load_raw_dataset(args: argparse.Namespace) -> DatasetDict:
    local_files = expand_local_files(args.data_files) if args.data_files else []
    all_local = bool(local_files) and all(os.path.exists(path) for path in local_files)
    data_files: dict[str, list[str]] | None = None

    if all_local and local_dataset_loader(local_files) == "json" and not args.eval_split:
        rows = load_local_json_records(local_files)
        train_rows, eval_rows = split_records(rows, test_size=args.test_size, seed=args.seed)
        return DatasetDict(train=Dataset.from_list(train_rows), eval=Dataset.from_list(eval_rows))

    def load_once() -> Dataset | DatasetDict:
        if all_local:
            extension = local_dataset_loader(local_files)
            data_files = {args.split: local_files}
            return load_dataset(extension, data_files=data_files)

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
        return load_dataset(args.dataset_name, data_files=data_files)

    def prepare_loaded_dataset() -> DatasetDict:
        loaded = load_once()
        if isinstance(loaded, Dataset):
            loaded = DatasetDict({args.split: loaded})

        if args.eval_split and args.eval_split in loaded:
            return DatasetDict(train=loaded[args.split], eval=loaded[args.eval_split])

        dataset = loaded[args.split]
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        eval_size = max(1, int(len(indices) * args.test_size))
        eval_indices = indices[:eval_size]
        train_indices = indices[eval_size:]
        return DatasetDict(
            train=dataset.select(train_indices, keep_in_memory=True),
            eval=dataset.select(eval_indices, keep_in_memory=True),
        )

    if is_distributed():
        with PartialState().main_process_first():
            return prepare_loaded_dataset()
    return prepare_loaded_dataset()


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


def assistant_token_count_after_truncation(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_seq_length: int,
) -> int | None:
    """Return assistant-mask token count after truncation, or None if unsupported."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )
    except Exception as exc:
        if is_main_process():
            print(
                "WARNING: Could not pre-check assistant label masks with tokenizer.apply_chat_template; "
                f"continuing without label filtering. Reason: {exc}"
            )
        return None

    for key in ("assistant_masks", "assistant_mask", "assistant_tokens_mask"):
        mask = encoded.get(key)
        if mask is not None:
            return int(sum(mask))
    if is_main_process():
        print(
            "WARNING: tokenizer.apply_chat_template did not return an assistant token mask; "
            "continuing without label filtering."
        )
    return None


def filter_empty_assistant_label_examples(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> DatasetDict:
    if not args.assistant_only_loss or not args.filter_empty_assistant_labels:
        return dataset

    filtered = DatasetDict()
    unsupported = False
    for split_name, split_dataset in dataset.items():
        keep_indices: list[int] = []
        dropped = 0
        min_seen: int | None = None

        for index, example in enumerate(split_dataset):
            count = assistant_token_count_after_truncation(
                tokenizer,
                example["messages"],
                args.max_seq_length,
            )
            if count is None:
                unsupported = True
                break
            min_seen = count if min_seen is None else min(min_seen, count)
            if count >= args.min_assistant_tokens:
                keep_indices.append(index)
            else:
                dropped += 1

        if unsupported:
            return dataset
        if not keep_indices:
            raise ValueError(
                f"All {split_name} examples have fewer than {args.min_assistant_tokens} assistant tokens "
                f"after max_seq_length={args.max_seq_length} truncation. Increase --max_seq_length, "
                "shorten prompts, or disable assistant-only loss."
            )

        filtered[split_name] = split_dataset.select(keep_indices)
        if is_main_process():
            print(
                f"{split_name}: kept {len(keep_indices)} examples, dropped {dropped} examples with "
                f"too few assistant tokens after truncation; min_assistant_tokens_seen={min_seen}."
            )

    return filtered


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

    validate_quantization_args(args.use_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(args.bf16, args.fp16),
        device_map=current_device_map(),
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
        optim=training_optim(args.use_4bit),
        report_to="none",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    if "ddp_find_unused_parameters" in supported and is_distributed():
        kwargs["ddp_find_unused_parameters"] = False
    if "gradient_checkpointing_kwargs" in supported:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if "max_seq_length" in supported:
        kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in supported:
        kwargs["max_length"] = args.max_seq_length

    if "assistant_only_loss" in supported:
        kwargs["assistant_only_loss"] = args.assistant_only_loss
    elif args.assistant_only_loss and is_main_process():
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
    try:
        args = parse_args()
        setup_npu()
        configure_local_dataset_cache()
        maybe_login()

        if is_main_process():
            mode = "distributed" if is_distributed() else "single-process"
            print(
                f"Starting {mode} LoRA SFT: world_size={get_world_size()}, "
                f"use_4bit={args.use_4bit}, output_dir={args.output_dir}"
            )

        raw = load_raw_dataset(args)
        dataset = prepare_messages_dataset(raw, args)
        model, tokenizer = build_model_and_tokenizer(args)
        dataset = filter_empty_assistant_label_examples(dataset, tokenizer, args)

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
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
