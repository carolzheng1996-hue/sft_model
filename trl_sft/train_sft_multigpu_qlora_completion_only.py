#!/usr/bin/env python3
"""Multi-GPU QLoRA SFT with completion-only loss via a data collator.

This script keeps the existing assistant_only trainer untouched. It formats
single-turn messages into text with tokenizer.apply_chat_template, then masks
prompt tokens from the loss.
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
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer


DEFAULT_LOCAL_MODEL_PATH = "../models/Qwen2.5-1.5B"
DEFAULT_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
IGNORE_INDEX = -100


class CompletionOnlyDataCollator:
    """Mask all labels through the assistant response marker.

    TRL removed DataCollatorForCompletionOnlyLM from newer releases. This local
    collator keeps the same behavior needed by this script without pinning TRL
    to an older version.
    """

    def __init__(
        self,
        response_template: str,
        tokenizer: PreTrainedTokenizerBase,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.response_token_ids = tokenizer(response_template, add_special_tokens=False)["input_ids"]
        if not self.response_token_ids:
            raise ValueError("response_template must tokenize to at least one token.")

    @staticmethod
    def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
        if len(subsequence) > len(sequence):
            return -1
        last_start = len(sequence) - len(subsequence)
        for start in range(last_start + 1):
            if sequence[start : start + len(subsequence)] == subsequence:
                return start
        return -1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = self.ignore_index

        for row_index, input_ids in enumerate(batch["input_ids"].tolist()):
            if attention_mask is not None:
                valid_length = int(attention_mask[row_index].sum().item())
                input_ids = input_ids[:valid_length]
            marker_start = self._find_subsequence(input_ids, self.response_token_ids)
            if marker_start < 0:
                decoded = self.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                raise ValueError(
                    "Could not find response_template token ids in a training example. "
                    f"Check --response_template. Example prefix: {decoded[:200]!r}"
                )
            response_start = marker_start + len(self.response_token_ids)
            labels[row_index, :response_start] = self.ignore_index

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL QLoRA SFT with completion-only loss.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--dataset_name", default="local")
    parser.add_argument("--data_files", nargs="*", default=["data/processed/train_cot_messages.jsonl"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval_split", default=None)
    parser.add_argument("--test_size", type=float, default=0.02)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--output_dir", default="outputs/qwen2.5-1.5b-train-cot-completion-only-multigpu-qlora")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--response_template", default=DEFAULT_RESPONSE_TEMPLATE)
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.add_argument(
        "--drop_overlong_assistant",
        action="store_true",
        help="Drop samples whose assistant completion alone cannot fit in max_seq_length.",
    )
    parser.add_argument(
        "--error_on_overlong_assistant",
        action="store_false",
        dest="drop_overlong_assistant",
        help="Raise an error instead of dropping samples whose assistant completion alone is too long.",
    )
    parser.set_defaults(use_4bit=True, drop_overlong_assistant=True)
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
    return get_global_rank() in {-1, 0}


def qlora_device_map() -> dict[str, int] | str:
    local_rank = get_local_rank()
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return {"": local_rank}
    return "auto"


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


def local_dataset_loader(files: list[str]) -> str:
    extension = os.path.splitext(files[0])[1].lower().lstrip(".")
    if extension == "jsonl":
        return "json"
    if extension in {"csv", "json", "parquet"}:
        return extension
    raise ValueError(f"Unsupported local file extension: {extension!r}.")


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
                rows.extend(parsed)
            elif isinstance(parsed, dict):
                rows.append(parsed)
            else:
                raise ValueError(f"{file_path} must contain JSON objects.")
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

    if all_local and local_dataset_loader(local_files) == "json" and not args.eval_split:
        rows = load_local_json_records(local_files)
        train_rows, eval_rows = split_records(rows, test_size=args.test_size, seed=args.seed)
        return DatasetDict(train=Dataset.from_list(train_rows), eval=Dataset.from_list(eval_rows))

    def prepare_loaded_dataset() -> DatasetDict:
        if all_local:
            loaded = load_dataset(local_dataset_loader(local_files), data_files={args.split: local_files})
        else:
            if args.dataset_name == "local":
                raise FileNotFoundError(f"Local dataset files were not found: {local_files}")
            loaded = load_dataset(args.dataset_name, data_files={args.split: args.data_files} if args.data_files else None)
        if isinstance(loaded, Dataset):
            loaded = DatasetDict({args.split: loaded})
        if args.eval_split and args.eval_split in loaded:
            return DatasetDict(train=loaded[args.split], eval=loaded[args.eval_split])
        dataset = loaded[args.split]
        indices = list(range(len(dataset)))
        random.Random(args.seed).shuffle(indices)
        eval_size = max(1, int(len(indices) * args.test_size))
        return DatasetDict(
            train=dataset.select(indices[eval_size:], keep_in_memory=True),
            eval=dataset.select(indices[:eval_size], keep_in_memory=True),
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
        raise ValueError(f"{split_name}[{index}].messages[{message_index}].role is invalid: {role!r}.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{split_name}[{index}].messages[{message_index}].content must be non-empty.")


def validate_single_turn_messages(example: dict[str, Any], split_name: str, index: int) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{split_name}[{index}].messages must be a non-empty list.")
    for message_index, message in enumerate(messages):
        validate_message(message, split_name, index, message_index)
    assistant_indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if len(assistant_indices) != 1 or assistant_indices[0] != len(messages) - 1:
        raise ValueError(
            f"{split_name}[{index}].messages must be single-turn with exactly one final assistant message."
        )
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"{split_name}[{index}].messages must include a user message.")


def select_requested_samples(raw: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    prepared = raw
    if args.max_train_samples:
        prepared["train"] = prepared["train"].select(range(min(args.max_train_samples, len(prepared["train"]))))
    if args.max_eval_samples and "eval" in prepared:
        prepared["eval"] = prepared["eval"].select(range(min(args.max_eval_samples, len(prepared["eval"]))))
    for split_name, dataset in prepared.items():
        if "messages" not in dataset.column_names:
            raise ValueError(f"Split {split_name!r} does not contain a 'messages' column.")
        for index, example in enumerate(dataset):
            validate_single_turn_messages(example, split_name, index)
    return prepared


def validate_local_model_dir(model_name_or_path: str) -> None:
    if not os.path.isdir(model_name_or_path):
        raise FileNotFoundError(f"Local model directory not found: {model_name_or_path!r}.")


def build_model_and_tokenizer(args: argparse.Namespace):
    validate_local_model_dir(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
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
        device_map=qlora_device_map() if args.use_4bit else None,
        local_files_only=True,
    )
    model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tokenizer


def token_ids(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def decode_ids(tokenizer: AutoTokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def build_completion_text(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_seq_length: int,
    response_template: str,
) -> str | None:
    prompt_messages = messages[:-1]
    assistant_message = messages[-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    assistant_text = assistant_message["content"]
    if tokenizer.eos_token and not assistant_text.endswith(tokenizer.eos_token):
        assistant_text += tokenizer.eos_token

    prompt_ids = token_ids(tokenizer, prompt_text)
    assistant_ids = token_ids(tokenizer, assistant_text)
    marker_ids = token_ids(tokenizer, response_template)
    if len(assistant_ids) + len(marker_ids) > max_seq_length:
        return None

    max_prompt_tokens = max_seq_length - len(assistant_ids)
    kept_prompt_ids = prompt_ids[-max_prompt_tokens:] if len(prompt_ids) > max_prompt_tokens else prompt_ids
    text = decode_ids(tokenizer, kept_prompt_ids) + decode_ids(tokenizer, assistant_ids)
    if response_template not in text:
        return None
    return text


def make_text_dataset(raw: DatasetDict, tokenizer: AutoTokenizer, args: argparse.Namespace) -> DatasetDict:
    text_dataset = DatasetDict()
    for split_name, dataset in raw.items():
        rows: list[dict[str, str]] = []
        dropped = 0
        for example in dataset:
            text = build_completion_text(tokenizer, example["messages"], args.max_seq_length, args.response_template)
            if text is None:
                dropped += 1
                if not args.drop_overlong_assistant:
                    raise ValueError(
                        f"{split_name} example cannot fit response_template plus assistant completion "
                        "within max_seq_length."
                    )
                continue
            if args.response_template not in text:
                raise ValueError(f"{split_name} example does not contain response_template after formatting.")
            rows.append({"text": text})
        if not rows:
            raise ValueError(f"All {split_name} examples were dropped during completion-only formatting.")
        text_dataset[split_name] = Dataset.from_list(rows)
        if is_main_process():
            print(f"{split_name}: formatted {len(rows)} examples, dropped {dropped} overlong assistant examples.")
    return text_dataset


def make_sft_config(args: argparse.Namespace) -> SFTConfig:
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
    if "dataset_text_field" in supported:
        kwargs["dataset_text_field"] = "text"
    if "ddp_find_unused_parameters" in supported and is_distributed():
        kwargs["ddp_find_unused_parameters"] = False
    if "gradient_checkpointing_kwargs" in supported:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "max_seq_length" in supported:
        kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in supported:
        kwargs["max_length"] = args.max_seq_length
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
    data_collator: Any,
) -> SFTTrainer:
    kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        peft_config=peft_config,
        data_collator=data_collator,
    )
    supported = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in supported:
        kwargs["dataset_text_field"] = "text"
    return SFTTrainer(**{key: value for key, value in kwargs.items() if key in supported})


def main() -> None:
    try:
        args = parse_args()
        configure_local_dataset_cache()
        maybe_login()
        if is_main_process():
            print(f"Starting completion-only QLoRA SFT: world_size={get_world_size()}, output_dir={args.output_dir}")

        raw = select_requested_samples(load_raw_dataset(args), args)
        model, tokenizer = build_model_and_tokenizer(args)
        dataset = make_text_dataset(raw, tokenizer, args)

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
        data_collator = CompletionOnlyDataCollator(response_template=args.response_template, tokenizer=tokenizer)
        trainer = make_trainer(model, tokenizer, training_args, dataset, peft_config, data_collator)
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
