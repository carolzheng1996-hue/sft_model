#!/usr/bin/env python3
"""Parallel full-set evaluation for local Time-MQA messages JSON on CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_PATH = "../models/Qwen2.5-1.5B"
DEFAULT_ADAPTER_PATH = "outputs/qwen2.5-1.5b-timemqa-local-multigpu-qlora"
DEFAULT_DATA_FILE = "data/processed/timemqa_local_eval.json"
DEFAULT_OUTPUT_FILE = "reports/timemqa_local_full_predictions_parallel.jsonl"
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
    parser = argparse.ArgumentParser(description="Parallel full Time-MQA evaluation for a local SFT model.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter_name_or_path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--use_adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max_samples", type=int, default=0, help="Use 0 for the full dataset.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--chat_template", choices=("auto", "qwen"), default="auto")
    parser.add_argument("--keep_part_files", action="store_true")
    return parser.parse_args()


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return global_rank() == 0


def setup_distributed() -> None:
    if world_size() <= 1:
        return
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def distributed_barrier() -> None:
    if world_size() <= 1:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def teardown_distributed() -> None:
    if world_size() <= 1:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def validate_messages_row(row: Any, index: int) -> tuple[list[dict[str, str]], str]:
    if not isinstance(row, dict):
        raise ValueError(f"Row {index} must be a JSON object.")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Row {index}.messages must be a non-empty messages list.")
    cleaned = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Row {index}.messages[{message_index}] must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Row {index}.messages[{message_index}].role is invalid: {role!r}.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Row {index}.messages[{message_index}].content must be a non-empty string.")
        cleaned.append({"role": role, "content": content.strip()})
    if cleaned[-1]["role"] != "assistant":
        raise ValueError(f"Row {index}.messages must end with the expected assistant answer.")
    if not any(message["role"] == "user" for message in cleaned[:-1]):
        raise ValueError(f"Row {index}.messages must contain a user prompt before the assistant answer.")
    return cleaned[:-1], cleaned[-1]["content"]


def load_items(args: argparse.Namespace) -> list[tuple[int, list[dict[str, str]], str]]:
    require_file(args.data_file, "Time-MQA messages JSON")
    rows = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--data_file must contain a JSON list of objects with a messages column.")
    end = None if args.max_samples == 0 else args.start_index + args.max_samples
    selected = rows[args.start_index : end]
    if not selected:
        raise ValueError("No examples selected. Check --start_index and --max_samples.")
    return [
        (args.start_index + offset, *validate_messages_row(row, args.start_index + offset))
        for offset, row in enumerate(selected)
    ]


def configure_chat_template(tokenizer: Any, model_path: str, mode: str) -> None:
    if mode == "qwen":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    elif not tokenizer.chat_template and "qwen" in model_path.lower():
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template. Use --chat_template qwen for Qwen-style models.")


def cuda_device_map() -> dict[str, int] | str | None:
    import torch

    rank = local_rank()
    if torch.cuda.is_available() and rank >= 0:
        torch.cuda.set_device(rank)
        return {"": rank}
    if torch.cuda.is_available():
        return "auto"
    return None


def model_input_device(model: Any):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_model_and_tokenizer(args: argparse.Namespace):
    require_dir(args.model_name_or_path, "Model")
    if args.use_adapter:
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

    device_map = cuda_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    if args.use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, local_files_only=True)
    if device_map is None and torch.cuda.is_available():
        model.to(torch.device("cuda", local_rank() if local_rank() >= 0 else 0))
    model.eval()
    return model, tokenizer


def part_file(output_file: str, rank: int) -> Path:
    output_path = Path(output_file)
    return output_path.parent / f".{output_path.name}.parts" / f"rank_{rank:05d}.jsonl"


def prepare_parts_dir(output_file: str) -> None:
    parts_dir = part_file(output_file, 0).parent
    parts_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        for path in parts_dir.glob("rank_*.jsonl"):
            path.unlink()


def evaluate_rank(args: argparse.Namespace) -> Path:
    import torch

    items = load_items(args)
    rank = global_rank()
    size = world_size()
    rank_items = [(index, prompt_messages, expected) for position, (index, prompt_messages, expected) in enumerate(items) if position % size == rank]

    model, tokenizer = load_model_and_tokenizer(args)
    input_device = model_input_device(model)
    output_path = part_file(args.output_file, rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for count, (index, prompt_messages, expected) in enumerate(rank_items, start=1):
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
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
            exact = normalize(prediction) == normalize(expected)
            contains = normalize(expected) in normalize(prediction)
            record = {
                "index": index,
                "prompt_messages": prompt_messages,
                "expected": expected,
                "prediction": prediction,
                "normalized_exact_match": exact,
                "expected_substring_match": contains,
                "rank": rank,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[rank {rank}/{size} item {count}/{len(rank_items)} index {index}] exact={exact}", flush=True)
    return output_path


def merge_outputs(args: argparse.Namespace) -> None:
    records = []
    for rank in range(world_size()):
        path = part_file(args.output_file, rank)
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation part file from rank {rank}: {path}")
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda record: record["index"])

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            record.pop("rank", None)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.keep_part_files:
        for rank in range(world_size()):
            part_file(args.output_file, rank).unlink(missing_ok=True)

    total = len(records)
    exact = sum(int(record["normalized_exact_match"]) for record in records)
    contains = sum(int(record["expected_substring_match"]) for record in records)
    print(f"Wrote predictions to {output_path}")
    print(f"Normalized exact match: {exact}/{total} = {exact / total:.4f}")
    print(f"Expected substring match: {contains}/{total} = {contains / total:.4f}")


def main() -> None:
    args = parse_args()
    setup_distributed()
    try:
        if is_main_process():
            print(f"Starting Time-MQA full parallel evaluation: world_size={world_size()}, data_file={args.data_file}")
        prepare_parts_dir(args.output_file)
        distributed_barrier()
        evaluate_rank(args)
        distributed_barrier()
        if is_main_process():
            merge_outputs(args)
        distributed_barrier()
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
