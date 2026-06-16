#!/usr/bin/env python3
"""Multi-GPU official-style TimeSeriesExam1 evaluation for a LoRA adapter.

Launch with accelerate or torchrun, for example:

  accelerate launch --num_processes 4 scripts/eval_exam1_qwen15b_lora_official_parallel.py ...

Each process evaluates a disjoint slice of examples. Rank 0 merges the part
files into the final JSONL and prints official flexible/strict accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exam1_official_eval import (
    add_common_args,
    build_user_content,
    configure_chat_template,
    correct_option_letter,
    load_rows,
    official_flexible_correct,
    official_strict_correct,
    require_dir,
)
from npu_utils import (
    current_device_map,
    get_inference_dtype,
    preferred_distributed_backend,
    setup_npu,
    validate_quantization_args,
)


DEFAULT_ADAPTER_PATH = "outputs/qwen2.5-1.5b-timemqa-local-lora"
DEFAULT_OUTPUT_FILE = "reports/timeseries_exam1_qwen15b_lora_official_predictions_parallel.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel official-style TimeSeriesExam1 evaluation for Qwen2.5-1.5B + LoRA."
    )
    add_common_args(parser)
    parser.add_argument("--adapter_name_or_path", default=DEFAULT_ADAPTER_PATH, help="Local LoRA adapter directory.")
    parser.add_argument(
        "--keep_part_files",
        action="store_true",
        help="Keep per-rank JSONL part files after rank 0 merges the final output.",
    )
    parser.set_defaults(output_file=DEFAULT_OUTPUT_FILE)
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
    dist.init_process_group(backend=preferred_distributed_backend())


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


def qlora_device_map() -> None:
    setup_npu(local_rank())
    return current_device_map()


def load_lora_model_and_tokenizer(args: argparse.Namespace):
    require_dir(args.model_name_or_path, "Model")
    require_dir(args.adapter_name_or_path, "Adapter")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
    configure_chat_template(tokenizer, args.model_name_or_path, args.chat_template)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    validate_quantization_args(args.load_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=get_inference_dtype(),
        device_map=qlora_device_map(),
    )
    model = PeftModel.from_pretrained(model, args.adapter_name_or_path, local_files_only=True)
    model.eval()
    return model, tokenizer


def selected_items(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    rows = load_rows(args)
    end = None if args.max_samples == 0 else args.start_index + args.max_samples
    selected = rows[args.start_index : end]
    if not selected:
        raise ValueError("No examples selected. Check --start_index and --max_samples.")
    return [(args.start_index + offset, row) for offset, row in enumerate(selected)]


def part_file(output_file: str, rank: int) -> Path:
    output_path = Path(output_file)
    parts_dir = output_path.parent / f".{output_path.name}.parts"
    return parts_dir / f"rank_{rank:05d}.jsonl"


def prepare_parts_dir(output_file: str) -> None:
    parts_dir = part_file(output_file, 0).parent
    parts_dir.mkdir(parents=True, exist_ok=True)
    if not is_main_process():
        return
    for path in parts_dir.glob("rank_*.jsonl"):
        path.unlink()


def model_input_device(model: Any):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def evaluate_rank(args: argparse.Namespace) -> Path:
    import torch

    items = selected_items(args)
    rank = global_rank()
    size = world_size()
    rank_items = [(index, row) for position, (index, row) in enumerate(items) if position % size == rank]

    model, tokenizer = load_lora_model_and_tokenizer(args)
    output_path = part_file(args.output_file, rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_device = model_input_device(model)
    with output_path.open("w", encoding="utf-8") as out:
        for count, (index, row) in enumerate(rank_items, start=1):
            messages = [{"role": "user", "content": build_user_content(row)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            flexible = official_flexible_correct(row, response)
            strict = official_strict_correct(row, response)
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
                "rank": rank,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[rank {rank}/{size} item {count}/{len(rank_items)} index {index}] "
                f"official_flexible_correct={flexible} official_strict_correct={strict}",
                flush=True,
            )
    return output_path


def merge_outputs(args: argparse.Namespace) -> None:
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for rank in range(world_size()):
        path = part_file(args.output_file, rank)
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation part file from rank {rank}: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))

    records.sort(key=lambda record: record["index"])
    flexible_correct = sum(int(record["official_flexible_correct"]) for record in records)
    strict_correct = sum(int(record["official_strict_correct"]) for record in records)
    total = len(records)

    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            record.pop("rank", None)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.keep_part_files:
        for rank in range(world_size()):
            part_file(args.output_file, rank).unlink(missing_ok=True)

    print(f"Wrote predictions to {output_path}")
    print(f"Official flexible accuracy: {flexible_correct}/{total} = {flexible_correct / total:.4f}")
    print(f"Official strict accuracy: {strict_correct}/{total} = {strict_correct / total:.4f}")


def main() -> None:
    args = parse_args()
    setup_distributed()
    try:
        if is_main_process():
            print(f"Starting parallel evaluation: world_size={world_size()}, output_file={args.output_file}")
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
