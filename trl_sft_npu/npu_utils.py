"""Ascend NPU helpers for TRL/Transformers scripts."""

from __future__ import annotations

import os

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))


def is_npu_available() -> bool:
    return torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()


def setup_npu(local_rank: int | None = None) -> None:
    """Bind this process to its Ascend device when torch-npu is available."""
    if not is_npu_available():
        return
    rank = get_local_rank() if local_rank is None else local_rank
    if rank >= 0:
        torch.npu.set_device(rank)


def preferred_distributed_backend() -> str:
    if is_npu_available():
        return "hccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def get_torch_dtype(use_bf16: bool, use_fp16: bool):
    if use_bf16:
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return "auto"


def get_inference_dtype():
    if is_npu_available() or torch.cuda.is_available():
        return torch.float16
    return torch.float32


def current_device_map() -> None:
    """Let Trainer/Accelerate place the model on NPU after torch.npu.set_device."""
    return None


def training_optim(_use_4bit: bool) -> str:
    return "adamw_torch"


def validate_quantization_args(use_4bit: bool) -> None:
    if use_4bit:
        raise ValueError(
            "Ascend 910B2 migration disables bitsandbytes 4-bit QLoRA. "
            "Use standard LoRA/full fine-tuning by omitting --use_4bit."
        )
