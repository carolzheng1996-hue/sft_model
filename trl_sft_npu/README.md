# Local Qwen2.5 SFT with TRL on Ascend 910B2 NPU

This directory is the Ascend NPU migration of `../trl_sft`. The original CUDA
project is left unchanged. The NPU version keeps the same TRL/PEFT training
workflow and dataset format, but removes CUDA-only bitsandbytes QLoRA.

## What Changed

- Added `npu_utils.py` for `torch_npu` import, local-rank device binding, HCCL
  backend selection, and shared dtype/device helpers.
- Training scripts call `setup_npu()` before loading models.
- `--use_4bit` and `--load_4bit` are kept only for CLI compatibility and raise
  a clear error if used.
- Optimizer defaults to `adamw_torch` instead of `paged_adamw_8bit`.
- Multi-process evaluation uses HCCL when NPU is available.
- Shell wrappers no longer pass `--use_4bit`.

## Environment

Install the Ascend software stack first:

1. Ascend driver/firmware for 910B2.
2. CANN toolkit and runtime.
3. Python environment with `torch` and `torch-npu` versions matching CANN.

Then install Python-side dependencies:

```bash
cd trl_sft_npu
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` intentionally does not pin `torch` or `torch-npu`, because
those wheels must match the target CANN/Python environment.

Before running, source the Ascend environment script if your deployment requires
it, for example:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Dataset Format

Training still expects TRL conversational `messages` rows:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
  ]
}
```

Inspect a prepared local dataset:

```bash
python scripts/inspect_dataset.py \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json
```

## Single-NPU LoRA Training

```bash
python train_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --num_train_epochs 2 \
  --output_dir outputs/qwen2.5-1.5b-timemqa-local-lora-npu
```

You can also use the wrapper:

```bash
bash scripts/train_timemqa_local_qlora.sh
```

The wrapper name is preserved for compatibility with the source project, but it
now runs standard LoRA on NPU.

## Multi-NPU Training

Use `accelerate launch`; each process binds to its `LOCAL_RANK` NPU via
`torch.npu.set_device`.

```bash
NUM_PROCESSES=8 bash scripts/train_timemqa_local_multigpu_qlora.sh
```

CoT variants are also available:

```bash
NUM_PROCESSES=8 bash scripts/train_cot_completion_only_multigpu_qlora.sh
NUM_PROCESSES=8 bash scripts/train_cot_assistant_only_multigpu_qlora.sh
NUM_PROCESSES=8 bash scripts/train_cot_full_loss_multigpu_qlora.sh
```

## Evaluation

Single-process:

```bash
python scripts/eval_exam1_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timeseries-exam1-lora \
  --use_adapter
```

Multi-process:

```bash
NUM_PROCESSES=8 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```

## Notes

- Do not pass `--use_4bit` or `--load_4bit` on Ascend; bitsandbytes 4-bit
  quantization is CUDA-specific in this project.
- If bf16 is unavailable in your software stack, use `--fp16` or omit both dtype
  flags to let Transformers choose automatically.
- Keep model paths local; these scripts use `local_files_only=True` for model
  and tokenizer loading.
