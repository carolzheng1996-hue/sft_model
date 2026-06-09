# Local Qwen2.5-1.5B SFT with TRL

This directory contains a minimal Hugging Face + TRL SFT pipeline for fine-tuning a locally downloaded Qwen2.5 model on datasets that have already been converted to TRL conversational `messages` format.

## Files

- `train_sft.py`: main TRL SFT script for `messages` datasets with LoRA/QLoRA support.
- `train_sft_multigpu_qlora.py`: multi-GPU QLoRA variant for `accelerate launch` or `torchrun`.
- `train_sft_multigpu_qlora_completion_only.py`: multi-GPU QLoRA for single-turn CoT `messages`; uses a local completion-only collator so only assistant replies contribute to loss.
- `train_sft_multigpu_qlora_full_loss.py`: multi-GPU QLoRA full-sequence loss baseline for CoT `messages`.
- `scripts/inspect_dataset.py`: prints dataset splits, columns, features, and examples before training.
- `scripts/eval_exam1_qwen15b_lora_official_parallel.py`: sample-parallel multi-GPU LoRA adapter evaluation on TimeSeriesExam1.
- `requirements.txt`: Python dependencies.
- `configs/qwen25_15b_tsqa_lora.yaml`: Time-MQA/TSQA reference hyperparameters.
- `configs/qwen25_15b_timemqa_local_lora.yaml`: local Time-MQA CSV reference hyperparameters.
- `configs/qwen25_15b_timeseries_exam1_lora.yaml`: TimeSeriesExam1 reference hyperparameters.

## Dataset Format

`train_sft.py` only accepts TRL conversational language modeling data. Each row must contain a `messages` column:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
  ]
}
```

Raw dataset schemas, packed QA strings, and dataset-specific context fields should be handled in a preparation script before training.

## Local Model

The training script only loads the base model from a local directory. It will not download model files during training.

Recommended layout:

```text
../models/Qwen2.5-1.5B/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors or model-*.safetensors
```

Use the default path above, or pass a different local model directory explicitly:

```bash
python train_sft.py --model_name_or_path /path/to/Qwen2.5-1.5B --use_4bit --bf16
```

For `Qwen2.5-1.5B` base model, keep `--chat_template qwen` or rely on the default `auto` fallback. For another chat model, its tokenizer should include a compatible chat template; otherwise set `--chat_template qwen` only if that model uses Qwen-style `<|im_start|>` formatting.

## Install

Use a CUDA Linux environment for practical training. QLoRA uses `bitsandbytes`, which is not reliable on macOS CPU/MPS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Inspect Dataset

Run this first to inspect local or prepared files:

```bash
python scripts/inspect_dataset.py --dataset_name local --data_files /path/to/train.json
```

Inspect the local TimeSeriesExam1 JSON:

```bash
python scripts/inspect_dataset.py \
  --dataset_name local \
  --data_files ../datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json
```

Parquet is also supported:

```bash
python scripts/inspect_dataset.py \
  --dataset_name local \
  --data_files ../datasets/AutonLab/TimeSeriesExam1/data/test-00000-of-00001.parquet
```

Inspect the local Time-MQA CSV example:

```bash
bash scripts/prepare_timemqa_local_data.sh
python scripts/inspect_dataset.py --dataset_name local --data_files data/processed/timemqa_local_train.json
```

## Train with QLoRA

Recommended starting point for a single 16-24 GB NVIDIA GPU, after preparing `data/processed/timemqa_local_train.json`:

```bash
python train_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json \
  --use_4bit \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --num_train_epochs 2 \
  --output_dir outputs/qwen2.5-1.5b-timemqa-local-lora
```

If your GPU does not support bf16, use `--fp16` instead of `--bf16`.

### Multi-GPU QLoRA

GPU count is controlled by the launcher, not by an `SFTTrainer` argument. The multi-GPU script keeps TRL `SFTTrainer` as the training framework and maps each distributed process to its own `LOCAL_RANK` GPU before loading the 4-bit model.

Run the provided wrapper:

```bash
NUM_PROCESSES=4 bash scripts/train_timemqa_local_multigpu_qlora.sh
```

Equivalent direct command:

```bash
accelerate launch --num_processes 4 train_sft_multigpu_qlora.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --num_train_epochs 2 \
  --output_dir outputs/qwen2.5-1.5b-timemqa-local-multigpu-qlora
```

For `N` GPUs, the effective training batch size is:

```text
per_device_train_batch_size * gradient_accumulation_steps * N
```

## Train on train_cot CoT Messages

The prepared CoT file is:

```text
data/processed/train_cot_messages.jsonl
```

There are three train_cot multi-GPU variants:

| Script | Loss scope | Data path | Recommended use |
| --- | --- | --- | --- |
| `train_sft_multigpu_qlora_completion_only.py` | Assistant reply text only | Converts single-turn `messages` to `text`, then masks everything through the assistant marker with a local collator | Default choice for train_cot CoT SFT, especially when prompts are long |
| `train_sft_multigpu_qlora_assistant_only.py` | Assistant regions marked by the chat template | Keeps `messages` and uses TRL `assistant_only_loss=True` with generation masks | Use when you want the native TRL conversational path or multi-turn data |
| `train_sft_multigpu_qlora_full_loss.py` | Full rendered sequence: `system`, `user`, and `assistant` | Converts complete `messages` to `text` and applies no label mask | Baseline/comparison run, not the default CoT SFT setting |

Recommended command for assistant-reply-only loss without `SFTConfig(assistant_only_loss=True)`:

```bash
NUM_PROCESSES=4 bash scripts/train_cot_completion_only_multigpu_qlora.sh
```

This script formats each single-turn `messages` example with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, appends the assistant content, and uses a local completion-only collator compatible with current TRL releases:

```text
CompletionOnlyDataCollator(response_template="<|im_start|>assistant\n")
```

The prompt side may be left-truncated when a sample exceeds `MAX_SEQ_LENGTH`, so the assistant marker and assistant reply are preserved for loss computation.

Full-sequence loss baseline:

```bash
NUM_PROCESSES=4 bash scripts/train_cot_full_loss_multigpu_qlora.sh
```

In this baseline, `system`, `user`, and `assistant` tokens all contribute to causal LM loss. Use it for comparison, not as the default CoT SFT setting.

Common overrides:

```bash
MODEL_NAME_OR_PATH=/data/sft_model/Qwen2.5-3B-Instruct \
DATA_FILES=data/processed/train_cot_messages.jsonl \
OUTPUT_DIR=outputs/qwen2.5-3b-train-cot-completion-only-multigpu-qlora \
NUM_PROCESSES=2 \
MAX_SEQ_LENGTH=4096 \
bash scripts/train_cot_completion_only_multigpu_qlora.sh
```

Quick smoke test:

```bash
python train_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --use_4bit \
  --bf16 \
  --max_train_samples 128 \
  --max_eval_samples 32 \
  --save_steps 50 \
  --eval_steps 50 \
  --output_dir outputs/smoke-test
```

## Train on Local Time-MQA CSV

The local `../timemqa/open_ended_QA.csv` file stores question and answer inside one `QA_list` string column. Keep that special parsing in the data preparation step, not in `train_sft.py`.

First convert it to TRL conversational `messages` JSON:

```bash
bash scripts/prepare_timemqa_local_data.sh
```

This writes:

```text
data/processed/timemqa_local_train.json
```

with one `messages` column. The preparation script keeps Time-MQA metadata such as `application_domain`, `task_type`, and `question_format` inside `user.content`.

Then train:

```bash
bash scripts/train_timemqa_local_qlora.sh
```

Equivalent command:

```bash
python train_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json \
  --use_4bit \
  --bf16 \
  --output_dir outputs/qwen2.5-1.5b-timemqa-local-lora
```

## Train on TimeSeriesExam1

Convert TimeSeriesExam1 to a JSON/JSONL/Parquet file with a `messages` column before training. `train_sft.py` no longer accepts raw TimeSeriesExam1 fields such as `ts1`, `ts2`, or `options`.

## Evaluate TimeSeriesExam1

TimeSeriesExam1 is a single-choice exam with 2-4 options per question. The evaluation scripts below adapt the official TimeSeriesExam GitHub scoring logic for local Qwen models: <https://github.com/moment-timeseries-foundation-model/TimeSeriesExam>. The primary metric is official flexible accuracy: a response is correct when it contains the formatted correct option, such as `B) No autocorrelation`.

Evaluate the untrained base model:

```bash
python scripts/eval_exam1_qwen15b_base_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --max_samples 50
```

Evaluate a LoRA adapter:

```bash
python scripts/eval_exam1_qwen15b_lora_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 50
```

Evaluate a LoRA adapter faster with multiple GPUs:

```bash
NUM_PROCESSES=4 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```

Equivalent direct command:

```bash
accelerate launch --num_processes 4 scripts/eval_exam1_qwen15b_lora_official_parallel.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 0
```

The parallel evaluation script splits selected examples by rank, writes temporary per-rank JSONL files, then rank 0 merges them into:

```text
reports/timeseries_exam1_qwen15b_lora_official_predictions_parallel.jsonl
```

The scripts write JSONL predictions to:

```text
reports/timeseries_exam1_qwen15b_base_official_predictions.jsonl
reports/timeseries_exam1_qwen15b_lora_official_predictions.jsonl
```

They print official flexible accuracy and official strict accuracy. Flexible scoring follows the official repository's default `evaluate_response(..., mode="flexible")` behavior. Use `--max_samples 0` to evaluate all examples.

For a future dataset, write a small preparation script that converts it to TRL conversational `messages` first. See `scripts/prepare_timemqa_local_data.py` for the local Time-MQA CSV example.

## Full Fine-Tuning

Full fine-tuning needs much more VRAM than LoRA/QLoRA:

```bash
python train_sft.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json \
  --no_lora \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --output_dir outputs/qwen2.5-1.5b-timemqa-local-full
```

## Output

Training writes LoRA adapter weights and tokenizer files to `--output_dir`. If you train with LoRA, load the adapter with PEFT or merge it into the base model before deployment.
