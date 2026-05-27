# LLaMA-Factory SFT: Local Qwen2.5-1.5B on Time-Series QA

This directory contains the LLaMA-Factory implementation for SFT on Time-MQA/TSQA and TimeSeriesExam1. The TRL implementation lives in `../trl_sft/`.

It follows the current LLaMA-Factory workflow:

- Convert custom data to Alpaca format.
- Register the dataset in `data/dataset_info.json`.
- Train with `llamafactory-cli train <config.yaml>`.
- Use `template: qwen` for Qwen2.5 chat models.

References:

- LLaMA-Factory SFT docs: <https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html>
- Qwen LLaMA-Factory docs: <https://qwen.readthedocs.io/en/v3.0/training/llama_factory.html>

## Files

- `prepare_timemqa_data.py`: converts Time-MQA/TSQA into LLaMA-Factory Alpaca JSON.
- `scripts/prepare_exam1_data.sh`: regenerates the TimeSeriesExam1 Alpaca JSON from the local downloaded data.
- `data/dataset_info.json`: registers `timemqa_tsqa_alpaca` and `timeseries_exam1_alpaca`.
- `configs/qwen25_15b_timemqa_qlora_sft.yaml`: QLoRA SFT config.
- `configs/qwen25_15b_timemqa_lora_sft.yaml`: LoRA SFT config without 4-bit loading.
- `configs/qwen25_15b_timemqa_export.yaml`: LoRA merge/export config.
- `configs/*_chat.yaml`: LLaMA-Factory native chat configs for adapter or merged-model inference.
- `scripts/infer.py`: local inference script for adapter or merged-model testing.
- `scripts/*.sh`: reproducible data preparation, smoke test, train, inference, and export commands.

## Install

Use a CUDA Linux environment for practical training. `bitsandbytes` QLoRA is not a good fit for macOS CPU/MPS.

```bash
cd llamafactory_sft
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If your CUDA environment needs a specific PyTorch wheel, install PyTorch first from the official PyTorch selector, then install this requirements file.

## Dataset Access

`Time-MQA/TSQA` is gated. If you have already downloaded the dataset and prepare from local files, no Hugging Face token is needed.

If you prepare data directly from Hugging Face:

1. Open <https://huggingface.co/datasets/Time-MQA/TSQA>.
2. Accept the dataset access terms.
3. Export your token in the shell:

```bash
export HF_TOKEN=hf_xxx
```

## TimeSeriesExam1 Dataset

The AutonLab TimeSeriesExam1 dataset is stored locally at:

```text
../datasets/AutonLab/TimeSeriesExam1/data/test-00000-of-00001.parquet
../datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json
```

It has been converted to LLaMA-Factory Alpaca format:

```text
data/timeseries_exam1_alpaca.json
```

Regenerate this file from the local downloaded JSON:

```bash
bash scripts/prepare_exam1_data.sh
```

and registered in `data/dataset_info.json` as:

```text
timeseries_exam1_alpaca
```

Before fine-tuning, test the local base model on Exam1:

```bash
bash scripts/eval_exam1_base_model.sh \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --max_samples 20
```

The script writes predictions to:

```text
reports/timeseries_exam1_base_predictions.jsonl
```

Train QLoRA on TimeSeriesExam1:

```bash
bash scripts/train_exam1_qlora.sh
```

Train non-quantized LoRA on TimeSeriesExam1:

```bash
bash scripts/train_exam1_lora.sh
```

## Local Model

Training and export configs expect the base model to already exist on disk. The default path is:

```text
../models/Qwen2.5-1.5B
```

The directory should contain the downloaded Hugging Face model files, for example `config.json`, tokenizer files, and `model.safetensors` or sharded `model-*.safetensors`.

To use a different local model directory, edit `model_name_or_path` in the YAML config. Despite the parameter name, use a local directory here.

## Prepare Data

Full conversion:

```bash
bash scripts/prepare_data.sh
```

Smoke conversion with 128 examples:

```bash
bash scripts/smoke_prepare_data.sh
```

Output:

```text
data/timemqa_tsqa_alpaca.json
```

Each record uses Alpaca fields:

```json
{
  "instruction": "question text",
  "input": "serialized time-series context and metadata",
  "output": "answer text",
  "system": "time-series QA system prompt",
  "history": []
}
```

The converter auto-detects question, answer, and context fields. If it fails, inspect the printed columns and edit `QUESTION_KEYS`, `ANSWER_KEYS`, or `CONTEXT_KEYS` in `prepare_timemqa_data.py`.

## Train

QLoRA, recommended starting point for one 16-24 GB NVIDIA GPU:

```bash
bash scripts/train_qlora.sh
```

Quick smoke train:

```bash
bash scripts/smoke_train_qlora.sh
```

LoRA without 4-bit quantized loading:

```bash
bash scripts/train_lora.sh
```

If your GPU does not support bf16, change `bf16: true` to `fp16: true` in the YAML and remove or set `bf16: false`.

## Test Inference

LLaMA-Factory can load the local base model and LoRA adapter directly through its own CLI.

Interactive chat with the QLoRA adapter:

```bash
bash scripts/chat_qlora.sh
```

Interactive chat with the non-QLoRA LoRA adapter:

```bash
bash scripts/chat_lora.sh
```

These call:

```bash
llamafactory-cli chat configs/qwen25_15b_timemqa_qlora_chat.yaml
```

The chat config points to:

```text
base model: ../models/Qwen2.5-1.5B
adapter:    saves/qwen2.5-1.5b/timemqa/qlora-sft
```

For repeatable one-shot tests or checking one converted sample against its expected answer, use the standalone inference script:

```bash
bash scripts/infer_qlora.sh \
  --instruction "What is the answer to this time-series question?" \
  --input "Paste the relevant time-series context here."
```

You can also test against one converted Alpaca example:

```bash
bash scripts/infer_qlora.sh \
  --example_file data/timemqa_tsqa_alpaca.json \
  --example_index 0
```

By default this loads:

```text
base model: ../models/Qwen2.5-1.5B
adapter:    saves/qwen2.5-1.5b/timemqa/qlora-sft
```

If you trained the non-QLoRA LoRA config, pass its adapter path:

```bash
bash scripts/infer_qlora.sh \
  --adapter_name_or_path saves/qwen2.5-1.5b/timemqa/lora-sft \
  --example_file data/timemqa_tsqa_alpaca.json \
  --example_index 0
```

## Merge LoRA Adapter

After QLoRA/LoRA training, merge the adapter into the base model:

```bash
bash scripts/export_merged.sh
```

Default merged output:

```text
saves/qwen2.5-1.5b/timemqa/merged
```

Then test the merged model without loading an adapter:

```bash
bash scripts/infer_merged.sh \
  --example_file data/timemqa_tsqa_alpaca.json \
  --example_index 0
```

Or use LLaMA-Factory native chat with the merged model:

```bash
bash scripts/chat_merged.sh
```

## Comparison With TRL Setup

Keep these parameters aligned with `../trl_sft/train_sft.py` when comparing frameworks:

- base model path: `../models/Qwen2.5-1.5B`
- effective batch size: `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`
- cutoff/max sequence length: `2048`
- LoRA rank/alpha/dropout: `16 / 32 / 0.05`
- epochs: `2.0`
- learning rate: `2e-4`
- scheduler: `cosine`
- validation split: `0.02`

The TRL trainer expects pre-converted conversational `messages` data. The Alpaca conversion and `QUESTION_KEYS` / `ANSWER_KEYS` / `CONTEXT_KEYS` notes above apply only to this LLaMA-Factory data preparation path.
