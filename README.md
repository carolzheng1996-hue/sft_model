# Time-Series SFT: TRL vs LLaMA-Factory

This workspace contains two independent SFT implementations for fine-tuning a locally downloaded Qwen2.5 model on time-series QA datasets. It includes examples for `Time-MQA/TSQA` and the local `TimeSeriesExam1` dataset.

## Directory Layout

```text
.
├── trl_sft/              # Hugging Face TRL implementation
├── llamafactory_sft/     # LLaMA-Factory implementation
└── pdf_page_image/       # Non-training artifact from prior PDF conversion work
```

## Recommended Experiment Workflow

For the workflow "test Qwen2.5-1.5B on TimeSeriesExam1, fine-tune on Time-MQA, then evaluate again on TimeSeriesExam1", see:

```text
EXPERIMENT_WORKFLOW.md
```

## Implementations

### TRL

See [trl_sft/README.md](trl_sft/README.md).

Key entry points:

```bash
cd trl_sft
pip install -r requirements.txt
bash scripts/prepare_timemqa_local_data.sh
bash scripts/train_timemqa_local_qlora.sh
```

Multi-GPU QLoRA training and parallel adapter evaluation are also available:

```bash
cd trl_sft
NUM_PROCESSES=4 bash scripts/train_timemqa_local_multigpu_qlora.sh
NUM_PROCESSES=4 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```

### LLaMA-Factory

See [llamafactory_sft/README.md](llamafactory_sft/README.md).

Key entry points:

```bash
cd llamafactory_sft
pip install -r requirements.txt
bash scripts/prepare_exam1_data.sh
bash scripts/train_exam1_qlora.sh
bash scripts/infer_qlora.sh --example_file data/timeseries_exam1_alpaca.json --example_index 0
```

## Shared Dataset Access

`Time-MQA/TSQA` is a gated Hugging Face dataset. If you load it directly from Hugging Face instead of local files, accept access on the dataset page, then export `HF_TOKEN` in your shell.

Dataset page: <https://huggingface.co/datasets/Time-MQA/TSQA>

## Local Exam Dataset

The TimeSeriesExam1 dataset has been downloaded locally:

```text
datasets/AutonLab/TimeSeriesExam1/data/test-00000-of-00001.parquet
datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json
```

For LLaMA-Factory, it has also been converted to Alpaca format and registered as `timeseries_exam1_alpaca`:

```text
llamafactory_sft/data/timeseries_exam1_alpaca.json
```

Regenerate that Alpaca file with:

```bash
cd llamafactory_sft
bash scripts/prepare_exam1_data.sh
```

Test a local base model on Exam1 before fine-tuning:

```bash
cd trl_sft
python scripts/eval_exam1_qwen15b_base_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --max_samples 20
```

This uses the official TimeSeriesExam flexible scoring rule from <https://github.com/moment-timeseries-foundation-model/TimeSeriesExam>: the main metric is `official_flexible_accuracy`, which checks whether the model response contains the correct formatted option, such as `B) No autocorrelation`.

After LoRA training, evaluate the adapter with the same official-style metric:

```bash
cd trl_sft
python scripts/eval_exam1_qwen15b_lora_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 50
```

For faster full-dataset evaluation on multiple GPUs, use sample-level parallel evaluation:

```bash
cd trl_sft
NUM_PROCESSES=4 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```

TRL training now requires a prepared conversational `messages` dataset. `trl_sft/train_sft.py` no longer trains directly from raw TimeSeriesExam1 columns such as `ts1`, `ts2`, or `options`; convert TimeSeriesExam1 to a file with a `messages` column before using the TRL trainer.

Train on this dataset with LLaMA-Factory:

```bash
cd llamafactory_sft
bash scripts/train_exam1_qlora.sh
```

## Local Base Model

Download the base model before training. The default project layout expects:

```text
models/Qwen2.5-1.5B
```

The TRL script always loads model/tokenizer files with local-only mode. LLaMA-Factory configs use the same local path through `model_name_or_path`.

Check that a downloaded local model can be loaded before training:

```bash
python scripts/check_local_model.py --model_name_or_path models/Qwen2.5-1.5B
```

For another local model directory:

```bash
python scripts/check_local_model.py --model_name_or_path /path/to/local/model
```

Minimal local-model workflow:

```bash
# 1. Put the downloaded model here.
mkdir -p models
# expected: models/Qwen2.5-1.5B/config.json, tokenizer files, and safetensors weights

# 2. Train with TRL on the prepared local Time-MQA example.
cd trl_sft
bash scripts/prepare_timemqa_local_data.sh
bash scripts/train_timemqa_local_qlora.sh
# optional multi-GPU variant:
# NUM_PROCESSES=4 bash scripts/train_timemqa_local_multigpu_qlora.sh

# 3. Or train with LLaMA-Factory on TimeSeriesExam1.
cd ../llamafactory_sft
bash scripts/prepare_exam1_data.sh
bash scripts/train_exam1_qlora.sh
bash scripts/infer_qlora.sh --example_file data/timeseries_exam1_alpaca.json --example_index 0
```

To use a different model directory with TRL, pass `--model_name_or_path /path/to/model`. To use a different model directory with LLaMA-Factory, edit `model_name_or_path` in the YAML config. For non-Qwen models, also update the chat template/template setting to match that model.

## Comparison Notes

When comparing the two frameworks, keep these aligned:

- Base model path: `models/Qwen2.5-1.5B`
- Max sequence length / cutoff length: `2048`
- LoRA rank / alpha / dropout: `16 / 32 / 0.05`
- Effective batch size: `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`
- Learning rate: `2e-4`
- Epochs: `2.0`
- Validation split: `0.02`
