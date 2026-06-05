# Qwen2.5-1.5B Time-Series SFT Experiment Workflow

本文档说明一个完整实验流程：

1. 先测试本地 Qwen2.5-1.5B 基座模型在 `TimeSeriesExam1` 数据集上的回复效果。
2. 使用 `Time-MQA/TSQA` 数据集对 Qwen2.5-1.5B 做 SFT。
3. 使用微调后的模型在一批 `TimeSeriesExam1` 样本上做评测。

本文默认模型本地路径为：

```text
models/Qwen2.5-1.5B
```

如果你的模型目录不同，把命令里的 `models/Qwen2.5-1.5B` 或 `../models/Qwen2.5-1.5B` 替换成你的本地路径。

## 目录约定

从仓库根目录看：

```text
.
├── models/Qwen2.5-1.5B                         # 本地基座模型
├── datasets/AutonLab/TimeSeriesExam1/          # 本地 TimeSeriesExam1 数据
├── trl_sft/                                    # TRL 训练与评测
├── llamafactory_sft/                           # LLaMA-Factory 训练与评测
└── scripts/check_local_model.py                # 通用本地模型检查脚本
```

`TimeSeriesExam1` 本地数据应包含：

```text
datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json
datasets/AutonLab/TimeSeriesExam1/data/test-00000-of-00001.parquet
```

## 0. 检查本地模型是否可用

先确认本地模型能被 Transformers 正常加载并生成文本：

```bash
cd /Users/monychen/Documents/sft
python scripts/check_local_model.py --model_name_or_path models/Qwen2.5-1.5B
```

成功时会看到：

```text
Tokenizer loaded...
Model loaded...
OK: local model can be loaded and used for generation.
```

如果模型路径不同：

```bash
python scripts/check_local_model.py --model_name_or_path /path/to/local/qwen2.5-1.5b
```

## 1. 测试基座模型在 TimeSeriesExam1 上的效果

推荐使用 TRL 目录下适配官方 TimeSeriesExam 评测逻辑的脚本：

```text
trl_sft/scripts/eval_exam1_qwen15b_base_official.py
```

它会读取本地原始 Exam1 JSON：

```text
datasets/AutonLab/TimeSeriesExam1/timeseries_exam1_test.json
```

并输出每条样本的：

```text
expected answer
model prediction
answer_option_letter
official_flexible_correct
official_strict_correct
```

运行：

```bash
cd /Users/monychen/Documents/sft/trl_sft

python scripts/eval_exam1_qwen15b_base_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --max_samples 20
```

输出文件：

```text
trl_sft/reports/timeseries_exam1_qwen15b_base_official_predictions.jsonl
```

### 1.1 当前准确率如何计算

`eval_exam1_qwen15b_base_official.py` 使用官方 GitHub 仓库 `evaluate/evaluate.py` 的默认 flexible 规则作为主指标：<https://github.com/moment-timeseries-foundation-model/TimeSeriesExam>。官方逻辑等价于检查模型回复中是否包含：

```text
<正确选项字母>) <标准答案文本>
```

例如标准答案是 B 选项 `No autocorrelation`，回复中包含下面文本就算正确：

```text
B) No autocorrelation
```

脚本也输出官方 strict 规则作为辅助指标：只检查最后一行是否包含标准答案文本。

如果要测试全部 746 条：

```bash
python scripts/eval_exam1_qwen15b_base_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --max_samples 0
```

建议先用 `--max_samples 20` 跑通，再扩大样本量。

## 2. 用 Time-MQA/TSQA 微调 Qwen2.5-1.5B

推荐先使用 QLoRA，显存占用更低。你可以选择 TRL 或 LLaMA-Factory。两者都使用本地模型，不会训练时下载模型。

### 2.0 Time-MQA 是否需要提前格式转换

使用 TRL 微调时，`Time-MQA/TSQA` 不需要转换成 Alpaca JSON，但必须先转换成 TRL conversational `messages` JSON。原始数据字段解析不放在 `train_sft.py` 中。

`trl_sft/train_sft.py` 只接收固定格式输入：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

核心流程是：

```text
原始 Time-MQA/TSQA 数据
        ↓ preparation script
标准 messages JSON
        ↓ train_sft.py 校验 messages
        ↓ SFTTrainer
开始训练
```

LLaMA-Factory 不同。LLaMA-Factory 需要先把数据转成它能识别的 Alpaca/sharegpt 格式，并在 `data/dataset_info.json` 里注册。因此 LLaMA-Factory 路径需要：

```bash
cd llamafactory_sft
bash scripts/prepare_data.sh
```

生成：

```text
llamafactory_sft/data/timemqa_tsqa_alpaca.json
```

### 方案 A：TRL 微调

进入 TRL 目录：

```bash
cd /Users/monychen/Documents/sft/trl_sft
```

如果你要从 Hugging Face 读取完整 `Time-MQA/TSQA` 并编写转换脚本，需要先在 shell 里设置：

```bash
export HF_TOKEN=hf_xxx
```

如果你已经把 `Time-MQA/TSQA` 下载到本地，转换脚本可以直接读取本地文件，不需要 `HF_TOKEN`。

当前仓库中已有一份本地 Time-MQA 示例 CSV：

```text
timemqa/open_ended_QA.csv
```

这个 CSV 的 `question` 和 `answer` 被放在同一个 `QA_list` 字符串列里。这个特殊解析不放在 `train_sft.py` 中，而是先用单独的数据准备脚本转成 TRL conversational `messages` JSON：

```bash
bash scripts/prepare_timemqa_local_data.sh
```

输出文件：

```text
trl_sft/data/processed/timemqa_local_train.json
```

标准列：

```text
messages
```

其中 `application_domain`、`task_type`、`question_format` 会被放进 `user.content` 的 `Context:` 部分。可以检查转换后的标准数据：

```bash
python scripts/inspect_dataset.py \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json
```

如果 GPU 不支持 bf16，改用：

```bash
--fp16
```

训练输出：

```text
trl_sft/outputs/qwen2.5-1.5b-timemqa-local-lora
```

这是 LoRA adapter，不会覆盖原始基座模型。

使用仓库中的本地 Time-MQA CSV，运行：

```bash
bash scripts/prepare_timemqa_local_data.sh
bash scripts/train_timemqa_local_qlora.sh
```

训练输出：

```text
trl_sft/outputs/qwen2.5-1.5b-timemqa-local-lora
```

如果要使用多卡 QLoRA，使用 `accelerate launch` 包装的多卡脚本。GPU 数量由 `NUM_PROCESSES` 控制，不是 `SFTTrainer` 的入参：

```bash
NUM_PROCESSES=4 bash scripts/train_timemqa_local_multigpu_qlora.sh
```

等价直接命令：

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

多卡训练输出仍然是 LoRA adapter，不需要因为多卡训练而手动合并每张卡的参数。评测时可以继续加载本地基座模型和 LoRA adapter。

### 2.1 使用 train_cot messages 数据做 assistant-only loss SFT

如果使用已经转换好的 CoT 数据：

```text
trl_sft/data/processed/train_cot_messages.jsonl
```

推荐使用 assistant-only loss 版本脚本训练。这样 `system` 和 `user` 部分只作为上下文输入，不参与 loss；loss 只计算在 `assistant.content` 上，也就是 `<think>...</think>` CoT 和最终答案部分。

单卡训练用：

```bash
cd trl_sft
python train_sft_assistant_only.py \
  --dataset_name local \
  --data_files data/processed/train_cot_messages.jsonl
```

多卡训练用：

```bash
cd trl_sft
accelerate launch --num_processes 4 train_sft_multigpu_qlora_assistant_only.py \
  --dataset_name local \
  --data_files data/processed/train_cot_messages.jsonl
```

### 2.2 TRL SFT 直接执行命令

如果只跑 TRL 路线，可以从仓库根目录按下面顺序执行：

```bash
cd /Users/monychen/Documents/sft/trl_sft

# 1. 把本地 Time-MQA CSV 转成 TRL messages 格式
bash scripts/prepare_timemqa_local_data.sh

# 2. 检查转换后的训练数据
python scripts/inspect_dataset.py \
  --dataset_name local \
  --data_files data/processed/timemqa_local_train.json

# 3. 用 QLoRA 做 TRL SFT
bash scripts/train_timemqa_local_qlora.sh

# 3b. 可选：多卡 QLoRA
# NUM_PROCESSES=4 bash scripts/train_timemqa_local_multigpu_qlora.sh

# 4. 用官方 TimeSeriesExam 评测规则测试 LoRA adapter，单进程
python scripts/eval_exam1_qwen15b_lora_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 50

# 4b. 可选：多卡并行评测，按样本切分到多张 GPU
# NUM_PROCESSES=4 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```

如果你的 GPU 不支持 bf16，需要编辑：

```text
trl_sft/scripts/train_timemqa_local_qlora.sh
```

把 `--bf16` 改成 `--fp16`。

### 方案 B：LLaMA-Factory 微调

进入 LLaMA-Factory 目录：

```bash
cd /Users/monychen/Documents/sft/llamafactory_sft
```

准备 Time-MQA Alpaca 数据：

```bash
bash scripts/prepare_data.sh
```

训练 QLoRA：

```bash
bash scripts/train_qlora.sh
```

训练输出：

```text
llamafactory_sft/saves/qwen2.5-1.5b/timemqa/qlora-sft
```

## 3. 用微调后的模型测试 TimeSeriesExam1

推荐继续使用 TRL 目录下适配官方 scoring 的评测脚本：

```text
trl_sft/scripts/eval_exam1_qwen15b_lora_official.py
```

它可以加载：

```text
本地基座模型 + LoRA adapter
```

并在原始 `TimeSeriesExam1` JSON 上生成预测。

### 3.1 测试 TRL 微调后的 adapter

如果第 2 步使用的是 TRL，并且输出目录是：

```text
trl_sft/outputs/qwen2.5-1.5b-timemqa-local-lora
```

运行：

```bash
cd /Users/monychen/Documents/sft/trl_sft

python scripts/eval_exam1_qwen15b_lora_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 50
```

输出：

```text
trl_sft/reports/timeseries_exam1_qwen15b_lora_official_predictions.jsonl
```

### 3.1.1 微调后评测的准确率计算

`eval_exam1_qwen15b_lora_official.py` 和第 1 步的基座模型评测脚本一样，使用官方 flexible accuracy 作为主指标：

```text
response contains "<正确选项字母>) <标准答案文本>"
```

它会输出：

```text
expected answer
model response
official_flexible_correct
official_strict_correct
Official flexible accuracy: correct/total
```

比较实验结果时，优先看基座模型和 LoRA 模型在相同样本范围上的 official flexible accuracy。

测试全部样本：

```bash
python scripts/eval_exam1_qwen15b_lora_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 0
```

如果要提高评测速度，可以使用多卡并行评测脚本。它会把选中的样本按 rank 切分，每张卡独立生成预测，最后由 rank 0 合并结果：

```bash
NUM_PROCESSES=4 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```

等价直接命令：

```bash
accelerate launch --num_processes 4 scripts/eval_exam1_qwen15b_lora_official_parallel.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora \
  --max_samples 0
```

输出：

```text
trl_sft/reports/timeseries_exam1_qwen15b_lora_official_predictions_parallel.jsonl
```

### 3.2 测试 LLaMA-Factory 微调后的 adapter

如果第 2 步使用的是 LLaMA-Factory，adapter 路径通常是：

```text
llamafactory_sft/saves/qwen2.5-1.5b/timemqa/qlora-sft
```

可以继续用 TRL 的评测脚本加载这个 PEFT adapter：

```bash
cd /Users/monychen/Documents/sft/trl_sft

python scripts/eval_exam1_qwen15b_lora_official.py \
  --model_name_or_path ../models/Qwen2.5-1.5B \
  --adapter_name_or_path ../llamafactory_sft/saves/qwen2.5-1.5b/timemqa/qlora-sft \
  --max_samples 50
```

如果你已经把 LoRA 合并成 merged model，则可以用基座模型评测入口直接指向 merged model：

```bash
python scripts/eval_exam1_qwen15b_base_official.py \
  --model_name_or_path ../llamafactory_sft/saves/qwen2.5-1.5b/timemqa/merged \
  --max_samples 50
```

## 4. 建议的实验记录规范

每次实验建议记录：

```text
base_model: models/Qwen2.5-1.5B
training_dataset: Time-MQA/TSQA
eval_dataset: TimeSeriesExam1
framework: TRL or LLaMA-Factory
method: QLoRA or LoRA
max_seq_length/cutoff_len: 2048
lora_rank/lora_alpha/lora_dropout: 16/32/0.05
learning_rate: 2e-4
epochs: 2
adapter_path: ...
prediction_file: ...
official_flexible_accuracy: ...
official_strict_accuracy: ...
```

建议按这个顺序比较：

1. 基座模型在 TimeSeriesExam1 上的 official flexible accuracy。
2. 使用 Time-MQA 微调后的模型在 TimeSeriesExam1 上的 official flexible accuracy。
3. 如果效果不好，再调整 `max_seq_length`、上下文字段格式、LoRA 参数或模型规模。

## 5. 常用命令汇总

检查模型：

```bash
python scripts/check_local_model.py --model_name_or_path models/Qwen2.5-1.5B
```

基座模型测试 Exam1：

```bash
cd trl_sft
python scripts/eval_exam1_qwen15b_base_official.py --model_name_or_path ../models/Qwen2.5-1.5B --max_samples 20
```

TRL 使用 Time-MQA 微调：

```bash
cd trl_sft
bash scripts/prepare_timemqa_local_data.sh
bash scripts/train_timemqa_local_qlora.sh
```

TRL 多卡 QLoRA 微调：

```bash
cd trl_sft
bash scripts/prepare_timemqa_local_data.sh
NUM_PROCESSES=4 bash scripts/train_timemqa_local_multigpu_qlora.sh
```

微调后测试 Exam1：

```bash
cd trl_sft
python scripts/eval_exam1_qwen15b_lora_official.py --adapter_name_or_path outputs/qwen2.5-1.5b-timemqa-local-lora --max_samples 50
```

微调后多卡并行测试 Exam1：

```bash
cd trl_sft
NUM_PROCESSES=4 bash scripts/eval_exam1_qwen15b_lora_official_parallel.sh
```
