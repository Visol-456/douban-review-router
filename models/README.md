# 模型权重（Releases）

本目录说明课题《商品评价的观点自动提取》最终发布的模型权重。**权重只随 GitHub Release
（v1.0.0 附件）发布，不入 Git 树**（`.gitignore` 排除 `*.pt/*.safetensors/*.bin`），
训练过程中的中间 checkpoint 亦不保留。

## 1. `bilstm_model.pt` — LSTM 情感/观点分类基线

- 大小：约 9.5 MB
- 结构：词嵌入 + BiLSTM(+注意力) + 线性分类头，作为 router 的轻量基线
- 训练/评测代码：`src/router/lstm_baseline.py`
- 评测图：`figures/lstm_train_curve.png`、`figures/lstm_confusion_matrix.png`

## 2. `qwen_adapter_final/` — Qwen3-0.6B QLoRA adapter

- 基座模型：`Qwen/Qwen3-0.6B`
- 方法：QLoRA（4-bit 量化 + LoRA adapter），最终只发布 adapter 权重
- 关键文件：
  - `adapter_model.safetensors`（约 40 MB）— LoRA adapter 权重
  - `adapter_config.json` — LoRA / PEFT 配置（r、alpha、target_modules 等）
  - `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` — 分词器与对话模板
- 训练代码：`src/router/qwen_finetune.py`
- 评测图：`figures/qwen_training_curve.png`

## 使用方式

```python
# LSTM 基线（从 Release 下载 bilstm_model.pt 放到 models/ 下）
import torch
state = torch.load("models/bilstm_model.pt", map_location="cpu")

# Qwen3-0.6B QLoRA adapter（从 Release 下载 qwen_adapter_final/ 放到 models/ 下；需先安装 peft/transformers）
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "models/qwen_adapter_final")
tok = AutoTokenizer.from_pretrained("models/qwen_adapter_final")
```

> 权重为课题实验产物，仅供复现与研究使用。
