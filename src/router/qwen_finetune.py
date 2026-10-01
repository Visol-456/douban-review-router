#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Route B - Qwen3-0.6B QLoRA fine-tune (System Two deep router).

Task: same binary decision as the LSTM baseline, on the same dataset, so the two
routes can be compared head-to-head (accuracy / latency / param count / cost).

STATUS: skeleton / placeholder. Not yet trained on the WSL2 box.
"""
import os
import sys
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import DESENS_DIR  # noqa: E402

BASE_MODEL = os.environ.get("DRR_QWEN_MODEL", "Qwen/Qwen3-0.6B")
TRAIN_CSV = os.path.join(DESENS_DIR, "train.csv")
VALID_CSV = os.path.join(DESENS_DIR, "valid.csv")

# 4-bit QLoRA so 0.6B fits in a 4 GB card (GTX 1050 Ti, Pascal sm_61).
# torch must be a CUDA 12.x build compatible with Pascal (2.4~2.6).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # adjust to the chosen Qwen3 layer names
MAX_LEN = 256
EPOCHS = 3
LR = 2e-4


def load_split(path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            X.append(r["text"])
            y.append(int(r["label"]))
    return X, y


def build_prompt(text):
    return (
        "判断下面这条电影评论是否属于可疑需深挖（水军/反讽/极端情绪/引战/AI生成）。"
        "只回答 true 或 false。\n\n评论：" + text
    )


def main():
    # 1. load Qwen3-0.6B with 4-bit quantization (bitsandbytes)
    # 2. attach LoRA adapters (peft) on the attention projection layers
    # 3. tokenize build_prompt(text) with the Qwen tokenizer
    # 4. train, log loss / accuracy / per-sample inference latency
    # 5. save adapter to outputs/ (gitignored); dump metrics to results/
    raise SystemExit("TODO: implement Qwen3-0.6B QLoRA fine-tune (env not provisioned yet)")


if __name__ == "__main__":
    main()
