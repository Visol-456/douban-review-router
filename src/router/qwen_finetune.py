#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route B - Qwen3-0.6B QLoRA fine-tune (System Two deep router).

Task: same binary decision as the LSTM baseline, on the same dataset, so the two
routes can be compared head-to-head (accuracy / latency / param count / cost).

This is the REAL, trained script that produced the reported results on the WSL2 box
(GTX 1050 Ti, Pascal sm_61). 4-bit QLoRA (nf4, double quant) + LoRA r=16 so 0.6B fits
in a 4 GB card; FP32 because Pascal (sm_61) does not support FP16 fast path.

Results (valid, 418 samples): acc 0.799, F1-macro 0.755, F1(label=1) 0.859.
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUT = os.environ.get("DRR_QLORA_OUT", "outputs/qlora")
os.makedirs(OUT, exist_ok=True)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
DATA_DIR = os.environ.get("DRR_DATA_DIR", "data/desensitized")

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)

def load_df(p):
    df = pd.read_csv(p)[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df

train_df = load_df(os.path.join(DATA_DIR, "train.csv"))
valid_df = load_df(os.path.join(DATA_DIR, "valid.csv"))

# Class weights for imbalance (inverse frequency, normalized)
counts = train_df["label"].value_counts().to_dict()
n = len(train_df)
weights = {c: n / (len(counts) * cnt) for c, cnt in counts.items()}
cw = torch.tensor([weights.get(i, 1.0) for i in range(2)], dtype=torch.float32)
print("train", len(train_df), "counts", counts, "class_weights", cw.tolist(), flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAXLEN = 128
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAXLEN, padding="max_length")

train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)
train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
valid_ds = valid_ds.map(tok, batched=True, remove_columns=["text"])
train_ds = train_ds.with_format("torch", columns=["input_ids", "attention_mask", "label"])
valid_ds = valid_ds.with_format("torch", columns=["input_ids", "attention_mask", "label"])

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.label2id = {str(k): int(k) for k in (0, 1)}
model.config.id2label = {int(k): str(k) for k in (0, 1)}

# Frozen-embedding-friendly prep
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="SEQ_CLS",
)
model = get_peft_model(model, lora)
print("trainable params:", model.print_trainable_parameters() or "", flush=True)

# Weighted loss to handle class imbalance
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = nn.CrossEntropyLoss(weight=cw.to(logits.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(ep: EvalPrediction):
    preds = np.argmax(ep.predictions, axis=-1)
    labels = ep.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_label1": f1_score(labels, preds, labels=[1], average=None, zero_division=0)[0],
        "precision_label1": precision_score(labels, preds, labels=[1], average=None, zero_division=0)[0],
        "recall_label1": recall_score(labels, preds, labels=[1], average=None, zero_division=0)[0],
    }

args = TrainingArguments(
    output_dir=os.path.join(OUT, "checkpoints"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_label1",
    greater_is_better=True,
    report_to=[],
    remove_unused_columns=False,
    fp16=False,
    dataloader_pin_memory=False,
    seed=42,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
)

t0 = time.time()
trainer.train()
print("TRAIN_TIME_SEC", round(time.time() - t0, 1), flush=True)

metrics = trainer.evaluate()
print("EVAL_METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)

# Best checkpoint metrics across epochs
hist = []
for log in trainer.state.log_history:
    if "eval_accuracy" in log:
        hist.append({k: log[k] for k in log if k in
                     ("epoch", "eval_accuracy", "eval_f1_macro", "eval_f1_label1",
                      "eval_precision_label1", "eval_recall_label1", "eval_loss")})
print("EPOCH_METRICS", json.dumps(hist, ensure_ascii=False), flush=True)

# Save final adapter
final = os.path.join(OUT, "adapter_final")
trainer.model.save_pretrained(final)
tokenizer.save_pretrained(final)
print("ADAPTER_SAVED", final, flush=True)

# Loss curve data from log_history
loss_hist = [{"step": l.get("step"), "loss": l.get("loss"), "epoch": l.get("epoch"),
              "eval_loss": l.get("eval_loss"), "eval_f1_label1": l.get("eval_f1_label1"),
              "eval_accuracy": l.get("eval_accuracy")} for l in trainer.state.log_history]
with open(os.path.join(OUT, "training_log.json"), "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics, "epoch_metrics": hist,
               "class_counts": counts, "class_weights": weights,
               "train_time_sec": round(time.time() - t0, 1),
               "log_history": loss_hist}, f, ensure_ascii=False, indent=2)
print("LOG_SAVED", os.path.join(OUT, "training_log.json"), flush=True)

# Save plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    steps = [l["step"] for l in loss_hist if l.get("loss") is not None]
    loss = [l["loss"] for l in loss_hist if l.get("loss") is not None]
    ax1.plot(steps, loss, label="train loss", color="tab:blue")
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss", color="tab:blue")
    ev = [l for l in hist]
    if ev:
        ax2 = ax1.twinx()
        ax2.plot([e["epoch"] * (len(steps)) if steps else 0 for e in ev], [e["eval_f1_label1"] for e in ev],
                 label="val F1(label=1)", color="tab:orange", marker="o")
        ax2.set_ylabel("val F1(label=1)", color="tab:orange")
    ax1.set_title("Qwen3-0.6B QLoRA training loss & val F1")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=120)
    print("CURVE_SAVED", os.path.join(OUT, "training_curve.png"), flush=True)
except Exception as e:
    print("CURVE_FAIL", repr(e), flush=True)

print("DONE", flush=True)
