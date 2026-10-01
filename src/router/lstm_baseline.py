#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route A - LSTM baseline (System One fast router).

Task: given a movie review, predict whether it is "可疑需深挖" (label 1) or plain (0).
Pipeline: jieba tokenize -> pretrained word-vector vocab -> embedding -> BiLSTM -> 2-class head.

This is the REAL, trained script (v2) that produced the reported results on the WSL2 box
(GTX 1050 Ti, Pascal sm_61). v2 fixes double imbalance correction, adds weight decay +
a lower LR, and supports an optional frozen embedding.

Results (valid, 418 samples): acc 0.7775, P 0.8419, R 0.8390, F1 0.8405, macro-F1 0.7364, AUC 0.7815.
Word-vector coverage: 73.0% (vocab 6712).
"""
import os, json, random, time
import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors

SEED = 42
DATA_DIR = os.environ.get("DRR_DATA_DIR", "data/desensitized")
VEC_PATH = os.path.join(DATA_DIR, "fasttext_zh.vec")
SUF = "_v2"
MAX_LEN = 200
MIN_FREQ = 2
EMB_DIM = 300
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH = 32
EPOCHS = 60
LR = 5e-4
WD = 1e-4
PATIENCE = 10
FREEZE_EMB = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("device:", DEVICE, "| ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).dropna(subset=["text"])
valid_df = pd.read_csv(os.path.join(DATA_DIR, "valid.csv")).dropna(subset=["text"])
train_df["text"] = train_df["text"].astype(str); valid_df["text"] = valid_df["text"].astype(str)
print(f"train={len(train_df)} valid={len(valid_df)}")

def tokenize(t): return [w for w in jieba.cut(t) if w.strip()]
from collections import Counter
cnt = Counter()
for t in train_df["text"]: cnt.update(tokenize(t))
words = [w for w, c in cnt.most_common() if c >= MIN_FREQ]
word2idx = {"<pad>": 0, "<unk>": 1}
for w in words: word2idx[w] = len(word2idx)
VOCAB = len(word2idx)
print("vocab", VOCAB)

kv = KeyedVectors.load_word2vec_format(VEC_PATH, binary=False)
emb = np.random.normal(0, 0.1, (VOCAB, EMB_DIM)).astype(np.float32); emb[0] = 0.0
hits = 0
for w, i in word2idx.items():
    if w in kv: emb[i] = kv[w]; hits += 1
coverage = hits / VOCAB
print(f"coverage {hits}/{VOCAB}={coverage:.1%}")
del kv

class TextDS(Dataset):
    def __init__(self, df):
        self.texts = [tokenize(t) for t in df["text"]]
        self.labels = df["label"].astype(np.float32).values
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        ids = [word2idx.get(w, 1) for w in self.texts[i][:MAX_LEN]] or [1]
        return ids, self.labels[i]

def collate(b):
    ids, ys = zip(*b); L = max(len(x) for x in ids)
    X = torch.zeros(len(ids), L, dtype=torch.long); M = torch.zeros(len(ids), L)
    for i, x in enumerate(ids):
        X[i, :len(x)] = torch.tensor(x); M[i, :len(x)] = 1.0
    return X, M, torch.tensor(ys)

train_ds, valid_ds = TextDS(train_df), TextDS(valid_df)
valid_dl = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, num_layers, dropout, emb_weights):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.emb.weight.data.copy_(torch.tensor(emb_weights))
        self.emb.weight.requires_grad = not FREEZE_EMB
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, bidirectional=True,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, 1)
    def forward(self, x, mask):
        e = self.drop(self.emb(x))
        out, _ = self.lstm(e)
        out = out * mask.unsqueeze(-1)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        mean = out.sum(1) / denom
        mx = out.masked_fill(mask.unsqueeze(-1) == 0, torch.finfo(out.dtype).min).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1))).squeeze(-1)

model = BiLSTMClassifier(VOCAB, EMB_DIM, HIDDEN, NUM_LAYERS, DROPOUT, emb).to(DEVICE)
y_train = train_df["label"].values
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
print("class weights", cw)
criterion = nn.BCEWithLogitsLoss()   # sampler handles balance; no pos_weight (avoid double-correct)
sample_w = np.array([cw[int(l)] for l in y_train], dtype=np.float64)
sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), len(sample_w), replacement=True)
train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, collate_fn=collate)
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

def evaluate(dl):
    model.eval(); ls, ys = [], []
    with torch.no_grad():
        for X, M, y in dl:
            ls.append(model(X.to(DEVICE), M.to(DEVICE)).cpu()); ys.append(y)
    logits = torch.cat(ls).numpy(); ys = torch.cat(ys).numpy().astype(int)
    probs = 1 / (1 + np.exp(-logits))
    return ys, (probs >= 0.5).astype(int), probs

hist = {"epoch": [], "train_loss": [], "valid_loss": [], "valid_acc": [], "valid_f1": []}
best_f1, best_state, wait = -1.0, None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); tot, n = 0.0, 0
    for X, M, y in train_dl:
        X, M, y = X.to(DEVICE), M.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = criterion(model(X, M), y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tot += loss.item() * len(y); n += len(y)
    tr = tot / n
    ys, preds, probs = evaluate(valid_dl)
    vl = nn.functional.binary_cross_entropy(torch.tensor(probs), torch.tensor(ys, dtype=torch.float32)).item()
    acc = accuracy_score(ys, preds)
    _, _, f1, _ = precision_recall_fscore_support(ys, preds, average="macro", zero_division=0)
    hist["epoch"].append(ep); hist["train_loss"].append(tr); hist["valid_loss"].append(vl)
    hist["valid_acc"].append(acc); hist["valid_f1"].append(f1)
    print(f"ep {ep:02d} | tr {tr:.4f} | va {vl:.4f} | acc {acc:.4f} | mF1 {f1:.4f}")
    sched.step(f1)
    if f1 > best_f1:
        best_f1 = f1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; wait = 0
    else:
        wait += 1
        if wait >= PATIENCE: print("early stop", ep, "best", best_f1); break

model.load_state_dict(best_state)
ys, preds, probs = evaluate(valid_dl)
acc = accuracy_score(ys, preds)
prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average="binary", pos_label=1, zero_division=0)
_, _, mf1, _ = precision_recall_fscore_support(ys, preds, average="macro", zero_division=0)
auc = roc_auc_score(ys, probs)
print(f"\nV2 FINAL acc {acc:.4f} P {prec:.4f} R {rec:.4f} F1 {f1:.4f} macroF1 {mf1:.4f} AUC {auc:.4f}")
print(classification_report(ys, preds, target_names=["普通(0)", "可疑(1)"], digits=4))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(hist["epoch"], hist["train_loss"], "o-", label="train loss")
ax1.plot(hist["epoch"], hist["valid_loss"], "s-", label="valid loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper right"); ax1.set_title("Training / Validation Loss")
ax2 = ax1.twinx(); ax2.plot(hist["epoch"], hist["valid_f1"], "^--", color="green", label="valid macro-F1")
ax2.set_ylabel("valid macro-F1"); ax2.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(DATA_DIR, f"train_curve{SUF}.png"), dpi=150); plt.close(fig)
cm = confusion_matrix(ys, preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (valid)")
fig.tight_layout(); fig.savefig(os.path.join(DATA_DIR, f"confusion_matrix{SUF}.png"), dpi=150); plt.close(fig)
torch.save({"state_dict": model.state_dict(), "word2idx": word2idx,
            "config": {"vocab": VOCAB, "emb_dim": EMB_DIM, "hidden": HIDDEN, "num_layers": NUM_LAYERS,
                       "dropout": DROPOUT, "max_len": MAX_LEN}},
           os.path.join(DATA_DIR, f"bilstm_model{SUF}.pt"))
json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "macro_f1": mf1, "auc": auc,
           "coverage": coverage, "vocab": VOCAB, "confusion_matrix": cm.tolist(), "history": hist,
           "best_epoch": int(np.argmax(hist["valid_f1"]) + 1)},
          open(os.path.join(DATA_DIR, f"metrics{SUF}.json"), "w"), ensure_ascii=False, indent=2)
print("DONE v2")
