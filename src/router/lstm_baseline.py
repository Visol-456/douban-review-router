#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Route A - LSTM baseline (System One fast router).

Task: given a review, predict whether it is "可疑需深挖" (label 1) or plain (0).
Pipeline: jieba tokenize -> pretrained word-vector vocab -> embedding -> BiLSTM -> 2-class head.

STATUS: skeleton / placeholder. Not yet trained on the WSL2 box.
This file encodes the intended architecture and training loop so the repo
structure and the experiment contract are fixed before results land.
"""
import os
import sys
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import DESENS_DIR  # noqa: E402

TRAIN_CSV = os.path.join(DESENS_DIR, "train.csv")
VALID_CSV = os.path.join(DESENS_DIR, "valid.csv")
EMBED_DIM = 200
HIDDEN_DIM = 128
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3


def load_split(path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            X.append(r["text"])
            y.append(int(r["label"]))
    return X, y


class BiLSTMClassifier:  # to be implemented with torch once the env is ready
    """Pretrained embedding -> BiLSTM -> mean-pool -> linear(2)."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim


def main():
    # 1. jieba tokenize each review (RNN path uses tokens + word index, NOT a transformer tokenizer)
    # 2. build vocab from a pretrained Chinese word-vector file loaded via gensim
    # 3. init the embedding layer from those vectors (better than random init on small data)
    # 4. train BiLSTM, log loss / accuracy / per-sample inference latency
    # 5. dump metrics to results/, weights to checkpoints/ (gitignored)
    raise SystemExit("TODO: implement LSTM baseline training (env not provisioned yet)")


if __name__ == "__main__":
    main()
