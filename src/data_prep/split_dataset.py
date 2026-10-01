#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stratified 8:2 split of the DeepSeek-labeled dataset -> train.csv / valid.csv.

Input : data/interim/labeled_full.csv   (columns text,label)
Output: data/desensitized/train.csv, data/desensitized/valid.csv  (columns text,label)
Only text + label leave this script; no reviewer names or ids are carried over.
"""
import os
import sys
import csv
import random
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import INTERIM_DIR, DESENS_DIR  # noqa: E402

SRC = os.path.join(INTERIM_DIR, "labeled_full.csv")
TRAIN = os.path.join(DESENS_DIR, "train.csv")
VALID = os.path.join(DESENS_DIR, "valid.csv")
SEED = 42


def main():
    rows = []
    seen = set()
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = r["text"].strip()
            if not t or t in seen:
                continue
            seen.add(t)
            rows.append((t, int(r["label"])))
    print("unique labeled:", len(rows))
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    print("class dist: 1=%d 0=%d" % (len(pos), len(neg)))

    rng = random.Random(SEED)
    train, valid = [], []
    for cls in (0, 1):
        grp = [r for r in rows if r[1] == cls]
        rng.shuffle(grp)
        n_valid = round(len(grp) * 0.2)
        valid.extend(grp[:n_valid])
        train.extend(grp[n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    for path, data in [(TRAIN, train), (VALID, valid)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(data)
    print("train:", len(train), dict(Counter(l for _, l in train)))
    print("valid:", len(valid), dict(Counter(l for _, l in valid)))
    print("WROTE", TRAIN, VALID)


if __name__ == "__main__":
    main()
