#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge existing labeled texts + freshly crawled comments into one deduped dataset.

Reads the already-labeled text column plus every data/raw/new_crawled csv,
writes data/interim/all_comments_merged.csv (single `text` column).
"""
import os
import sys
import csv
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import INTERIM_DIR, RAW_DIR, DESENS_DIR  # noqa: E402

EXISTING = [
    os.path.join(DESENS_DIR, "train.csv"),
    os.path.join(DESENS_DIR, "valid.csv"),
]
NEW_DIR = os.path.join(RAW_DIR, "new_crawled")
OUT = os.path.join(INTERIM_DIR, "all_comments_merged.csv")


def read_texts(path):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return out
    h = [x.strip("\ufeff").strip() for x in rows[0]]
    ci = h.index("内容") if "内容" in h else h.index("text") if "text" in h else len(h) - 1
    for r in rows[1:]:
        if len(r) > ci and r[ci].strip():
            out.append(r[ci].strip())
    return out


def main():
    all_texts = []
    seen = set()
    src = {}

    existing_raw = 0
    for f in EXISTING:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            t = r["text"].strip()
            existing_raw += 1
            if t and t not in seen:
                seen.add(t)
                all_texts.append(t)
                src["existing"] = src.get("existing", 0) + 1
    print(f'existing: raw={existing_raw} unique={src.get("existing", 0)}')

    new_raw = 0
    for f in sorted(glob.glob(os.path.join(NEW_DIR, "douban_comments_*.csv"))):
        texts = read_texts(f)
        new_raw += len(texts)
        name = os.path.basename(f).replace("douban_comments_", "").replace(".csv", "")
        for c in texts:
            if c not in seen:
                seen.add(c)
                all_texts.append(c)
                src[name] = src.get(name, 0) + 1
    print(f"new crawled: raw={new_raw}")
    print("new unique added per movie:", {k: v for k, v in src.items() if k != "existing"})
    print("TOTAL merged unique:", len(all_texts))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        for c in all_texts:
            w.writerow([c])
    print("WROTE", OUT, len(all_texts))


if __name__ == "__main__":
    main()
