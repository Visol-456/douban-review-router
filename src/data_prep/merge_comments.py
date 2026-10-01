#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge old + new douban comment csv files into one deduped, text-only dataset.

Reads the per-movie csv from data/raw/new_crawled (and optionally a legacy dir),
writes data/interim/all_comments_merged.csv with a single `text` column.
"""
import os
import sys
import csv
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import INTERIM_DIR, RAW_DIR  # noqa: E402

NEW_DIR = os.path.join(RAW_DIR, "new_crawled")
OLD_DIR = os.environ.get("DRR_LEGACY_COMMENTS_DIR", "")  # optional legacy dir holding douban_comments_*.csv
OUT = os.path.join(INTERIM_DIR, "all_comments_merged.csv")


def read_comments(path, encoding="utf-8-sig"):
    """Extract the 内容 column from a douban comments csv."""
    out = []
    try:
        with open(path, encoding=encoding, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return out
        header = [h.strip("\ufeff").strip() for h in rows[0]]
        ci = header.index("内容") if "内容" in header else len(header) - 1
        for r in rows[1:]:
            if len(r) <= ci:
                continue
            c = r[ci].strip()
            if c:
                out.append(c)
    except Exception as e:
        print(f"  ERR {path}: {e}")
    return out


def main():
    all_texts = []
    seen = set()
    source_count = {}

    dirs = [("old", OLD_DIR), ("new", NEW_DIR)]
    for tag, d in dirs:
        if not d or not os.path.isdir(d):
            continue
        files_ = sorted(glob.glob(os.path.join(d, "douban_comments_*.csv")))
        total = 0
        for f in files_:
            texts = read_comments(f)
            total += len(texts)
            name = os.path.basename(f).replace("douban_comments_", "").replace(".csv", "")
            for c in texts:
                if c not in seen:
                    seen.add(c)
                    all_texts.append(c)
                    source_count[name] = source_count.get(name, 0) + 1
        print(f"{tag}: files={len(files_)} raw rows={total}")
    print("source breakdown:", source_count)
    print("TOTAL merged unique texts:", len(all_texts))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        for c in all_texts:
            w.writerow([c])
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
