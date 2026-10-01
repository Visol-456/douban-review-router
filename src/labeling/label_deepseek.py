#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weak-label merged comments with DeepSeek: 可疑需深挖 = 1 else 0.

Input : data/interim/all_comments_merged.csv
Output: data/interim/labeled_full.csv
The API key is read from the DEEPSEEK_API_KEY environment variable only.
"""
import os
import re
import sys
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import INTERIM_DIR, get_deepseek_client  # noqa: E402

MERGED = os.path.join(INTERIM_DIR, "all_comments_merged.csv")
OUT = os.path.join(INTERIM_DIR, "labeled_full.csv")

PROMPT_TPL = "这条评论是否可疑需深挖？只回答 true 或 false\n\n评论：{c}"


def parse_bool(s):
    if s is None:
        return None
    s = s.strip().lower()
    if s.startswith("true") or s == "1" or s[:1] == "是":
        return 1
    if s.startswith("false") or s == "0":
        return 0
    m = re.search(r"\b(true|false)\b", s)
    if m:
        return 1 if m.group(1) == "true" else 0
    return None


def label_one(client, text, retries=4):
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": PROMPT_TPL.format(c=text)}],
                max_tokens=5,
                temperature=0,
            )
            content = r.choices[0].message.content
            v = parse_bool(content)
            if v is not None:
                return v, content
        except Exception as e:
            if i == retries - 1:
                return None, f"ERR:{str(e)[:80]}"
            time.sleep(1.5 * (i + 1))
    return None, "unparsed"


def main():
    client = get_deepseek_client()
    rows = []
    with open(MERGED, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r["text"])
    print("to label:", len(rows), flush=True)
    results = [None] * len(rows)
    errors = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(label_one, client, t): i for i, t in enumerate(rows)}
        for fut in as_completed(futs):
            i = futs[fut]
            v, raw = fut.result()
            if v is None:
                errors.append((i, raw[:80]))
            results[i] = (v if v is not None else 0, raw)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    if errors:
        print("retrying", len(errors), "failed", flush=True)
        for i, _ in errors:
            v, raw = label_one(client, rows[i], retries=5)
            if v is not None:
                results[i] = (v, raw)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for t, (v, raw) in zip(rows, results):
            w.writerow([t, v])
    from collections import Counter
    cnt = Counter(v for v, _ in results)
    print("done in %.0fs" % (time.time() - t0))
    print("label dist:", dict(cnt))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
