#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 relabel with an explicit 6-criteria "可疑需深挖" standard.

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

PROMPT_TPL = (
    '判断下面这条电影评论是否属于"可疑需深挖"。满足以下任一标准即为可疑(true)：\n'
    "1 反讽/阴阳怪气：表面褒奖实际贬低（如\"神作\"\"必看\"\"五星好评\"配负面内容）\n"
    "2 极端情绪宣泄：无理由的极致吹捧或贬低（\"史上最烂\"\"宇宙第一\"）\n"
    "3 人身攻击：攻击演员/导演/其他观众（辱骂、扣帽子）\n"
    "4 疑似水军：公式化吹捧、刷分话术、无实质内容的跟风好评\n"
    "5 疑似AI生成：机械、模板化、缺乏个人体验的长篇评论\n"
    "6 争议性/引战：粉丝互撕、拉踩、带节奏\n"
    "普通评论为 false：真诚、说理、有具体个人体验的正常评论。\n"
    "只回答 true 或 false，不要解释。\n\n评论：{c}"
)


def parse_bool(s):
    if s is None:
        return None
    s = s.strip().lower()
    if s.startswith("true") or s == "1":
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
                results[i] = (0, raw)
            else:
                results[i] = (v, raw)
            done += 1
            if done % 200 == 0:
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
    print("unparsed/err remaining:", sum(1 for v, raw in results if v is None))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
