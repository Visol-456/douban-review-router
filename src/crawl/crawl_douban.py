#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crawl douban comments via the mobile rexxar /interests API.

Output: data/raw/new_crawled/douban_comments_<mid>_<name>.csv
Columns: 用户,评分,时间,内容  (contains reviewer names -> data/raw is gitignored)
"""
import os
import sys
import csv
import json
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.config import RAW_DIR  # noqa: E402

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36",
    "Referer": "https://m.douban.com/",
}

MOVIES = [
    ("38581618", "牛来"),
    ("26747919", "749局"),
    ("26581837", "上海堡垒"),
    ("1292052", "肖申克的救赎"),
    ("26266893", "流浪地球"),
    ("34780991", "哪吒2"),
    ("30513783", "无职转生"),
]

OUT_DIR = os.path.join(RAW_DIR, "new_crawled")
os.makedirs(OUT_DIR, exist_ok=True)
COUNT = 20


def fetch_interests(mid, start):
    url = f"https://m.douban.com/rexxar/api/v2/movie/{mid}/interests?count={COUNT}&start={start}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    summary = {}
    for mid, name in MOVIES:
        rows = []
        start = 0
        total = None
        empty_streak = 0
        seen_comments = set()
        for _ in range(300):
            try:
                d = fetch_interests(mid, start)
            except Exception as e:
                print(f"  [{name}] err at start={start}: {str(e)[:80]}", flush=True)
                time.sleep(2)
                empty_streak += 1
                if empty_streak >= 3:
                    break
                continue
            if total is None:
                total = d.get("total")
            interests = d.get("interests", [])
            added = 0
            for it in interests:
                c = (it.get("comment") or "").strip()
                if not c:
                    continue
                u = (it.get("user") or {}).get("name", "")
                star = (it.get("rating") or {}).get("value", "")
                ct = it.get("create_time", "")
                key = (u, c)
                if key in seen_comments:
                    continue
                seen_comments.add(key)
                rows.append([u, star, ct, c])
                added += 1
            empty_streak = 0 if added else empty_streak + 1
            start += COUNT
            if empty_streak >= 3 or (total and start >= min(total, 2000)):
                break
            time.sleep(0.3)
        seen = set()
        dedup = []
        for r_ in rows:
            if r_[3] in seen:
                continue
            seen.add(r_[3])
            dedup.append(r_)
        path = os.path.join(OUT_DIR, f"douban_comments_{mid}_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["用户", "评分", "时间", "内容"])
            w.writerows(dedup)
        summary[name] = len(dedup)
        print(f"[{name}] total={total} saved={len(dedup)}", flush=True)
        time.sleep(0.5)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
