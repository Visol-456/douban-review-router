#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 crawl: more douban comments for additional movies via the rexxar API."""
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
    ("35766491", "满江红"),
    ("26363254", "战狼2"),
    ("4811774", "阿凡达2"),
    ("35660795", "消失的她"),
    ("36035676", "长安三万里"),
    ("36081094", "热辣滚烫"),
    ("36151692", "周处除三害"),
    ("35267224", "孤注一掷"),
]

OUT_DIR = os.path.join(RAW_DIR, "new_crawled")
os.makedirs(OUT_DIR, exist_ok=True)
COUNT = 20
PER_MOVIE = 160


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
        empty_streak = 0
        seen = set()
        while start < PER_MOVIE:
            try:
                d = fetch_interests(mid, start)
            except Exception as e:
                print(f"  [{name}] err start={start}: {str(e)[:80]}", flush=True)
                time.sleep(2)
                empty_streak += 1
                if empty_streak >= 3:
                    break
                continue
            interests = d.get("interests", [])
            if not interests:
                break
            added = 0
            for it in interests:
                c = (it.get("comment") or "").strip()
                if not c or c in seen:
                    continue
                seen.add(c)
                u = (it.get("user") or {}).get("name", "")
                star = (it.get("rating") or {}).get("value", "")
                ct = it.get("create_time", "")
                rows.append([u, star, ct, c])
                added += 1
            empty_streak = 0 if added else empty_streak + 1
            if empty_streak >= 3:
                break
            start += COUNT
            time.sleep(0.3)
        path = os.path.join(OUT_DIR, f"douban_comments_{mid}_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["用户", "评分", "时间", "内容"])
            w.writerows(rows)
        summary[name] = len(rows)
        print(f"[{name}] saved={len(rows)}", flush=True)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
