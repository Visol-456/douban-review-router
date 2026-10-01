# Desensitized dataset

These CSVs are safe to commit: they contain **only** `text,label` — no reviewer
names, no user ids, no timestamps.

| file | rows | columns | meaning |
|------|------|---------|---------|
| `labeled_full.csv` | full corpus | `text,label` | DeepSeek weak-labeled, `1` = 可疑需深挖, `0` = 普通 |
| `train.csv` | 80% | `text,label` | stratified train split (seed 42) |
| `valid.csv` | 20% | `text,label` | stratified validation split (seed 42) |

Raw crawl output (`data/raw/new_crawled/*.csv`, which includes the `用户` column)
is **gitignored** and never pushed.
