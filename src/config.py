#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared paths and client factory for the douban-review-router project.

Security rule: every secret is read from an ENVIRONMENT VARIABLE here.
Never hardcode a key in this file or anywhere else in the repo.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DRR_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

RAW_DIR      = os.path.join(DATA_ROOT, "raw")           # per-movie crawl csv (has reviewer names -> gitignored)
INTERIM_DIR  = os.path.join(DATA_ROOT, "interim")       # merged / labeled intermediates (gitignored)
DESENS_DIR   = os.path.join(DATA_ROOT, "desensitized")  # text+label only, no PII -> safe to commit

for _d in (RAW_DIR, INTERIM_DIR, DESENS_DIR):
    os.makedirs(_d, exist_ok=True)

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def get_deepseek_client():
    """Build an OpenAI-compatible client from the env var only.

    Set DEEPSEEK_API_KEY in your shell or a local .env (which is gitignored).
    """
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set. Export it or put it in a local .env "
            "(see .env.example). Never hardcode it."
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
