#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP server exposing tools for the agent layer.

Tools:
- run_python_code   : execute arbitrary python (like an interactive kernel)
- search_movie_info : look up a movie's Douban reputation (score, raters) as external
                      reference for judging whether a review is credible.

Security rule: every secret is read from an ENVIRONMENT VARIABLE, never hardcoded.
Douban public pages/APIs are fetched anonymously (no key involved).
"""
from mcp.server.fastmcp import FastMCP

import io
import contextlib
import traceback

import requests

mcp = FastMCP("Demo")

DOUBAN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36",
    "Referer": "https://m.douban.com/",
}


@mcp.tool()
def run_python_code(code: str) -> str:
    """Execute arbitrary Python code and return its printed output / error."""
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    exec_namespace = {}
    try:
        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            exec(code, exec_namespace)
    except Exception:
        stderr_io.write(traceback.format_exc())
    content = str(stdout_io.getvalue())
    error = str(stderr_io.getvalue())
    if error:
        content += f"\nError: {error}"
    return content


@mcp.tool()
def search_movie_info(movie_title: str) -> str:
    """Search Douban for a movie's public reputation and return a compact summary.

    Fields returned (when available): title, year, genre/director/cast snippet,
    Douban rating (score) and number of raters. This is ground truth the model can
    compare a review against to decide whether the review is suspicious/credible.

    Uses Douban's public mobile rexxar search API (anonymous, no key).
    """
    import urllib.parse

    q = urllib.parse.quote(movie_title)
    url = f"https://m.douban.com/rexxar/api/v2/search?q={q}&type=movie"
    try:
        resp = requests.get(url, headers=DOUBAN_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"搜索失败（可能被限流或网络问题）: {str(e)[:200]}"

    subjects = data.get("subjects") or {}
    items = subjects.get("items") or []
    if not items:
        return f"未在豆瓣找到电影「{movie_title}」的相关信息。"

    # Prefer an exact title match; otherwise take the top result.
    exact = next(
        (it for it in items if (it.get("target") or {}).get("title") == movie_title),
        None,
    )
    target = ((exact or items[0]).get("target")) or {}

    title = target.get("title", movie_title)
    year = target.get("year", "未知年份")
    subtitle = target.get("card_subtitle", "")
    rating = target.get("rating") or {}
    score = rating.get("value")
    count = rating.get("count")
    mid = target.get("id", "")

    lines = [
        f"片名: {title}",
        f"年份: {year}",
    ]
    if subtitle:
        lines.append(f"类型/主创: {subtitle}")
    if score is not None:
        lines.append(f"豆瓣评分: {score} / 10")
    else:
        lines.append("豆瓣评分: 暂无评分")
    if count:
        lines.append(f"评分人数: {count:,}")
    if mid:
        lines.append(f"豆瓣ID: {mid}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
