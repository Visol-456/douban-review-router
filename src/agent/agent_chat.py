#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive MCP agent loop that routes deep analysis to DeepSeek.

This agent wires together three capabilities:

1. MCP Tools   — an MCP stdio server (src/agent/mcp_server.py) exposes
                 `run_python_code` and `search_movie_info` (Douban reputation).
2. Tools       — the model can autonomously call a DeepSeek **Responses API**
                 native web_search tool, plus a Metaso third-party search tool,
                 to look up a movie's real-world reputation before judging a
                 review as credible / suspicious.
3. Skills      — a domain-expert skill (skills/movie_review_analyst.md) is
                 injected into the system prompt on every session.

Security rule: every secret is read from an ENVIRONMENT VARIABLE ONLY:
    DEEPSEEK_API_KEY   DeepSeek API key (required)
    DEEPSEEK_BASE_URL  default https://api.deepseek.com
    DEEPSEEK_MODEL     default deepseek-chat
    METASO_API_KEY     optional Metaso search key (fallback search)
(Original prototype hardcoded a key here - that has been removed.)
"""
import os
import json

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

API_KEY = os.environ.get("DEEPSEEK_API_KEY")          # never hardcode
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
METASO_API_KEY = os.environ.get("METASO_API_KEY", "")
# Model used for the native DeepSeek Responses-API web_search tool (server-side
# search is only available on v4-pro per DeepSeek; override if it changes).
SEARCH_MODEL = os.environ.get("DEEPSEEK_SEARCH_MODEL", "deepseek-v4-pro")

server_params = StdioServerParameters(
    command="python",
    args=["src/agent/mcp_server.py"],
    env=None,
)


def _load_skill(path: str) -> str:
    """Load a skill markdown file from the skills/ dir; return '' if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _build_system_prompt() -> str:
    """Compose the system prompt, injecting the '影评分析专家' skill."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    skill = _load_skill(os.path.join(repo_root, "skills", "movie_review_analyst.md"))

    base = """你是一个 AI Agent，你的任务是完成用户的需求，你可以使用 Python 工具完成大部分事情。

注意事项
1. 使用 Python 工具时，把你需要看到的内容用 print 打印出来，运行完成后会返回所有打印日志和错误日志。
2. Python 将运行在用户的电脑上，你有充足的权限，干各类任务。
3. 当你在评判一条电影/剧集评论是否可信、是否可疑（水军/反讽/刷分）时，必须先调用 search_movie_info 或搜索工具查该电影的真实口碑，再下结论。
"""
    if skill:
        base += (
            "\n\n===== 注入的领域专家 Skill：影评分析专家 =====\n"
            + skill
            + "\n===== Skill 注入结束 =====\n"
        )
    return base


system_prompt = _build_system_prompt()


def _search_web_deepseek(query: str) -> str:
    """Use DeepSeek Responses API's native server-side web_search tool.

    POST /v1/responses with tools=[{"type":"web_search"}]. Returns the model's
    search-grounded answer. Returns an error string on any failure.
    """
    if not API_KEY:
        return "错误: DEEPSEEK_API_KEY 未设置，无法调用 DeepSeek 联网搜索。"
    url = BASE_URL.rstrip("/") + "/v1/responses"
    payload = {
        "model": SEARCH_MODEL,
        "input": (
            f"联网搜索「{query}」，查找该电影/剧集的真实口碑（豆瓣评分、评分人数、"
            f"口碑争议等），最多 5 条，返回带链接的简要结果，无解释。"
        ),
        "tools": [{"type": "web_search"}],
        "max_tool_calls": 3,
        "reasoning": {"effort": "low"},
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        # Collect all output text items (incl. web_search_call result citations).
        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            # fall back to top-level output_text
            text = (data.get("output_text") or "").strip()
        return text or "DeepSeek 联网搜索未返回可用文本结果。"
    except Exception as e:
        return f"DeepSeek 联网搜索失败: {str(e)[:200]}"


def _search_web_metaso(query: str) -> str:
    """Search via the Metaso (秘塔) API as a third-party fallback."""
    if not METASO_API_KEY:
        return "错误: METASO_API_KEY 未设置，无法调用 Metaso 搜索。"
    url = "https://metaso.cn/api/v1/search"
    payload = {
        "q": query,
        "scope": "webpage",
        "includeSummary": True,
        "size": "5",
        "includeRawContent": False,
        "conciseSnippet": True,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {METASO_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("webpages", [])
        if not pages:
            return f"Metaso 未找到「{query}」的相关结果。"
        lines = []
        for i, p in enumerate(pages, 1):
            title = p.get("title", "")
            link = p.get("link", "")
            summary = p.get("summary") or p.get("snippet") or ""
            lines.append(f"{i}. {title}\n   链接: {link}\n   摘要: {summary}")
        return "\n".join(lines)
    except Exception as e:
        return f"Metaso 搜索失败: {str(e)[:200]}"


# Function tools the model can call, beyond the MCP tools.
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "deepseek_web_search",
            "description": "调用 DeepSeek 服务端内置联网搜索，查询某部电影/剧集的真实口碑（豆瓣评分、评分人数、争议）。返回带链接的搜索结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的电影/剧集名称及问题，如「流浪地球2 豆瓣评分 口碑」"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "metaso_search",
            "description": "调用秘塔 Metaso 联网搜索，查询某部电影/剧集的真实口碑。返回带链接的搜索结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的电影/剧集名称及问题"},
                },
                "required": ["query"],
            },
        },
    },
]

# Client-side dispatcher for the extra function tools (not backed by MCP).
_EXTRA_TOOL_HANDLERS = {
    "deepseek_web_search": lambda a: _search_web_deepseek(a.get("query", "")),
    "metaso_search": lambda a: _search_web_metaso(a.get("query", "")),
}


async def run():
    if not API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set; export it or use a local .env.")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP Server\n")

            raw_tools = await session.list_tools()

            def _convert_tool(t):
                name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                description = getattr(t, "description", "") or (t.get("description") if isinstance(t, dict) else "")
                input_schema = (
                    getattr(t, "input_schema", None)
                    or getattr(t, "inputSchema", None)
                    or (t.get("input_schema") if isinstance(t, dict) else None)
                    or (t.get("inputSchema") if isinstance(t, dict) else None)
                ) or {"type": "object", "properties": {}}
                return {
                    "type": "function",
                    "function": {"name": name, "description": description, "parameters": input_schema},
                }

            tool_items = getattr(raw_tools, "tools", raw_tools)
            # Combine MCP tools + the extra search function tools into one list.
            tools = [_convert_tool(t) for t in tool_items] + list(FUNCTION_TOOLS)
            print("Available tools:")
            for t in tools:
                print(f"- {t['function']['name']}")
            print()

            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

            while True:
                user_input = input("You: ")
                if user_input == "exit":
                    break
                messages.append({"role": "user", "content": user_input})

                while True:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        stream=True,
                    )
                    print()
                    print("Assistant: ", end="", flush=True)
                    content = ""
                    full_tool_calls: list[dict] = []
                    tool_call_index = 0
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            content += delta.content
                            print(delta.content, end="", flush=True)
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index if tc_delta.index is not None else tool_call_index
                                while len(full_tool_calls) <= idx:
                                    full_tool_calls.append(None)
                                if full_tool_calls[idx] is None:
                                    full_tool_calls[idx] = {
                                        "id": tc_delta.id or "",
                                        "type": "function",
                                        "function": {
                                            "name": tc_delta.function.name if tc_delta.function else "",
                                            "arguments": tc_delta.function.arguments if tc_delta.function else "",
                                        },
                                    }
                                else:
                                    if tc_delta.function and tc_delta.function.name:
                                        full_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                                    if tc_delta.function and tc_delta.function.arguments:
                                        full_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments
                                tool_call_index = max(tool_call_index, idx + 1)
                    print()
                    print()

                    assistant_message = {"role": "assistant", "content": content or None}
                    valid_tool_calls = [tc for tc in full_tool_calls if tc]
                    if valid_tool_calls:
                        assistant_message["tool_calls"] = valid_tool_calls
                    messages.append(assistant_message)

                    if valid_tool_calls:
                        for tc in valid_tool_calls:
                            func_name = tc["function"]["name"]
                            func_args_str = tc["function"]["arguments"]
                            try:
                                func_args = json.loads(func_args_str) if func_args_str else {}
                            except json.JSONDecodeError:
                                func_args = {}
                            print(f"\n[ToolExec] {func_name} args={func_args}", flush=True)
                            # Route: extra search tools run client-side; everything
                            # else (MCP tools) goes to the MCP server.
                            if func_name in _EXTRA_TOOL_HANDLERS:
                                try:
                                    text_content = _EXTRA_TOOL_HANDLERS[func_name](func_args)
                                except Exception as e:
                                    text_content = f"Error executing {func_name}: {e}"
                            else:
                                try:
                                    result = await session.call_tool(func_name, arguments=func_args)
                                    text_content = result.content[0].text
                                except Exception as e:
                                    text_content = f"Error executing {func_name}: {e}"
                            print(f"\n[ToolResult] {text_content}\n", flush=True)
                            messages.append({"role": "tool", "content": text_content, "tool_call_id": tc["id"]})
                        continue
                    break
            print()
            print("Bye!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
