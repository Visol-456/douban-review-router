#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive MCP agent loop that routes deep analysis to DeepSeek.

The API key is read from the DEEPSEEK_API_KEY environment variable ONLY.
(Original prototype hardcoded a key here - that has been removed.)
"""
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

API_KEY = os.environ.get("DEEPSEEK_API_KEY")          # never hardcode
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")

server_params = StdioServerParameters(
    command="python",
    args=["src/agent/mcp_server.py"],
    env=None,
)

system_prompt = """
你是一个 AI Agent，你的任务是完成用户的需求，你可以使用 Python 工具完成大部分事情。

注意事项
1. 使用 Python 工具时，把你需要看到的内容用 print 打印出来，运行完成后会返回所有打印日志和错误日志。
2. Python 将运行在用户的电脑上，你有充足的权限，干各类任务。
"""


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
            tools = [_convert_tool(t) for t in tool_items]
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
                        import json
                        for tc in valid_tool_calls:
                            func_name = tc["function"]["name"]
                            func_args_str = tc["function"]["arguments"]
                            try:
                                func_args = json.loads(func_args_str) if func_args_str else {}
                            except json.JSONDecodeError:
                                func_args = {}
                            print(f"\n[ToolExec] {func_name} args={func_args}", flush=True)
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
