#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal MCP server exposing a Python-execution tool for the agent layer."""
from mcp.server.fastmcp import FastMCP

import io
import contextlib
import traceback

mcp = FastMCP("Demo")


@mcp.tool()
def run_python_code(code: str) -> str:
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
