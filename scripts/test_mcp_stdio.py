"""MCP stdio test helper.

Sends the initialize handshake followed by a tools/list or tools/call
request to the Product Discovery MCP server.

Usage:
    python scripts/test_mcp_stdio.py list
    python scripts/test_mcp_stdio.py call "best wireless earbuds under 100"
"""

from __future__ import annotations

import json
import re
import subprocess
import sys


def _encode_message(message: dict) -> bytes:
    """Encode one MCP stdio message using Content-Length framing."""
    body = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _build_stdin_data(messages: list[dict]) -> bytes:
    """Build a framed MCP stdio request stream."""
    return b"".join(_encode_message(message) for message in messages)


def _iter_framed_messages(data: bytes):
    """Yield decoded JSON messages from a Content-Length framed stream."""
    cursor = 0
    while cursor < len(data):
        header_end = data.find(b"\r\n\r\n", cursor)
        if header_end == -1:
            break

        header = data[cursor:header_end].decode("ascii", errors="ignore")
        match = re.search(r"Content-Length:\s*(\d+)", header, re.IGNORECASE)
        if not match:
            cursor = header_end + 4
            continue

        content_length = int(match.group(1))
        body_start = header_end + 4
        body_end = body_start + content_length
        if body_end > len(data):
            break

        yield json.loads(data[body_start:body_end].decode("utf-8"))
        cursor = body_end


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_mcp_stdio.py [list|call] [query]")
        sys.exit(1)

    action = sys.argv[1]

    # Build the sequence of JSON-RPC messages
    messages = []

    # 1. Initialize handshake (required by MCP)
    messages.append({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    })

    # 2. Initialized notification
    messages.append({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })

    if action == "list":
        # 3. List tools
        messages.append({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        })
    elif action == "call":
        query = sys.argv[2] if len(sys.argv) > 2 else "best wireless earbuds"
        # 3. Call tool
        messages.append({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_products",
                "arguments": {"query": query, "max_results": 3},
            },
        })
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

    stdin_data = _build_stdin_data(messages)

    proc = subprocess.run(
        [sys.executable, "-m", "agents.product_discovery.mcp_server"],
        input=stdin_data,
        capture_output=True,
        timeout=60,
    )

    # Parse and print responses
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0 and stderr_text:
        # Filter out log lines (they go to stderr)
        for line in stderr_text.splitlines():
            if "ERROR" in line or "Traceback" in line:
                print(f"STDERR: {line}", file=sys.stderr)

    for data in _iter_framed_messages(proc.stdout):
        try:
            msg_id = data.get("id")
            if msg_id == 0:
                print("Initialize: OK")
            elif msg_id == 1:
                print(json.dumps(data, indent=2))
        except json.JSONDecodeError:
            pass  # skip non-JSON lines


if __name__ == "__main__":
    main()
