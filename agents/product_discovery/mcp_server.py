"""Product Discovery Agent — MCP server.

Exposes the canonical `search_products` capability as an MCP tool
via stdio transport.

Usage:
    python -m agents.product_discovery.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agents.product_discovery.agent import (
    FINAL_MAX_PRODUCTS,
    search_products,
    search_products_with_debug,
)
from agents.product_discovery.progress import PROGRESS_STORE
from common.config import configure_logging, get_settings

logger = logging.getLogger(__name__)


def _coerce_max_results(value: object) -> int:
    """Validate the MCP max_results argument."""
    if value is None:
        return FINAL_MAX_PRODUCTS

    try:
        max_results = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_results must be an integer") from exc

    if max_results < 1 or max_results > FINAL_MAX_PRODUCTS:
        raise ValueError(f"max_results must be between 1 and {FINAL_MAX_PRODUCTS}")

    return max_results


def _coerce_trace_id(value: object) -> str | None:
    """Validate the optional MCP trace correlation id."""
    if value is None:
        return None

    trace_id = str(value).strip()
    if not trace_id:
        return None
    if len(trace_id) > 200:
        raise ValueError("trace_id must be 200 characters or fewer")
    return trace_id

# ── MCP Server Setup ──────────────────────────────────────────────

mcp = Server("product-discovery")


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise the search_products tool."""
    return [
        Tool(
            name="search_products",
            description=(
                "Retrieve and normalize product candidates for a shopper query. "
                "Returns structured JSON with product names, prices, ratings, "
                "source URLs, and confidence scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language shopping query (e.g., 'best wireless earbuds under $100')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of products to return (default: 3, max: 3)",
                        "default": FINAL_MAX_PRODUCTS,
                    },
                    "trace_id": {
                        "type": "string",
                        "description": "Optional trace correlation id for live progress snapshots",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations."""
    if name != "search_products":
        raise ValueError(f"Unknown tool: {name}")

    arguments = arguments or {}
    query = arguments.get("query", "")
    max_results = _coerce_max_results(arguments.get("max_results", FINAL_MAX_PRODUCTS))
    trace_id = _coerce_trace_id(arguments.get("trace_id"))

    if not query:
        raise ValueError("query parameter is required")

    settings = get_settings()
    if trace_id:
        async def publish_debug(stage: str, message: str, payload: dict) -> None:
            snapshot = dict(payload)
            snapshot["stage"] = stage
            snapshot["message"] = message
            snapshot["interface"] = "mcp"
            PROGRESS_STORE.publish(trace_id, snapshot)

        result, debug_snapshot = await search_products_with_debug(
            query=query,
            settings=settings,
            max_results=max_results,
            on_step=publish_debug,
        )
        final_snapshot = dict(debug_snapshot)
        final_snapshot["message"] = f"Normalized {len(result.products)} products for the final result"
        final_snapshot["interface"] = "mcp"
        PROGRESS_STORE.publish(trace_id, final_snapshot)
    else:
        result = await search_products(query=query, settings=settings, max_results=max_results)

    return [
        TextContent(
            type="text",
            text=result.model_dump_json(indent=2),
        )
    ]


# ── Entry point ───────────────────────────────────────────────────

async def main():
    configure_logging()
    logger.info("Starting Product Discovery MCP server (stdio)")
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
