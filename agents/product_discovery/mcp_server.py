"""Product Discovery Agent - MCP server."""

from __future__ import annotations

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agents.product_discovery.agent import FINAL_MAX_PRODUCTS, search_products
from agents.product_discovery.catalog import find_similar_products
from common.a2a_models import SimilarProductsRequest
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


mcp = Server("product-discovery")


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise the MCP tools exposed by Product Discovery."""
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
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="find_similar_products",
            description=(
                "Look up similar products from the structured mock product catalog using "
                "an extracted product payload from the YouTube review agent."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product identified from the review video",
                    },
                    "category": {
                        "type": ["string", "null"],
                        "description": "Optional product category",
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Structured key features extracted from the review video",
                    },
                    "price": {
                        "type": ["string", "null"],
                        "description": "Price mentioned in the video, if any",
                    },
                    "source_video_url": {
                        "type": "string",
                        "description": "The YouTube review URL that produced the product details",
                    },
                    "constraints": {
                        "type": ["object", "null"],
                        "description": "Optional structured lookup constraints",
                    },
                },
                "required": ["product_name", "source_video_url"],
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations."""
    arguments = arguments or {}
    if name == "search_products":
        query = arguments.get("query", "")
        max_results = _coerce_max_results(arguments.get("max_results", FINAL_MAX_PRODUCTS))

        if not query:
            raise ValueError("query parameter is required")

        settings = get_settings()
        result = await search_products(query=query, settings=settings, max_results=max_results)
    elif name == "find_similar_products":
        request = SimilarProductsRequest(
            product_name=str(arguments.get("product_name", "")).strip(),
            category=arguments.get("category"),
            features=[str(item) for item in arguments.get("features", [])],
            price=arguments.get("price"),
            source_video_url=str(arguments.get("source_video_url", "")).strip(),
            constraints=arguments.get("constraints"),
        )
        if not request.product_name:
            raise ValueError("product_name parameter is required")
        if not request.source_video_url:
            raise ValueError("source_video_url parameter is required")
        result = find_similar_products(request, max_results=FINAL_MAX_PRODUCTS)
    else:
        raise ValueError(f"Unknown tool: {name}")

    return [
        TextContent(
            type="text",
            text=result.model_dump_json(indent=2),
        )
    ]


async def main():
    configure_logging()
    logger.info("Starting Product Discovery MCP server (stdio)")
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
