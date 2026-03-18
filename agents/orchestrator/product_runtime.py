"""Product Discovery transport helpers for the orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import timedelta

from agents.orchestrator.runtime import extract_named_artifact_json
from common.a2a_models import AgentCard, ProductDiscoveryResult, Task
from common.config import Settings

logger = logging.getLogger(__name__)


async def call_product_discovery_via_mcp(
    *,
    query: str,
    settings: Settings,
    agent_card: AgentCard | None = None,
    mcp_url: str | None = None,
    max_results: int = 3,
) -> ProductDiscoveryResult:
    """Invoke Product Discovery through its MCP search_products tool."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("MCP client dependencies are not installed") from exc

    resolved_mcp_url = mcp_url or resolve_product_mcp_url(agent_card)

    async with streamable_http_client(resolved_mcp_url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=settings.A2A_CLIENT_TIMEOUT_SECONDS),
        ) as session:
            await session.initialize()

            tools = await session.list_tools()
            if not any(tool.name == "search_products" for tool in tools.tools):
                raise RuntimeError("Product Discovery MCP server did not advertise search_products")

            result = await session.call_tool(
                "search_products",
                {
                    "query": query,
                    "max_results": max_results,
                },
            )

    if result.isError:
        raise RuntimeError("Product Discovery MCP tool returned an error")

    if result.structuredContent:
        return ProductDiscoveryResult(**result.structuredContent)

    text_parts = [item.text for item in result.content if getattr(item, "type", None) == "text"]
    if not text_parts:
        raise RuntimeError("Product Discovery MCP tool returned no text content")

    return ProductDiscoveryResult(**json.loads(text_parts[0]))


def resolve_product_mcp_url(agent_card: AgentCard | None) -> str:
    """Resolve the Product Discovery MCP URL from the discovered Agent Card."""
    if not agent_card or not agent_card.additionalInterfaces:
        raise RuntimeError("Product Discovery agent card did not advertise an MCP interface")

    for interface in agent_card.additionalInterfaces:
        if interface.transport.upper() != "MCP":
            continue
        if interface.url.startswith(("http://", "https://")):
            return interface.url

    raise RuntimeError("Product Discovery agent card did not advertise an HTTP MCP interface")


def extract_product_result(task: Task) -> ProductDiscoveryResult | None:
    """Extract ProductDiscoveryResult from a completed Task's artifacts."""
    if task.status.state != "completed":
        logger.warning("Product task not completed: state=%s", task.status.state)
        return None

    data = extract_named_artifact_json(task, "product-discovery-result")
    if data is None:
        return None

    try:
        return ProductDiscoveryResult(**data)
    except Exception as exc:
        logger.debug("Failed to parse product result artifact: %s", exc)
        return None
