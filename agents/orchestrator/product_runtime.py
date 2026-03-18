"""Product Discovery transport and trace helpers for the orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import httpx

from agents.orchestrator.runtime import TASK_POLL_INTERVAL_SECONDS, extract_named_artifact_json
from agents.orchestrator.trace import TraceRecorder
from common.a2a_models import AgentCard, ProductDiscoveryResult, Task
from common.config import Settings

logger = logging.getLogger(__name__)


async def call_product_discovery_via_mcp(
    *,
    query: str,
    settings: Settings,
    agent_card: AgentCard | None = None,
    trace: TraceRecorder | None = None,
    mcp_url: str | None = None,
    trace_id: str | None = None,
    max_results: int = 3,
) -> ProductDiscoveryResult:
    """Invoke Product Discovery through its MCP search_products tool."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("MCP client dependencies are not installed") from exc

    resolved_mcp_url = mcp_url or resolve_product_mcp_url(agent_card)

    if trace:
        trace.add_event(
            "Resolved Product Discovery MCP interface",
            status="done",
            detail=resolved_mcp_url,
            agent="product-discovery",
            data={"transport": "streamable-http"},
        )

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
                    **({"trace_id": trace_id} if trace_id else {}),
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


def build_product_debug_url(mcp_url: str, trace_id: str) -> str:
    """Derive the Product Discovery debug endpoint from the advertised MCP URL."""
    parsed = urlparse(mcp_url)
    debug_path = f"/debug/product-discovery/{quote(trace_id, safe='')}"
    return urlunparse(parsed._replace(path=debug_path, params="", query="", fragment=""))


async def poll_product_discovery_debug(
    debug_url: str,
    trace: TraceRecorder,
    stop_event: asyncio.Event,
) -> dict[str, Any] | None:
    """Mirror Product Discovery progress snapshots into the orchestrator trace."""
    latest_payload: dict[str, Any] | None = None

    async with httpx.AsyncClient(timeout=2) as client:
        while not stop_event.is_set():
            payload = await fetch_product_debug_snapshot(client, debug_url)
            if payload and payload != latest_payload:
                latest_payload = payload
                trace.set_product_trace(payload)
            await asyncio.sleep(TASK_POLL_INTERVAL_SECONDS)

        payload = await fetch_product_debug_snapshot(client, debug_url)
        if payload and payload != latest_payload:
            latest_payload = payload
            trace.set_product_trace(payload)

    return latest_payload


async def fetch_product_debug_snapshot(
    client: httpx.AsyncClient,
    debug_url: str,
) -> dict[str, Any] | None:
    try:
        response = await client.get(debug_url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("Product Discovery debug poll failed: %s", exc)
        return None

    return response.json()


async def clear_product_discovery_debug(debug_url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            await client.delete(debug_url)
    except httpx.HTTPError as exc:
        logger.debug("Failed to clear Product Discovery debug snapshot: %s", exc)


def build_mcp_product_trace(query: str, product_result: ProductDiscoveryResult) -> dict[str, Any]:
    """Build a compact trace payload when Product Discovery runs through MCP."""
    finalized_products = [
        {
            "name": product.name,
            "price": product.price,
            "rating": product.rating,
            "source": product.source,
            "url": product.url,
            "confidence": product.confidence,
            "key_features": product.key_features[:3],
            "evidence_urls": product.evidence_urls[:3],
        }
        for product in product_result.products
    ]
    return {
        "query": query,
        "stage": "mcp-complete",
        "search_query": query,
        "search_queries": [query],
        "candidate_count": 0,
        "candidates": [],
        "enriched_candidates": [],
        "candidate_entities": [],
        "filtered_candidates": [],
        "normalized_products": finalized_products,
        "finalized_products": finalized_products,
        "shortlist_reasons": [],
        "rejected_generic_products": [],
        "rejected_entities": [],
        "review_targets": [],
        "summary": product_result.summary,
        "interface": "mcp",
    }


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


def extract_product_trace(task: Task) -> dict[str, Any] | None:
    return extract_named_artifact_json(task, "product-discovery-debug")
