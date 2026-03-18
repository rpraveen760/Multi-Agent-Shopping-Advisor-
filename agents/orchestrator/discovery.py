"""Shared downstream discovery and readiness helpers for the orchestrator."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from common.a2a_client import A2AClient
from common.a2a_models import AgentCard
from common.config import Settings


def agent_endpoints(settings: Settings) -> dict[str, str]:
    return {
        "product-discovery": settings.PRODUCT_AGENT_CARD_URL,
        "youtube-review": settings.YOUTUBE_AGENT_CARD_URL,
    }


def resolve_mcp_interface_url(card: AgentCard | None) -> str | None:
    interfaces = getattr(card, "additionalInterfaces", None) or []
    for interface in interfaces:
        if getattr(interface, "transport", "").upper() != "MCP":
            continue
        url = getattr(interface, "url", "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
    return None


def resolve_readiness_url(card: AgentCard | None) -> str | None:
    if card is None or not getattr(card, "url", None):
        return None

    parsed = urlparse(card.url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, "/readiness", "", "", ""))


async def verify_product_discovery_mcp(
    card: AgentCard,
    *,
    timeout_seconds: int = 5,
) -> tuple[bool, str | None]:
    """Verify that the discovered Product Discovery MCP endpoint is usable."""
    mcp_url = resolve_mcp_interface_url(card)
    if not mcp_url:
        return False, "Agent Card did not advertise an HTTP MCP interface"

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        return False, "MCP client dependencies are unavailable in the orchestrator"

    try:
        async with streamable_http_client(mcp_url) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
    except Exception as exc:
        return False, f"MCP interface check failed: {exc}"

    tool_names = {tool.name for tool in tools.tools}
    if "search_products" not in tool_names:
        return False, "MCP interface did not advertise search_products"
    if "find_similar_products" not in tool_names:
        return False, "MCP interface did not advertise find_similar_products"

    return True, None


async def verify_youtube_review_runtime(
    card: AgentCard,
    *,
    timeout_seconds: int = 5,
) -> tuple[bool, str | None]:
    skill_ids = {skill.id for skill in getattr(card, "skills", [])}
    if "youtube-video-analysis" not in skill_ids:
        return False, "Agent Card did not advertise youtube-video-analysis"

    readiness_url = resolve_readiness_url(card)
    if not readiness_url:
        return False, "Could not derive the YouTube agent readiness endpoint"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(readiness_url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return False, f"YouTube readiness check failed: {exc}"

    status = payload.get("status")
    if status == "available":
        return True, None

    issues = payload.get("issues") or []
    if isinstance(issues, list) and issues:
        return False, "; ".join(str(item) for item in issues)
    return False, "YouTube readiness endpoint reported degraded status"


async def assess_agent_readiness(name: str, card: AgentCard) -> tuple[str, str]:
    """Return the runtime readiness status and description for a discovered agent."""
    description = getattr(card, "description", "") or ""

    if name == "youtube-review":
        ok, detail = await verify_youtube_review_runtime(card)
        if ok:
            return "available", description
        degraded_description = description
        if detail:
            degraded_description = f"{description} (video-analysis unavailable: {detail})".strip()
        return "degraded", degraded_description

    if name != "product-discovery":
        return "available", description

    ok, detail = await verify_product_discovery_mcp(card)
    if ok:
        return "available", description

    degraded_description = description
    if detail:
        degraded_description = f"{description} (MCP unavailable: {detail})".strip()
    return "degraded", degraded_description


async def discover_seeded_agents(
    settings: Settings,
    client: A2AClient,
) -> tuple[dict[str, AgentCard], dict[str, str], list[str]]:
    """Discover downstream agent cards from configured URLs."""
    discovered: dict[str, AgentCard] = {}
    card_urls: dict[str, str] = {}
    errors: list[str] = []

    for name, card_url in agent_endpoints(settings).items():
        try:
            card = await client.discover_agent(card_url)
            discovered[name] = card
            card_urls[name] = card_url
        except Exception as exc:
            errors.append(f"Failed to discover {name} at {card_url}: {exc}")

    return discovered, card_urls, errors


async def discover_agent_statuses(
    settings: Settings,
    client: A2AClient,
) -> list[dict[str, Any]]:
    """Discover agents and return readiness-rich status rows for the UI."""
    agents: list[dict[str, Any]] = []
    for name, card_url in agent_endpoints(settings).items():
        try:
            card = await client.discover_agent(card_url)
            readiness, description = await assess_agent_readiness(name, card)
            agents.append(
                {
                    "name": card.name,
                    "url": card.url,
                    "description": description,
                    "status": readiness,
                    "skills": [skill.model_dump() for skill in card.skills],
                }
            )
        except Exception as exc:
            agents.append(
                {
                    "name": name,
                    "url": card_url,
                    "description": f"Agent unavailable: {exc}",
                    "status": "unavailable",
                    "skills": [],
                }
            )

    return agents
