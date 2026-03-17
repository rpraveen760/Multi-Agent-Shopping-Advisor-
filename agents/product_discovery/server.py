"""Product Discovery Agent — A2A server + MCP facade.

The A2A facade delegates to the same canonical `search_products` capability
that the MCP server exposes. One source of truth, two access methods.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from agents.product_discovery.agent import search_products
from common.a2a_models import (
    AgentCard,
    AgentCardCapabilities,
    AgentCardInterface,
    AgentCardSkill,
    SendMessageRequest,
    Task,
    TaskStatus,
    make_agent_message,
    make_artifact,
)
from common.a2a_server import InMemoryTaskStore, create_a2a_routes
from common.config import configure_logging, get_settings

logger = logging.getLogger(__name__)

# ── Agent Card ────────────────────────────────────────────────────

AGENT_CARD = AgentCard(
    protocolVersion="0.3.0",
    name="Product Discovery Agent",
    description=(
        "Discovers products matching a query with normalized pricing, ratings, "
        "provenance, and purchase links. Also exposed as an MCP tool."
    ),
    url="http://localhost:5002/a2a/v1",
    preferredTransport="JSONRPC",
    additionalInterfaces=[
        AgentCardInterface(url="http://localhost:5002/a2a/v1", transport="JSONRPC"),
    ],
    version="1.0.0",
    capabilities=AgentCardCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["application/json"],
    skills=[
        AgentCardSkill(
            id="product-search",
            name="Product Search",
            description=(
                "Given a shopping query, returns normalized product candidates "
                "with price, rating, source URLs, and confidence scores."
            ),
            inputModes=["text/plain"],
            outputModes=["application/json"],
        ),
    ],
)

# ── Handler ───────────────────────────────────────────────────────


async def handle_send_message(
    request: SendMessageRequest,
    task_store: InMemoryTaskStore,
    task: Task,
) -> Task:
    """Process an A2A SendMessage request via the canonical search_products capability."""
    # Extract query from the incoming message
    query = "unknown query"
    for part in request.message.parts:
        if part.type == "text" and part.text.strip():
            query = part.text.strip()
            break

    logger.info("Received product search for: %s", query)
    settings = get_settings()
    task_store.update_status(
        task.id,
        "working",
        make_agent_message(f"Searching for products matching: {query}"),
    )

    try:
        result = await search_products(query=query, settings=settings)

        artifact = make_artifact(
            name="product-discovery-result",
            text=result.model_dump_json(indent=2),
            metadata={"schema": "ProductDiscoveryResult"},
        )

        task.status = TaskStatus(
            state="completed",
            message=make_agent_message(f"Product search results for: {query}"),
        )
        task.artifacts = [artifact]

    except Exception as exc:
        logger.exception("Product discovery failed for: %s", query)
        task.status = TaskStatus(
            state="failed",
            message=make_agent_message(f"Failed to search products for {query}: {exc}"),
        )
        task.artifacts = []

    return task


# ── FastAPI App ───────────────────────────────────────────────────

configure_logging()

app = FastAPI(title="Product Discovery Agent", version="1.0.0")

a2a_router = create_a2a_routes(
    agent_card_dict=AGENT_CARD.model_dump(),
    handle_send_message=handle_send_message,
)
app.include_router(a2a_router)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "product-discovery"}
