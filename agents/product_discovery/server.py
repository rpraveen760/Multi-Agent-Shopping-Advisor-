"""Product Discovery Agent — A2A server + MCP facade.

The A2A facade delegates to the same canonical `search_products` capability
that the MCP server exposes. One source of truth, two access methods.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

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

try:
    from agents.product_discovery.mcp_server import mcp as product_discovery_mcp
except ImportError:  # pragma: no cover - depends on optional MCP package
    product_discovery_mcp = None

try:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
except ImportError:  # pragma: no cover - depends on optional MCP package
    StreamableHTTPSessionManager = None

logger = logging.getLogger(__name__)


class ProductDiscoveryMCPApp:
    """Mounted ASGI bridge that serves MCP requests from the FastAPI process."""

    def __init__(self, fastapi_app: FastAPI) -> None:
        self.fastapi_app = fastapi_app

    async def __call__(self, scope, receive, send) -> None:
        manager = getattr(self.fastapi_app.state, "product_mcp_manager", None)
        if manager is None:
            response = JSONResponse(
                {"detail": "Product Discovery MCP transport is unavailable"},
                status_code=503,
            )
            await response(scope, receive, send)
            return

        await manager.handle_request(scope, receive, send)

# ── Agent Card ────────────────────────────────────────────────────

AGENT_CARD = AgentCard(
    protocolVersion="0.3.0",
    name="Product Discovery Agent",
    description=(
        "Discovers products matching a query and exposes MCP tools for both legacy product search "
        "and structured similar-product lookup backed by a mock catalog."
    ),
    url="http://localhost:5002/a2a/v1",
    preferredTransport="JSONRPC",
    additionalInterfaces=[
        AgentCardInterface(url="http://localhost:5002/a2a/v1", transport="JSONRPC"),
        AgentCardInterface(url="http://localhost:5002/mcp", transport="MCP"),
    ],
    version="1.0.0",
    capabilities=AgentCardCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    defaultInputModes=["application/json", "text/plain"],
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
        AgentCardSkill(
            id="similar-product-lookup",
            name="Similar Product Lookup",
            description=(
                "Accepts structured product details from the YouTube review agent and returns similar "
                "products from the mock product catalog through the MCP interface."
            ),
            inputModes=["application/json"],
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
        result = await search_products(
            query=query,
            settings=settings,
        )
        result_artifact = make_artifact(
            name="product-discovery-result",
            text=result.model_dump_json(indent=2),
            metadata={"schema": "ProductDiscoveryResult"},
        )

        task.status = TaskStatus(
            state="completed",
            message=make_agent_message(f"Product search results for: {query}"),
        )
        task.artifacts = [result_artifact]

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    if StreamableHTTPSessionManager is None or product_discovery_mcp is None:
        app.state.product_mcp_manager = None
        yield
        return

    manager = StreamableHTTPSessionManager(
        app=product_discovery_mcp,
        json_response=False,
        stateless=False,
    )
    app.state.product_mcp_manager = manager
    async with manager.run():
        yield
    app.state.product_mcp_manager = None


app = FastAPI(title="Product Discovery Agent", version="1.0.0", lifespan=lifespan)

a2a_router = create_a2a_routes(
    agent_card_dict=AGENT_CARD.model_dump(),
    handle_send_message=handle_send_message,
)
app.include_router(a2a_router)
app.mount("/mcp", ProductDiscoveryMCPApp(app), name="product-discovery-mcp")


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "product-discovery"}
