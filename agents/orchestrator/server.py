"""Central Orchestrator - FastAPI server.

Endpoints:
    GET  /         - Serve the frontend UI.
    POST /query    - Run the full orchestration pipeline for a shopping query.
    GET  /health   - Health check for the orchestrator process.
    GET  /status   - Aggregated downstream agent readiness.
    GET  /agents   - List discovered downstream agents.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.orchestrator.graph import run_query
from agents.orchestrator.trace import QueryTraceSnapshot, TraceRecorder, TraceStore
from common.a2a_client import A2AClient
from common.a2a_models import UnifiedResponse
from common.config import configure_logging, get_settings

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
TRACE_INDEX = FRONTEND_DIR / "trace.html"
TRACE_STORE = TraceStore()

# ── FastAPI App ──────────────────────────────────────────────────

configure_logging()

app = FastAPI(
    title="Central Orchestrator",
    description="Federated multi-agent shopping advisor with LangGraph orchestration",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Request / Response Models ────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str = Field(
        ...,
        description="Natural-language shopping query",
        min_length=1,
        max_length=500,
        examples=["best wireless earbuds under $100"],
    )


class QueryResponse(BaseModel):
    """Response body from the /query endpoint."""
    query: str
    recommendations: list[dict]
    sources: list[dict]
    partial: bool = False
    notes: str | None = None


class AgentStatusItem(BaseModel):
    name: str
    url: str
    description: str
    status: str
    skills: list[dict]


class StatusResponse(BaseModel):
    status: str
    orchestrator: str
    available_agents: int
    total_agents: int
    agents: list[AgentStatusItem]


class TraceStartResponse(BaseModel):
    run_id: str
    query: str
    status: str
    status_url: str


def _agent_endpoints(settings) -> dict[str, str]:
    return {
        "product-discovery": settings.PRODUCT_AGENT_CARD_URL,
        "youtube-review": settings.YOUTUBE_AGENT_CARD_URL,
    }


def _resolve_mcp_interface_url(card) -> str | None:
    interfaces = getattr(card, "additionalInterfaces", None) or []
    for interface in interfaces:
        if getattr(interface, "transport", "").upper() != "MCP":
            continue
        url = getattr(interface, "url", "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
    return None


async def _verify_product_discovery_mcp(card) -> tuple[bool, str | None]:
    """Verify that the discovered Product Discovery MCP endpoint is usable."""
    mcp_url = _resolve_mcp_interface_url(card)
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
                read_timeout_seconds=timedelta(seconds=5),
            ) as session:
                await session.initialize()
                tools = await session.list_tools()
    except Exception as exc:
        return False, f"MCP interface check failed: {exc}"

    if not any(tool.name == "search_products" for tool in tools.tools):
        return False, "MCP interface did not advertise search_products"

    return True, None


async def _assess_agent_readiness(name: str, card) -> tuple[str, str]:
    """Return the runtime readiness status and description for a discovered agent."""
    description = getattr(card, "description", "") or ""

    if name != "product-discovery":
        return "available", description

    ok, detail = await _verify_product_discovery_mcp(card)
    if ok:
        return "available", description

    degraded_description = description
    if detail:
        degraded_description = f"{description} (MCP unavailable: {detail})".strip()
    return "degraded", degraded_description


async def _discover_agents() -> list[dict]:
    settings = get_settings()
    client = A2AClient()
    agents = []

    try:
        for name, card_url in _agent_endpoints(settings).items():
            try:
                card = await client.discover_agent(card_url)
                readiness, description = await _assess_agent_readiness(name, card)
                agents.append(
                    {
                        "name": card.name,
                        "url": card.url,
                        "description": description,
                        "status": readiness,
                        "skills": [s.model_dump() for s in card.skills],
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
    finally:
        await client.close()

    return agents


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/")
async def frontend_index():
    """Serve the submission UI."""
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend index not found")
    return FileResponse(FRONTEND_INDEX)


@app.get("/trace")
async def trace_index():
    """Serve the live execution trace UI."""
    if not TRACE_INDEX.exists():
        raise HTTPException(status_code=404, detail="Trace frontend not found")
    return FileResponse(TRACE_INDEX)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Run the orchestration pipeline for a shopping query.

    Discovers agents, routes the query, executes agent calls in parallel
    where possible, and synthesizes a ranked response.
    """
    logger.info("Received query: %s", request.query)

    try:
        settings = get_settings()
        result: UnifiedResponse = await run_query(
            user_query=request.query,
            settings=settings,
        )

        return QueryResponse(
            query=result.query,
            recommendations=[r.model_dump() for r in result.recommendations],
            sources=[s.model_dump() for s in result.sources],
            partial=result.partial,
            notes=result.notes,
        )

    except Exception as exc:
        logger.exception("Query pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Orchestration failed: {exc}",
        )


async def _run_traced_query(run_id: str, query: str) -> None:
    """Execute a query in the background while publishing live trace events."""
    recorder = TraceRecorder(TRACE_STORE, run_id)
    recorder.mark_running()
    recorder.add_event(
        "Trace started",
        status="done",
        detail=f"Queued live trace for query: {query}",
    )

    try:
        settings = get_settings()
        result = await run_query(
            user_query=query,
            settings=settings,
            trace_recorder=recorder,
        )
        if isinstance(result, UnifiedResponse):
            recorder.set_final_response(result)
        recorder.mark_completed()
    except Exception as exc:
        logger.exception("Traced query pipeline failed: %s", exc)
        recorder.mark_failed(str(exc))
        recorder.add_event(
            "Trace failed",
            status="error",
            detail=str(exc),
        )


def _launch_traced_query(run_id: str, query: str) -> None:
    """Launch a traced query without blocking the request thread."""
    asyncio.create_task(_run_traced_query(run_id, query))


@app.post("/query/trace", response_model=TraceStartResponse, status_code=202)
async def start_trace_query(request: QueryRequest):
    """Start a traced query run and return a polling handle."""
    snapshot = TRACE_STORE.create_run(request.query)
    _launch_traced_query(snapshot.run_id, request.query)
    return TraceStartResponse(
        run_id=snapshot.run_id,
        query=request.query,
        status=snapshot.status,
        status_url=f"/query/trace/{snapshot.run_id}",
    )


@app.get("/query/trace/{run_id}", response_model=QueryTraceSnapshot)
async def get_trace_query(run_id: str):
    """Return the latest live trace snapshot for a run."""
    snapshot = TRACE_STORE.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Trace run not found")
    return snapshot


@app.get("/health")
async def health():
    """Health check endpoint for the orchestrator process."""
    return {"status": "ok", "agent": "orchestrator"}


@app.get("/status", response_model=StatusResponse)
async def status():
    """Aggregate downstream agent readiness for the UI status badge."""
    agents = await _discover_agents()
    available_agents = sum(1 for agent in agents if agent["status"] == "available")
    total_agents = len(agents)

    if available_agents == total_agents:
        overall_status = "ok"
    elif available_agents > 0:
        overall_status = "degraded"
    else:
        overall_status = "unavailable"

    return StatusResponse(
        status=overall_status,
        orchestrator="ok",
        available_agents=available_agents,
        total_agents=total_agents,
        agents=[AgentStatusItem(**agent) for agent in agents],
    )


@app.get("/agents")
async def list_agents():
    """Discover and list downstream agents."""
    agents = await _discover_agents()
    return {"agents": agents}
