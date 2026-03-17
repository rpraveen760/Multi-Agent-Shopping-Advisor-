"""Central Orchestrator - FastAPI server.

Endpoints:
    GET  /         - Serve the frontend UI.
    POST /query    - Run the full orchestration pipeline for a shopping query.
    GET  /health   - Health check for the orchestrator process.
    GET  /status   - Aggregated downstream agent readiness.
    GET  /agents   - List discovered downstream agents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.orchestrator.graph import run_query
from common.a2a_client import A2AClient
from common.a2a_models import UnifiedResponse
from common.config import configure_logging, get_settings

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

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


def _agent_endpoints(settings) -> dict[str, str]:
    return {
        "product-discovery": settings.PRODUCT_AGENT_CARD_URL,
        "youtube-review": settings.YOUTUBE_AGENT_CARD_URL,
    }


async def _discover_agents() -> list[dict]:
    settings = get_settings()
    client = A2AClient()
    agents = []

    try:
        for name, card_url in _agent_endpoints(settings).items():
            try:
                card = await client.discover_agent(card_url)
                agents.append(
                    {
                        "name": card.name,
                        "url": card.url,
                        "description": card.description,
                        "status": "available",
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
