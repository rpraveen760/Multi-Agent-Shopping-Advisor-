"""Central Orchestrator - FastAPI server."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from agents.orchestrator.discovery import (
    agent_endpoints,
    resolve_mcp_interface_url,
    verify_product_discovery_mcp,
    verify_youtube_review_runtime,
)
from agents.orchestrator.graph import run_query
from agents.orchestrator.runtime import resolve_send_message_result, task_status_text
from agents.orchestrator.video_runtime import build_video_chat_payload, extract_video_chat_response
from common.a2a_client import A2AClient
from common.a2a_models import Task, UnifiedResponse, VideoChatRequest, VideoChatResponse
from common.config import configure_logging, get_settings
from common.runtime_helpers import format_exception_detail

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
    query: str | None = Field(
        default=None,
        description="Legacy natural-language shopping query",
        min_length=1,
        max_length=500,
        examples=["best wireless earbuds under $100"],
    )
    youtube_url: str | None = Field(
        default=None,
        description="Primary PDF flow: a YouTube product review URL",
        min_length=5,
        max_length=1000,
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier used when refreshing an existing video transcript session",
        min_length=1,
        max_length=200,
    )
    chat_message: str | None = Field(
        default=None,
        description="Optional follow-up question to answer from transcript evidence",
        min_length=1,
        max_length=500,
    )
    find_similar_products: bool = Field(
        default=False,
        description="Whether the YouTube agent should trigger Product Discovery via MCP for similar products",
    )

    @model_validator(mode="after")
    def validate_inputs(self):
        if not (self.query or self.youtube_url):
            raise ValueError("Either query or youtube_url is required")
        return self


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
    finally:
        await client.close()

    return agents


def _agent_endpoints(settings) -> dict[str, str]:
    return agent_endpoints(settings)


def _resolve_mcp_interface_url(card) -> str | None:
    return resolve_mcp_interface_url(card)


async def _verify_product_discovery_mcp(card) -> tuple[bool, str | None]:
    return await verify_product_discovery_mcp(card)


async def _verify_youtube_review_runtime(card) -> tuple[bool, str | None]:
    return await verify_youtube_review_runtime(card)


async def _assess_agent_readiness(name: str, card) -> tuple[str, str]:
    description = getattr(card, "description", "") or ""

    if name == "youtube-review":
        ok, detail = await _verify_youtube_review_runtime(card)
        if ok:
            return "available", description
        degraded_description = description
        if detail:
            degraded_description = f"{description} (video-analysis unavailable: {detail})".strip()
        return "degraded", degraded_description

    if name != "product-discovery":
        return "available", description

    ok, detail = await _verify_product_discovery_mcp(card)
    if ok:
        return "available", description

    degraded_description = description
    if detail:
        degraded_description = f"{description} (MCP unavailable: {detail})".strip()
    return "degraded", degraded_description


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/")
async def frontend_index():
    """Serve the submission UI."""
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend index not found")
    return FileResponse(FRONTEND_INDEX)


@app.post("/query", response_model=UnifiedResponse)
async def query_endpoint(request: QueryRequest):
    """Run the orchestrator for either a legacy shopping query or a YouTube review URL."""
    effective_query = (request.query or request.youtube_url or "").strip()
    logger.info("Received orchestrator request: %s", effective_query)

    try:
        settings = get_settings()
        result: UnifiedResponse = await run_query(
            user_query=effective_query,
            settings=settings,
            youtube_url=request.youtube_url,
            session_id=request.session_id,
            chat_message=request.chat_message,
            find_similar_products=request.find_similar_products,
        )
        return result

    except Exception as exc:
        logger.exception("Query pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Orchestration failed: {exc}",
        )


@app.post("/video/chat", response_model=VideoChatResponse)
async def video_chat_endpoint(request: VideoChatRequest):
    """Continue a transcript-grounded conversation for an existing YouTube session."""
    settings = get_settings()
    client = A2AClient()

    try:
        youtube_card = await client.discover_agent(settings.YOUTUBE_AGENT_CARD_URL)
        skill_ids = {skill.id for skill in getattr(youtube_card, "skills", [])}
        if "youtube-transcript-chat" not in skill_ids:
            raise HTTPException(
                status_code=503,
                detail="The discovered YouTube agent does not advertise transcript chat support.",
            )
        payload = build_video_chat_payload(
            session_id=request.session_id,
            chat_message=request.chat_message,
        )
        result = await client.send_message(youtube_card.url, payload)
        result = await resolve_send_message_result(
            result,
            client=client,
            agent_url=youtube_card.url,
            timeout_seconds=settings.A2A_CLIENT_TIMEOUT_SECONDS,
        )

        if isinstance(result, Task):
            parsed = extract_video_chat_response(result)
            if parsed is not None:
                return parsed

            detail = task_status_text(result) or "Transcript chat returned no parseable response."
            if result.status.state == "failed":
                status_code = 404 if "No transcript chat session exists" in detail else 502
                raise HTTPException(status_code=status_code, detail=detail)
            raise HTTPException(status_code=502, detail=detail)

        raise HTTPException(status_code=502, detail="Transcript chat returned no task payload.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcript chat pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Transcript chat failed: {format_exception_detail(exc)}",
        )
    finally:
        await client.close()


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
