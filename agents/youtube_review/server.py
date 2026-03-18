"""YouTube Product Review Agent A2A server."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI

from agents.youtube_review.agent import get_review_summary
from agents.youtube_review.video_analysis import analyze_video_review, chat_with_video_session
from common.a2a_models import (
    AgentCard,
    AgentCardCapabilities,
    AgentCardInterface,
    AgentCardSkill,
    SendMessageRequest,
    Task,
    TaskStatus,
    VideoChatRequest,
    YouTubeVideoRequest,
    make_agent_message,
    make_artifact,
)
from common.a2a_server import InMemoryTaskStore, create_a2a_routes
from common.config import configure_logging, get_settings
from common.runtime_helpers import format_exception_detail

logger = logging.getLogger(__name__)

_READINESS_STATE: dict[str, Any] = {
    "last_runtime_issue": None,
    "last_runtime_status": "idle",
}


AGENT_CARD = AgentCard(
    protocolVersion="0.3.0",
    name="YouTube Product Review Agent",
    description=(
        "Accepts a YouTube product review URL, extracts transcript evidence, indexes it for grounded chat, "
        "extracts structured product details, and can call Product Discovery via MCP for similar products."
    ),
    url="http://localhost:5001/a2a/v1",
    preferredTransport="JSONRPC",
    additionalInterfaces=[
        AgentCardInterface(url="http://localhost:5001/a2a/v1", transport="JSONRPC"),
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
            id="youtube-video-analysis",
            name="YouTube Video Analysis",
            description=(
                "Given a YouTube product review URL, returns transcript-grounded product details, "
                "optional chat answers, and optionally similar products looked up via Product Discovery MCP."
            ),
            inputModes=["application/json", "text/plain"],
            outputModes=["application/json"],
        ),
        AgentCardSkill(
            id="youtube-product-review",
            name="Legacy Product Review Lookup",
            description=(
                "Compatibility skill that accepts a product name and returns a structured summary "
                "of recent YouTube reviews."
            ),
            inputModes=["text/plain"],
            outputModes=["application/json"],
        ),
        AgentCardSkill(
            id="youtube-transcript-chat",
            name="Transcript Chat",
            description=(
                "Continues a transcript-grounded conversation for an existing YouTube analysis session "
                "and keeps memory scoped to that session."
            ),
            inputModes=["application/json"],
            outputModes=["application/json"],
        ),
    ],
)


def _parse_video_request(request: SendMessageRequest) -> YouTubeVideoRequest | None:
    raw_text = ""
    for part in request.message.parts:
        if part.type == "text" and part.text.strip():
            raw_text = part.text.strip()
            break

    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or not parsed.get("youtube_url"):
        return None

    metadata = request.metadata or {}
    return YouTubeVideoRequest(
        youtube_url=str(parsed.get("youtube_url", "")).strip(),
        chat_message=parsed.get("chat_message"),
        find_similar_products=bool(parsed.get("find_similar_products", False)),
        product_mcp_url=parsed.get("product_mcp_url") or metadata.get("product_mcp_url"),
        legacy_query=parsed.get("legacy_query"),
        session_id=parsed.get("session_id"),
    )


def _parse_chat_request(request: SendMessageRequest) -> VideoChatRequest | None:
    raw_text = ""
    for part in request.message.parts:
        if part.type == "text" and part.text.strip():
            raw_text = part.text.strip()
            break

    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    session_value = parsed.get("session_id")
    chat_value = parsed.get("chat_message")
    if not isinstance(session_value, str) or not isinstance(chat_value, str) or parsed.get("youtube_url"):
        return None

    session_id = session_value.strip()
    chat_message = chat_value.strip()
    if not session_id or not chat_message:
        return None

    return VideoChatRequest(session_id=session_id, chat_message=chat_message)


async def _handle_video_request(
    video_request: YouTubeVideoRequest,
    task_store: InMemoryTaskStore,
    task: Task,
) -> Task:
    settings = get_settings()

    async def publish_video_debug(stage: str, message: str, payload: dict) -> None:
        task_store.update_status(task.id, "working", make_agent_message(message))

    await publish_video_debug("accepted", f"Analyzing YouTube review URL: {video_request.youtube_url}", {})

    try:
        result, _debug_snapshot = await analyze_video_review(
            video_request,
            settings,
            on_step=publish_video_debug,
        )
        if result.transcript_status == "unavailable":
            _READINESS_STATE["last_runtime_status"] = "degraded"
            _READINESS_STATE["last_runtime_issue"] = result.notes
        else:
            _READINESS_STATE["last_runtime_status"] = "available"
            _READINESS_STATE["last_runtime_issue"] = None
        result_artifact = make_artifact(
            name="video-analysis-result",
            text=result.model_dump_json(indent=2),
            metadata={"schema": "UnifiedVideoAnalysisResponse"},
        )

        task.status = TaskStatus(
            state="completed",
            message=make_agent_message(f"Video analysis for {video_request.youtube_url}"),
        )
        task.artifacts = [result_artifact]
        return task
    except Exception as exc:
        logger.exception("Video analysis pipeline failed for: %s", video_request.youtube_url)
        detail = format_exception_detail(exc)
        _READINESS_STATE["last_runtime_status"] = "degraded"
        _READINESS_STATE["last_runtime_issue"] = detail
        task.status = TaskStatus(
            state="failed",
            message=make_agent_message(f"Failed to analyze {video_request.youtube_url}: {detail}"),
        )
        task.artifacts = []
        return task


async def _handle_chat_request(
    chat_request: VideoChatRequest,
    task_store: InMemoryTaskStore,
    task: Task,
) -> Task:
    settings = get_settings()

    async def publish_chat_debug(stage: str, message: str, payload: dict) -> None:
        task_store.update_status(task.id, "working", make_agent_message(message))

    await publish_chat_debug("accepted", f"Continuing transcript chat for session {chat_request.session_id}", {})

    try:
        result, _debug_snapshot = await chat_with_video_session(
            chat_request,
            settings,
            on_step=publish_chat_debug,
        )
        result_artifact = make_artifact(
            name="video-chat-response",
            text=result.model_dump_json(indent=2),
            metadata={"schema": "VideoChatResponse"},
        )
        task.status = TaskStatus(
            state="completed",
            message=make_agent_message(f"Transcript chat response for session {chat_request.session_id}"),
        )
        task.artifacts = [result_artifact]
        return task
    except Exception as exc:
        logger.exception("Transcript chat failed for session: %s", chat_request.session_id)
        detail = format_exception_detail(exc)
        task.status = TaskStatus(
            state="failed",
            message=make_agent_message(f"Failed transcript chat for {chat_request.session_id}: {detail}"),
        )
        task.artifacts = []
        return task


async def _handle_legacy_review_request(
    product_name: str,
    task_store: InMemoryTaskStore,
    task: Task,
) -> Task:
    settings = get_settings()

    async def publish_debug(stage: str, message: str, payload: dict) -> None:
        task_store.update_status(task.id, "working", make_agent_message(message))

    await publish_debug("accepted", f"Collecting YouTube reviews for {product_name}", {})

    try:
        try:
            review = await get_review_summary(product_name, settings, on_step=publish_debug)
        except TypeError as exc:
            if "on_step" not in str(exc):
                raise
            review = await get_review_summary(product_name, settings)

        task.status = TaskStatus(
            state="completed",
            message=make_agent_message(f"Review summary for {product_name}"),
        )
        task.artifacts = [
            make_artifact(
                name="review-summary",
                text=review.model_dump_json(indent=2),
                metadata={"schema": "ReviewSummary"},
            ),
        ]
        return task
    except Exception as exc:
        logger.exception("Legacy review pipeline failed for: %s", product_name)
        detail = format_exception_detail(exc)
        task.status = TaskStatus(
            state="failed",
            message=make_agent_message(f"Failed to get reviews for {product_name}: {detail}"),
        )
        task.artifacts = []
        return task


async def handle_send_message(
    request: SendMessageRequest,
    task_store: InMemoryTaskStore,
    task: Task,
) -> Task:
    """Process either the PDF-aligned video flow or the legacy product-name flow."""
    video_request = _parse_video_request(request)
    if video_request:
        logger.info("Received YouTube video analysis request for: %s", video_request.youtube_url)
        return await _handle_video_request(video_request, task_store, task)

    chat_request = _parse_chat_request(request)
    if chat_request:
        logger.info("Received transcript chat request for session: %s", chat_request.session_id)
        return await _handle_chat_request(chat_request, task_store, task)

    product_name = "Unknown Product"
    for part in request.message.parts:
        if part.type == "text" and part.text.strip():
            product_name = part.text.strip()
            break

    logger.info("Received legacy review request for: %s", product_name)
    return await _handle_legacy_review_request(product_name, task_store, task)


configure_logging()

app = FastAPI(title="YouTube Product Review Agent", version="1.0.0")

a2a_router = create_a2a_routes(
    agent_card_dict=AGENT_CARD.model_dump(),
    handle_send_message=handle_send_message,
)
app.include_router(a2a_router)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "youtube-review"}


@app.get("/readiness")
async def readiness():
    settings = get_settings()
    issues: list[str] = []

    if not getattr(settings, "OPENAI_API_KEY", ""):
        issues.append("OPENAI_API_KEY is not configured")
    if not getattr(settings, "YOUTUBE_API_KEY", ""):
        issues.append("YOUTUBE_API_KEY is not configured")
    if getattr(settings, "ENABLE_PINECONE", False) and not getattr(settings, "PINECONE_API_KEY", ""):
        issues.append("PINECONE_API_KEY is not configured; transcript indexing will fall back to memory")

    last_runtime_issue = _READINESS_STATE.get("last_runtime_issue")
    if last_runtime_issue:
        issues.append(str(last_runtime_issue))

    status = "available" if not issues else "degraded"
    return {
        "status": status,
        "agent": "youtube-review",
        "supports_video_analysis": any(skill.id == "youtube-video-analysis" for skill in AGENT_CARD.skills),
        "supports_transcript_chat": any(skill.id == "youtube-transcript-chat" for skill in AGENT_CARD.skills),
        "supports_legacy_review": any(skill.id == "youtube-product-review" for skill in AGENT_CARD.skills),
        "pinecone_enabled": bool(getattr(settings, "ENABLE_PINECONE", False)),
        "runtime_status": _READINESS_STATE.get("last_runtime_status", "idle"),
        "issues": issues,
    }
