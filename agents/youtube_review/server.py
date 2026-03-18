"""YouTube Product Review Agent — A2A server.

Accepts a product name via A2A SendMessage, searches YouTube for recent reviews,
extracts transcripts, and returns a structured ReviewSummary via GPT-4o-mini.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI

from agents.youtube_review.agent import get_review_summary
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


def _format_exception_detail(exc: Exception) -> str:
    text = str(exc).strip()
    exc_name = type(exc).__name__
    if not text:
        return exc_name
    if text.startswith(exc_name):
        return text
    return f"{exc_name}: {text}"

# ── Agent Card ────────────────────────────────────────────────────

AGENT_CARD = AgentCard(
    protocolVersion="0.3.0",
    name="YouTube Product Review Agent",
    description="Searches YouTube for recent product reviews and returns structured review evidence.",
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
    defaultInputModes=["text/plain"],
    defaultOutputModes=["application/json"],
    skills=[
        AgentCardSkill(
            id="youtube-product-review",
            name="YouTube Product Review",
            description=(
                "Given a product name, returns a structured summary of recent YouTube reviews "
                "with pros, cons, sentiment, confidence, and source links."
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
    """Process an A2A SendMessage request with real YouTube + GPT-4o-mini logic."""
    # Extract product name from the incoming message
    product_name = "Unknown Product"
    for part in request.message.parts:
        if part.type == "text" and part.text.strip():
            product_name = part.text.strip()
            break

    logger.info("Received review request for: %s", product_name)
    settings = get_settings()
    debug_snapshot = {
        "product_name": product_name,
        "search_query": "",
        "stage": "accepted",
        "initial_videos": [],
        "ranked_videos": [],
        "evidence": [],
        "summary": None,
    }

    async def publish_debug(stage: str, message: str, payload: dict) -> None:
        debug_snapshot.update(payload)
        debug_snapshot["stage"] = stage
        task_store.update_status(
            task.id,
            "working",
            make_agent_message(message),
        )
        task_store.set_artifacts(
            task.id,
            [
                make_artifact(
                    name="review-debug",
                    text=json.dumps(debug_snapshot, indent=2),
                    metadata={"schema": "ReviewDebugTrace", "stage": stage},
                )
            ],
        )

    await publish_debug(
        "accepted",
        f"Collecting YouTube reviews for {product_name}",
        debug_snapshot,
    )

    try:
        try:
            review = await get_review_summary(
                product_name,
                settings,
                on_step=publish_debug,
            )
        except TypeError as exc:
            if "on_step" not in str(exc):
                raise
            review = await get_review_summary(
                product_name,
                settings,
            )
        result_artifact = make_artifact(
            name="review-summary",
            text=review.model_dump_json(indent=2),
            metadata={"schema": "ReviewSummary"},
        )
        debug_artifact = make_artifact(
            name="review-debug",
            text=json.dumps(debug_snapshot, indent=2),
            metadata={"schema": "ReviewDebugTrace", "stage": "completed"},
        )

        task.status = TaskStatus(
            state="completed",
            message=make_agent_message(f"Review summary for {product_name}"),
        )
        task.artifacts = [result_artifact, debug_artifact]

    except Exception as exc:
        logger.exception("Review pipeline failed for: %s", product_name)
        detail = _format_exception_detail(exc)
        task.status = TaskStatus(
            state="failed",
            message=make_agent_message(f"Failed to get reviews for {product_name}: {detail}"),
        )
        task.artifacts = []

    return task


# ── FastAPI App ───────────────────────────────────────────────────

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
