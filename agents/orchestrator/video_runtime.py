"""Video-analysis transport and parsing helpers for the orchestrator."""

from __future__ import annotations

import json

from agents.orchestrator.runtime import extract_named_artifact_json
from common.a2a_models import (
    Task,
    UnifiedVideoAnalysisResponse,
    VideoChatRequest,
    VideoChatResponse,
    YouTubeVideoRequest,
)


def build_video_analysis_payload(
    *,
    youtube_url: str,
    chat_message: str | None = None,
    find_similar_products: bool = False,
    product_mcp_url: str | None = None,
    legacy_query: str | None = None,
    session_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    request = YouTubeVideoRequest(
        youtube_url=youtube_url,
        chat_message=chat_message,
        find_similar_products=find_similar_products,
        product_mcp_url=product_mcp_url,
        legacy_query=legacy_query,
        session_id=session_id,
    )
    return request.model_dump_json(indent=2), {"product_mcp_url": product_mcp_url or ""}


def build_video_chat_payload(*, session_id: str, chat_message: str) -> str:
    request = VideoChatRequest(session_id=session_id, chat_message=chat_message)
    return request.model_dump_json(indent=2)


def extract_video_analysis(task: Task) -> UnifiedVideoAnalysisResponse | None:
    if task.status.state != "completed":
        return None

    data = extract_named_artifact_json(task, "video-analysis-result")
    if data is None:
        return None

    try:
        return UnifiedVideoAnalysisResponse(**data)
    except Exception:
        return None


def extract_video_chat_response(task: Task) -> VideoChatResponse | None:
    if task.status.state != "completed":
        return None

    data = extract_named_artifact_json(task, "video-chat-response")
    if data is None:
        return None

    try:
        return VideoChatResponse(**data)
    except Exception:
        return None


def build_video_review_trace(result: UnifiedVideoAnalysisResponse) -> dict[str, object]:
    return {
        "youtube_url": result.youtube_url,
        "session_id": result.session_id,
        "stage": "video-analysis-complete",
        "video": result.video.model_dump() if result.video else None,
        "transcript_status": result.transcript_status,
        "indexing_status": result.indexing_status,
        "transcript_chunks": [chunk.model_dump() for chunk in result.transcript_chunks[:8]],
        "extracted_product": result.extracted_product.model_dump() if result.extracted_product else None,
        "chat_response": result.chat_response.model_dump() if result.chat_response else None,
        "similar_products": result.similar_products.model_dump() if result.similar_products else None,
        "summary": result.summary,
        "partial": result.partial,
        "notes": result.notes,
    }
