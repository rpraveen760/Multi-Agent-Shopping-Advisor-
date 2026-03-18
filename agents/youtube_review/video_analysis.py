"""YouTube URL ingestion, transcript indexing, chat, and MCP similar-product lookup."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlparse

from googleapiclient.discovery import build
from openai import AsyncOpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from agents.youtube_review.rag import answer_chat_over_transcript, index_transcript
from agents.youtube_review.sessions import SESSION_STORE
from common.a2a_models import (
    ExtractedProductDetails,
    SimilarProductsRequest,
    SimilarProductsResponse,
    TranscriptChunk,
    VideoChatRequest,
    VideoChatResponse,
    UnifiedVideoAnalysisResponse,
    VideoMetadata,
    YouTubeVideoRequest,
)
from common.config import Settings
from common.runtime_helpers import close_async_resource, format_exception_detail

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, dict[str, Any]], Awaitable[None] | None]

PRODUCT_DETAILS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtractedProductDetails",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "category": {"type": ["string", "null"]},
                "features": {"type": "array", "items": {"type": "string"}},
                "price": {"type": ["string", "null"]},
                "summary": {"type": "string"},
                "evidence_quotes": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": [
                "product_name",
                "category",
                "features",
                "price",
                "summary",
                "evidence_quotes",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}


async def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    result = callback(stage, message, json.loads(json.dumps(payload)))
    if asyncio.iscoroutine(result):
        await result


def _parse_youtube_url(youtube_url: str) -> str:
    parsed = urlparse(youtube_url.strip())
    host = parsed.netloc.lower()
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    if "youtube.com" not in host:
        raise ValueError("Only youtube.com and youtu.be URLs are supported")

    if parsed.path.startswith("/watch"):
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if video_id:
            return video_id
    if parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
        video_id = parsed.path.strip("/").split("/")[1]
        if video_id:
            return video_id

    raise ValueError("Could not extract a YouTube video id from the supplied URL")


def _fetch_video_metadata(video_id: str, api_key: str, youtube_url: str) -> VideoMetadata:
    youtube = build("youtube", "v3", developerKey=api_key)
    response = (
        youtube.videos()
        .list(id=video_id, part="snippet,statistics")
        .execute()
    )
    items = response.get("items", [])
    if not items:
        raise ValueError("YouTube metadata lookup returned no video for that URL")

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    return VideoMetadata(
        video_id=video_id,
        video_url=youtube_url,
        title=snippet.get("title", video_id),
        channel=snippet.get("channelTitle", "Unknown channel"),
        published_at=(snippet.get("publishedAt") or "")[:10] or None,
        view_count=stats.get("viewCount"),
    )


def _fetch_transcript_text(video_id: str) -> str:
    transcript_api = YouTubeTranscriptApi()
    transcript = transcript_api.fetch(video_id)
    segments = getattr(transcript, "snippets", transcript)
    parts: list[str] = []
    for snippet in segments:
        text = getattr(snippet, "text", "") or ""
        if text.strip():
            parts.append(text.strip())
    return " ".join(parts).strip()


async def _extract_product_details(
    video: VideoMetadata,
    transcript_chunks: list[TranscriptChunk],
    settings: Settings,
    *,
    client: AsyncOpenAI | None = None,
) -> ExtractedProductDetails:
    evidence = "\n\n".join(
        f"[Chunk {chunk.chunk_index}] {chunk.text}"
        for chunk in transcript_chunks[:6]
    )
    if not evidence:
        raise ValueError("No transcript evidence was available to extract product details")

    owned_client = client is None
    runtime_client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        response = await runtime_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=0.1,
            response_format=PRODUCT_DETAILS_SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured product details from a single YouTube review transcript. "
                        "Only use facts grounded in the transcript snippets."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Video title: {video.title}\nChannel: {video.channel}\nVideo URL: {video.video_url}\n\n"
                        f"Transcript evidence:\n{evidence}\n\n"
                        "Extract the reviewed product, category, features, any mentioned price, a short grounded summary, and direct evidence quotes."
                    ),
                },
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return ExtractedProductDetails(**data)
    finally:
        if owned_client:
            await close_async_resource(runtime_client)


async def _call_product_discovery_mcp(
    request: SimilarProductsRequest,
    *,
    mcp_url: str,
    settings: Settings,
) -> SimilarProductsResponse:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MCP client dependencies are not installed") from exc

    async with streamable_http_client(mcp_url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            if not any(tool.name == "find_similar_products" for tool in tools.tools):
                raise RuntimeError("Product Discovery MCP server did not advertise find_similar_products")
            result = await session.call_tool("find_similar_products", request.model_dump())

    if result.isError:
        raise RuntimeError("Product Discovery MCP tool returned an error")
    if result.structuredContent:
        return SimilarProductsResponse(**result.structuredContent)

    text_parts = [item.text for item in result.content if getattr(item, "type", None) == "text"]
    if not text_parts:
        raise RuntimeError("find_similar_products returned no text content")
    return SimilarProductsResponse(**json.loads(text_parts[0]))


async def analyze_video_review(
    request: YouTubeVideoRequest,
    settings: Settings,
    *,
    on_step: ProgressCallback | None = None,
) -> tuple[UnifiedVideoAnalysisResponse, dict[str, Any]]:
    """Analyze a YouTube review URL and optionally request similar products."""
    debug_snapshot: dict[str, Any] = {
        "youtube_url": request.youtube_url,
        "session_id": request.session_id,
        "stage": "accepted",
        "video": None,
        "transcript_status": "accepted",
        "indexing_status": "pending",
        "transcript_chunks": [],
        "extracted_product": None,
        "chat_response": None,
        "similar_products": None,
        "product_mcp_url": request.product_mcp_url,
    }
    await _emit_progress(on_step, "accepted", "Accepted YouTube video analysis request.", debug_snapshot)

    try:
        video_id = _parse_youtube_url(request.youtube_url)
    except Exception as exc:
        detail = format_exception_detail(exc)
        response = UnifiedVideoAnalysisResponse(
            youtube_url=request.youtube_url,
            transcript_status="rejected",
            indexing_status="not-started",
            partial=True,
            notes=detail,
        )
        debug_snapshot["stage"] = "failed"
        debug_snapshot["notes"] = detail
        return response, debug_snapshot

    await _emit_progress(on_step, "metadata-started", "Resolving video metadata from YouTube.", debug_snapshot)
    try:
        video = await asyncio.to_thread(
            _fetch_video_metadata,
            video_id,
            settings.YOUTUBE_API_KEY,
            request.youtube_url,
        )
    except Exception as exc:
        detail = format_exception_detail(exc)
        response = UnifiedVideoAnalysisResponse(
            youtube_url=request.youtube_url,
            transcript_status="metadata-failed",
            indexing_status="not-started",
            partial=True,
            notes=detail,
        )
        debug_snapshot["stage"] = "failed"
        debug_snapshot["notes"] = detail
        return response, debug_snapshot

    debug_snapshot["video"] = video.model_dump()
    await _emit_progress(on_step, "metadata-complete", f"Loaded video metadata for {video.title}.", debug_snapshot)

    await _emit_progress(on_step, "transcript-started", "Fetching transcript or subtitles for the video.", debug_snapshot)
    try:
        transcript_text = await asyncio.to_thread(_fetch_transcript_text, video_id)
    except Exception as exc:
        detail = format_exception_detail(exc)
        response = UnifiedVideoAnalysisResponse(
            youtube_url=request.youtube_url,
            video=video,
            transcript_status="unavailable",
            indexing_status="not-started",
            partial=True,
            notes=f"Transcript extraction failed: {detail}",
        )
        debug_snapshot["stage"] = "transcript-failed"
        debug_snapshot["notes"] = response.notes
        return response, debug_snapshot

    debug_snapshot["transcript_status"] = "transcript-complete"
    await _emit_progress(on_step, "transcript-complete", "Transcript extraction succeeded.", debug_snapshot)

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
    try:
        indexed, reindexed = await index_transcript(
            video=video,
            transcript_text=transcript_text,
            settings=settings,
            client=client,
        )
        debug_snapshot["indexing_status"] = "indexed" if reindexed else "duplicate-skip"
        debug_snapshot["transcript_chunks"] = [chunk.model_dump() for chunk in indexed.chunks[:8]]
        await _emit_progress(
            on_step,
            "indexed",
            "Transcript chunks were indexed for grounded retrieval.",
            debug_snapshot,
        )

        product_details = await _extract_product_details(
            video,
            indexed.chunks,
            settings,
            client=client,
        )
        debug_snapshot["extracted_product"] = product_details.model_dump()
        session = SESSION_STORE.create(
            video=video,
            transcript_hash=getattr(indexed, "transcript_hash", video.video_id),
            extracted_product=product_details,
            session_id=request.session_id,
        )
        debug_snapshot["session_id"] = session.session_id
        await _emit_progress(
            on_step,
            "product-details-complete",
            f"Extracted structured product details for {product_details.product_name}.",
            debug_snapshot,
        )

        chat_response = None
        if request.chat_message:
            raw_chat_response = await answer_chat_over_transcript(
                video=video,
                chat_message=request.chat_message,
                settings=settings,
                client=client,
                conversation_history=list(session.history),
            )
            SESSION_STORE.append_turn(session.session_id, role="user", content=request.chat_message)
            SESSION_STORE.append_turn(session.session_id, role="assistant", content=raw_chat_response.answer)
            session = SESSION_STORE.get(session.session_id) or session
            chat_response = raw_chat_response.model_copy(
                update={
                    "session_id": session.session_id,
                    "history": [turn.model_copy() for turn in session.history],
                }
            )
            debug_snapshot["chat_response"] = chat_response.model_dump()
            await _emit_progress(
                on_step,
                "chat-complete",
                "Answered the follow-up chat question from transcript evidence.",
                debug_snapshot,
            )

        similar_products = None
        notes: list[str] = []
        if request.find_similar_products:
            if not request.product_mcp_url:
                notes.append("Product Discovery MCP URL was not supplied by the orchestrator.")
            else:
                similar_request = SimilarProductsRequest(
                    product_name=product_details.product_name,
                    category=product_details.category,
                    features=product_details.features,
                    price=product_details.price,
                    source_video_url=request.youtube_url,
                    constraints={"legacy_query": request.legacy_query} if request.legacy_query else None,
                )
                await _emit_progress(
                    on_step,
                    "similar-products-started",
                    "Calling Product Discovery over MCP for similar products.",
                    debug_snapshot,
                )
                try:
                    similar_products = await _call_product_discovery_mcp(
                        similar_request,
                        mcp_url=request.product_mcp_url,
                        settings=settings,
                    )
                    debug_snapshot["similar_products"] = similar_products.model_dump()
                    await _emit_progress(
                        on_step,
                        "similar-products-complete",
                        f"Received {len(similar_products.products)} similar products from Product Discovery.",
                        debug_snapshot,
                    )
                except Exception as exc:
                    detail = format_exception_detail(exc)
                    notes.append(f"Similar-product lookup failed: {detail}")
                    debug_snapshot["similar_products_error"] = detail
                    await _emit_progress(
                        on_step,
                        "similar-products-failed",
                        f"Similar-product lookup failed: {detail}",
                        debug_snapshot,
                    )

        response = UnifiedVideoAnalysisResponse(
            youtube_url=request.youtube_url,
            video=video,
            session_id=session.session_id,
            transcript_status="indexed",
            indexing_status=debug_snapshot["indexing_status"],
            transcript_chunks=indexed.chunks[:8],
            extracted_product=product_details,
            chat_response=chat_response,
            similar_products=similar_products,
            summary=product_details.summary,
            partial=bool(notes),
            notes=" ".join(notes) if notes else None,
        )
        debug_snapshot["stage"] = "completed"
        return response, debug_snapshot
    finally:
        if client is not None:
            await close_async_resource(client)


async def chat_with_video_session(
    request: VideoChatRequest,
    settings: Settings,
    *,
    on_step: ProgressCallback | None = None,
) -> tuple[VideoChatResponse, dict[str, Any]]:
    """Answer a follow-up transcript question using persisted session history."""
    debug_snapshot: dict[str, Any] = {
        "session_id": request.session_id,
        "stage": "accepted",
        "chat_message": request.chat_message,
        "video": None,
        "history": [],
        "chat_response": None,
    }
    await _emit_progress(on_step, "accepted", "Accepted follow-up transcript chat request.", debug_snapshot)

    session = SESSION_STORE.get(request.session_id)
    if session is None:
        raise ValueError(f"No transcript chat session exists for {request.session_id}")

    debug_snapshot["video"] = session.video.model_dump()
    debug_snapshot["history"] = [turn.model_dump() for turn in session.history]
    await _emit_progress(
        on_step,
        "chat-started",
        f"Answering a follow-up transcript question for {session.video.title}.",
        debug_snapshot,
    )

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
    try:
        raw_chat_response = await answer_chat_over_transcript(
            video=session.video,
            chat_message=request.chat_message,
            settings=settings,
            client=client,
            conversation_history=list(session.history),
        )
        SESSION_STORE.append_turn(request.session_id, role="user", content=request.chat_message)
        SESSION_STORE.append_turn(request.session_id, role="assistant", content=raw_chat_response.answer)
        session = SESSION_STORE.get(request.session_id) or session
        response = raw_chat_response.model_copy(
            update={
                "session_id": session.session_id,
                "history": [turn.model_copy() for turn in session.history],
            }
        )
        debug_snapshot["history"] = [turn.model_dump() for turn in session.history]
        debug_snapshot["chat_response"] = response.model_dump()
        debug_snapshot["stage"] = "chat-complete"
        await _emit_progress(
            on_step,
            "chat-complete",
            "Answered the follow-up transcript question with session memory.",
            debug_snapshot,
        )
        return response, debug_snapshot
    finally:
        if client is not None:
            await close_async_resource(client)
