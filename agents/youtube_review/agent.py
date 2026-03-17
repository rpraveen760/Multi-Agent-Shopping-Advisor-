"""YouTube Product Review Agent — core logic.

Pipeline:
1. Search YouTube Data API v3 for recent product reviews (18-month window).
2. Fetch video metadata (views, publish date) and rerank candidates.
3. Extract transcripts (fallback to descriptions).
4. Summarize with GPT-4o-mini using structured output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone

from googleapiclient.discovery import build
from openai import AsyncOpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from common.a2a_models import ReviewSummary, VideoSource
from common.config import Settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

RECENCY_WINDOW_MONTHS = 18
SEARCH_MAX_RESULTS = 8
TOP_K_VIDEOS = 5

REVIEW_SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ReviewSummary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "overall_sentiment": {"type": "string", "enum": ["positive", "mixed", "negative"]},
                "score": {"type": "number"},
                "pros": {"type": "array", "items": {"type": "string"}},
                "cons": {"type": "array", "items": {"type": "string"}},
                "key_quotes": {"type": "array", "items": {"type": "string"}},
                "recommendation": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "product_name", "overall_sentiment", "score",
                "pros", "cons", "key_quotes", "recommendation", "confidence",
            ],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are a product review analyst. You will receive transcript excerpts and metadata from YouTube product review videos.

Your task:
1. Synthesize the evidence into a structured review summary.
2. Only use information explicitly present in the provided evidence. Do not invent claims.
3. Extract direct quotes where possible for the key_quotes field.
4. Rate the product on a scale of 1.0 to 10.0 based on the evidence.
5. Assign a confidence score (0.0 to 1.0) based on how much evidence was available.
6. Keep pros and cons to a maximum of 5 each, ordered by importance.
7. The recommendation should be 2-3 sentences summarizing the overall verdict."""


# ── Main entry point ──────────────────────────────────────────────

async def get_review_summary(product_name: str, settings: Settings) -> ReviewSummary:
    """Retrieve and summarize YouTube reviews for a product.

    Args:
        product_name: The product to search reviews for.
        settings: Application settings (API keys, model config).

    Returns:
        A structured ReviewSummary with sentiment, pros/cons, and sources.
    """
    logger.info("Starting review pipeline for: %s", product_name)

    # Step 1: Search YouTube
    videos = await asyncio.to_thread(
        _search_youtube,
        product_name,
        settings.YOUTUBE_API_KEY,
    )
    if not videos:
        logger.warning("No YouTube videos found for: %s", product_name)
        return _empty_review(product_name, "No YouTube review videos found.")

    # Step 2: Fetch metadata and rerank
    enriched = await asyncio.to_thread(
        _fetch_video_metadata,
        videos,
        settings.YOUTUBE_API_KEY,
    )
    ranked = _rerank_candidates(enriched)[:TOP_K_VIDEOS]
    logger.info("Reranked to top %d videos", len(ranked))

    # Step 3: Extract transcripts
    evidence = await asyncio.to_thread(_extract_transcripts, ranked)
    usable_evidence = _filter_usable_evidence(evidence)

    # Step 4: Summarize with GPT-4o-mini
    if usable_evidence:
        summary = await _summarize_reviews(product_name, usable_evidence, settings)
    else:
        logger.warning("No transcript or description evidence available for: %s", product_name)
        summary = _empty_review(
            product_name,
            "No transcript or description evidence was available from recent review videos.",
        )

    # Attach video sources
    summary.sources = _build_video_sources(ranked)

    return summary


# ── Step 1: YouTube Search ────────────────────────────────────────

def _search_youtube(product_name: str, api_key: str) -> list[dict]:
    """Search YouTube for product review videos within the recency window."""
    youtube = build("youtube", "v3", developerKey=api_key)

    published_after = (
        datetime.now(timezone.utc) - timedelta(days=RECENCY_WINDOW_MONTHS * 30)
    ).isoformat()

    query = f"{product_name} review"
    logger.info("YouTube search: q='%s', publishedAfter=%s", query, published_after[:10])

    try:
        response = (
            youtube.search()
            .list(
                q=query,
                type="video",
                part="snippet",
                order="relevance",
                maxResults=SEARCH_MAX_RESULTS,
                publishedAfter=published_after,
            )
            .execute()
        )
    except Exception as exc:
        logger.error("YouTube search failed: %s", exc)
        return []

    videos = []
    for item in response.get("items", []):
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "description": item["snippet"]["description"],
            "published_at": item["snippet"]["publishedAt"][:10],
        })

    logger.info("YouTube search returned %d videos", len(videos))
    return videos


# ── Step 2: Metadata + Reranking ──────────────────────────────────

def _fetch_video_metadata(videos: list[dict], api_key: str) -> list[dict]:
    """Fetch view counts and exact publish dates via videos.list()."""
    if not videos:
        return []

    youtube = build("youtube", "v3", developerKey=api_key)
    video_ids = [v["video_id"] for v in videos]

    try:
        response = (
            youtube.videos()
            .list(
                id=",".join(video_ids),
                part="snippet,statistics",
            )
            .execute()
        )
    except Exception as exc:
        logger.error("Video metadata fetch failed: %s", exc)
        return videos  # Return without metadata enrichment

    metadata_map = {}
    for item in response.get("items", []):
        vid_id = item["id"]
        stats = item.get("statistics", {})
        metadata_map[vid_id] = {
            "view_count": stats.get("viewCount", "0"),
            "published_at": item["snippet"]["publishedAt"][:10],
        }

    for v in videos:
        meta = metadata_map.get(v["video_id"], {})
        v["view_count"] = meta.get("view_count", "0")
        v["published_at"] = meta.get("published_at", v.get("published_at", "unknown"))

    return videos


def _rerank_candidates(videos: list[dict]) -> list[dict]:
    """Rerank by combining relevance order, recency, and view count.

    Relevance score: convert YouTube's returned order into a descending score.
    Recency score: days since publish normalized to [0,1] within the set.
    View score: log-normalized view count.
    """
    if not videos:
        return []

    now = datetime.now(timezone.utc)
    n = len(videos)

    # Relevance: position 0 gets highest score, position n-1 gets lowest
    for i, v in enumerate(videos):
        v["_relevance"] = (n - i) / n

    # Recency: more recent = higher score
    for v in videos:
        try:
            pub_date = datetime.fromisoformat(v["published_at"]).replace(tzinfo=timezone.utc)
            days_ago = (now - pub_date).days
        except (ValueError, TypeError):
            days_ago = 365  # default to ~1 year if unparseable
        v["_days_ago"] = days_ago

    max_days = max(v["_days_ago"] for v in videos) or 1
    for v in videos:
        v["_recency"] = 1.0 - (v["_days_ago"] / max_days)

    # View score: log-normalized
    for v in videos:
        try:
            views = int(v.get("view_count", "0"))
        except (ValueError, TypeError):
            views = 0
        v["_view_log"] = math.log1p(views)

    max_view_log = max(v["_view_log"] for v in videos) or 1
    for v in videos:
        v["_view_score"] = v["_view_log"] / max_view_log

    # Combined score: 0.4 * relevance + 0.3 * recency + 0.3 * views
    for v in videos:
        v["_score"] = (
            0.4 * v["_relevance"]
            + 0.3 * v["_recency"]
            + 0.3 * v["_view_score"]
        )

    videos.sort(key=lambda v: v["_score"], reverse=True)

    # Clean up temp keys
    for v in videos:
        for k in list(v.keys()):
            if k.startswith("_"):
                del v[k]

    return videos


# ── Step 3: Transcript Extraction ─────────────────────────────────

def _extract_transcripts(videos: list[dict]) -> list[dict]:
    """Extract transcripts for each video, falling back to description."""
    ytt_api = YouTubeTranscriptApi()
    for v in videos:
        try:
            transcript = ytt_api.fetch(v["video_id"])
            # Combine transcript snippets into text (limit to ~3000 chars)
            text = " ".join(
                snippet.text for snippet in transcript.snippets
            )[:3000]
            v["transcript"] = text
            v["evidence_source"] = "transcript"
            logger.info("Transcript fetched for: %s", v["video_id"])
        except Exception as exc:
            logger.info(
                "Transcript unavailable for %s (%s), using description",
                v["video_id"],
                type(exc).__name__,
            )
            v["transcript"] = v.get("description", "")
            v["evidence_source"] = "description"

    return videos


def _filter_usable_evidence(videos: list[dict]) -> list[dict]:
    """Keep only videos that produced usable transcript or description text."""
    usable = []
    for video in videos:
        text = video.get("transcript", "").strip()
        if text:
            usable.append(video)
            continue

        logger.info(
            "Skipping video %s because no transcript or description evidence was available",
            video.get("video_id", "unknown"),
        )

    return usable


def _build_video_sources(videos: list[dict]) -> list[VideoSource]:
    """Render ranked videos into the structured source-link format."""
    return [
        VideoSource(
            title=video["title"],
            channel=video["channel"],
            url=f"https://youtube.com/watch?v={video['video_id']}",
            published_at=video.get("published_at", "unknown"),
            view_count=video.get("view_count"),
        )
        for video in videos
    ]


# ── Step 4: GPT-4o-mini Summarization ─────────────────────────────

async def _summarize_reviews(
    product_name: str,
    evidence: list[dict],
    settings: Settings,
) -> ReviewSummary:
    """Use GPT-4o-mini to synthesize review evidence into a structured summary."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Build evidence text for the prompt
    evidence_parts = []
    for i, v in enumerate(evidence, 1):
        evidence_parts.append(
            f"--- Video {i}: \"{v['title']}\" by {v['channel']} "
            f"(views: {v.get('view_count', 'N/A')}, published: {v.get('published_at', 'N/A')}) ---\n"
            f"[Source: {v['evidence_source']}]\n"
            f"{v.get('transcript', 'No content available.')}\n"
        )

    user_prompt = (
        f"Product: {product_name}\n\n"
        f"Evidence from {len(evidence)} YouTube review videos:\n\n"
        + "\n".join(evidence_parts)
        + "\n\nBased on the above evidence only, produce a structured review summary."
    )

    logger.info("Calling GPT-4o-mini for review synthesis (%d evidence sources)", len(evidence))

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=REVIEW_SUMMARY_SCHEMA,
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        return ReviewSummary(**data)

    except Exception as exc:
        logger.error("GPT-4o-mini summarization failed: %s", exc)
        return _empty_review(product_name, f"Summarization failed: {exc}")


# ── Fallback ──────────────────────────────────────────────────────

def _empty_review(product_name: str, reason: str) -> ReviewSummary:
    """Return a minimal ReviewSummary when the pipeline cannot produce results."""
    return ReviewSummary(
        product_name=product_name,
        overall_sentiment="mixed",
        score=0.0,
        pros=[],
        cons=[],
        key_quotes=[],
        recommendation=reason,
        confidence=0.0,
        sources=[],
    )
