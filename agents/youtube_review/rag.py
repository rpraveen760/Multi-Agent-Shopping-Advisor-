"""Transcript chunking, indexing, and retrieval helpers for video-grounded chat."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from common.a2a_models import ConversationTurn, TranscriptChunk, VideoChatResponse, VideoMetadata
from common.config import Settings
from common.runtime_helpers import close_async_resource

try:  # pragma: no cover - optional dependency
    from pinecone import Pinecone, ServerlessSpec
except Exception:  # pragma: no cover - optional dependency
    Pinecone = None
    ServerlessSpec = None


@dataclass
class IndexedTranscript:
    video: VideoMetadata
    transcript_hash: str
    chunks: list[TranscriptChunk]
    embeddings: list[list[float]]
    backend: str


_MEMORY_INDEX: dict[str, IndexedTranscript] = {}


def transcript_namespace(video_id: str) -> str:
    return f"video-{video_id}"


def transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_transcript(
    transcript_text: str,
    *,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[TranscriptChunk]:
    """Split transcript text into overlapping chunks."""
    cleaned = " ".join(transcript_text.split())
    if not cleaned:
        return []

    chunks: list[TranscriptChunk] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        text = cleaned[start:end].strip()
        if text:
            chunks.append(TranscriptChunk(chunk_index=index, text=text))
            index += 1
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _lexical_vector(text: str) -> Counter[str]:
    return Counter(token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_score(query: str, chunk_text: str) -> float:
    query_counts = _lexical_vector(query)
    chunk_counts = _lexical_vector(chunk_text)
    if not query_counts or not chunk_counts:
        return 0.0
    overlap = sum((query_counts & chunk_counts).values())
    return overlap / max(len(query_counts), 1)


async def _embed_texts(
    texts: list[str],
    settings: Settings,
    *,
    client: AsyncOpenAI | None = None,
) -> list[list[float]]:
    if not texts or not settings.OPENAI_API_KEY:
        return [[] for _ in texts]

    owned_client = client is None
    runtime_client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        request_kwargs: dict[str, Any] = {
            "model": settings.OPENAI_EMBEDDING_MODEL,
            "input": texts,
        }
        if settings.OPENAI_EMBEDDING_DIMENSIONS > 0:
            request_kwargs["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSIONS
        response = await runtime_client.embeddings.create(**request_kwargs)
        return [list(item.embedding) for item in response.data]
    finally:
        if owned_client:
            await close_async_resource(runtime_client)


async def _get_pinecone_index(settings: Settings, *, create_if_missing: bool = True):
    if not settings.ENABLE_PINECONE or not settings.PINECONE_API_KEY or Pinecone is None or ServerlessSpec is None:
        return None

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_list = pc.list_indexes()
    index_names = [item["name"] for item in index_list]
    if settings.PINECONE_INDEX_NAME not in index_names:
        if not create_if_missing:
            return None
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.OPENAI_EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )
    else:
        for item in index_list:
            if item["name"] == settings.PINECONE_INDEX_NAME:
                existing_dimension = item.get("dimension")
                if existing_dimension and existing_dimension != settings.OPENAI_EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        "Pinecone index dimension does not match OPENAI_EMBEDDING_DIMENSIONS "
                        f"({existing_dimension} != {settings.OPENAI_EMBEDDING_DIMENSIONS})"
                    )
                break
    return pc.Index(settings.PINECONE_INDEX_NAME)


async def clear_transcript_index(*, video_id: str, settings: Settings) -> bool:
    """Remove transcript chunks for a prior session from memory and Pinecone."""
    namespace = transcript_namespace(video_id)
    cleared = _MEMORY_INDEX.pop(namespace, None) is not None

    pinecone_index = await _get_pinecone_index(settings, create_if_missing=False)
    if pinecone_index is None:
        return cleared

    try:
        pinecone_index.delete(namespace=namespace, delete_all=True)
        return True
    except Exception:
        return cleared


async def index_transcript(
    *,
    video: VideoMetadata,
    transcript_text: str,
    settings: Settings,
    client: AsyncOpenAI | None = None,
) -> tuple[IndexedTranscript, bool]:
    """Index transcript chunks and skip duplicate embedding when the hash is unchanged."""
    namespace = transcript_namespace(video.video_id)
    current_hash = transcript_hash(transcript_text)
    existing = _MEMORY_INDEX.get(namespace)
    if existing and existing.transcript_hash == current_hash:
        return existing, False

    chunks = split_transcript(
        transcript_text,
        chunk_size=settings.YOUTUBE_TRANSCRIPT_CHUNK_SIZE,
        overlap=settings.YOUTUBE_TRANSCRIPT_CHUNK_OVERLAP,
    )
    embeddings = await _embed_texts([chunk.text for chunk in chunks], settings, client=client)
    backend = "memory"

    pinecone_index = await _get_pinecone_index(settings)
    if pinecone_index is not None and chunks and all(embedding for embedding in embeddings):
        vectors = [
            {
                "id": f"{video.video_id}:{chunk.chunk_index}",
                "values": embedding,
                "metadata": {
                    "video_id": video.video_id,
                    "video_url": video.video_url,
                    "chunk_index": chunk.chunk_index,
                    "transcript_hash": current_hash,
                    "channel": video.channel,
                    "published_at": video.published_at or "",
                    "text": chunk.text,
                },
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        try:
            pinecone_index.delete(namespace=namespace, delete_all=True)
        except Exception:
            # The first write for a namespace can report "not found"; upsert is still safe.
            pass
        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        backend = "pinecone"

    record = IndexedTranscript(
        video=video,
        transcript_hash=current_hash,
        chunks=chunks,
        embeddings=embeddings,
        backend=backend,
    )
    _MEMORY_INDEX[namespace] = record
    return record, True


async def answer_chat_over_transcript(
    *,
    video: VideoMetadata,
    chat_message: str,
    settings: Settings,
    client: AsyncOpenAI | None = None,
    top_k: int = 4,
    conversation_history: list[ConversationTurn] | None = None,
) -> VideoChatResponse:
    """Answer a follow-up question using transcript chunks from the indexed video only."""
    namespace = transcript_namespace(video.video_id)
    indexed = _MEMORY_INDEX.get(namespace)
    if indexed is None or not indexed.chunks:
        return VideoChatResponse(
            youtube_url=video.video_url,
            answer="Transcript content is not indexed for this video yet.",
            citations=[],
            confidence=0.0,
        )

    query_embedding = (await _embed_texts([chat_message], settings, client=client))[0]
    scored: list[tuple[float, TranscriptChunk]] = []
    for chunk, chunk_embedding in zip(indexed.chunks, indexed.embeddings):
        score = _cosine_similarity(query_embedding, chunk_embedding) if query_embedding and chunk_embedding else _lexical_score(chat_message, chunk.text)
        scored.append((score, chunk))

    top_chunks = [chunk for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k] if score > 0]
    if not top_chunks:
        top_chunks = indexed.chunks[: min(top_k, len(indexed.chunks))]

    context = "\n\n".join(
        f"[Chunk {chunk.chunk_index}]\n{chunk.text}"
        for chunk in top_chunks
    )

    history = conversation_history or []
    recent_history = history[-6:]
    history_block = "\n".join(
        f"{turn.role.title()}: {turn.content}"
        for turn in recent_history
        if turn.content.strip()
    )

    if settings.OPENAI_API_KEY:
        owned_client = client is None
        runtime_client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        try:
            response = await runtime_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the user's question using only the transcript context from the single YouTube video. "
                            "Do not invent details. Cite chunk numbers in plain language when helpful."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Video: {video.title}\n"
                            f"Conversation history:\n{history_block or 'No prior transcript chat yet.'}\n"
                            f"Question: {chat_message}\n\nTranscript context:\n{context}\n\n"
                            "Return a concise answer grounded in the transcript."
                        ),
                    },
                ],
            )
            answer = response.choices[0].message.content or "No grounded answer was produced."
        finally:
            if owned_client:
                await close_async_resource(runtime_client)
    else:
        answer = top_chunks[0].text[:500]

    citations = [f"Chunk {chunk.chunk_index}: {chunk.text[:140]}..." for chunk in top_chunks]
    confidence = round(min(0.95, 0.45 + 0.12 * len(top_chunks)), 2)
    return VideoChatResponse(
        youtube_url=video.video_url,
        answer=answer,
        citations=citations,
        confidence=confidence,
    )
