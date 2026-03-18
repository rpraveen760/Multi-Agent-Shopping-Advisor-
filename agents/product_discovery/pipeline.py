"""Thin staged orchestration helpers for Product Discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


ProgressEmitter = Callable[[str, str, dict[str, Any]], Awaitable[None]]
SearchFn = Callable[[str], list[dict[str, Any]]]
SnapshotFn = Callable[[dict[str, Any]], dict[str, Any]]
ShortlistFn = Callable[[str, list[dict[str, Any]], int], tuple[list[dict[str, Any]], list[dict[str, Any]]]]


@dataclass(slots=True)
class SearchStageResult:
    candidates: list[dict[str, Any]]


@dataclass(slots=True)
class EnrichmentStageResult:
    enriched_candidates: list[dict[str, Any]]


@dataclass(slots=True)
class ShortlistStageResult:
    filtered_candidates: list[dict[str, Any]]
    candidate_entities: list[dict[str, Any]]


async def run_search_stage(
    *,
    query: str,
    debug_snapshot: dict[str, Any],
    emit_progress: ProgressEmitter,
    search_fn: SearchFn,
    snapshot_fn: SnapshotFn,
    search_max_results: int,
) -> SearchStageResult:
    candidates = search_fn(query)
    debug_snapshot["stage"] = "search-complete"
    debug_snapshot["candidate_count"] = len(candidates)
    debug_snapshot["candidates"] = [snapshot_fn(item) for item in candidates[:search_max_results]]
    await emit_progress(
        "search-complete",
        f"Collected {len(candidates)} web search candidates",
        debug_snapshot,
    )
    return SearchStageResult(candidates=candidates)


async def run_enrichment_stage(
    *,
    candidates: list[dict[str, Any]],
    debug_snapshot: dict[str, Any],
    emit_progress: ProgressEmitter,
    enrich_fn,
    snapshot_fn: SnapshotFn,
    enrich_top_n: int,
) -> EnrichmentStageResult:
    await emit_progress(
        "enrichment-started",
        f"Enriching the top {min(len(candidates), enrich_top_n)} candidate pages",
        debug_snapshot,
    )
    enriched = await enrich_fn(candidates[:enrich_top_n])
    usable = [candidate for candidate in enriched if candidate.get("enriched_title") or candidate.get("snippet")]
    debug_snapshot["stage"] = "enrichment-complete"
    debug_snapshot["enriched_candidates"] = [snapshot_fn(item) for item in usable[:enrich_top_n]]
    await emit_progress(
        "enrichment-complete",
        f"Prepared {len(usable)} enriched candidates for normalization",
        debug_snapshot,
    )
    return EnrichmentStageResult(enriched_candidates=usable)


async def run_shortlist_stage(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    max_results: int,
    debug_snapshot: dict[str, Any],
    emit_progress: ProgressEmitter,
    shortlist_fn: ShortlistFn,
    snapshot_fn: SnapshotFn,
    entity_snapshot_fn: SnapshotFn,
) -> ShortlistStageResult:
    await emit_progress(
        "filtering-started",
        "Scoring enriched candidates and building a concrete shortlist",
        debug_snapshot,
    )
    filtered, candidate_entities = shortlist_fn(query, candidates, max_results)
    debug_snapshot["stage"] = "filtering-complete"
    debug_snapshot["candidate_entities"] = [entity_snapshot_fn(item) for item in candidate_entities]
    debug_snapshot["filtered_candidates"] = [snapshot_fn(item) for item in filtered]
    await emit_progress(
        "filtering-complete",
        f"Retained {len(filtered)} shortlist candidates for normalization",
        debug_snapshot,
    )
    return ShortlistStageResult(filtered_candidates=filtered, candidate_entities=candidate_entities)
