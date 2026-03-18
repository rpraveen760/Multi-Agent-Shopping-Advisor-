"""Canonical trace builders for Product Discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.a2a_models import Product


@dataclass(slots=True)
class CandidateTraceRecord:
    title: str
    candidate_entity: str
    snippet: str
    url: str
    merchant: str
    matched_query: str
    enriched_title: str
    price: str | None
    rating: str | None
    shortlist_score: float | None
    shortlist_reasons: list[str]
    page_excerpt: str


@dataclass(slots=True)
class EntityTraceRecord:
    name: str
    support: int
    product_page_support: int
    price_support: int
    rating_support: int
    candidate_urls: list[str]


@dataclass(slots=True)
class ProductTraceRecord:
    name: str
    price: str | None
    rating: str | None
    source: str
    url: str
    confidence: float
    key_features: list[str]
    evidence_urls: list[str]


def candidate_snapshot(candidate: dict[str, Any], merchant_resolver) -> dict[str, Any]:
    """Trim raw candidate data for trace/debug views."""
    matched_queries = candidate.get("matched_queries") or []
    record = CandidateTraceRecord(
        title=candidate.get("title", ""),
        candidate_entity=candidate.get("candidate_entity", ""),
        snippet=candidate.get("snippet", "")[:240],
        url=candidate.get("url", ""),
        merchant=candidate.get("merchant") or merchant_resolver(candidate.get("url", "")),
        matched_query=", ".join(matched_queries[:2]) or candidate.get("matched_query", ""),
        enriched_title=candidate.get("enriched_title", ""),
        price=candidate.get("extracted_price"),
        rating=candidate.get("extracted_rating"),
        shortlist_score=candidate.get("shortlist_score"),
        shortlist_reasons=candidate.get("shortlist_reasons", [])[:3],
        page_excerpt=candidate.get("page_text_excerpt", "")[:240],
    )
    return asdict(record)


def entity_snapshot(entity: dict[str, Any]) -> dict[str, Any]:
    record = EntityTraceRecord(
        name=entity.get("name", ""),
        support=entity.get("support", 0),
        product_page_support=entity.get("product_page_support", 0),
        price_support=entity.get("price_support", 0),
        rating_support=entity.get("rating_support", 0),
        candidate_urls=list(entity.get("candidate_urls", [])[:3]),
    )
    return asdict(record)


def product_snapshot(product: Product) -> dict[str, Any]:
    record = ProductTraceRecord(
        name=product.name,
        price=product.price,
        rating=product.rating,
        source=product.source,
        url=product.url,
        confidence=product.confidence,
        key_features=product.key_features[:3],
        evidence_urls=product.evidence_urls[:3],
    )
    return asdict(record)


def build_debug_snapshot(query: str, search_query: str, search_queries: list[str]) -> dict[str, Any]:
    return {
        "query": query,
        "search_query": search_query,
        "search_queries": search_queries,
        "candidate_count": 0,
        "stage": "search-started",
        "candidates": [],
        "enriched_candidates": [],
        "candidate_entities": [],
        "filtered_candidates": [],
        "normalized_products": [],
        "finalized_products": [],
        "shortlist_reasons": [],
        "rejected_generic_products": [],
        "rejected_entities": [],
        "summary": "",
    }


def attach_finalized_products(debug_snapshot: dict[str, Any], products: list[Product]) -> dict[str, Any]:
    finalized = [product_snapshot(product) for product in products]
    debug_snapshot["normalized_products"] = finalized
    debug_snapshot["finalized_products"] = finalized
    return debug_snapshot
