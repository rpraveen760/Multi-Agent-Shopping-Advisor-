"""Product Discovery Agent — core logic.

Pipeline:
1. Candidate retrieval via DuckDuckGo search.
2. Deterministic enrichment: fetch top pages, parse title/domain/price/rating.
3. GPT-4o-mini normalization: dedupe, summarize features, assign confidence.

This is the canonical capability shared by both the A2A facade and the MCP tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import AsyncOpenAI

from agents.product_discovery.pipeline import (
    run_enrichment_stage,
    run_search_stage,
    run_shortlist_stage,
)
from agents.product_discovery.traces import (
    build_debug_snapshot,
    candidate_snapshot as build_candidate_trace_snapshot,
    entity_snapshot as build_entity_trace_snapshot,
    product_snapshot as build_product_trace_snapshot,
)
from common.a2a_models import Product, ProductDiscoveryResult
from common.config import Settings
from common.runtime_helpers import close_async_resource

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, dict[str, Any]], Awaitable[None] | None]

# ── Constants ─────────────────────────────────────────────────────

SEARCH_MAX_RESULTS = 12
ENRICH_TOP_N = 8
ENRICH_TIMEOUT = 8  # seconds per page fetch
FINAL_MAX_PRODUCTS = 3
FILTERED_CANDIDATE_LIMIT = 6
SHORTLIST_SCORE_THRESHOLD = 2.3
DETAIL_SEARCH_MAX_RESULTS = 6

_GENERIC_PRODUCT_NAME_TOKENS = {
    "best",
    "budget",
    "beginner",
    "beginners",
    "entry",
    "level",
    "top",
    "rated",
    "gaming",
    "wireless",
    "wired",
    "screen",
    "display",
    "tablet",
    "drawing",
    "graphics",
    "monitor",
    "headphones",
    "earbuds",
    "keyboard",
    "mouse",
    "laptop",
    "product",
    "products",
    "option",
    "options",
    "with",
    "for",
    "under",
    "review",
    "reviews",
    "price",
    "prices",
    "model",
    "models",
    "buy",
    "pick",
    "picks",
}

_GENERIC_ENTITY_LABEL_TOKENS = _GENERIC_PRODUCT_NAME_TOKENS | {
    "the",
    "a",
    "an",
    "our",
    "ours",
    "weve",
    "tested",
    "reviewed",
    "expert",
    "experts",
    "winner",
    "winners",
    "standout",
    "online",
    "india",
    "every",
    "style",
    "styles",
    "latest",
    "new",
}

_MERCHANT_DOMAINS = {
    "amazon",
    "amazon india",
    "best buy",
    "walmart",
    "target",
    "newegg",
    "b&h photo",
    "flipkart",
    "ebay",
}

_EDITORIAL_DOMAIN_CUES = {
    "techradar",
    "cnet",
    "pcmag",
    "rtings",
    "tomsguide",
    "wirecutter",
    "trustedreviews",
    "theverge",
    "reviewgeek",
}

_PRODUCT_URL_CUES = (
    "/dp/",
    "/gp/product/",
    "/product/",
    "/products/",
    "/item/",
    "/p/",
    "/buy",
)

_EDITORIAL_TEXT_CUES = (
    "best ",
    "top ",
    "review",
    "reviews",
    "vs",
    "guide",
    "guides",
    "roundup",
)

_MODEL_TOKEN_PATTERN = re.compile(r"\b(?:[a-z]+\d+[a-z0-9-]*|\d+[a-z]+[a-z0-9-]*)\b", re.IGNORECASE)
_TITLE_SPLIT_PATTERN = re.compile(r"\s[\-|–—:]\s|\|")
_TRAILING_CONTEXT_PATTERN = re.compile(
    r"\b(?:review|reviews|price|prices|buy|guide|comparison|compare|vs|deal|deals|sale|top picks?|best)\b.*$",
    re.IGNORECASE,
)
_ENTITY_PHRASE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9&+./-]*(?:\s+[A-Z0-9][A-Za-z0-9&+./-]*){1,5})\b"
)

PRODUCT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ProductDiscoveryResult",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": ["string", "null"]},
                            "rating": {"type": ["string", "null"]},
                            "url": {"type": "string"},
                            "key_features": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "name", "price", "rating", "url",
                            "key_features", "source", "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["products", "summary"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are a product discovery assistant. You will receive raw web search results and enrichment data for a shopping query.

Your task:
1. Identify distinct real products from the evidence (deduplicate variants of the same product).
2. For each product, extract: name, price, rating, source merchant, and up to 3 key features.
3. NEVER invent a price or rating. If the evidence does not contain a clear price or rating, set the field to null.
4. Assign a confidence score (0.0 to 1.0) based on how much evidence supports each product.
5. The source field should be the merchant or site name (e.g., "Amazon", "Best Buy", "Walmart").
6. Return at most 3 products, ordered by confidence descending.
7. Only return concrete brand/model products that are explicitly supported by the evidence.
8. NEVER return generic category labels such as "Drawing Tablet with Screen", "Budget Mechanical Keyboard", or "Gaming Mouse".
9. Prefer 1-3 strong, reviewable candidates over a long generic list.
10. Write a short summary (1-2 sentences) of what was found."""


async def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    snapshot: dict[str, Any],
) -> None:
    """Publish a progress snapshot if a callback was provided."""
    if callback is None:
        return

    result = callback(stage, message, json.loads(json.dumps(snapshot)))
    if asyncio.iscoroutine(result):
        await result


async def _close_async_client(client: Any) -> None:
    """Best-effort close for SDK clients used in async helpers."""
    await close_async_resource(client)


def _build_search_query(query: str) -> str:
    return _build_search_queries(query)[0]


def _build_search_queries(query: str) -> list[str]:
    """Build a small generic set of product-focused search queries."""
    cleaned = " ".join(query.split())
    queries = [
        f"{cleaned} best models price",
        f"{cleaned} buy review",
        f"{cleaned} top picks",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        normalized = " ".join(item.split())
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    return deduped[:3]


def _build_detail_search_queries(product_name: str) -> list[str]:
    """Build exact-product follow-up queries for link grounding."""
    cleaned = " ".join(product_name.split())
    queries = [
        f"\"{cleaned}\" product",
        f"\"{cleaned}\" buy",
        f"\"{cleaned}\" price",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def _candidate_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    """Trim raw candidate data for trace/debug views."""
    return build_candidate_trace_snapshot(candidate, merchant_resolver=_extract_domain)


def _product_snapshot(product: Product) -> dict[str, Any]:
    return build_product_trace_snapshot(product)


# ── Main entry point ──────────────────────────────────────────────

async def search_products(
    query: str,
    settings: Settings,
    max_results: int = FINAL_MAX_PRODUCTS,
    on_step: ProgressCallback | None = None,
) -> ProductDiscoveryResult:
    """Retrieve and normalize product candidates for a shopper query.

    This function is the canonical capability used by both the A2A facade
    and the MCP tool.

    Args:
        query: Natural-language shopping query.
        settings: Application settings (API keys, model config).
        max_results: Maximum products to return (default 3).

    Returns:
        Structured ProductDiscoveryResult with provenance.
    """
    result, _debug = await search_products_with_debug(
        query=query,
        settings=settings,
        max_results=max_results,
        on_step=on_step,
    )
    return result


async def search_products_with_debug(
    query: str,
    settings: Settings,
    max_results: int = FINAL_MAX_PRODUCTS,
    on_step: ProgressCallback | None = None,
) -> tuple[ProductDiscoveryResult, dict[str, Any]]:
    """Run product discovery and return both the final result and debug snapshots."""
    logger.info("Starting product discovery for: %s", query)
    search_queries = _build_search_queries(query)
    debug_snapshot: dict[str, Any] = build_debug_snapshot(
        query=query,
        search_query=_build_search_query(query),
        search_queries=search_queries,
    )

    await _emit_progress(
        on_step,
        "search-started",
        f"Searching the web for product candidates matching '{query}'",
        debug_snapshot,
    )

    search_stage = await run_search_stage(
        query=query,
        debug_snapshot=debug_snapshot,
        emit_progress=lambda stage, message, snapshot: _emit_progress(on_step, stage, message, snapshot),
        search_fn=lambda q: _search_duckduckgo(q),
        snapshot_fn=_candidate_snapshot,
        search_max_results=SEARCH_MAX_RESULTS,
    )
    candidates = search_stage.candidates
    if not candidates:
        logger.warning("No search results for: %s", query)
        result = ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"No product results found for '{query}'.",
        )
        debug_snapshot["summary"] = result.summary
        return result, debug_snapshot

    enrichment_stage = await run_enrichment_stage(
        candidates=candidates,
        debug_snapshot=debug_snapshot,
        emit_progress=lambda stage, message, snapshot: _emit_progress(on_step, stage, message, snapshot),
        enrich_fn=_enrich_candidates,
        snapshot_fn=_candidate_snapshot,
        enrich_top_n=ENRICH_TOP_N,
    )
    usable = enrichment_stage.enriched_candidates

    if not usable:
        logger.warning("No usable enrichment data for: %s", query)
        result = ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"Search returned results but no usable product data for '{query}'.",
        )
        debug_snapshot["summary"] = result.summary
        return result, debug_snapshot

    shortlist_stage = await run_shortlist_stage(
        query=query,
        candidates=usable,
        max_results=max_results,
        debug_snapshot=debug_snapshot,
        emit_progress=lambda stage, message, snapshot: _emit_progress(on_step, stage, message, snapshot),
        shortlist_fn=_filter_candidates_for_shortlist,
        snapshot_fn=_candidate_snapshot,
        entity_snapshot_fn=build_entity_trace_snapshot,
    )
    filtered = shortlist_stage.filtered_candidates

    if not filtered:
        logger.warning("No shortlist-quality candidates for: %s", query)
        result = ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"Search returned results but no concrete shortlist candidates for '{query}'.",
        )
        debug_snapshot["summary"] = result.summary
        return result, debug_snapshot

    # Step 4: GPT-4o-mini normalization
    await _emit_progress(
        on_step,
        "normalization-started",
        "Normalizing candidate data into distinct products",
        debug_snapshot,
    )
    result, normalized_debug = await _normalize_products_with_debug(query, filtered, settings, max_results)
    debug_snapshot["stage"] = "normalization-complete"
    debug_snapshot["normalized_products"] = normalized_debug.get("normalized_products", [])
    debug_snapshot["finalized_products"] = normalized_debug.get("finalized_products", [])
    debug_snapshot["shortlist_reasons"] = normalized_debug.get("shortlist_reasons", [])
    debug_snapshot["rejected_generic_products"] = normalized_debug.get("rejected_generic_products", [])
    debug_snapshot["rejected_entities"] = normalized_debug.get("rejected_entities", [])
    debug_snapshot["summary"] = result.summary
    await _emit_progress(
        on_step,
        "normalization-complete",
        f"Normalized {len(result.products)} products for the final result",
        debug_snapshot,
    )
    return result, debug_snapshot


# ── Step 1: DuckDuckGo Search ─────────────────────────────────────

def _search_duckduckgo(query: str) -> list[dict]:
    """Search DuckDuckGo for product candidates."""
    search_queries = _build_search_queries(query)
    logger.info("DuckDuckGo search queries: %s", search_queries)

    try:
        with DDGS() as ddgs:
            candidates: list[dict[str, Any]] = []
            candidates_by_url: dict[str, dict[str, Any]] = {}
            per_query_limit = max(6, SEARCH_MAX_RESULTS // max(len(search_queries), 1))

            for search_query in search_queries:
                results = list(ddgs.text(search_query, max_results=per_query_limit))
                for r in results:
                    url = r.get("href", "")
                    if not _is_web_url(url):
                        continue

                    existing = candidates_by_url.get(url)
                    if existing is not None:
                        existing.setdefault("matched_queries", [])
                        if search_query not in existing["matched_queries"]:
                            existing["matched_queries"].append(search_query)
                        continue

                    candidate = {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": url,
                        "matched_query": search_query,
                        "matched_queries": [search_query],
                    }
                    candidates_by_url[url] = candidate
                    candidates.append(candidate)
                    if len(candidates) >= SEARCH_MAX_RESULTS:
                        break

                if len(candidates) >= SEARCH_MAX_RESULTS:
                    break
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)
        return []

    logger.info("DuckDuckGo returned %d results", len(candidates))
    return candidates


# ── Step 2: Deterministic Enrichment ──────────────────────────────

async def _enrich_candidates(candidates: list[dict]) -> list[dict]:
    """Fetch top candidate pages and extract title, domain, price, rating."""
    async with httpx.AsyncClient(
        timeout=ENRICH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ProductBot/1.0)"},
    ) as client:

        async def _enrich_one(candidate: dict) -> dict:
            url = candidate.get("url", "")
            if not _is_web_url(url):
                return candidate

            domain = _extract_domain(url)
            candidate["merchant"] = domain

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text[:50_000]  # limit to 50KB

                soup = BeautifulSoup(html, "html.parser")

                # Extract page title
                title_tag = soup.find("title")
                candidate["enriched_title"] = title_tag.get_text(strip=True) if title_tag else ""

                # Try to find price patterns in the page text
                page_text = soup.get_text(" ", strip=True)[:5000]
                candidate["page_text_excerpt"] = page_text

                price = _extract_price(page_text)
                if price:
                    candidate["extracted_price"] = price

                rating = _extract_rating(page_text)
                if rating:
                    candidate["extracted_rating"] = rating

            except Exception as exc:
                logger.info("Enrichment failed for %s: %s", url[:60], type(exc).__name__)
                candidate["enriched_title"] = ""
                candidate["page_text_excerpt"] = ""

            return candidate

        tasks = [_enrich_one(c) for c in candidates]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, item in enumerate(enriched):
        if isinstance(item, Exception):
            logger.info("Enrichment exception for candidate %d: %s", i, item)
            results.append(candidates[i])
        else:
            results.append(item)

    return results


def _is_web_url(url: str) -> bool:
    """Return True for absolute http/https URLs."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_text(text: str) -> str:
    """Normalize text for loose matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _canonicalize_candidate_url(url: str | None) -> str | None:
    """Normalize candidate URLs for stable grounding and dedupe."""
    if not url or not _is_web_url(url):
        return None
    return url.split("#", 1)[0]


def _extract_domain(url: str) -> str:
    """Extract a clean merchant name from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        domain = re.sub(r"^www\.", "", domain)
        # Map common domains to merchant names
        domain_map = {
            "amazon.com": "Amazon",
            "amazon.in": "Amazon India",
            "bestbuy.com": "Best Buy",
            "walmart.com": "Walmart",
            "target.com": "Target",
            "newegg.com": "Newegg",
            "bhphotovideo.com": "B&H Photo",
            "flipkart.com": "Flipkart",
            "ebay.com": "eBay",
        }
        return domain_map.get(domain, domain)
    except Exception:
        return "unknown"


def _extract_price(text: str) -> str | None:
    """Try to find a price pattern like $49.99 or Rs. 4,999 in text."""
    # USD pattern
    match = re.search(r"\$\d{1,5}(?:,\d{3})*(?:\.\d{2})?", text)
    if match:
        return match.group(0)
    # INR pattern
    match = re.search(r"(?:Rs\.?|INR|₹)\s?\d{1,6}(?:,\d{3})*(?:\.\d{2})?", text)
    if match:
        return match.group(0)
    return None


def _looks_like_specific_product_name(name: str, candidates: list[dict], query: str) -> bool:
    """Reject broad category labels and keep only evidence-backed real product names."""
    normalized_name = _normalize_text(name)
    if not normalized_name:
        return False

    if _looks_like_budget_bucket_name(name, query):
        return False

    evidence_candidates = _find_evidence_candidates(name, candidates)
    if not evidence_candidates:
        return False

    if _has_model_signal(name) or _has_specific_product_tokens(name, query):
        return True

    normalized_evidence_names = {
        _normalize_text(candidate.get("candidate_entity", "")) for candidate in evidence_candidates
    }
    normalized_evidence_names.discard("")
    if normalized_name in normalized_evidence_names and len(evidence_candidates) >= 2:
        return True

    non_generic_words = [
        word
        for word in normalized_name.split()
        if len(word) > 2 and word not in _GENERIC_PRODUCT_NAME_TOKENS
    ]
    if len(non_generic_words) < 2:
        return False

    supported_words = set()
    for candidate in evidence_candidates:
        text = _candidate_text(candidate)
        supported_words.update(word for word in non_generic_words if word in text)

    return len(supported_words) >= 2 and len(evidence_candidates) >= 2


def _extract_rating(text: str) -> str | None:
    """Try to find a rating pattern like 4.5/5 or 4.5 out of 5 in text."""
    match = re.search(r"(\d(?:\.\d)?)\s*(?:/|out of)\s*5", text)
    if match:
        return f"{match.group(1)}/5"
    return None


def _candidate_text(candidate: dict[str, Any]) -> str:
    return _normalize_text(
        " ".join(
            [
                candidate.get("title", ""),
                candidate.get("snippet", ""),
                candidate.get("enriched_title", ""),
                candidate.get("page_text_excerpt", "")[:500],
            ]
        )
    )


def _query_content_tokens(query: str) -> set[str]:
    return {
        token
        for token in _normalize_text(query).split()
        if token
        and token not in _GENERIC_PRODUCT_NAME_TOKENS
        and not token.isdigit()
    }


def _query_is_concrete_lookup(query: str) -> bool:
    normalized = _normalize_text(query)
    if any(term in normalized.split() for term in ("best", "budget", "under", "compare", "vs", "top")):
        return False
    return 1 < len(_query_content_tokens(query)) <= 5


def _extract_candidate_entity_name(candidate: dict[str, Any]) -> str:
    raw_title = candidate.get("enriched_title") or candidate.get("title") or ""
    blocked_tokens = _build_blocked_entity_tokens(candidate)
    blocked_phrases = _build_blocked_entity_phrases(candidate)

    best_phrase = ""
    best_score = -1.0
    seen_phrases: set[str] = set()

    for text in (
        candidate.get("snippet", ""),
        candidate.get("title", ""),
        candidate.get("enriched_title", ""),
    ):
        for match in _ENTITY_PHRASE_PATTERN.finditer(text or ""):
            phrase = _clean_entity_phrase(match.group(1))
            normalized = _normalize_text(phrase)
            if not normalized or normalized in seen_phrases:
                continue
            seen_phrases.add(normalized)
            score = _score_entity_phrase(
                phrase,
                candidate,
                blocked_tokens=blocked_tokens,
                blocked_phrases=blocked_phrases,
            )
            if score > best_score:
                best_phrase = phrase
                best_score = score

    if best_phrase:
        return best_phrase

    segments = [segment.strip(" -|:") for segment in _TITLE_SPLIT_PATTERN.split(raw_title) if segment.strip()]
    if not segments:
        segments = [raw_title.strip()]

    for segment in segments:
        cleaned = _clean_entity_phrase(segment)
        if not cleaned:
            continue
        score = _score_entity_phrase(
            cleaned,
            candidate,
            blocked_tokens=blocked_tokens,
            blocked_phrases=blocked_phrases,
        )
        if score > best_score:
            best_phrase = cleaned
            best_score = score

    return best_phrase


def _build_blocked_entity_tokens(candidate: dict[str, Any]) -> set[str]:
    blocked = set(_GENERIC_ENTITY_LABEL_TOKENS)
    domain = _extract_domain(candidate.get("url", ""))
    blocked.update(
        token
        for token in _normalize_text(domain).split()
        if token and token not in {"www", "com", "co", "in", "net", "org"}
    )
    return blocked


def _build_blocked_entity_phrases(candidate: dict[str, Any]) -> set[str]:
    raw_title = candidate.get("enriched_title") or candidate.get("title") or ""
    segments = [segment.strip(" -|:") for segment in _TITLE_SPLIT_PATTERN.split(raw_title) if segment.strip()]
    blocked: set[str] = set()
    if len(segments) > 1:
        tail = _normalize_text(segments[-1])
        if tail and len(tail.split()) <= 3 and not _has_model_signal(segments[-1]):
            blocked.add(tail)
    domain = _normalize_text(_extract_domain(candidate.get("url", "")))
    if domain:
        blocked.add(domain)
    return blocked


def _clean_entity_phrase(value: str) -> str:
    cleaned = value.strip(" -|:,.()[]{}")
    cleaned = re.sub(r"^(?:the|our)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _TRAILING_CONTEXT_PATTERN.sub("", cleaned).strip(" -|:,.()[]{}")
    return cleaned


def _score_entity_phrase(
    phrase: str,
    candidate: dict[str, Any],
    *,
    blocked_tokens: set[str],
    blocked_phrases: set[str],
) -> float:
    normalized = _normalize_text(phrase)
    if not normalized or normalized in blocked_phrases:
        return -1.0

    raw_tokens = [token for token in normalized.split() if token]
    meaningful_tokens = [
        token for token in raw_tokens if token not in blocked_tokens and not token.isdigit()
    ]
    if len(meaningful_tokens) < 2 and not _has_model_signal(phrase):
        return -1.0

    if all(token in blocked_tokens or token.isdigit() for token in raw_tokens):
        return -1.0

    score = float(len(meaningful_tokens))
    if _has_model_signal(phrase):
        score += 2.0

    snippet_text = _normalize_text(candidate.get("snippet", ""))
    title_text = _normalize_text(
        f"{candidate.get('title', '')} {candidate.get('enriched_title', '')}"
    )
    if normalized in snippet_text:
        score += 1.0
    elif normalized in title_text:
        score += 0.5

    return score


def _build_candidate_entities(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        entity_name = _extract_candidate_entity_name(candidate)
        candidate["candidate_entity"] = entity_name
        key = _normalize_text(entity_name)
        if not key:
            continue

        cluster = entities.setdefault(
            key,
            {
                "name": entity_name,
                "support": 0,
                "product_page_support": 0,
                "price_support": 0,
                "rating_support": 0,
                "candidate_urls": [],
            },
        )
        cluster["support"] += 1
        cluster["product_page_support"] += int(_has_product_page_signal(candidate))
        cluster["price_support"] += int(bool(candidate.get("extracted_price")))
        cluster["rating_support"] += int(bool(candidate.get("extracted_rating")))
        if candidate.get("url") and candidate["url"] not in cluster["candidate_urls"]:
            cluster["candidate_urls"].append(candidate["url"])

    return sorted(
        entities.values(),
        key=lambda item: (
            -item["support"],
            -item["product_page_support"],
            -item["price_support"],
            item["name"],
        ),
    )


def _has_specific_product_tokens(name: str, query: str) -> bool:
    normalized = _normalize_text(name)
    if not normalized:
        return False
    if any(char.isdigit() for char in name):
        return True

    name_tokens = [
        token
        for token in normalized.split()
        if token not in _GENERIC_PRODUCT_NAME_TOKENS and not token.isdigit()
    ]
    if not name_tokens:
        return False

    query_tokens = _query_content_tokens(query)
    return any(token not in query_tokens for token in name_tokens)


def _looks_like_budget_bucket_name(name: str, query: str) -> bool:
    normalized = _normalize_text(name)
    if not normalized:
        return True

    query_normalized = _normalize_text(query)
    if _query_is_concrete_lookup(query) and normalized in query_normalized:
        return False

    if normalized == query_normalized:
        return True

    name_tokens = [
        token
        for token in normalized.split()
        if token not in _GENERIC_PRODUCT_NAME_TOKENS and not token.isdigit()
    ]
    if not name_tokens:
        return True

    query_tokens = _query_content_tokens(query)
    return all(token in query_tokens for token in name_tokens)


def _has_model_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    if _MODEL_TOKEN_PATTERN.search(normalized):
        return True

    tokens = normalized.split()
    for previous, current in zip(tokens, tokens[1:]):
        if previous in _GENERIC_PRODUCT_NAME_TOKENS:
            continue
        if current.isdigit():
            return True
    return False


def _has_product_page_signal(candidate: dict[str, Any]) -> bool:
    url = candidate.get("url", "").lower()
    title_text = _normalize_text(
        f"{candidate.get('title', '')} {candidate.get('enriched_title', '')}"
    )
    return any(cue in url for cue in _PRODUCT_URL_CUES) or "product page" in title_text


def _is_editorial_candidate(candidate: dict[str, Any]) -> bool:
    domain = _normalize_text(_extract_domain(candidate.get("url", "")))
    if any(cue in domain for cue in _EDITORIAL_DOMAIN_CUES):
        return True

    if _has_product_page_signal(candidate):
        return False

    title_text = _normalize_text(
        f"{candidate.get('title', '')} {candidate.get('enriched_title', '')}"
    )
    if not title_text:
        return False

    return any(cue in title_text for cue in _EDITORIAL_TEXT_CUES)


def _is_product_detail_candidate(candidate: dict[str, Any]) -> bool:
    """Return True for grounded product/detail pages, not editorial roundups."""
    if _has_product_page_signal(candidate):
        return True

    if _is_editorial_candidate(candidate):
        return False

    entity_name = _normalize_text(candidate.get("candidate_entity", ""))
    title_text = _normalize_text(
        f"{candidate.get('title', '')} {candidate.get('enriched_title', '')}"
    )
    if entity_name and entity_name in title_text:
        return True

    return bool(
        candidate.get("extracted_price")
        and candidate.get("candidate_entity")
        and len(_normalize_text(candidate.get("candidate_entity", "")).split()) >= 2
    )


def _score_evidence_candidate(product_name: str, candidate: dict[str, Any]) -> float:
    normalized_name = _normalize_text(product_name)
    candidate_entity = _normalize_text(candidate.get("candidate_entity", ""))
    title_text = _normalize_text(
        f"{candidate.get('title', '')} {candidate.get('enriched_title', '')}"
    )
    text = _candidate_text(candidate)
    merchant = _normalize_text(candidate.get("merchant") or _extract_domain(candidate.get("url", "")))

    score = 0.0
    if candidate_entity and candidate_entity == normalized_name:
        score += 3.0
    elif normalized_name and normalized_name in text:
        score += 2.0

    if _is_product_detail_candidate(candidate):
        score += 3.0
    elif _has_product_page_signal(candidate):
        score += 1.5

    if candidate.get("extracted_price"):
        score += 0.8
    if candidate.get("extracted_rating"):
        score += 0.3
    if any(domain in merchant for domain in _MERCHANT_DOMAINS):
        score += 0.8
    if len(candidate.get("matched_queries", [])) > 1:
        score += 0.2
    if _is_editorial_candidate(candidate):
        score -= 0.8
    if normalized_name and normalized_name in title_text:
        score += 0.5

    return score


def _has_strong_product_signal(query: str, candidate: dict[str, Any], entity_support: int) -> bool:
    text = _candidate_text(candidate)
    return any(
        (
            _has_specific_product_tokens(candidate.get("candidate_entity", ""), query),
            entity_support >= 2 and not _looks_like_budget_bucket_name(candidate.get("candidate_entity", ""), query),
            _has_model_signal(text),
            _has_product_page_signal(candidate),
            bool(candidate.get("extracted_price")),
            bool(candidate.get("extracted_rating")),
        )
    )


def _score_candidate_for_shortlist(
    query: str,
    candidate: dict[str, Any],
    entity_supports: dict[str, int],
) -> tuple[float, list[str]]:
    text = _candidate_text(candidate)
    title_text = _normalize_text(f"{candidate.get('title', '')} {candidate.get('enriched_title', '')}")
    merchant = _normalize_text(candidate.get("merchant") or _extract_domain(candidate.get("url", "")))
    entity_name = candidate.get("candidate_entity", "")
    entity_key = _normalize_text(entity_name)
    entity_support = entity_supports.get(entity_key, 0)
    reasons: list[str] = []
    score = 0.0

    if _has_specific_product_tokens(entity_name, query):
        score += 1.8
        reasons.append("specific product entity")
    elif entity_support >= 2 and not _looks_like_budget_bucket_name(entity_name, query):
        score += 1.2
        reasons.append("repeated entity across sources")

    if _has_model_signal(title_text):
        score += 2.0
        reasons.append("model-like title")
    elif _has_model_signal(text):
        score += 1.2
        reasons.append("model-like evidence")

    if _has_product_page_signal(candidate):
        score += 1.4
        reasons.append("product page cues")

    if any(domain in merchant for domain in _MERCHANT_DOMAINS):
        score += 0.9
        reasons.append("merchant domain")

    if candidate.get("extracted_price"):
        score += 0.9
        reasons.append("price detected")

    if candidate.get("extracted_rating"):
        score += 0.5
        reasons.append("rating detected")

    if entity_support >= 2:
        score += 0.8
        reasons.append("cross-source entity support")

    if len(candidate.get("matched_queries", [])) > 1:
        score += 0.4
        reasons.append("matched multiple search variants")

    if _is_editorial_candidate(candidate):
        score -= 1.1
        reasons.append("editorial roundup")

    if _looks_like_budget_bucket_name(entity_name, query):
        score -= 1.6
        reasons.append("generic budget/category bucket")

    if not _has_strong_product_signal(query, candidate, entity_support):
        score -= 0.8
        reasons.append("weak product evidence")

    return score, list(dict.fromkeys(reasons))


def _filter_candidates_for_shortlist(
    query: str,
    candidates: list[dict[str, Any]],
    max_results: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score enriched candidates and keep only shortlist-worthy product evidence."""
    candidate_entities = _build_candidate_entities(candidates)
    entity_supports = {
        _normalize_text(item["name"]): item["support"]
        for item in candidate_entities
    }
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        working = dict(candidate)
        working["candidate_entity"] = working.get("candidate_entity") or _extract_candidate_entity_name(working)
        score, reasons = _score_candidate_for_shortlist(query, working, entity_supports)
        working["shortlist_score"] = round(score, 2)
        working["shortlist_reasons"] = reasons[:4]
        working["is_editorial"] = _is_editorial_candidate(working)
        working["has_strong_product_signal"] = _has_strong_product_signal(
            query,
            working,
            entity_supports.get(_normalize_text(working.get("candidate_entity", "")), 0),
        )
        scored.append(working)

    scored.sort(
        key=lambda item: (
            -(item.get("shortlist_score") or 0.0),
            item.get("is_editorial", False),
            -(1 if item.get("extracted_price") else 0),
            -(1 if _has_product_page_signal(item) else 0),
            item.get("title", ""),
        )
    )

    has_non_editorial_winner = any(
        item.get("shortlist_score", 0.0) >= SHORTLIST_SCORE_THRESHOLD and not item.get("is_editorial")
        for item in scored
    )

    filtered: list[dict[str, Any]] = []
    for item in scored:
        if item.get("shortlist_score", 0.0) < SHORTLIST_SCORE_THRESHOLD:
            continue
        if not item.get("has_strong_product_signal"):
            continue
        if has_non_editorial_winner and item.get("is_editorial") and not _has_product_page_signal(item):
            continue

        filtered.append(item)
        if len(filtered) >= min(len(scored), max(FILTERED_CANDIDATE_LIMIT, max_results * 2)):
            break

    if filtered:
        return filtered, candidate_entities

    fallback = [item for item in scored if item.get("has_strong_product_signal")]
    return fallback[: min(len(fallback), max(FILTERED_CANDIDATE_LIMIT, max_results * 2))], candidate_entities


# ── Step 3: GPT-4o-mini Normalization ─────────────────────────────

async def _normalize_products(
    query: str,
    candidates: list[dict],
    settings: Settings,
    max_results: int,
) -> ProductDiscoveryResult:
    """Use GPT-4o-mini to normalize raw candidates into structured products."""
    result, _debug = await _normalize_products_with_debug(
        query=query,
        candidates=candidates,
        settings=settings,
        max_results=max_results,
    )
    return result


async def _normalize_products_with_debug(
    query: str,
    candidates: list[dict],
    settings: Settings,
    max_results: int,
) -> tuple[ProductDiscoveryResult, dict[str, Any]]:
    """Use GPT-4o-mini to normalize raw candidates into structured products."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Build evidence for the prompt
    evidence_parts = []
    for i, c in enumerate(candidates, 1):
        parts = [f"--- Candidate {i} ---"]
        parts.append(f"URL: {c.get('url', 'N/A')}")
        parts.append(f"Merchant: {c.get('merchant', 'N/A')}")
        parts.append(f"Search title: {c.get('title', 'N/A')}")
        parts.append(f"Search snippet: {c.get('snippet', 'N/A')}")
        if c.get("enriched_title"):
            parts.append(f"Page title: {c['enriched_title']}")
        if c.get("extracted_price"):
            parts.append(f"Extracted price: {c['extracted_price']}")
        if c.get("extracted_rating"):
            parts.append(f"Extracted rating: {c['extracted_rating']}")
        if c.get("page_text_excerpt"):
            # Only include first 500 chars of page text to keep prompt size reasonable
            parts.append(f"Page excerpt: {c['page_text_excerpt'][:500]}")
        evidence_parts.append("\n".join(parts))

    user_prompt = (
        f"Shopping query: \"{query}\"\n"
        f"Maximum products to return: {max_results}\n\n"
        f"Raw candidate data from web search and page enrichment:\n\n"
        + "\n\n".join(evidence_parts)
        + "\n\nBased on the above evidence only, produce a normalized product list with concrete brand/model names."
    )

    logger.info("Calling GPT-4o-mini for product normalization (%d candidates)", len(candidates))

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=PRODUCT_SCHEMA,
            temperature=0.2,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        # Build Product objects with evidence_urls
        products = []
        rejected_generic_products: list[dict[str, str]] = []
        shortlist_reasons: list[dict[str, Any]] = []
        for p in data.get("products", [])[:max_results]:
            product_name = p.get("name", "")
            if not _looks_like_specific_product_name(product_name, candidates, query):
                logger.info(
                    "Skipping normalized product '%s' because it was too generic",
                    product_name or "unknown",
                )
                rejected_generic_products.append(
                    {
                        "name": product_name or "unknown",
                        "reason": "generic-or-unreviewable product label",
                    }
                )
                continue

            evidence_candidates = _find_evidence_candidates(product_name, candidates)
            resolved_detail_candidates: list[dict[str, Any]] = []
            if not any(_is_product_detail_candidate(candidate) for candidate in evidence_candidates):
                resolved_detail_candidates = await _resolve_product_detail_candidates(product_name)

            combined_candidates = list(candidates)
            for candidate in resolved_detail_candidates:
                url = _canonicalize_candidate_url(candidate.get("url"))
                if not url:
                    continue
                if any(_canonicalize_candidate_url(existing.get("url")) == url for existing in combined_candidates):
                    continue
                combined_candidates.append(candidate)

            evidence_candidates = _find_evidence_candidates(product_name, combined_candidates)
            canonical_candidate = _select_canonical_evidence_candidate(
                p.get("url"),
                evidence_candidates,
            )
            if not canonical_candidate:
                logger.info(
                    "Skipping normalized product '%s' because no grounded evidence URL was found",
                    p.get("name", "unknown"),
                )
                continue

            grounded_url = _canonicalize_candidate_url(canonical_candidate.get("url"))
            evidence_urls = _find_evidence_urls(product_name, combined_candidates)
            if grounded_url and grounded_url not in evidence_urls:
                evidence_urls = [grounded_url, *evidence_urls][:3]
            if not grounded_url:
                continue

            product_shortlist_reasons = _build_product_shortlist_reasons(
                product_name,
                combined_candidates,
                evidence_urls,
            )
            grounding_reason = (
                "resolved product detail page"
                if resolved_detail_candidates and _is_product_detail_candidate(canonical_candidate)
                else "grounded product detail page"
                if _is_product_detail_candidate(canonical_candidate)
                else "grounded evidence page"
            )
            if grounding_reason not in product_shortlist_reasons:
                product_shortlist_reasons.append(grounding_reason)
            shortlist_reasons.append(
                {
                    "product_name": product_name,
                    "reasons": product_shortlist_reasons,
                    "confidence": p.get("confidence", 0.0),
                    "evidence_urls": evidence_urls[:3],
                }
            )
            products.append(
                Product(
                    name=product_name,
                    price=p.get("price"),
                    rating=p.get("rating"),
                    url=grounded_url,
                    key_features=p.get("key_features", [])[:3],
                    source=canonical_candidate.get("merchant") or p.get("source", "unknown"),
                    evidence_urls=evidence_urls,
                    confidence=p.get("confidence", 0.0),
                )
            )

        products.sort(
            key=lambda product: (
                -(product.confidence or 0.0),
                -len(product.evidence_urls),
                product.name,
            )
        )
        products = products[:max_results]

        result = ProductDiscoveryResult(
            query=query,
            products=products,
            summary=data.get("summary", ""),
        )
        return result, {
            "normalized_products": [_product_snapshot(product) for product in products],
            "finalized_products": [_product_snapshot(product) for product in products],
            "shortlist_reasons": shortlist_reasons,
            "rejected_generic_products": rejected_generic_products,
            "rejected_entities": rejected_generic_products,
            "summary": result.summary,
        }

    except Exception as exc:
        logger.error("GPT-4o-mini normalization failed: %s", exc)
        result = ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"Product normalization failed: {exc}",
        )
        return result, {
            "normalized_products": [],
            "finalized_products": [],
            "shortlist_reasons": [],
            "rejected_generic_products": [],
            "rejected_entities": [],
            "summary": result.summary,
            "error": str(exc),
        }
    finally:
        await _close_async_client(client)


def _build_product_shortlist_reasons(
    product_name: str,
    candidates: list[dict[str, Any]],
    evidence_urls: list[str],
) -> list[str]:
    reasons: list[str] = []
    evidence_url_set = {
        _canonicalize_candidate_url(url)
        for url in evidence_urls
        if _canonicalize_candidate_url(url)
    }

    for candidate in candidates:
        if _canonicalize_candidate_url(candidate.get("url")) not in evidence_url_set:
            continue
        for reason in candidate.get("shortlist_reasons", []):
            if reason not in reasons:
                reasons.append(reason)

    if len(evidence_urls) > 1 and "multi-source support" not in reasons:
        reasons.append("multi-source support")
    if any(char.isdigit() for char in product_name) and "concrete model name" not in reasons:
        reasons.append("concrete model name")

    return reasons[:4]


def _find_evidence_candidates(product_name: str, candidates: list[dict]) -> list[dict[str, Any]]:
    """Find candidate rows that likely support the given normalized product name."""
    normalized_name = _normalize_text(product_name)
    if not normalized_name:
        return []

    name_words = [word for word in normalized_name.split() if len(word) > 2]
    threshold = 1 if len(name_words) <= 1 else 2
    matched: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for candidate in candidates:
        url = _canonicalize_candidate_url(candidate.get("url"))
        if not url or url in seen_urls:
            continue

        text = _normalize_text(
            " ".join(
                [
                    candidate.get("candidate_entity", ""),
                    candidate.get("title", ""),
                    candidate.get("snippet", ""),
                    candidate.get("enriched_title", ""),
                ]
            )
        )
        if not text:
            continue

        if normalized_name in text:
            seen_urls.add(url)
            matched.append(candidate)
            continue

        matches = sum(1 for word in name_words if word in text)
        if matches >= threshold:
            seen_urls.add(url)
            matched.append(candidate)

    return sorted(
        matched,
        key=lambda item: (
            -_score_evidence_candidate(product_name, item),
            -len(item.get("matched_queries", [])),
            item.get("url", ""),
        ),
    )


def _select_canonical_evidence_candidate(
    proposed_url: str | None,
    evidence_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the grounded canonical evidence page, preferring product/detail URLs."""
    proposed = _canonicalize_candidate_url(proposed_url)
    if proposed:
        for candidate in evidence_candidates:
            if (
                _canonicalize_candidate_url(candidate.get("url")) == proposed
                and _is_product_detail_candidate(candidate)
            ):
                return candidate

    for candidate in evidence_candidates:
        if _is_product_detail_candidate(candidate):
            return candidate

    for candidate in evidence_candidates:
        if _canonicalize_candidate_url(candidate.get("url")):
            return candidate

    return None


def _find_evidence_urls(product_name: str, candidates: list[dict]) -> list[str]:
    """Find candidate URLs that likely relate to the given product name."""
    urls: list[str] = []
    for candidate in _find_evidence_candidates(product_name, candidates):
        url = _canonicalize_candidate_url(candidate.get("url"))
        if url and url not in urls:
            urls.append(url)
    return urls[:3]


def _search_product_detail_candidates(product_name: str) -> list[dict[str, Any]]:
    """Search for grounded detail pages for an already-known product."""
    search_queries = _build_detail_search_queries(product_name)
    logger.info("Detail grounding queries for %s: %s", product_name, search_queries)

    try:
        with DDGS() as ddgs:
            candidates: list[dict[str, Any]] = []
            seen_urls: set[str] = set()
            per_query_limit = max(2, DETAIL_SEARCH_MAX_RESULTS // max(len(search_queries), 1))

            for search_query in search_queries:
                results = list(ddgs.text(search_query, max_results=per_query_limit))
                for row in results:
                    url = _canonicalize_candidate_url(row.get("href", ""))
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    candidates.append(
                        {
                            "title": row.get("title", ""),
                            "snippet": row.get("body", ""),
                            "url": url,
                            "matched_query": search_query,
                            "matched_queries": [search_query],
                            "candidate_entity": product_name,
                        }
                    )
                    if len(candidates) >= DETAIL_SEARCH_MAX_RESULTS:
                        return candidates
    except Exception as exc:
        logger.info(
            "Detail grounding search failed for %s: %s",
            product_name,
            type(exc).__name__,
        )
        return []

    return candidates


async def _resolve_product_detail_candidates(product_name: str) -> list[dict[str, Any]]:
    """Run a second-pass exact-product search to improve product-link grounding."""
    raw_candidates = await asyncio.to_thread(_search_product_detail_candidates, product_name)
    if not raw_candidates:
        return []

    enriched = await _enrich_candidates(raw_candidates[:DETAIL_SEARCH_MAX_RESULTS])
    for candidate in enriched:
        candidate["candidate_entity"] = candidate.get("candidate_entity") or product_name
        candidate["shortlist_score"] = round(_score_evidence_candidate(product_name, candidate), 2)
        candidate["shortlist_reasons"] = list(
            dict.fromkeys(
                [
                    reason
                    for reason, condition in (
                        ("resolved detail search", True),
                        ("product page cues", _has_product_page_signal(candidate)),
                        (
                            "merchant domain",
                            any(
                                domain in _normalize_text(
                                    candidate.get("merchant") or _extract_domain(candidate.get("url", ""))
                                )
                                for domain in _MERCHANT_DOMAINS
                            ),
                        ),
                        ("price detected", bool(candidate.get("extracted_price"))),
                    )
                    if condition
                ]
            )
        )

    return [candidate for candidate in _find_evidence_candidates(product_name, enriched) if _canonicalize_candidate_url(candidate.get("url"))]
