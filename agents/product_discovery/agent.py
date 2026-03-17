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
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import AsyncOpenAI

from common.a2a_models import Product, ProductDiscoveryResult
from common.config import Settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

SEARCH_MAX_RESULTS = 12
ENRICH_TOP_N = 8
ENRICH_TIMEOUT = 8  # seconds per page fetch
FINAL_MAX_PRODUCTS = 5

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
6. Return at most 5 products, ordered by confidence descending.
7. Write a short summary (1-2 sentences) of what was found."""


# ── Main entry point ──────────────────────────────────────────────

async def search_products(
    query: str,
    settings: Settings,
    max_results: int = FINAL_MAX_PRODUCTS,
) -> ProductDiscoveryResult:
    """Retrieve and normalize product candidates for a shopper query.

    This function is the canonical capability used by both the A2A facade
    and the MCP tool.

    Args:
        query: Natural-language shopping query.
        settings: Application settings (API keys, model config).
        max_results: Maximum products to return (default 5).

    Returns:
        Structured ProductDiscoveryResult with provenance.
    """
    logger.info("Starting product discovery for: %s", query)

    # Step 1: Candidate retrieval
    candidates = await asyncio.to_thread(_search_duckduckgo, query)
    if not candidates:
        logger.warning("No search results for: %s", query)
        return ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"No product results found for '{query}'.",
        )

    # Step 2: Deterministic enrichment
    enriched = await _enrich_candidates(candidates[:ENRICH_TOP_N])
    usable = [c for c in enriched if c.get("enriched_title") or c.get("snippet")]

    if not usable:
        logger.warning("No usable enrichment data for: %s", query)
        return ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"Search returned results but no usable product data for '{query}'.",
        )

    # Step 3: GPT-4o-mini normalization
    result = await _normalize_products(query, usable, settings, max_results)
    return result


# ── Step 1: DuckDuckGo Search ─────────────────────────────────────

def _search_duckduckgo(query: str) -> list[dict]:
    """Search DuckDuckGo for product candidates."""
    search_query = f"{query} buy price review rating"
    logger.info("DuckDuckGo search: '%s'", search_query)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=SEARCH_MAX_RESULTS))
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)
        return []

    candidates = []
    for r in results:
        url = r.get("href", "")
        if not _is_web_url(url):
            continue

        candidates.append({
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": url,
        })

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


def _ground_product_url(
    proposed_url: str | None,
    evidence_urls: list[str],
    candidate_urls: set[str],
) -> str | None:
    """Choose a product URL that is grounded in retrieved candidate data."""
    if proposed_url and proposed_url in candidate_urls:
        return proposed_url

    if evidence_urls:
        return evidence_urls[0]

    return None


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


def _extract_rating(text: str) -> str | None:
    """Try to find a rating pattern like 4.5/5 or 4.5 out of 5 in text."""
    match = re.search(r"(\d(?:\.\d)?)\s*(?:/|out of)\s*5", text)
    if match:
        return f"{match.group(1)}/5"
    return None


# ── Step 3: GPT-4o-mini Normalization ─────────────────────────────

async def _normalize_products(
    query: str,
    candidates: list[dict],
    settings: Settings,
    max_results: int,
) -> ProductDiscoveryResult:
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
        + "\n\nBased on the above evidence only, produce a normalized product list."
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
        candidate_urls = {
            candidate["url"]
            for candidate in candidates
            if _is_web_url(candidate.get("url", ""))
        }

        # Build Product objects with evidence_urls
        products = []
        for p in data.get("products", [])[:max_results]:
            # Find matching candidate URLs for evidence
            evidence_urls = _find_evidence_urls(p.get("name", ""), candidates)
            grounded_url = _ground_product_url(
                p.get("url"),
                evidence_urls,
                candidate_urls,
            )
            if not grounded_url:
                logger.info(
                    "Skipping normalized product '%s' because no grounded source URL was found",
                    p.get("name", "unknown"),
                )
                continue

            if grounded_url not in evidence_urls:
                evidence_urls = [grounded_url, *evidence_urls][:3]

            products.append(
                Product(
                    name=p["name"],
                    price=p.get("price"),
                    rating=p.get("rating"),
                    url=grounded_url,
                    key_features=p.get("key_features", [])[:3],
                    source=p.get("source", "unknown"),
                    evidence_urls=evidence_urls,
                    confidence=p.get("confidence", 0.0),
                )
            )

        return ProductDiscoveryResult(
            query=query,
            products=products,
            summary=data.get("summary", ""),
        )

    except Exception as exc:
        logger.error("GPT-4o-mini normalization failed: %s", exc)
        return ProductDiscoveryResult(
            query=query,
            products=[],
            summary=f"Product normalization failed: {exc}",
        )


def _find_evidence_urls(product_name: str, candidates: list[dict]) -> list[str]:
    """Find candidate URLs that likely relate to the given product name."""
    normalized_name = _normalize_text(product_name)
    if not normalized_name:
        return []

    name_words = [word for word in normalized_name.split() if len(word) > 2]
    threshold = 1 if len(name_words) <= 1 else 2
    urls = []
    for c in candidates:
        url = c.get("url", "")
        if not _is_web_url(url):
            continue

        text = _normalize_text(
            f"{c.get('title', '')} {c.get('snippet', '')} {c.get('enriched_title', '')}"
        )
        if not text:
            continue

        if normalized_name in text:
            urls.append(url)
            continue

        if not name_words:
            continue

        matches = sum(1 for word in name_words if word in text)
        if matches >= threshold:
            urls.append(url)

    return list(dict.fromkeys(urls))[:3]
