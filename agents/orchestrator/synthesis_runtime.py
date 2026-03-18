"""Unified-response synthesis helpers for the orchestrator."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from openai import AsyncOpenAI

from common.a2a_models import (
    ProductDiscoveryResult,
    RankedRecommendation,
    ReviewSummary,
    UnifiedResponse,
    UnifiedSourceLink,
)
from common.config import Settings
from common.runtime_helpers import close_async_resource

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a shopping advisor synthesizer. You will receive product discovery data (prices, ratings, features) and YouTube review summaries (sentiment, pros, cons).

Your task:
1. Merge all evidence into a ranked list of product recommendations.
2. For each product, combine discovery data (price, rating, features) with review data (sentiment, pros, cons) where available.
3. Assign a composite score (0.0 - 1.0) considering: price-value ratio, ratings, review sentiment, and confidence.
4. Write a clear rationale for each product's ranking.
5. Return 1-3 recommendations only, ordered by score descending.
6. Use the exact product names from PRODUCT DISCOVERY DATA whenever discovery data is present.
7. If a product has no review data, note that in the rationale and lower the score slightly.
8. Keep pros and cons to a maximum of 3 each per product, prioritizing the most impactful ones.
9. NEVER invent data. If something is not in the evidence, do not include it."""

SYNTHESIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "SynthesisResult",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "product_name": {"type": "string"},
                            "price": {"type": ["string", "null"]},
                            "rating": {"type": ["string", "null"]},
                            "sentiment": {"type": ["string", "null"]},
                            "score": {"type": "number"},
                            "rationale": {"type": "string"},
                            "pros": {"type": "array", "items": {"type": "string"}},
                            "cons": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "rank",
                            "product_name",
                            "price",
                            "rating",
                            "sentiment",
                            "score",
                            "rationale",
                            "pros",
                            "cons",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "notes": {"type": ["string", "null"]},
            },
            "required": ["recommendations", "notes"],
            "additionalProperties": False,
        },
    },
}


def build_source_links(
    product_result: ProductDiscoveryResult | None,
    review_results: dict[str, ReviewSummary],
) -> list[UnifiedSourceLink]:
    """Build grounded source links from actual agent-retrieved data."""
    sources: list[UnifiedSourceLink] = []
    seen_urls: set[str] = set()

    if product_result:
        for product in product_result.products:
            if product.url and product.url not in seen_urls:
                sources.append(
                    UnifiedSourceLink(
                        type="product",
                        title=f"{product.name} on {product.source}",
                        url=product.url,
                        agent="product-discovery",
                    )
                )
                seen_urls.add(product.url)

            for evidence_url in product.evidence_urls:
                if evidence_url and evidence_url not in seen_urls:
                    sources.append(
                        UnifiedSourceLink(
                            type="product",
                            title=f"Evidence for {product.name}",
                            url=evidence_url,
                            agent="product-discovery",
                        )
                    )
                    seen_urls.add(evidence_url)

    for review in review_results.values():
        for video in review.sources:
            if video.url and video.url not in seen_urls:
                sources.append(
                    UnifiedSourceLink(
                        type="video",
                        title=f"{video.title} by {video.channel}",
                        url=video.url,
                        agent="youtube-review",
                    )
                )
                seen_urls.add(video.url)

    return sources


def fallback_synthesis(
    query: str,
    product_result: ProductDiscoveryResult | None,
    review_results: dict[str, ReviewSummary],
    errors: list[str],
    *,
    max_final_recommendations: int = 3,
) -> UnifiedResponse:
    """Build a basic UnifiedResponse without LLM synthesis as fallback."""
    recommendations = []

    if product_result:
        for i, product in enumerate(product_result.products[:max_final_recommendations], 1):
            review = review_results.get(product.name)
            recommendations.append(
                RankedRecommendation(
                    rank=i,
                    product_name=product.name,
                    price=product.price,
                    rating=product.rating,
                    sentiment=review.overall_sentiment if review else None,
                    score=product.confidence,
                    rationale=f"Product from {product.source} with confidence {product.confidence:.2f}",
                    pros=review.pros[:3] if review else product.key_features[:3],
                    cons=review.cons[:3] if review else [],
                    confidence=product.confidence,
                )
            )
    elif review_results:
        for i, (product_name, review) in enumerate(list(review_results.items())[:max_final_recommendations], 1):
            recommendations.append(
                RankedRecommendation(
                    rank=i,
                    product_name=product_name,
                    price=None,
                    rating=None,
                    sentiment=review.overall_sentiment,
                    score=review.confidence,
                    rationale="Recommendation derived from YouTube review evidence only.",
                    pros=review.pros[:3],
                    cons=review.cons[:3],
                    confidence=review.confidence,
                )
            )

    sources = build_source_links(product_result, review_results)

    return UnifiedResponse(
        query=query,
        recommendations=recommendations,
        sources=sources,
        partial=True,
        notes="Fallback synthesis (LLM unavailable). " + "; ".join(errors),
    )


async def synthesize_response(
    *,
    query: str,
    settings: Settings,
    product_result: ProductDiscoveryResult | None,
    review_results: dict[str, ReviewSummary],
    decision: Any,
    trace: Any,
    errors: list[str],
    llm_client: AsyncOpenAI | None = None,
    llm_client_factory: Callable[..., AsyncOpenAI] = AsyncOpenAI,
    max_final_recommendations: int = 3,
    match_known_product_name_fn: Callable[[str, list[str]], str | None],
) -> dict[str, Any]:
    """Merge product discovery and review data into a UnifiedResponse."""
    has_products = product_result and len(product_result.products) > 0
    has_reviews = len(review_results) > 0
    requested_products = bool(decision and decision.needs_product_discovery)
    requested_reviews = bool(decision and decision.needs_youtube_reviews)
    missing_requested_products = requested_products and not has_products
    missing_requested_reviews = requested_reviews and not has_reviews

    if trace:
        trace.add_event(
            "Synthesizing unified response",
            status="active",
            detail="Merging product discovery and YouTube review evidence.",
            data={
                "product_count": len(product_result.products) if product_result else 0,
                "review_count": len(review_results),
            },
        )
        trace.set_synthesis(
            {
                "status": "running",
                "product_count": len(product_result.products) if product_result else 0,
                "review_count": len(review_results),
                "requested_products": requested_products,
                "requested_reviews": requested_reviews,
            }
        )

    if not has_products and not has_reviews:
        if requested_products:
            note = "No concrete products could be finalized from product discovery."
            if errors:
                note = note + " " + "; ".join(errors)
        else:
            note = (
                "No data was available from any agent. " + "; ".join(errors)
                if errors
                else "No data was available from any agent."
            )

        unified = UnifiedResponse(
            query=query,
            recommendations=[],
            sources=[],
            partial=True,
            notes=note,
        )
        if trace:
            trace.set_synthesis(
                {
                    "status": "completed",
                    "product_count": 0,
                    "review_count": 0,
                    "partial": True,
                    "notes": unified.notes,
                }
            )
            trace.set_final_response(unified)
            trace.add_event(
                "Unified response ready",
                status="done",
                detail=unified.notes,
            )
        return {"unified_response": unified, "errors": errors}

    evidence_parts: list[str] = []
    if has_products:
        evidence_parts.append("=== PRODUCT DISCOVERY DATA ===")
        for index, product in enumerate(product_result.products, 1):
            parts = [f"--- Product {index} ---"]
            parts.append(f"Name: {product.name}")
            parts.append(f"Price: {product.price or 'N/A'}")
            parts.append(f"Rating: {product.rating or 'N/A'}")
            parts.append(f"Source: {product.source}")
            parts.append(f"URL: {product.url}")
            parts.append(f"Features: {', '.join(product.key_features) if product.key_features else 'N/A'}")
            parts.append(f"Confidence: {product.confidence}")
            evidence_parts.append("\n".join(parts))

    if has_reviews:
        evidence_parts.append("\n=== YOUTUBE REVIEW DATA ===")
        for product_name, review in review_results.items():
            parts = [f"--- Reviews for: {product_name} ---"]
            parts.append(f"Sentiment: {review.overall_sentiment}")
            parts.append(f"Review Score: {review.score}/10")
            parts.append(f"Pros: {', '.join(review.pros) if review.pros else 'N/A'}")
            parts.append(f"Cons: {', '.join(review.cons) if review.cons else 'N/A'}")
            parts.append(f"Recommendation: {review.recommendation}")
            parts.append(f"Confidence: {review.confidence}")
            if review.key_quotes:
                parts.append(f"Key Quotes: {'; '.join(review.key_quotes[:3])}")
            evidence_parts.append("\n".join(parts))

    user_prompt = (
        f"Shopping query: \"{query}\"\n\n"
        + "\n\n".join(evidence_parts)
        + "\n\nBased on ALL the above evidence, produce a ranked recommendation list."
    )

    owned_client = llm_client is None
    runtime_client = llm_client or llm_client_factory(api_key=settings.OPENAI_API_KEY)

    try:
        response = await runtime_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SYNTHESIS_SCHEMA,
            temperature=0.2,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        recommendations = []
        known_product_names = [product.name for product in product_result.products] if product_result else []
        for item in data.get("recommendations", [])[:max_final_recommendations]:
            resolved_name = item["product_name"]
            if known_product_names:
                matched_name = match_known_product_name_fn(item["product_name"], known_product_names)
                if matched_name is None:
                    logger.info(
                        "Skipping synthesized recommendation '%s' because it did not match a discovered product",
                        item["product_name"],
                    )
                    continue
                resolved_name = matched_name

            recommendations.append(
                RankedRecommendation(
                    rank=item["rank"],
                    product_name=resolved_name,
                    price=item.get("price"),
                    rating=item.get("rating"),
                    sentiment=item.get("sentiment"),
                    score=item.get("score", 0.0),
                    rationale=item["rationale"],
                    pros=item.get("pros", [])[:3],
                    cons=item.get("cons", [])[:3],
                    confidence=item.get("confidence", 0.0),
                )
            )

        if not recommendations and has_products:
            raise ValueError("Synthesis returned no recommendations that matched discovered products")

        sources = build_source_links(product_result, review_results)

        partial = missing_requested_products or missing_requested_reviews
        notes = data.get("notes")
        if partial and not notes:
            missing = []
            if missing_requested_products:
                missing.append("product discovery")
            if missing_requested_reviews:
                missing.append("YouTube reviews")
            notes = f"Partial results - {', '.join(missing)} data was unavailable."

        unified = UnifiedResponse(
            query=query,
            recommendations=recommendations,
            sources=sources,
            partial=partial,
            notes=notes,
        )

        if trace:
            trace.set_synthesis(
                {
                    "status": "completed",
                    "product_count": len(product_result.products) if product_result else 0,
                    "review_count": len(review_results),
                    "partial": partial,
                    "notes": notes,
                    "recommendation_count": len(recommendations),
                    "source_count": len(sources),
                }
            )
            trace.set_final_response(unified)
            trace.add_event(
                "Unified response ready",
                status="done",
                detail=f"Built {len(recommendations)} ranked recommendations.",
            )

        logger.info(
            "Synthesis complete: %d recommendations, %d sources",
            len(recommendations),
            len(sources),
        )
        return {"unified_response": unified, "errors": errors}

    except Exception as exc:
        msg = f"Synthesis LLM call failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        if trace:
            trace.append_error(msg)
            trace.add_event(
                "Synthesis failed, using fallback",
                status="error",
                detail=str(exc),
            )

        unified = fallback_synthesis(
            query=query,
            product_result=product_result,
            review_results=review_results,
            errors=errors,
            max_final_recommendations=max_final_recommendations,
        )
        if trace:
            trace.set_synthesis(
                {
                    "status": "fallback",
                    "product_count": len(product_result.products) if product_result else 0,
                    "review_count": len(review_results),
                    "partial": True,
                    "notes": unified.notes,
                    "recommendation_count": len(unified.recommendations),
                    "source_count": len(unified.sources),
                }
            )
            trace.set_final_response(unified)
        return {"unified_response": unified, "errors": errors}
    finally:
        if owned_client:
            await close_async_resource(runtime_client)
