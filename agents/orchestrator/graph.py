"""Central Orchestrator — LangGraph StateGraph.

Implements the 5-node workflow:
    discover_agents → route_tasks → [execute_product_discovery, execute_youtube_reviews] → synthesize

Key design decisions:
- Deterministic routing (no LLM for routing).
- Dependency-injected a2a_client and llm_client for testability.
- Bounded concurrency for YouTube fan-out via asyncio.Semaphore.
- Structured UnifiedResponse produced first, then rendered.
- All URLs in the final response are grounded in agent-retrieved data.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from agents.orchestrator.routing import RoutingDecision, route_query
from agents.orchestrator.trace import TraceRecorder
from common.a2a_client import A2AClient, A2AError
from common.a2a_models import (
    AgentCard,
    Message,
    Product,
    ProductDiscoveryResult,
    RankedRecommendation,
    ReviewSummary,
    Task,
    UnifiedResponse,
    UnifiedSourceLink,
)
from common.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TERMINAL_TASK_STATES = {"completed", "failed", "canceled", "input-required"}
_TASK_POLL_INTERVAL_SECONDS = 0.25


# ═══════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════

class OrchestratorState(TypedDict, total=False):
    """Typed state flowing through the LangGraph nodes."""
    # Input
    query: str
    settings: Settings

    # Dependency injection (optional overrides)
    a2a_client: A2AClient | None
    llm_client: AsyncOpenAI | None
    trace_recorder: TraceRecorder | None

    # Discovery
    discovered_agents: dict[str, AgentCard]
    card_urls: dict[str, str]

    # Routing
    routing_decision: RoutingDecision

    # Execution results
    product_result: ProductDiscoveryResult | None
    review_results: dict[str, ReviewSummary]  # product_name → ReviewSummary

    # Synthesis
    unified_response: UnifiedResponse | None

    # Error tracking
    errors: list[str]


# ═══════════════════════════════════════════════════════════════════
# Node 1: Discover Agents
# ═══════════════════════════════════════════════════════════════════

async def discover_agents(state: OrchestratorState) -> dict[str, Any]:
    """Fetch Agent Cards from configured discovery URLs."""
    settings = state["settings"]
    client = state.get("a2a_client") or A2AClient()
    trace = state.get("trace_recorder")
    errors: list[str] = list(state.get("errors", []))

    discovered: dict[str, AgentCard] = {}
    card_urls: dict[str, str] = {}

    if trace:
        trace.add_event(
            "Discovering agents",
            status="active",
            detail="Fetching seeded Agent Cards for downstream services.",
        )

    # Map of agent name -> card URL from settings
    agent_endpoints = {
        "product-discovery": settings.PRODUCT_AGENT_CARD_URL,
        "youtube-review": settings.YOUTUBE_AGENT_CARD_URL,
    }

    for name, card_url in agent_endpoints.items():
        try:
            card = await client.discover_agent(card_url)
            discovered[name] = card
            card_urls[name] = card_url
            logger.info("Discovered agent: %s at %s", card.name, card.url)
            if trace:
                trace.add_event(
                    f"Discovered {card.name}",
                    status="done",
                    detail=card.url,
                    agent=name,
                )
        except Exception as exc:
            msg = f"Failed to discover {name} at {card_url}: {exc}"
            logger.warning(msg)
            errors.append(msg)
            if trace:
                trace.append_error(msg)
                trace.add_event(
                    f"Failed to discover {name}",
                    status="error",
                    detail=str(exc),
                    agent=name,
                )

    if trace:
        trace.set_discovered_agents(discovered, card_urls)

    return {
        "discovered_agents": discovered,
        "card_urls": card_urls,
        "a2a_client": client,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════
# Node 2: Route Tasks
# ═══════════════════════════════════════════════════════════════════

async def route_tasks(state: OrchestratorState) -> dict[str, Any]:
    """Apply deterministic routing to determine which agents to call."""
    query = state["query"]
    discovered = state.get("discovered_agents", {})
    card_urls = state.get("card_urls", {})
    trace = state.get("trace_recorder")

    if trace:
        trace.add_event(
            "Routing query",
            status="active",
            detail="Resolving which agents should execute this request.",
        )

    decision = route_query(query, discovered, card_urls)

    if trace:
        trace.set_routing(decision)
        trace.add_event(
            "Routing complete",
            status="done",
            detail=f"Resolved {len(decision.routes)} downstream routes.",
            data={
                "needs_product_discovery": decision.needs_product_discovery,
                "needs_youtube_reviews": decision.needs_youtube_reviews,
            },
        )

    return {"routing_decision": decision}


# ═══════════════════════════════════════════════════════════════════
# Node 3a: Execute Product Discovery
# ═══════════════════════════════════════════════════════════════════

async def execute_product_discovery(state: OrchestratorState) -> dict[str, Any]:
    """Call the Product Discovery agent via A2A SendMessage."""
    decision = state.get("routing_decision")
    trace = state.get("trace_recorder")
    errors: list[str] = list(state.get("errors", []))

    if not decision or not decision.needs_product_discovery:
        return {"product_result": None, "errors": errors}

    client = state.get("a2a_client") or A2AClient()

    # Find the product-discovery route
    route = next(
        (r for r in decision.routes if r.agent_name == "product-discovery"),
        None,
    )
    if not route:
        return {"product_result": None, "errors": errors}

    try:
        logger.info("Calling Product Discovery agent: %s", route.agent_url)
        if trace:
            trace.add_event(
                "Calling Product Discovery agent",
                status="active",
                detail=route.agent_url,
                agent="product-discovery",
                data={"query_text": route.query_text},
            )
        result = await client.send_message(route.agent_url, route.query_text)
        result = await _resolve_send_message_result(
            result=result,
            client=client,
            agent_url=route.agent_url,
            settings=state["settings"],
            task_observer=(
                lambda task: _record_agent_task_snapshot(
                    trace,
                    agent_name="product-discovery",
                    task=task,
                )
            ) if trace else None,
        )

        # Parse the result — it should be a Task with an artifact
        if isinstance(result, Task):
            product_result = _extract_product_result(result)
            if product_result:
                logger.info(
                    "Product Discovery returned %d products",
                    len(product_result.products),
                )
                if trace:
                    trace.add_event(
                        "Product discovery completed",
                        status="done",
                        detail=f"Returned {len(product_result.products)} normalized products.",
                        agent="product-discovery",
                    )
                return {"product_result": product_result, "errors": errors}

        msg = "Product Discovery agent returned no parseable product data"
        logger.warning(msg)
        errors.append(msg)
        if trace:
            trace.append_error(msg)
            trace.add_event(
                "Product discovery returned no parseable data",
                status="error",
                detail=msg,
                agent="product-discovery",
            )
        return {"product_result": None, "errors": errors}

    except (A2AError, Exception) as exc:
        msg = f"Product Discovery agent failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        if trace:
            trace.append_error(msg)
            trace.add_event(
                "Product discovery failed",
                status="error",
                detail=str(exc),
                agent="product-discovery",
            )
        return {"product_result": None, "errors": errors}


def _extract_product_result(task: Task) -> ProductDiscoveryResult | None:
    """Extract ProductDiscoveryResult from a completed Task's artifacts."""
    if task.status.state != "completed":
        logger.warning("Product task not completed: state=%s", task.status.state)
        return None

    if not task.artifacts:
        return None

    data = _extract_named_artifact_json(task, "product-discovery-result")
    if data is None:
        return None

    try:
        return ProductDiscoveryResult(**data)
    except Exception as exc:
        logger.debug("Failed to parse product result artifact: %s", exc)
        return None


def _extract_product_trace(task: Task) -> dict[str, Any] | None:
    return _extract_named_artifact_json(task, "product-discovery-debug")


# ═══════════════════════════════════════════════════════════════════
# Node 3b: Execute YouTube Reviews (with bounded parallelism)
# ═══════════════════════════════════════════════════════════════════

async def execute_youtube_reviews(state: OrchestratorState) -> dict[str, Any]:
    """Call the YouTube Review agent for each discovered product.

    Uses bounded concurrency (asyncio.Semaphore) to avoid overwhelming
    the YouTube agent with too many parallel requests.
    """
    decision = state.get("routing_decision")
    product_result = state.get("product_result")
    trace = state.get("trace_recorder")
    errors: list[str] = list(state.get("errors", []))
    settings = state["settings"]

    if not decision or not decision.needs_youtube_reviews:
        return {"review_results": {}, "errors": errors}

    client = state.get("a2a_client") or A2AClient()

    # Find the youtube-review route
    route = next(
        (r for r in decision.routes if r.agent_name == "youtube-review"),
        None,
    )
    if not route:
        return {"review_results": {}, "errors": errors}

    # Determine product names to fetch reviews for
    product_names: list[str] = []
    if product_result and product_result.products:
        product_names = list(dict.fromkeys(p.name for p in product_result.products if p.name))
    else:
        # If no products were discovered, use the original query
        product_names = [state["query"]]

    # Bounded parallel fan-out
    semaphore = asyncio.Semaphore(settings.YOUTUBE_REVIEW_CONCURRENCY)
    review_results: dict[str, ReviewSummary] = {}

    async def _fetch_review(product_name: str) -> tuple[str, ReviewSummary | None, str | None]:
        async with semaphore:
            try:
                logger.info("Requesting YouTube review for: %s", product_name)
                if trace:
                    trace.add_event(
                        f"Calling YouTube Review agent for {product_name}",
                        status="active",
                        detail=route.agent_url,
                        agent="youtube-review",
                        data={"product_name": product_name},
                    )
                result = await client.send_message(route.agent_url, product_name)
                result = await _resolve_send_message_result(
                    result=result,
                    client=client,
                    agent_url=route.agent_url,
                    settings=settings,
                    task_observer=(
                        lambda task: _record_agent_task_snapshot(
                            trace,
                            agent_name="youtube-review",
                            task=task,
                            product_name=product_name,
                        )
                    ) if trace else None,
                )

                if isinstance(result, Task):
                    summary = _extract_review_summary(result)
                    if summary:
                        if trace:
                            trace.add_event(
                                f"YouTube review completed for {product_name}",
                                status="done",
                                detail=f"Captured {len(summary.sources)} ranked videos.",
                                agent="youtube-review",
                                data={"product_name": product_name},
                            )
                        return (product_name, summary, None)

                return (product_name, None, f"No parseable review data for {product_name}")
            except (A2AError, Exception) as exc:
                return (product_name, None, f"YouTube review failed for {product_name}: {exc}")

    tasks = [_fetch_review(name) for name in product_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            errors.append(f"YouTube review error: {item}")
            continue

        name, summary, error = item
        if summary:
            review_results[name] = summary
        if error:
            errors.append(error)
            if trace:
                trace.append_error(error)
                trace.add_event(
                    f"YouTube review issue for {name}",
                    status="error",
                    detail=error,
                    agent="youtube-review",
                    data={"product_name": name},
                )

    logger.info("YouTube reviews completed: %d/%d successful", len(review_results), len(product_names))

    return {"review_results": review_results, "errors": errors}


def _extract_review_summary(task: Task) -> ReviewSummary | None:
    """Extract ReviewSummary from a completed Task's artifacts."""
    if task.status.state != "completed":
        logger.warning("Review task not completed: state=%s", task.status.state)
        return None

    if not task.artifacts:
        return None

    data = _extract_named_artifact_json(task, "review-summary")
    if data is None:
        return None

    try:
        return ReviewSummary(**data)
    except Exception as exc:
        logger.debug("Failed to parse review summary artifact: %s", exc)
        return None


def _extract_review_trace(task: Task) -> dict[str, Any] | None:
    return _extract_named_artifact_json(task, "review-debug")


async def _resolve_send_message_result(
    result: Task | Message,
    client: A2AClient,
    agent_url: str,
    settings: Settings,
    task_observer: Callable[[Task], None] | None = None,
) -> Task | Message:
    """Poll non-terminal A2A tasks until they reach a terminal state."""
    if not isinstance(result, Task):
        return result

    task = result
    deadline = asyncio.get_running_loop().time() + settings.A2A_CLIENT_TIMEOUT_SECONDS
    last_signature = _task_signature(task)

    if task_observer:
        task_observer(task)

    while task.status.state not in _TERMINAL_TASK_STATES:
        now = asyncio.get_running_loop().time()
        if now >= deadline:
            raise TimeoutError(
                f"Task {task.id} did not reach a terminal state within "
                f"{settings.A2A_CLIENT_TIMEOUT_SECONDS} seconds"
            )

        await asyncio.sleep(min(_TASK_POLL_INTERVAL_SECONDS, deadline - now))
        task = await client.get_task(agent_url, task.id)
        signature = _task_signature(task)
        if task_observer and signature != last_signature:
            task_observer(task)
        last_signature = signature

    return task


def _extract_named_artifact_json(task: Task, artifact_name: str) -> dict[str, Any] | None:
    """Extract JSON content from a named task artifact."""
    if not task.artifacts:
        return None

    for artifact in task.artifacts:
        if artifact.name != artifact_name:
            continue
        for part in artifact.parts:
            try:
                return json.loads(part.text)
            except json.JSONDecodeError as exc:
                logger.debug("Failed to parse %s artifact as JSON: %s", artifact_name, exc)
                return None

    return None


def _task_status_text(task: Task) -> str | None:
    """Extract a flat status message from a task."""
    message = task.status.message
    if message is None:
        return None

    parts = [part.text.strip() for part in message.parts if part.type == "text" and part.text.strip()]
    if not parts:
        return None

    return " ".join(parts)


def _task_signature(task: Task) -> tuple[str, str, str]:
    """Build a compact signature so trace observers only react to real changes."""
    debug_payload = _extract_product_trace(task) or _extract_review_trace(task) or {}
    debug_json = json.dumps(debug_payload, sort_keys=True) if debug_payload else ""
    return (
        task.status.state,
        _task_status_text(task) or "",
        debug_json,
    )


def _map_task_state_to_event_status(task_state: str) -> str:
    if task_state == "completed":
        return "done"
    if task_state == "failed":
        return "error"
    if task_state in {"working", "submitted"}:
        return "active"
    return "info"


def _record_agent_task_snapshot(
    trace: TraceRecorder | None,
    *,
    agent_name: str,
    task: Task,
    product_name: str | None = None,
) -> None:
    """Record in-flight A2A task updates into the live trace store."""
    if trace is None:
        return

    message = _task_status_text(task)
    if message:
        title = "Product Discovery task update"
        if agent_name == "youtube-review":
            label = product_name or "review target"
            title = f"YouTube review update for {label}"

        trace.add_event(
            title,
            status=_map_task_state_to_event_status(task.status.state),
            detail=message,
            agent=agent_name,
            data={
                "task_id": task.id,
                "task_state": task.status.state,
                "product_name": product_name,
            },
        )

    if agent_name == "product-discovery":
        payload = _extract_product_trace(task)
        if payload:
            trace.set_product_trace(payload)
        return

    payload = _extract_review_trace(task)
    if payload:
        trace.set_review_trace(
            payload.get("product_name") or product_name or "unknown",
            payload,
        )


# ═══════════════════════════════════════════════════════════════════
# Node 4: Synthesize — GPT-4o-mini merges all agent data
# ═══════════════════════════════════════════════════════════════════

_SYNTHESIS_SYSTEM_PROMPT = """You are a shopping advisor synthesizer. You will receive product discovery data (prices, ratings, features) and YouTube review summaries (sentiment, pros, cons).

Your task:
1. Merge all evidence into a ranked list of product recommendations.
2. For each product, combine discovery data (price, rating, features) with review data (sentiment, pros, cons) where available.
3. Assign a composite score (0.0 - 1.0) considering: price-value ratio, ratings, review sentiment, and confidence.
4. Write a clear rationale for each product's ranking.
5. Order by score descending (best recommendation first).
6. If a product has no review data, note that in the rationale and lower the score slightly.
7. Keep pros and cons to a maximum of 3 each per product, prioritizing the most impactful ones.
8. NEVER invent data. If something is not in the evidence, do not include it."""

_SYNTHESIS_SCHEMA = {
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
                            "rank", "product_name", "price", "rating",
                            "sentiment", "score", "rationale", "pros", "cons", "confidence",
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


async def synthesize(state: OrchestratorState) -> dict[str, Any]:
    """Merge product discovery and review data into a UnifiedResponse.

    Produces structured UnifiedResponse first, then the API layer renders it.
    """
    query = state["query"]
    settings = state["settings"]
    product_result = state.get("product_result")
    review_results = state.get("review_results", {})
    decision = state.get("routing_decision")
    trace = state.get("trace_recorder")
    errors: list[str] = list(state.get("errors", []))

    # Determine if we have partial data
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
        # Nothing to synthesize
        unified = UnifiedResponse(
            query=query,
            recommendations=[],
            sources=[],
            partial=True,
            notes="No data was available from any agent. " + "; ".join(errors) if errors else "No data was available from any agent.",
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

    # If we only have products (no reviews), or if we have both, use LLM synthesis
    llm_client = state.get("llm_client") or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Build evidence for the LLM
    evidence_parts = []

    if has_products:
        evidence_parts.append("=== PRODUCT DISCOVERY DATA ===")
        for i, p in enumerate(product_result.products, 1):
            parts = [f"--- Product {i} ---"]
            parts.append(f"Name: {p.name}")
            parts.append(f"Price: {p.price or 'N/A'}")
            parts.append(f"Rating: {p.rating or 'N/A'}")
            parts.append(f"Source: {p.source}")
            parts.append(f"URL: {p.url}")
            parts.append(f"Features: {', '.join(p.key_features) if p.key_features else 'N/A'}")
            parts.append(f"Confidence: {p.confidence}")
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

    try:
        response = await llm_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_SYNTHESIS_SCHEMA,
            temperature=0.2,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Build recommendations
        recommendations = []
        for r in data.get("recommendations", []):
            recommendations.append(
                RankedRecommendation(
                    rank=r["rank"],
                    product_name=r["product_name"],
                    price=r.get("price"),
                    rating=r.get("rating"),
                    sentiment=r.get("sentiment"),
                    score=r.get("score", 0.0),
                    rationale=r["rationale"],
                    pros=r.get("pros", [])[:3],
                    cons=r.get("cons", [])[:3],
                    confidence=r.get("confidence", 0.0),
                )
            )

        # Build grounded source links
        sources = _build_source_links(product_result, review_results)

        partial = missing_requested_products or missing_requested_reviews
        notes = data.get("notes")
        if partial and not notes:
            missing = []
            if missing_requested_products:
                missing.append("product discovery")
            if missing_requested_reviews:
                missing.append("YouTube reviews")
            notes = f"Partial results — {', '.join(missing)} data was unavailable."

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

        # Fallback: build a basic response from raw product data
        unified = _fallback_synthesis(query, product_result, review_results, errors)
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


def _build_source_links(
    product_result: ProductDiscoveryResult | None,
    review_results: dict[str, ReviewSummary],
) -> list[UnifiedSourceLink]:
    """Build grounded source links from actual agent-retrieved data.

    Every URL in the output traces back to agent-retrieved data.
    No URLs are invented.
    """
    sources: list[UnifiedSourceLink] = []
    seen_urls: set[str] = set()

    # Product source links
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

    # Video source links
    for _product_name, review in review_results.items():
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


def _fallback_synthesis(
    query: str,
    product_result: ProductDiscoveryResult | None,
    review_results: dict[str, ReviewSummary],
    errors: list[str],
) -> UnifiedResponse:
    """Build a basic UnifiedResponse without LLM synthesis as fallback."""
    recommendations = []

    if product_result:
        for i, p in enumerate(product_result.products, 1):
            review = review_results.get(p.name)
            recommendations.append(
                RankedRecommendation(
                    rank=i,
                    product_name=p.name,
                    price=p.price,
                    rating=p.rating,
                    sentiment=review.overall_sentiment if review else None,
                    score=p.confidence,
                    rationale=f"Product from {p.source} with confidence {p.confidence:.2f}",
                    pros=review.pros[:3] if review else p.key_features[:3],
                    cons=review.cons[:3] if review else [],
                    confidence=p.confidence,
                )
            )
    elif review_results:
        for i, (product_name, review) in enumerate(review_results.items(), 1):
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

    sources = _build_source_links(product_result, review_results)

    return UnifiedResponse(
        query=query,
        recommendations=recommendations,
        sources=sources,
        partial=True,
        notes="Fallback synthesis (LLM unavailable). " + "; ".join(errors),
    )


# ═══════════════════════════════════════════════════════════════════
# Conditional Edge: Skip YouTube if no products found
# ═══════════════════════════════════════════════════════════════════

def _should_fetch_reviews(state: OrchestratorState) -> str:
    """Conditional edge: skip YouTube reviews if no products and no review route."""
    decision = state.get("routing_decision")

    if decision and decision.needs_youtube_reviews:
        return "execute_youtube_reviews"

    return "synthesize"


# ═══════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for the orchestrator.

    Graph structure:
        discover_agents
            ↓
        route_tasks
            ↓
        execute_product_discovery
            ↓
        [conditional] → execute_youtube_reviews → synthesize
                      → synthesize (skip reviews)
    """
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("discover_agents", discover_agents)
    graph.add_node("route_tasks", route_tasks)
    graph.add_node("execute_product_discovery", execute_product_discovery)
    graph.add_node("execute_youtube_reviews", execute_youtube_reviews)
    graph.add_node("synthesize", synthesize)

    # Define edges
    graph.set_entry_point("discover_agents")
    graph.add_edge("discover_agents", "route_tasks")
    graph.add_edge("route_tasks", "execute_product_discovery")

    # Conditional: after product discovery, either fetch reviews or go straight to synthesis
    graph.add_conditional_edges(
        "execute_product_discovery",
        _should_fetch_reviews,
        {
            "execute_youtube_reviews": "execute_youtube_reviews",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("execute_youtube_reviews", "synthesize")
    graph.add_edge("synthesize", END)

    return graph


# ═══════════════════════════════════════════════════════════════════
# Public API: run_query
# ═══════════════════════════════════════════════════════════════════

async def run_query(
    user_query: str,
    settings: Settings | None = None,
    a2a_client: A2AClient | None = None,
    llm_client: AsyncOpenAI | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> UnifiedResponse:
    """Execute the full orchestrator pipeline for a user query.

    This is the main entry point. Supports dependency injection for
    a2a_client and llm_client (Execution Contract rule 3).

    Args:
        user_query: Natural-language shopping query.
        settings: Application settings. If None, loads from env.
        a2a_client: Optional injected A2A client.
        llm_client: Optional injected OpenAI client.

    Returns:
        Structured UnifiedResponse with ranked recommendations and sources.
    """
    settings = settings or get_settings()
    owned_a2a_client = a2a_client is None
    owned_llm_client = llm_client is None
    runtime_a2a_client = a2a_client or A2AClient()
    runtime_llm_client = llm_client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        initial_state: OrchestratorState = {
            "query": user_query,
            "settings": settings,
            "a2a_client": runtime_a2a_client,
            "llm_client": runtime_llm_client,
            "trace_recorder": trace_recorder,
            "discovered_agents": {},
            "card_urls": {},
            "routing_decision": RoutingDecision(query=user_query),
            "product_result": None,
            "review_results": {},
            "unified_response": None,
            "errors": [],
        }

        graph = build_graph()
        compiled = graph.compile()

        final_state = await compiled.ainvoke(initial_state)

        response = final_state.get("unified_response")
        if response is None:
            response = UnifiedResponse(
                query=user_query,
                recommendations=[],
                sources=[],
                partial=True,
                notes="Orchestrator pipeline produced no response.",
            )

        if trace_recorder:
            trace_recorder.set_final_response(response)

        return response
    finally:
        if owned_a2a_client:
            await runtime_a2a_client.close()
        if owned_llm_client:
            await _close_async_resource(runtime_llm_client)


async def _close_async_resource(resource: Any) -> None:
    """Best-effort close for async client objects."""
    close_method = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close_method is None:
        return

    result = close_method()
    if asyncio.iscoroutine(result):
        await result
