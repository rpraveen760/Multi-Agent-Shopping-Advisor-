"""Central Orchestrator - LangGraph wiring plus thin node entrypoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from agents.orchestrator.discovery import agent_endpoints, discover_seeded_agents
from agents.orchestrator.product_runtime import (
    call_product_discovery_via_mcp,
    extract_product_result,
    resolve_product_mcp_url,
)
from agents.orchestrator.review_runtime import (
    extract_video_analysis,
    extract_review_summary,
    match_known_product_name,
    normalize_name_key,
    select_review_targets,
)
from agents.orchestrator.routing import AgentRoute, RoutingDecision, route_query
from agents.orchestrator.runtime import (
    extract_named_artifact_json,
    format_exception_detail,
    resolve_send_message_result,
    task_signature,
    task_status_text,
)
from agents.orchestrator.synthesis_runtime import (
    build_source_links,
    fallback_synthesis,
    synthesize_response,
)
from agents.orchestrator.video_runtime import build_video_analysis_payload, build_video_review_trace
from common.a2a_client import A2AClient, A2AError
from common.a2a_models import (
    AgentCard,
    Message,
    ProductDiscoveryResult,
    ReviewSummary,
    Task,
    UnifiedResponse,
    UnifiedVideoAnalysisResponse,
)
from common.config import Settings, get_settings
from common.runtime_helpers import close_async_resource

logger = logging.getLogger(__name__)

_MAX_REVIEW_PRODUCTS = 3
_MAX_FINAL_RECOMMENDATIONS = 3


class OrchestratorState(TypedDict, total=False):
    """Typed state flowing through the LangGraph nodes."""

    query: str
    youtube_url: str | None
    chat_message: str | None
    find_similar_products: bool
    settings: Settings
    a2a_client: A2AClient | None
    llm_client: AsyncOpenAI | None
    discovered_agents: dict[str, AgentCard]
    card_urls: dict[str, str]
    routing_decision: RoutingDecision
    product_result: ProductDiscoveryResult | None
    review_results: dict[str, ReviewSummary]
    video_analysis: UnifiedVideoAnalysisResponse | None
    unified_response: UnifiedResponse | None
    errors: list[str]


async def discover_agents(state: OrchestratorState) -> dict[str, Any]:
    """Fetch Agent Cards from configured discovery URLs."""
    settings = state["settings"]
    client = state.get("a2a_client") or A2AClient()
    discovered: dict[str, AgentCard] = {}
    card_urls: dict[str, str] = {}
    errors: list[str] = list(state.get("errors", []))

    raw_discovered, raw_card_urls, discovery_errors = await discover_seeded_agents(settings, client)
    discovered.update(raw_discovered)
    card_urls.update(raw_card_urls)
    errors.extend(discovery_errors)

    for name, card in discovered.items():
        logger.info("Discovered agent: %s at %s", card.name, card.url)

    for error in discovery_errors:
        logger.warning(error)

    return {
        "discovered_agents": discovered,
        "card_urls": card_urls,
        "a2a_client": client,
        "errors": errors,
    }


async def route_tasks(state: OrchestratorState) -> dict[str, Any]:
    """Use one LLM understanding step plus deterministic route assembly."""
    query = state["query"]
    discovered = state.get("discovered_agents", {})
    card_urls = state.get("card_urls", {})
    settings = state["settings"]
    llm_client = state.get("llm_client")
    youtube_url = state.get("youtube_url")

    if youtube_url:
        decision = RoutingDecision(
            query=query,
            routes=[],
            needs_product_discovery=False,
            needs_youtube_reviews=False,
            routing_reasoning=(
                "Primary PDF flow: the orchestrator delegates the YouTube review URL to the "
                "YouTube Product Review Agent, which may call Product Discovery via MCP for similar products."
            ),
        )
        youtube_card = discovered.get("youtube-review")
        if youtube_card:
            decision.routes.append(
                AgentRoute(
                    agent_name="youtube-review",
                    agent_url=youtube_card.url,
                    card_url=card_urls.get("youtube-review", ""),
                    skill_id="youtube-video-analysis",
                    query_text=youtube_url,
                )
            )
        return {"routing_decision": decision}

    decision = await route_query(
        query,
        discovered,
        card_urls,
        settings=settings,
        llm_client=llm_client,
    )
    return {"routing_decision": decision}


async def execute_product_discovery(state: OrchestratorState) -> dict[str, Any]:
    """Call Product Discovery via MCP, with A2A fallback for resilience."""
    decision = state.get("routing_decision")
    errors: list[str] = list(state.get("errors", []))

    if not decision or not decision.needs_product_discovery:
        return {"product_result": None, "errors": errors}

    client = state.get("a2a_client") or A2AClient()
    route = next((route for route in decision.routes if route.agent_name == "product-discovery"), None)
    if not route:
        return {"product_result": None, "errors": errors}

    try:
        agent_card = state.get("discovered_agents", {}).get("product-discovery")
        logger.info("Calling Product Discovery via MCP for query: %s", route.query_text)
        product_result = await _call_product_discovery_via_mcp(
            query=route.query_text,
            settings=state["settings"],
            agent_card=agent_card,
        )
        logger.info("Product Discovery returned %d products via MCP", len(product_result.products))
        return {"product_result": product_result, "errors": errors}
    except Exception as mcp_exc:
        fallback_msg = f"Product Discovery MCP call failed, falling back to A2A: {mcp_exc}"
        logger.warning(fallback_msg)
        errors.append(fallback_msg)

    try:
        logger.info("Falling back to Product Discovery agent over A2A: %s", route.agent_url)
        result = await client.send_message(route.agent_url, route.query_text)
        result = await _resolve_send_message_result(
            result=result,
            client=client,
            agent_url=route.agent_url,
            settings=state["settings"],
        )

        if isinstance(result, Task):
            product_result = _extract_product_result(result)
            if product_result:
                logger.info(
                    "Product Discovery returned %d products via A2A fallback",
                    len(product_result.products),
                )
                return {"product_result": product_result, "errors": errors}

        msg = "Product Discovery agent returned no parseable product data"
        logger.warning(msg)
        errors.append(msg)
        return {"product_result": None, "errors": errors}

    except (A2AError, Exception) as exc:
        msg = f"Product Discovery agent failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {"product_result": None, "errors": errors}


async def execute_youtube_video_analysis(state: OrchestratorState) -> dict[str, Any]:
    """Delegate a YouTube review URL to the YouTube agent and capture the unified video-analysis result."""
    youtube_url = state.get("youtube_url")
    errors: list[str] = list(state.get("errors", []))
    if not youtube_url:
        return {"video_analysis": None, "product_result": None, "errors": errors}

    decision = state.get("routing_decision")
    client = state.get("a2a_client") or A2AClient()
    route = next((route for route in (decision.routes if decision else []) if route.agent_name == "youtube-review"), None)
    if route is None:
        msg = "YouTube review agent was not available for the requested video-analysis flow."
        errors.append(msg)
        return {"video_analysis": None, "product_result": None, "errors": errors}

    product_card = state.get("discovered_agents", {}).get("product-discovery")
    try:
        product_mcp_url = _resolve_product_mcp_url(product_card) if product_card else None
    except Exception as exc:
        product_mcp_url = None
        errors.append(f"Unable to resolve Product Discovery MCP interface: {exc}")
    payload, metadata = build_video_analysis_payload(
        youtube_url=youtube_url,
        chat_message=state.get("chat_message"),
        find_similar_products=state.get("find_similar_products", False),
        product_mcp_url=product_mcp_url,
        legacy_query=state.get("query"),
    )

    try:
        result = await client.send_message(route.agent_url, payload, metadata=metadata)
        result = await _resolve_send_message_result(
            result=result,
            client=client,
            agent_url=route.agent_url,
            settings=state["settings"],
        )
        if isinstance(result, Task):
            video_analysis = _extract_video_analysis(result)
            if video_analysis:
                product_result = None
                if video_analysis.similar_products:
                    product_result = ProductDiscoveryResult(
                        query=video_analysis.extracted_product.product_name if video_analysis.extracted_product else youtube_url,
                        products=video_analysis.similar_products.products,
                        summary=video_analysis.similar_products.summary,
                    )
                return {
                    "video_analysis": video_analysis,
                    "product_result": product_result,
                    "review_results": {},
                    "errors": errors,
                }

            task_message = _task_status_text(result) or "Video analysis task completed without a parseable result."
            raise RuntimeError(task_message)

        raise RuntimeError("YouTube video analysis returned no task payload")
    except Exception as exc:
        detail = _format_exception_detail(exc)
        errors.append(detail)
        return {"video_analysis": None, "product_result": None, "review_results": {}, "errors": errors}


async def execute_youtube_reviews(state: OrchestratorState) -> dict[str, Any]:
    """Call the YouTube Review agent for each finalized product."""
    decision = state.get("routing_decision")
    product_result = state.get("product_result")
    errors: list[str] = list(state.get("errors", []))
    settings = state["settings"]

    if not decision or not decision.needs_youtube_reviews:
        return {"review_results": {}, "errors": errors}

    client = state.get("a2a_client") or A2AClient()
    route = next((route for route in decision.routes if route.agent_name == "youtube-review"), None)
    if not route:
        return {"review_results": {}, "errors": errors}

    product_names = _select_review_targets(product_result)
    if not product_names:
        return {"review_results": {}, "errors": errors}

    semaphore = asyncio.Semaphore(settings.YOUTUBE_REVIEW_CONCURRENCY)
    review_results: dict[str, ReviewSummary] = {}

    async def _fetch_review(product_name: str) -> tuple[str, ReviewSummary | None, str | None]:
        async with semaphore:
            try:
                logger.info("Requesting YouTube review for: %s", product_name)
                result = await client.send_message(route.agent_url, product_name)
                result = await _resolve_send_message_result(
                    result=result,
                    client=client,
                    agent_url=route.agent_url,
                    settings=settings,
                )

                if isinstance(result, Task):
                    summary = _extract_review_summary(result)
                    if summary:
                        return product_name, summary, None

                    task_message = _task_status_text(result)
                    if task_message:
                        return product_name, None, task_message
                    return (
                        product_name,
                        None,
                        f"YouTube review {result.status.state} for {product_name} without a structured summary.",
                    )

                return product_name, None, f"No parseable review data for {product_name}"
            except (A2AError, Exception) as exc:
                detail = _format_exception_detail(exc)
                return product_name, None, f"YouTube review failed for {product_name}: {detail}"

    results = await asyncio.gather(*[_fetch_review(name) for name in product_names], return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            errors.append(f"YouTube review error: {item}")
            continue

        name, summary, error = item
        if summary:
            review_results[name] = summary
        if error:
            errors.append(error)

    logger.info("YouTube reviews completed: %d/%d successful", len(review_results), len(product_names))
    return {"review_results": review_results, "errors": errors}


async def synthesize(state: OrchestratorState) -> dict[str, Any]:
    """Merge agent outputs into a single UnifiedResponse."""
    return await synthesize_response(
        query=state["query"],
        settings=state["settings"],
        product_result=state.get("product_result"),
        review_results=state.get("review_results", {}),
        video_analysis=state.get("video_analysis"),
        decision=state.get("routing_decision"),
        errors=list(state.get("errors", [])),
        llm_client=state.get("llm_client"),
        llm_client_factory=AsyncOpenAI,
        max_final_recommendations=_MAX_FINAL_RECOMMENDATIONS,
        match_known_product_name_fn=_match_known_product_name,
    )


async def _call_product_discovery_via_mcp(
    query: str,
    settings: Settings,
    agent_card: AgentCard | None = None,
    mcp_url: str | None = None,
) -> Any:
    return await call_product_discovery_via_mcp(
        query=query,
        settings=settings,
        agent_card=agent_card,
        mcp_url=mcp_url,
        max_results=_MAX_FINAL_RECOMMENDATIONS,
    )


def _resolve_product_mcp_url(agent_card: AgentCard | None) -> str:
    return resolve_product_mcp_url(agent_card)


def _extract_product_result(task: Task) -> ProductDiscoveryResult | None:
    return extract_product_result(task)


def _extract_review_summary(task: Task) -> ReviewSummary | None:
    return extract_review_summary(task)


def _extract_video_analysis(task: Task) -> UnifiedVideoAnalysisResponse | None:
    return extract_video_analysis(task)


def _build_video_review_trace(result: UnifiedVideoAnalysisResponse) -> dict[str, Any]:
    return build_video_review_trace(result)


def _normalize_name_key(value: str) -> list[str]:
    return normalize_name_key(value)


def _match_known_product_name(candidate_name: str, known_names: list[str]) -> str | None:
    return match_known_product_name(candidate_name, known_names)


def _select_review_targets(product_result: ProductDiscoveryResult | None) -> list[str]:
    return select_review_targets(product_result, max_review_products=_MAX_REVIEW_PRODUCTS)


async def _resolve_send_message_result(
    result: Task | Message,
    client: A2AClient,
    agent_url: str,
    settings: Settings,
) -> Task | Message:
    return await resolve_send_message_result(
        result,
        client=client,
        agent_url=agent_url,
        timeout_seconds=settings.A2A_CLIENT_TIMEOUT_SECONDS,
        sleep_fn=asyncio.sleep,
    )


def _extract_named_artifact_json(task: Task, artifact_name: str) -> dict[str, Any] | None:
    return extract_named_artifact_json(task, artifact_name)


def _task_status_text(task: Task) -> str | None:
    return task_status_text(task)


def _format_exception_detail(exc: Exception) -> str:
    return format_exception_detail(exc)


def _task_signature(task: Task) -> tuple[str, str, str]:
    return task_signature(task)


def _build_source_links(product_result: ProductDiscoveryResult | None, review_results: dict[str, ReviewSummary]) -> list[Any]:
    return build_source_links(product_result, review_results)


def _fallback_synthesis(
    query: str,
    product_result: ProductDiscoveryResult | None,
    review_results: dict[str, ReviewSummary],
    errors: list[str],
) -> UnifiedResponse:
    return fallback_synthesis(
        query=query,
        product_result=product_result,
        review_results=review_results,
        errors=errors,
        max_final_recommendations=_MAX_FINAL_RECOMMENDATIONS,
    )


def _should_fetch_reviews(state: OrchestratorState) -> str:
    decision = state.get("routing_decision")
    product_result = state.get("product_result")
    if decision and decision.needs_youtube_reviews and product_result and product_result.products:
        return "execute_youtube_reviews"
    return "synthesize"


def _select_primary_execution(state: OrchestratorState) -> str:
    if state.get("youtube_url"):
        return "execute_youtube_video_analysis"
    return "execute_product_discovery"


def build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)
    graph.add_node("discover_agents", discover_agents)
    graph.add_node("route_tasks", route_tasks)
    graph.add_node("execute_youtube_video_analysis", execute_youtube_video_analysis)
    graph.add_node("execute_product_discovery", execute_product_discovery)
    graph.add_node("execute_youtube_reviews", execute_youtube_reviews)
    graph.add_node("synthesize", synthesize)
    graph.set_entry_point("discover_agents")
    graph.add_edge("discover_agents", "route_tasks")
    graph.add_conditional_edges(
        "route_tasks",
        _select_primary_execution,
        {
            "execute_youtube_video_analysis": "execute_youtube_video_analysis",
            "execute_product_discovery": "execute_product_discovery",
        },
    )
    graph.add_edge("execute_youtube_video_analysis", "synthesize")
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


async def run_query(
    user_query: str,
    settings: Settings | None = None,
    a2a_client: A2AClient | None = None,
    llm_client: AsyncOpenAI | None = None,
    youtube_url: str | None = None,
    chat_message: str | None = None,
    find_similar_products: bool = False,
) -> UnifiedResponse:
    """Execute the full orchestrator pipeline for a user query."""
    settings = settings or get_settings()
    owned_a2a_client = a2a_client is None
    owned_llm_client = llm_client is None
    runtime_a2a_client = a2a_client or A2AClient()
    runtime_llm_client = llm_client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        initial_state: OrchestratorState = {
            "query": user_query,
            "youtube_url": youtube_url,
            "chat_message": chat_message,
            "find_similar_products": find_similar_products,
            "settings": settings,
            "a2a_client": runtime_a2a_client,
            "llm_client": runtime_llm_client,
            "discovered_agents": {},
            "card_urls": {},
            "routing_decision": RoutingDecision(query=user_query),
            "product_result": None,
            "review_results": {},
            "video_analysis": None,
            "unified_response": None,
            "errors": [],
        }

        compiled = build_graph().compile()
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
        return response
    finally:
        if owned_a2a_client:
            await runtime_a2a_client.close()
        if owned_llm_client:
            await _close_async_resource(runtime_llm_client)


async def _close_async_resource(resource: Any) -> None:
    await close_async_resource(resource)
