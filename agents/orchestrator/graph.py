"""Central Orchestrator - LangGraph wiring plus thin node entrypoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from agents.orchestrator.discovery import agent_endpoints, discover_seeded_agents
from agents.orchestrator.product_runtime import (
    build_mcp_product_trace,
    build_product_debug_url,
    call_product_discovery_via_mcp,
    clear_product_discovery_debug,
    extract_product_result,
    extract_product_trace,
    poll_product_discovery_debug,
    resolve_product_mcp_url,
)
from agents.orchestrator.review_runtime import (
    extract_review_summary,
    extract_review_trace,
    match_known_product_name,
    normalize_name_key,
    select_review_targets,
)
from agents.orchestrator.routing import RoutingDecision, route_query
from agents.orchestrator.runtime import (
    extract_named_artifact_json,
    format_exception_detail,
    map_task_state_to_event_status,
    record_agent_task_snapshot,
    resolve_send_message_result,
    task_signature,
    task_status_text,
)
from agents.orchestrator.synthesis_runtime import (
    build_source_links,
    fallback_synthesis,
    synthesize_response,
)
from agents.orchestrator.trace import TraceRecorder
from common.a2a_client import A2AClient, A2AError
from common.a2a_models import AgentCard, Message, ProductDiscoveryResult, ReviewSummary, Task, UnifiedResponse
from common.config import Settings, get_settings
from common.runtime_helpers import close_async_resource

logger = logging.getLogger(__name__)

_MAX_REVIEW_PRODUCTS = 3
_MAX_FINAL_RECOMMENDATIONS = 3


class OrchestratorState(TypedDict, total=False):
    """Typed state flowing through the LangGraph nodes."""

    query: str
    settings: Settings
    a2a_client: A2AClient | None
    llm_client: AsyncOpenAI | None
    trace_recorder: TraceRecorder | None
    discovered_agents: dict[str, AgentCard]
    card_urls: dict[str, str]
    routing_decision: RoutingDecision
    product_result: ProductDiscoveryResult | None
    review_results: dict[str, ReviewSummary]
    unified_response: UnifiedResponse | None
    errors: list[str]


async def discover_agents(state: OrchestratorState) -> dict[str, Any]:
    """Fetch Agent Cards from configured discovery URLs."""
    settings = state["settings"]
    client = state.get("a2a_client") or A2AClient()
    trace = state.get("trace_recorder")
    discovered: dict[str, AgentCard] = {}
    card_urls: dict[str, str] = {}
    errors: list[str] = list(state.get("errors", []))

    if trace:
        trace.add_event(
            "Discovering agents",
            status="active",
            detail="Fetching seeded Agent Cards for downstream services.",
        )

    configured_endpoints = agent_endpoints(settings)
    raw_discovered, raw_card_urls, discovery_errors = await discover_seeded_agents(settings, client)
    discovered.update(raw_discovered)
    card_urls.update(raw_card_urls)
    errors.extend(discovery_errors)

    for name, card in discovered.items():
        logger.info("Discovered agent: %s at %s", card.name, card.url)
        if trace:
            trace.add_event(
                f"Discovered {card.name}",
                status="done",
                detail=card.url,
                agent=name,
            )

    for error in discovery_errors:
        logger.warning(error)
        if trace:
            trace.append_error(error)
            for name, card_url in configured_endpoints.items():
                if name in discovered:
                    continue
                if card_url in error:
                    trace.add_event(
                        f"Failed to discover {name}",
                        status="error",
                        detail=error,
                        agent=name,
                    )
                    break

    if trace:
        trace.set_discovered_agents(discovered, card_urls)

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
    trace = state.get("trace_recorder")

    if trace:
        trace.add_event(
            "Understanding query via LLM",
            status="active",
            detail="GPT-4o-mini is interpreting the query for deterministic orchestration.",
        )

    decision = await route_query(
        query,
        discovered,
        card_urls,
        settings=settings,
        llm_client=llm_client,
    )

    if trace:
        trace.set_routing(decision)
        understanding = decision.query_understanding
        trace.add_event(
            "Query understood",
            status="done",
            detail=understanding.reasoning if understanding else "No query understanding available.",
            data={
                "reformulated_query": understanding.reformulated_query if understanding else query,
                "product_category": understanding.product_category if understanding else None,
                "budget": understanding.budget if understanding else None,
                "intent": understanding.intent if understanding else "unknown",
                "key_terms": understanding.key_terms if understanding else [],
            },
        )
        trace.add_event(
            "Routing complete",
            status="done",
            detail=f"Resolved {len(decision.routes)} downstream routes. {decision.routing_reasoning}",
            data={
                "needs_product_discovery": decision.needs_product_discovery,
                "needs_youtube_reviews": decision.needs_youtube_reviews,
                "routing_reasoning": decision.routing_reasoning,
            },
        )

    return {"routing_decision": decision}


async def execute_product_discovery(state: OrchestratorState) -> dict[str, Any]:
    """Call Product Discovery via MCP, with A2A fallback for resilience."""
    decision = state.get("routing_decision")
    trace = state.get("trace_recorder")
    errors: list[str] = list(state.get("errors", []))

    if not decision or not decision.needs_product_discovery:
        return {"product_result": None, "errors": errors}

    client = state.get("a2a_client") or A2AClient()
    route = next((route for route in decision.routes if route.agent_name == "product-discovery"), None)
    if not route:
        return {"product_result": None, "errors": errors}

    latest_trace = None
    try:
        agent_card = state.get("discovered_agents", {}).get("product-discovery")
        mcp_url = _resolve_product_mcp_url(agent_card)
        debug_trace_id = None
        debug_url = None
        poll_stop = None
        poll_task = None

        logger.info("Calling Product Discovery via MCP for query: %s", route.query_text)
        if trace:
            debug_trace_id = f"{trace.run_id}-product-discovery"
            debug_url = _build_product_debug_url(mcp_url, debug_trace_id)
            poll_stop = asyncio.Event()
            poll_task = asyncio.create_task(_poll_product_discovery_debug(debug_url, trace, poll_stop))
            trace.add_event(
                "Calling Product Discovery via MCP",
                status="active",
                detail="Invoking search_products on the MCP interface.",
                agent="product-discovery",
                data={"query_text": route.query_text},
            )

        try:
            product_result = await _call_product_discovery_via_mcp(
                query=route.query_text,
                settings=state["settings"],
                agent_card=agent_card,
                trace=trace,
                mcp_url=mcp_url,
                trace_id=debug_trace_id,
            )
        finally:
            if poll_task and poll_stop and debug_url:
                poll_stop.set()
                try:
                    latest_trace = await poll_task
                except Exception as exc:
                    logger.debug("Failed to poll Product Discovery debug progress: %s", exc)
                await _clear_product_discovery_debug(debug_url)

        logger.info("Product Discovery returned %d products via MCP", len(product_result.products))
        if trace:
            trace.set_product_trace(latest_trace or _build_mcp_product_trace(route.query_text, product_result))
            trace.add_event(
                "Product discovery completed",
                status="done",
                detail=f"Returned {len(product_result.products)} normalized products via MCP.",
                agent="product-discovery",
            )
        return {"product_result": product_result, "errors": errors}

    except Exception as mcp_exc:
        fallback_msg = f"Product Discovery MCP call failed, falling back to A2A: {mcp_exc}"
        logger.warning(fallback_msg)
        errors.append(fallback_msg)
        if trace:
            trace.append_error(fallback_msg)
            trace.add_event(
                "Product Discovery MCP call failed",
                status="error",
                detail=str(mcp_exc),
                agent="product-discovery",
            )

    try:
        logger.info("Falling back to Product Discovery agent over A2A: %s", route.agent_url)
        if trace:
            trace.add_event(
                "Falling back to Product Discovery A2A wrapper",
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
            )
            if trace
            else None,
        )

        if isinstance(result, Task):
            product_result = _extract_product_result(result)
            if product_result:
                logger.info(
                    "Product Discovery returned %d products via A2A fallback",
                    len(product_result.products),
                )
                if trace:
                    trace.add_event(
                        "Product discovery completed via A2A fallback",
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


async def execute_youtube_reviews(state: OrchestratorState) -> dict[str, Any]:
    """Call the YouTube Review agent for each finalized product."""
    decision = state.get("routing_decision")
    product_result = state.get("product_result")
    trace = state.get("trace_recorder")
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
        if trace:
            trace.merge_product_trace({"review_targets": []})
            trace.add_event(
                "Skipping YouTube review fan-out",
                status="done",
                detail="Product Discovery did not finalize any concrete products to review.",
                agent="youtube-review",
            )
        return {"review_results": {}, "errors": errors}

    if trace and product_result and product_result.products:
        trace.merge_product_trace({"review_targets": product_names})
        trace.add_event(
            "Selected review targets",
            status="done",
            detail=f"Sending {len(product_names)} shortlisted products to YouTube review analysis.",
            agent="youtube-review",
            data={"product_names": product_names},
        )

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
                    )
                    if trace
                    else None,
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


async def synthesize(state: OrchestratorState) -> dict[str, Any]:
    """Merge agent outputs into a single UnifiedResponse."""
    return await synthesize_response(
        query=state["query"],
        settings=state["settings"],
        product_result=state.get("product_result"),
        review_results=state.get("review_results", {}),
        decision=state.get("routing_decision"),
        trace=state.get("trace_recorder"),
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
    trace: TraceRecorder | None = None,
    mcp_url: str | None = None,
    trace_id: str | None = None,
) -> Any:
    return await call_product_discovery_via_mcp(
        query=query,
        settings=settings,
        agent_card=agent_card,
        trace=trace,
        mcp_url=mcp_url,
        trace_id=trace_id,
        max_results=_MAX_FINAL_RECOMMENDATIONS,
    )


def _resolve_product_mcp_url(agent_card: AgentCard | None) -> str:
    return resolve_product_mcp_url(agent_card)


def _build_product_debug_url(mcp_url: str, trace_id: str) -> str:
    return build_product_debug_url(mcp_url, trace_id)


async def _poll_product_discovery_debug(debug_url: str, trace: TraceRecorder, stop_event: asyncio.Event) -> dict[str, Any] | None:
    return await poll_product_discovery_debug(debug_url, trace, stop_event)


async def _clear_product_discovery_debug(debug_url: str) -> None:
    await clear_product_discovery_debug(debug_url)


def _build_mcp_product_trace(query: str, product_result: ProductDiscoveryResult) -> dict[str, Any]:
    return build_mcp_product_trace(query, product_result)


def _extract_product_result(task: Task) -> ProductDiscoveryResult | None:
    return extract_product_result(task)


def _extract_product_trace(task: Task) -> dict[str, Any] | None:
    return extract_product_trace(task)


def _extract_review_summary(task: Task) -> ReviewSummary | None:
    return extract_review_summary(task)


def _extract_review_trace(task: Task) -> dict[str, Any] | None:
    return extract_review_trace(task)


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
    task_observer: Callable[[Task], None] | None = None,
) -> Task | Message:
    return await resolve_send_message_result(
        result,
        client=client,
        agent_url=agent_url,
        timeout_seconds=settings.A2A_CLIENT_TIMEOUT_SECONDS,
        task_observer=task_observer,
        sleep_fn=asyncio.sleep,
    )


def _extract_named_artifact_json(task: Task, artifact_name: str) -> dict[str, Any] | None:
    return extract_named_artifact_json(task, artifact_name)


def _task_status_text(task: Task) -> str | None:
    return task_status_text(task)


def _format_exception_detail(exc: Exception) -> str:
    return format_exception_detail(exc)


def _task_signature(task: Task) -> tuple[str, str, str]:
    return task_signature(
        task,
        product_trace_extractor=_extract_product_trace,
        review_trace_extractor=_extract_review_trace,
    )


def _map_task_state_to_event_status(task_state: str) -> str:
    return map_task_state_to_event_status(task_state)


def _record_agent_task_snapshot(
    trace: TraceRecorder | None,
    *,
    agent_name: str,
    task: Task,
    product_name: str | None = None,
) -> None:
    record_agent_task_snapshot(
        trace,
        agent_name=agent_name,
        task=task,
        product_name=product_name,
        product_trace_extractor=_extract_product_trace,
        review_trace_extractor=_extract_review_trace,
    )


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


def build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)
    graph.add_node("discover_agents", discover_agents)
    graph.add_node("route_tasks", route_tasks)
    graph.add_node("execute_product_discovery", execute_product_discovery)
    graph.add_node("execute_youtube_reviews", execute_youtube_reviews)
    graph.add_node("synthesize", synthesize)
    graph.set_entry_point("discover_agents")
    graph.add_edge("discover_agents", "route_tasks")
    graph.add_edge("route_tasks", "execute_product_discovery")
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
    trace_recorder: TraceRecorder | None = None,
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

        if trace_recorder:
            trace_recorder.set_final_response(response)
        return response
    finally:
        if owned_a2a_client:
            await runtime_a2a_client.close()
        if owned_llm_client:
            await _close_async_resource(runtime_llm_client)


async def _close_async_resource(resource: Any) -> None:
    await close_async_resource(resource)
