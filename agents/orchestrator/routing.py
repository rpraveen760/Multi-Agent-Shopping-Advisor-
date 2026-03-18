"""LLM-assisted query understanding plus deterministic orchestrator routing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from common.a2a_models import AgentCard
from common.config import Settings
from common.runtime_helpers import close_async_resource

logger = logging.getLogger(__name__)


@dataclass
class AgentRoute:
    """A resolved routing decision for a single downstream agent."""

    agent_name: str
    agent_url: str
    card_url: str
    skill_id: str
    query_text: str


@dataclass
class QueryUnderstanding:
    """Structured interpretation of the user query produced by the LLM."""

    original_query: str
    reformulated_query: str
    product_category: str | None = None
    budget: str | None = None
    intent: str = "product_search"
    key_terms: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class RoutingDecision:
    """Complete routing decision for a user query."""

    query: str
    routes: list[AgentRoute] = field(default_factory=list)
    needs_product_discovery: bool = False
    needs_youtube_reviews: bool = False
    query_understanding: QueryUnderstanding | None = None
    routing_reasoning: str = ""


_QUERY_UNDERSTANDING_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "QueryUnderstanding",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reformulated_query": {"type": "string"},
                "product_category": {"type": ["string", "null"]},
                "budget": {"type": ["string", "null"]},
                "intent": {
                    "type": "string",
                    "enum": ["product_search", "review_lookup", "comparison", "general"],
                },
                "key_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reasoning": {"type": "string"},
            },
            "required": [
                "reformulated_query",
                "product_category",
                "budget",
                "intent",
                "key_terms",
                "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}

_QUERY_UNDERSTANDING_SYSTEM = """You are a shopping query analyst. Given a user's natural-language shopping query, produce a structured interpretation.

Your task:
1. Reformulate the query to be clearer and more specific for a product search engine.
1a. For broad category searches, reformulate toward concrete, reviewable products rather than generic category labels.
1b. Include useful shopping synonyms when they improve retrieval.
2. Extract the product category, budget constraints, and key product terms.
3. Classify the user's primary intent.
4. Be concise in your reasoning."""


async def understand_query(
    query: str,
    settings: Settings,
    llm_client: AsyncOpenAI | None = None,
) -> QueryUnderstanding:
    """Use GPT-4o-mini to interpret and structure the user query."""
    owned_client = llm_client is None
    client = llm_client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _QUERY_UNDERSTANDING_SYSTEM},
                {"role": "user", "content": query},
            ],
            response_format=_QUERY_UNDERSTANDING_SCHEMA,
            temperature=0.1,
        )

        data = json.loads(response.choices[0].message.content)
        return QueryUnderstanding(
            original_query=query,
            reformulated_query=data["reformulated_query"],
            product_category=data.get("product_category"),
            budget=data.get("budget"),
            intent=data.get("intent", "product_search"),
            key_terms=data.get("key_terms", []),
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning("LLM query understanding failed, using raw query: %s", exc)
        return QueryUnderstanding(
            original_query=query,
            reformulated_query=query,
            intent="product_search",
            reasoning=f"Fallback: LLM query understanding failed ({exc})",
        )
    finally:
        if owned_client:
            await close_async_resource(client)


def _build_routing_reasoning(
    *,
    query_understanding: QueryUnderstanding,
    has_product_agent: bool,
    has_youtube_agent: bool,
) -> str:
    reasons: list[str] = []
    if query_understanding.reasoning:
        reasons.append(query_understanding.reasoning)

    if has_product_agent:
        reasons.append(
            "Product Discovery is always used when available so the orchestrator can finalize a concrete shortlist first."
        )
    else:
        reasons.append("Product Discovery was unavailable, so no product shortlist route could be created.")

    if has_product_agent and has_youtube_agent:
        reasons.append(
            "YouTube reviews stay eligible whenever the review agent is available, but review fan-out only happens after Product Discovery finalizes concrete products."
        )
    elif not has_youtube_agent:
        reasons.append("YouTube review routing was skipped because the review agent was unavailable.")

    return " ".join(reasons)


async def route_query(
    query: str,
    discovered_agents: dict[str, AgentCard],
    card_urls: dict[str, str],
    settings: Settings | None = None,
    llm_client: AsyncOpenAI | None = None,
    query_understanding: QueryUnderstanding | None = None,
) -> RoutingDecision:
    """Understand the query once, then deterministically assemble routes."""
    from common.config import get_settings

    settings = settings or get_settings()
    if query_understanding is None:
        query_understanding = await understand_query(query, settings, llm_client)

    has_product_agent = "product-discovery" in discovered_agents
    has_youtube_agent = "youtube-review" in discovered_agents

    decision = RoutingDecision(
        query=query,
        query_understanding=query_understanding,
        needs_product_discovery=has_product_agent,
        needs_youtube_reviews=has_product_agent and has_youtube_agent,
        routing_reasoning=_build_routing_reasoning(
            query_understanding=query_understanding,
            has_product_agent=has_product_agent,
            has_youtube_agent=has_youtube_agent,
        ),
    )

    effective_query = query_understanding.reformulated_query or query

    if decision.needs_product_discovery:
        card = discovered_agents["product-discovery"]
        decision.routes.append(
            AgentRoute(
                agent_name="product-discovery",
                agent_url=card.url,
                card_url=card_urls.get("product-discovery", ""),
                skill_id="product-search",
                query_text=effective_query,
            )
        )

    if decision.needs_youtube_reviews:
        card = discovered_agents["youtube-review"]
        decision.routes.append(
            AgentRoute(
                agent_name="youtube-review",
                agent_url=card.url,
                card_url=card_urls.get("youtube-review", ""),
                skill_id="youtube-product-review",
                query_text=effective_query,
            )
        )

    logger.info(
        "Deterministic routing for '%s': %d routes [products=%s, reviews=%s]",
        query,
        len(decision.routes),
        decision.needs_product_discovery,
        decision.needs_youtube_reviews,
    )
    return decision
