"""LLM-powered query understanding and routing for the Central Orchestrator.

Two-phase approach:
1. Query Understanding — GPT-4o-mini interprets the user query, extracts the
   product category, budget constraints, and key intent signals.
2. Agent Routing — GPT-4o-mini decides which downstream agents should be
   invoked based on the understood intent and the set of discovered agents.

Both phases use structured output (JSON schema) so the orchestrator receives
machine-readable decisions it can act on deterministically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from common.a2a_models import AgentCard
from common.config import Settings

logger = logging.getLogger(__name__)


# ── Route Definitions ────────────────────────────────────────────

@dataclass
class AgentRoute:
    """A resolved routing decision for a single downstream agent."""
    agent_name: str
    agent_url: str       # Full JSON-RPC endpoint from AgentCard.url
    card_url: str        # Discovery URL used to fetch the card
    skill_id: str
    query_text: str      # The text to send to this agent


@dataclass
class QueryUnderstanding:
    """Structured interpretation of the user query produced by the LLM."""
    original_query: str
    reformulated_query: str
    product_category: str | None = None
    budget: str | None = None
    intent: str = "product_search"  # product_search | review_lookup | comparison | general
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


# ── JSON Schemas for Structured Output ───────────────────────────

_QUERY_UNDERSTANDING_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "QueryUnderstanding",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reformulated_query": {
                    "type": "string",
                    "description": "A clearer, more specific version of the query optimized for product search.",
                },
                "product_category": {
                    "type": ["string", "null"],
                    "description": "The product category (e.g., 'GPU', 'monitor', 'mouse', 'CPU', 'headphones'). Null if unclear.",
                },
                "budget": {
                    "type": ["string", "null"],
                    "description": "Any budget constraint mentioned (e.g., 'under $100', 'around Rs 5000'). Null if none.",
                },
                "intent": {
                    "type": "string",
                    "enum": ["product_search", "review_lookup", "comparison", "general"],
                    "description": "The primary intent of the query.",
                },
                "key_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Important product names, brands, or model numbers extracted from the query.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of how you interpreted the query.",
                },
            },
            "required": [
                "reformulated_query", "product_category", "budget",
                "intent", "key_terms", "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}

_ROUTING_DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "RoutingDecision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "needs_product_discovery": {
                    "type": "boolean",
                    "description": "Whether to invoke the Product Discovery agent to find and compare products.",
                },
                "needs_youtube_reviews": {
                    "type": "boolean",
                    "description": "Whether to invoke the YouTube Review agent for video-based review evidence.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why these agents were selected.",
                },
            },
            "required": ["needs_product_discovery", "needs_youtube_reviews", "reasoning"],
            "additionalProperties": False,
        },
    },
}

_QUERY_UNDERSTANDING_SYSTEM = """You are a shopping query analyst. Given a user's natural-language shopping query, produce a structured interpretation.

Your task:
1. Reformulate the query to be clearer and more specific for a product search engine.
1a. For broad category searches, reformulate toward concrete, reviewable products rather than generic category labels.
1b. Include useful shopping synonyms when they improve retrieval (for example, "drawing tablet with screen" -> "pen display drawing tablet").
2. Extract the product category, budget constraints, and key product terms.
3. Classify the user's primary intent.
4. Be concise in your reasoning.

Examples:
- "RTX 5070" → reformulated: "NVIDIA GeForce RTX 5070 graphics card", category: "GPU", intent: "product_search", key_terms: ["RTX 5070", "NVIDIA"]
- "best gaming mouse under 2000" → reformulated: "best gaming mouse under Rs 2000", category: "mouse", budget: "under Rs 2000", intent: "product_search"
- "best drawing tablet with screen for beginners" → reformulated: "best beginner pen display drawing tablet with screen", category: "drawing tablet", intent: "product_search", key_terms: ["drawing tablet", "pen display"]
- "is the Razer Cobra worth it" → reformulated: "Razer Cobra gaming mouse review and value assessment", category: "mouse", intent: "review_lookup", key_terms: ["Razer Cobra"]"""

_ROUTING_SYSTEM = """You are an agent routing coordinator for a shopping advisor system. Based on the user's query understanding and the available agents, decide which agents should be invoked.

Available agents and their capabilities:
{agent_descriptions}

Routing guidelines:
- Product Discovery: Invoke when the user wants to find, compare, or learn about products. This covers most shopping queries.
- YouTube Reviews: Invoke when review evidence, real-world opinions, pros/cons, or video demonstrations would be valuable. This is useful for most product queries where sentiment matters.
- For simple product lookups (e.g. just a model number), product discovery alone may suffice.
- For review-heavy queries ("is X worth it", "X vs Y", "pros and cons"), both agents are valuable.
- When in doubt, invoke both agents — more evidence leads to better recommendations.

Be concise in your reasoning."""


async def understand_query(
    query: str,
    settings: Settings,
    llm_client: AsyncOpenAI | None = None,
) -> QueryUnderstanding:
    """Use GPT-4o-mini to interpret and structure the user query."""
    client = llm_client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    owned_client = llm_client is None

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
            await client.close()


def _build_agent_descriptions(discovered_agents: dict[str, AgentCard]) -> str:
    """Format discovered agent capabilities for the routing prompt."""
    lines = []
    for name, card in discovered_agents.items():
        skills = ", ".join(s.name for s in card.skills) if card.skills else "no declared skills"
        lines.append(f"- {name}: {card.description} (skills: {skills})")
    return "\n".join(lines) if lines else "No agents are currently available."


async def route_query(
    query: str,
    discovered_agents: dict[str, AgentCard],
    card_urls: dict[str, str],
    settings: Settings | None = None,
    llm_client: AsyncOpenAI | None = None,
    query_understanding: QueryUnderstanding | None = None,
) -> RoutingDecision:
    """Use GPT-4o-mini to decide which agents should handle a user query.

    The LLM receives the query understanding and the list of available agents,
    then returns a structured routing decision.

    Falls back to routing both agents if the LLM call fails.

    Args:
        query: The user's natural-language shopping query.
        discovered_agents: Map of agent name -> AgentCard.
        card_urls: Map of agent name -> card discovery URL.
        settings: Application settings (needed for LLM call).
        llm_client: Optional injected OpenAI client.
        query_understanding: Pre-computed query understanding (if available).

    Returns:
        A RoutingDecision with resolved routes and reasoning.
    """
    from common.config import get_settings
    settings = settings or get_settings()

    has_product_agent = "product-discovery" in discovered_agents
    has_youtube_agent = "youtube-review" in discovered_agents

    # Phase 1: Query understanding (if not already provided)
    if query_understanding is None:
        query_understanding = await understand_query(query, settings, llm_client)

    # Phase 2: LLM-based routing
    needs_products = True
    needs_reviews = True
    routing_reasoning = ""

    try:
        client = llm_client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        owned_client = llm_client is None

        agent_descriptions = _build_agent_descriptions(discovered_agents)
        routing_system = _ROUTING_SYSTEM.format(agent_descriptions=agent_descriptions)

        routing_prompt = (
            f"User query: \"{query}\"\n\n"
            f"Query understanding:\n"
            f"- Reformulated: {query_understanding.reformulated_query}\n"
            f"- Category: {query_understanding.product_category or 'unknown'}\n"
            f"- Budget: {query_understanding.budget or 'none specified'}\n"
            f"- Intent: {query_understanding.intent}\n"
            f"- Key terms: {', '.join(query_understanding.key_terms) if query_understanding.key_terms else 'none'}\n\n"
            f"Which agents should be invoked?"
        )

        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": routing_system},
                    {"role": "user", "content": routing_prompt},
                ],
                response_format=_ROUTING_DECISION_SCHEMA,
                temperature=0.1,
            )

            data = json.loads(response.choices[0].message.content)
            needs_products = data.get("needs_product_discovery", True)
            needs_reviews = data.get("needs_youtube_reviews", True)
            routing_reasoning = data.get("reasoning", "")
        finally:
            if owned_client:
                await client.close()

    except Exception as exc:
        logger.warning("LLM routing failed, defaulting to both agents: %s", exc)
        needs_products = True
        needs_reviews = True
        routing_reasoning = f"Fallback: LLM routing failed ({exc}), invoking all available agents."

    # Keep review evidence enabled for concrete product lookups/comparisons even if
    # the routing model tries to optimize them down to product discovery only.
    if (
        has_youtube_agent
        and needs_products
        and not needs_reviews
        and query_understanding.intent in {"product_search", "comparison", "review_lookup"}
    ):
        needs_reviews = True
        if routing_reasoning:
            routing_reasoning += " Review evidence was kept on because concrete product lookups benefit from downstream review context."
        else:
            routing_reasoning = "Review evidence was kept on because concrete product lookups benefit from downstream review context."

    # Build the decision
    # Use the reformulated query for downstream agents
    effective_query = query_understanding.reformulated_query or query

    decision = RoutingDecision(
        query=query,
        query_understanding=query_understanding,
        routing_reasoning=routing_reasoning,
    )

    decision.needs_product_discovery = needs_products and has_product_agent
    decision.needs_youtube_reviews = needs_reviews and has_youtube_agent

    # ── Resolve routes ────────────────────────────────────────────

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
        "LLM routing for '%s': %d routes [products=%s, reviews=%s] — %s",
        query,
        len(decision.routes),
        decision.needs_product_discovery,
        decision.needs_youtube_reviews,
        routing_reasoning[:120],
    )

    return decision
