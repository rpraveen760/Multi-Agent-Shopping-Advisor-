"""Deterministic capability-based routing for the Central Orchestrator.

Maps a user query to downstream agent capabilities based on keyword
analysis and Agent Card skill metadata. No LLM is used for routing —
this is a pure rule-based decision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from common.a2a_models import AgentCard

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
class RoutingDecision:
    """Complete routing decision for a user query."""
    query: str
    routes: list[AgentRoute] = field(default_factory=list)
    needs_product_discovery: bool = False
    needs_youtube_reviews: bool = False


# ── Capability Matchers ──────────────────────────────────────────

# Product discovery is triggered for any shopping / product query.
# This is the default route — most queries will trigger it.
_PRODUCT_KEYWORDS = re.compile(
    r"\b(buy|price|cheap|affordable|expensive|best|top|recommend|compare|deal|"
    r"product|shop|purchase|review|worth|budget|under|over|between|vs|versus|"
    r"alternative|option|rating|rated|earbuds|headphone|laptop|phone|camera|"
    r"tablet|monitor|keyboard|mouse|speaker|watch|tv|television)\b",
    re.IGNORECASE,
)

# Review intent is detected from explicit language, but for shopping queries
# we still want review evidence whenever the YouTube agent is available.
_REVIEW_KEYWORDS = re.compile(
    r"\b(review|opinion|recommend|worth|should i|pros and cons|"
    r"hands on|experience|compared|comparison|vs|versus|best|top)\b",
    re.IGNORECASE,
)


def route_query(
    query: str,
    discovered_agents: dict[str, AgentCard],
    card_urls: dict[str, str],
) -> RoutingDecision:
    """Determine which agents should handle a user query.

    This is a deterministic, keyword-based router. No LLM is used.

    For shopping queries (the primary use case), both agents are almost
    always triggered because:
    - Product Discovery finds products with prices and ratings.
    - YouTube Reviews adds real-world sentiment from video reviews.

    Args:
        query: The user's natural-language shopping query.
        discovered_agents: Map of agent name -> AgentCard.
        card_urls: Map of agent name -> card discovery URL.

    Returns:
        A RoutingDecision with resolved routes.
    """
    decision = RoutingDecision(query=query)

    has_product_agent = "product-discovery" in discovered_agents
    has_youtube_agent = "youtube-review" in discovered_agents

    # Determine capabilities needed
    needs_products = bool(_PRODUCT_KEYWORDS.search(query))
    explicit_review_intent = bool(_REVIEW_KEYWORDS.search(query))

    # For a shopping assistant, if neither keyword matches, default to
    # triggering product discovery (the user typed *something*).
    if not needs_products and not explicit_review_intent:
        needs_products = True

    # If only reviews are requested but we have product discovery,
    # trigger it anyway so we have structured product data to merge.
    if explicit_review_intent and not needs_products and has_product_agent:
        needs_products = True

    # For shopping queries, product discovery and review evidence are meant
    # to work together. Once we know we're evaluating products, opt into
    # YouTube review analysis when that agent is available.
    needs_reviews = explicit_review_intent or needs_products

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
                query_text=query,
            )
        )

    if decision.needs_youtube_reviews:
        card = discovered_agents["youtube-review"]
        # YouTube agent expects a product name, not the full query.
        # We pass the full query and let the agent extract the product name.
        decision.routes.append(
            AgentRoute(
                agent_name="youtube-review",
                agent_url=card.url,
                card_url=card_urls.get("youtube-review", ""),
                skill_id="youtube-product-review",
                query_text=query,
            )
        )

    logger.info(
        "Routing decision for '%s': %d routes [products=%s, reviews=%s]",
        query,
        len(decision.routes),
        decision.needs_product_discovery,
        decision.needs_youtube_reviews,
    )

    return decision
