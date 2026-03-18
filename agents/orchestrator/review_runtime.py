"""YouTube review extraction and targeting helpers for the orchestrator."""

from __future__ import annotations

import logging

from agents.orchestrator.runtime import extract_named_artifact_json
from common.a2a_models import ProductDiscoveryResult, ReviewSummary, Task, UnifiedVideoAnalysisResponse

logger = logging.getLogger(__name__)


def extract_review_summary(task: Task) -> ReviewSummary | None:
    """Extract ReviewSummary from a completed Task's artifacts."""
    if task.status.state != "completed":
        logger.warning("Review task not completed: state=%s", task.status.state)
        return None

    data = extract_named_artifact_json(task, "review-summary")
    if data is None:
        return None

    try:
        return ReviewSummary(**data)
    except Exception as exc:
        logger.debug("Failed to parse review summary artifact: %s", exc)
        return None
def extract_video_analysis(task: Task) -> UnifiedVideoAnalysisResponse | None:
    """Extract the structured video-analysis result from a completed Task."""
    if task.status.state != "completed":
        logger.warning("Video analysis task not completed: state=%s", task.status.state)
        return None

    data = extract_named_artifact_json(task, "video-analysis-result")
    if data is None:
        return None

    try:
        return UnifiedVideoAnalysisResponse(**data)
    except Exception as exc:
        logger.debug("Failed to parse video analysis artifact: %s", exc)
        return None


def normalize_name_key(value: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()


def match_known_product_name(candidate_name: str, known_names: list[str]) -> str | None:
    """Resolve a synthesized product name back to a discovered concrete product name."""
    candidate_tokens = normalize_name_key(candidate_name)
    if not candidate_tokens:
        return None

    candidate_key = " ".join(candidate_tokens)
    exact = {
        " ".join(normalize_name_key(name)): name
        for name in known_names
    }
    if candidate_key in exact:
        return exact[candidate_key]

    candidate_set = set(candidate_tokens)
    best_name = None
    best_score = 0
    for known_name in known_names:
        known_tokens = set(normalize_name_key(known_name))
        overlap = len(candidate_set & known_tokens)
        if overlap > best_score:
            best_score = overlap
            best_name = known_name

    if best_score >= 2:
        return best_name
    return None


def select_review_targets(
    product_result: ProductDiscoveryResult | None,
    *,
    max_review_products: int = 3,
) -> list[str]:
    """Choose the top concrete products to send through the review pipeline."""
    if not product_result or not product_result.products:
        return []

    ordered = sorted(
        product_result.products,
        key=lambda product: (
            -(product.confidence or 0.0),
            -len(product.evidence_urls),
            product.name,
        ),
    )

    names: list[str] = []
    seen: set[str] = set()
    for product in ordered:
        key = " ".join(normalize_name_key(product.name))
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(product.name)
        if len(names) >= max_review_products:
            break
    return names
