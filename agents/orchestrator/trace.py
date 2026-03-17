"""Live trace storage for orchestrator runs.

Keeps short-lived in-memory snapshots that power the frontend trace page.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from agents.orchestrator.routing import RoutingDecision
from common.a2a_models import AgentCard, UnifiedResponse


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_utc_now)
    status: str = "info"
    title: str
    detail: str | None = None
    agent: str | None = None
    data: dict[str, Any] | None = None


class QueryTraceSnapshot(BaseModel):
    run_id: str
    query: str
    status: str = "queued"
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    events: list[TraceEvent] = Field(default_factory=list)
    discovered_agents: list[dict[str, Any]] = Field(default_factory=list)
    routing: dict[str, Any] | None = None
    product_trace: dict[str, Any] | None = None
    review_traces: dict[str, dict[str, Any]] = Field(default_factory=dict)
    synthesis: dict[str, Any] | None = None
    final_response: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class TraceStore:
    """Thread-safe in-memory storage for live query traces."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, QueryTraceSnapshot] = {}

    def create_run(self, query: str) -> QueryTraceSnapshot:
        run_id = str(uuid.uuid4())
        snapshot = QueryTraceSnapshot(run_id=run_id, query=query)
        with self._lock:
            self._runs[run_id] = snapshot
        return snapshot.model_copy(deep=True)

    def get_run(self, run_id: str) -> QueryTraceSnapshot | None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return None
            return snapshot.model_copy(deep=True)

    def mark_running(self, run_id: str) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.status = "running"
            snapshot.updated_at = _utc_now()

    def mark_completed(self, run_id: str) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.status = "completed"
            snapshot.updated_at = _utc_now()

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.status = "failed"
            snapshot.errors.append(error)
            snapshot.updated_at = _utc_now()

    def add_event(
        self,
        run_id: str,
        *,
        title: str,
        status: str = "info",
        detail: str | None = None,
        agent: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.events.append(
                TraceEvent(
                    title=title,
                    status=status,
                    detail=detail,
                    agent=agent,
                    data=data,
                )
            )
            snapshot.updated_at = _utc_now()

    def append_error(self, run_id: str, error: str) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.errors.append(error)
            snapshot.updated_at = _utc_now()

    def set_discovered_agents(
        self,
        run_id: str,
        discovered_agents: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.discovered_agents = discovered_agents
            snapshot.updated_at = _utc_now()

    def set_routing(self, run_id: str, routing: dict[str, Any]) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.routing = routing
            snapshot.updated_at = _utc_now()

    def set_product_trace(self, run_id: str, trace: dict[str, Any]) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.product_trace = trace
            snapshot.updated_at = _utc_now()

    def set_review_trace(self, run_id: str, product_name: str, trace: dict[str, Any]) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.review_traces[product_name] = trace
            snapshot.updated_at = _utc_now()

    def set_synthesis(self, run_id: str, synthesis: dict[str, Any]) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.synthesis = synthesis
            snapshot.updated_at = _utc_now()

    def set_final_response(self, run_id: str, response: dict[str, Any]) -> None:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                return
            snapshot.final_response = response
            snapshot.updated_at = _utc_now()


class TraceRecorder:
    """Convenience wrapper for updating a single run."""

    def __init__(self, store: TraceStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    def mark_running(self) -> None:
        self.store.mark_running(self.run_id)

    def mark_completed(self) -> None:
        self.store.mark_completed(self.run_id)

    def mark_failed(self, error: str) -> None:
        self.store.mark_failed(self.run_id, error)

    def add_event(
        self,
        title: str,
        *,
        status: str = "info",
        detail: str | None = None,
        agent: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_event(
            self.run_id,
            title=title,
            status=status,
            detail=detail,
            agent=agent,
            data=data,
        )

    def append_error(self, error: str) -> None:
        self.store.append_error(self.run_id, error)

    def set_discovered_agents(
        self,
        discovered_agents: dict[str, AgentCard],
        card_urls: dict[str, str],
    ) -> None:
        self.store.set_discovered_agents(
            self.run_id,
            [
                {
                    "agent_name": name,
                    "card_url": card_urls.get(name, ""),
                    "url": card.url,
                    "name": card.name,
                    "description": card.description,
                    "skills": [skill.model_dump() for skill in card.skills],
                }
                for name, card in discovered_agents.items()
            ],
        )

    def set_routing(self, decision: RoutingDecision) -> None:
        self.store.set_routing(
            self.run_id,
            {
                "query": decision.query,
                "needs_product_discovery": decision.needs_product_discovery,
                "needs_youtube_reviews": decision.needs_youtube_reviews,
                "routes": [
                    {
                        "agent_name": route.agent_name,
                        "agent_url": route.agent_url,
                        "card_url": route.card_url,
                        "skill_id": route.skill_id,
                        "query_text": route.query_text,
                    }
                    for route in decision.routes
                ],
            },
        )

    def set_product_trace(self, trace: dict[str, Any]) -> None:
        self.store.set_product_trace(self.run_id, trace)

    def set_review_trace(self, product_name: str, trace: dict[str, Any]) -> None:
        self.store.set_review_trace(self.run_id, product_name, trace)

    def set_synthesis(self, synthesis: dict[str, Any]) -> None:
        self.store.set_synthesis(self.run_id, synthesis)

    def set_final_response(self, response: UnifiedResponse) -> None:
        self.store.set_final_response(self.run_id, response.model_dump())
