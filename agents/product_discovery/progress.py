"""Lightweight in-memory progress store for Product Discovery debug snapshots."""

from __future__ import annotations

import copy
from threading import Lock
from typing import Any


class ProgressStore:
    """Thread-safe storage for the latest progress payload per trace id."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._payloads: dict[str, dict[str, Any]] = {}

    def publish(self, trace_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payloads[trace_id] = copy.deepcopy(payload)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._payloads.get(trace_id)
            if payload is None:
                return None
            return copy.deepcopy(payload)

    def clear(self, trace_id: str) -> None:
        with self._lock:
            self._payloads.pop(trace_id, None)


PROGRESS_STORE = ProgressStore()
