"""Shared runtime helpers for async resources and readable error details."""

from __future__ import annotations

import asyncio
from typing import Any


def format_exception_detail(exc: Exception) -> str:
    """Return a stable, non-empty exception detail string."""
    text = str(exc).strip()
    exc_name = type(exc).__name__
    if not text:
        return exc_name
    if text.startswith(exc_name):
        return text
    return f"{exc_name}: {text}"


async def close_async_resource(resource: Any) -> None:
    """Best-effort close for async client-like objects."""
    close_method = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close_method is None:
        return

    result = close_method()
    if asyncio.iscoroutine(result):
        await result
