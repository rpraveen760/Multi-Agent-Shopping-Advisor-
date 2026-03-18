"""Shared A2A task/runtime helpers for the orchestrator pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from common.a2a_client import A2AClient
from common.a2a_models import Message, Task
from common.runtime_helpers import format_exception_detail

logger = logging.getLogger(__name__)

TERMINAL_TASK_STATES = {"completed", "failed", "canceled", "input-required"}
TASK_POLL_INTERVAL_SECONDS = 0.25

TaskObserver = Callable[[Task], None]
SleepFunc = Callable[[float], Awaitable[None]]


async def resolve_send_message_result(
    result: Task | Message,
    *,
    client: A2AClient,
    agent_url: str,
    timeout_seconds: int,
    task_observer: TaskObserver | None = None,
    sleep_fn: SleepFunc = asyncio.sleep,
    poll_interval_seconds: float = TASK_POLL_INTERVAL_SECONDS,
    loop_time: Callable[[], float] | None = None,
) -> Task | Message:
    """Poll non-terminal A2A tasks until they reach a terminal state."""
    if not isinstance(result, Task):
        return result

    task = result
    current_time = loop_time or asyncio.get_running_loop().time
    deadline = current_time() + timeout_seconds
    last_signature = task_signature(task)

    if task_observer:
        task_observer(task)

    while task.status.state not in TERMINAL_TASK_STATES:
        now = current_time()
        if now >= deadline:
            raise TimeoutError(
                f"Task {task.id} did not reach a terminal state within {timeout_seconds} seconds"
            )

        await sleep_fn(min(poll_interval_seconds, deadline - now))
        task = await client.get_task(agent_url, task.id)
        signature = task_signature(task)
        if task_observer and signature != last_signature:
            task_observer(task)
        last_signature = signature

    return task


def extract_named_artifact_json(task: Task, artifact_name: str) -> dict[str, Any] | None:
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


def task_status_text(task: Task) -> str | None:
    """Extract a flat status message from a task."""
    message = task.status.message
    if message is None:
        return None

    parts = [part.text.strip() for part in message.parts if part.type == "text" and part.text.strip()]
    if not parts:
        return None

    return " ".join(parts)


def task_signature(
    task: Task,
) -> tuple[str, str, str]:
    """Build a compact signature so observers only react to meaningful changes."""
    return (
        task.status.state,
        task_status_text(task) or "",
        json.dumps(
            {
                "artifacts": [artifact.name for artifact in task.artifacts or []],
                "message": task_status_text(task) or "",
            },
            sort_keys=True,
        ),
    )


__all__ = [
    "TASK_POLL_INTERVAL_SECONDS",
    "TERMINAL_TASK_STATES",
    "extract_named_artifact_json",
    "format_exception_detail",
    "resolve_send_message_result",
    "task_signature",
    "task_status_text",
]
