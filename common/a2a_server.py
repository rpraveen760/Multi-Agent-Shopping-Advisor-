"""Reusable A2A server components for agent services.

Provides:
- InMemoryTaskStore: simple task state management
- create_a2a_routes(): FastAPI router factory with Agent Card + JSON-RPC dispatcher
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Awaitable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from common.a2a_models import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    Task,
    TaskStatus,
    Message,
    SendMessageRequest,
    make_task,
    METHOD_NOT_FOUND,
    INVALID_REQUEST,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    PARSE_ERROR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# In-Memory Task Store
# ═══════════════════════════════════════════════════════════════════

class InMemoryTaskStore:
    """Simple in-memory store for A2A task state management."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, task: Task) -> Task:
        """Store a new task."""
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        """Retrieve a task by ID, or None if not found."""
        return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        state: str,
        message: Message | None = None,
    ) -> Task | None:
        """Update a task's status state and optional message."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus(state=state, message=message)
        return task

    def set_artifacts(self, task_id: str, artifacts: list) -> Task | None:
        """Set the artifacts on a task."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.artifacts = artifacts
        return task


# ═══════════════════════════════════════════════════════════════════
# Type alias for the agent handler
# ═══════════════════════════════════════════════════════════════════

# Supported handler signatures:
#   async def handler(request) -> Task
#   async def handler(request, task) -> Task
#   async def handler(request, task_store, task) -> Task
SendMessageHandler = Callable[..., Awaitable[Task]]


# ═══════════════════════════════════════════════════════════════════
# A2A Router Factory
# ═══════════════════════════════════════════════════════════════════

def create_a2a_routes(
    agent_card_dict: dict[str, Any],
    handle_send_message: SendMessageHandler,
    task_store: InMemoryTaskStore | None = None,
) -> APIRouter:
    """Create a FastAPI router with A2A-compliant endpoints.

    Endpoints:
        GET  /.well-known/agent-card.json  — Agent Card discovery
        POST /a2a/v1                       — JSON-RPC dispatcher (SendMessage, GetTask)

    Args:
        agent_card_dict: The agent card as a plain dict (serialized to JSON).
        handle_send_message: Async callable that processes a SendMessage request.
        task_store: Optional task store; created internally if not provided.
    """
    store = task_store or InMemoryTaskStore()
    router = APIRouter()

    # ── Agent Card discovery ──────────────────────────────────────

    @router.get("/.well-known/agent-card.json")
    async def get_agent_card() -> JSONResponse:
        return JSONResponse(content=agent_card_dict)

    # ── JSON-RPC dispatcher ───────────────────────────────────────

    @router.post("/a2a/v1")
    async def jsonrpc_endpoint(request: Request) -> JSONResponse:
        # Parse raw body
        try:
            body = await request.json()
        except Exception:
            return _error_response(None, PARSE_ERROR, "Invalid JSON")

        if not isinstance(body, dict):
            return _error_response(None, INVALID_REQUEST, "JSON-RPC request must be an object")

        # Validate JSON-RPC envelope
        try:
            rpc_req = JsonRpcRequest(**body)
        except Exception as exc:
            return _error_response(
                body.get("id"), INVALID_REQUEST, f"Invalid JSON-RPC request: {exc}"
            )

        method = rpc_req.method
        rpc_id = rpc_req.id

        # ── SendMessage ───────────────────────────────────────────
        if method == "SendMessage":
            try:
                send_req = SendMessageRequest(**rpc_req.params)
            except Exception as exc:
                return _error_response(
                    rpc_id, INVALID_PARAMS, f"Invalid SendMessage params: {exc}"
                )

            try:
                task = _make_initial_task(send_req)
                store.create(task)

                final_task = await _invoke_send_message_handler(
                    handle_send_message,
                    send_req,
                    store,
                    task,
                )

                # Preserve the pre-created task identity so GetTask can observe
                # the same task while work is still in progress.
                final_task.id = task.id
                final_task.contextId = task.contextId
                store.create(final_task)
                return _success_response(rpc_id, {"task": final_task.model_dump()})
            except Exception as exc:
                logger.exception("SendMessage handler error")
                return _error_response(
                    rpc_id, INTERNAL_ERROR, f"Agent error: {exc}"
                )

        # ── GetTask ───────────────────────────────────────────────
        elif method == "GetTask":
            task_id = rpc_req.params.get("taskId") or rpc_req.params.get("task_id")
            if not task_id:
                return _error_response(
                    rpc_id, INVALID_PARAMS, "Missing taskId parameter"
                )
            task = store.get(task_id)
            if task is None:
                return _error_response(
                    rpc_id, INVALID_PARAMS, f"Task not found: {task_id}"
                )
            return _success_response(rpc_id, task.model_dump())

        # ── Unknown method ────────────────────────────────────────
        else:
            return _error_response(
                rpc_id, METHOD_NOT_FOUND, f"Unknown method: {method}"
            )

    return router


# ═══════════════════════════════════════════════════════════════════
# Response helpers
# ═══════════════════════════════════════════════════════════════════

def _success_response(rpc_id: str | int | None, result: dict) -> JSONResponse:
    resp = JsonRpcResponse(
        id=rpc_id or 0,
        result=result,
    )
    return JSONResponse(content=resp.model_dump(exclude_none=True))


def _error_response(rpc_id: str | int | None, code: int, message: str) -> JSONResponse:
    resp = JsonRpcResponse(
        id=rpc_id or 0,
        error=JsonRpcError(code=code, message=message),
    )
    return JSONResponse(content=resp.model_dump(exclude_none=True))


def _make_initial_task(send_req: SendMessageRequest) -> Task:
    """Create the submitted task before handler work begins."""
    task = make_task(send_req.message, state="submitted")
    context_id = (send_req.configuration or {}).get("contextId")
    if context_id:
        task.contextId = context_id
    return task


async def _invoke_send_message_handler(
    handle_send_message: SendMessageHandler,
    send_req: SendMessageRequest,
    task_store: InMemoryTaskStore,
    task: Task,
) -> Task:
    """Call the handler using the richest signature it supports."""
    positional_params = [
        param
        for param in inspect.signature(handle_send_message).parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    argc = len(positional_params)

    if argc >= 3:
        return await handle_send_message(send_req, task_store, task)
    if argc == 2:
        return await handle_send_message(send_req, task)
    return await handle_send_message(send_req)
