"""A2A client for agent discovery and task delegation.

Follows the Execution Contract:
- send_message() POSTs directly to the supplied agent_url (never appends /a2a/v1).
- get_task() POSTs directly to the supplied agent_url.
"""

from __future__ import annotations

import logging
import uuid
from typing import TypeAlias

import httpx

from common.a2a_models import (
    AgentCard,
    JsonRpcResponse,
    Message,
    Task,
    make_user_message,
)
from common.config import get_settings

logger = logging.getLogger(__name__)

SendMessageResult: TypeAlias = Task | Message


class A2AClient:
    """HTTP client for A2A v0.3 protocol interactions."""

    def __init__(self, timeout: int | None = None) -> None:
        settings = get_settings()
        self._timeout = timeout or settings.A2A_CLIENT_TIMEOUT_SECONDS
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Discovery ─────────────────────────────────────────────────

    async def discover_agent(self, card_url: str) -> AgentCard:
        """Fetch and parse an Agent Card from the given URL.

        Args:
            card_url: Full URL to the agent-card.json endpoint.
                      e.g. http://localhost:5001/.well-known/agent-card.json
        """
        client = await self._get_client()
        logger.info("Discovering agent at %s", card_url)
        resp = await client.get(card_url)
        resp.raise_for_status()
        return AgentCard(**resp.json())

    # ── SendMessage ───────────────────────────────────────────────

    async def send_message(
        self,
        agent_url: str,
        text: str,
        context_id: str | None = None,
    ) -> SendMessageResult:
        """Send a message to an agent via A2A SendMessage.

        IMPORTANT: Posts directly to agent_url as per Execution Contract.
        The agent_url comes from AgentCard.url and already contains the
        full JSON-RPC endpoint path.

        Args:
            agent_url: The agent's JSON-RPC endpoint (AgentCard.url).
            text: The user message text to send.
            context_id: Optional context ID for grouping related tasks.

        Returns:
            Either a Task or an immediate Message, depending on the
            SendMessage response shape returned by the agent.
        """
        client = await self._get_client()

        message = make_user_message(text)

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "SendMessage",
            "params": {
                "message": message.model_dump(),
            },
        }

        if context_id:
            payload["params"]["configuration"] = {"contextId": context_id}

        logger.info("SendMessage to %s: %.80s...", agent_url, text)
        resp = await client.post(agent_url, json=payload)
        resp.raise_for_status()

        rpc_resp = JsonRpcResponse(**resp.json())

        if rpc_resp.error:
            raise A2AError(
                f"Agent returned error: [{rpc_resp.error.code}] {rpc_resp.error.message}"
            )

        if rpc_resp.result is None:
            raise A2AError("Agent returned empty result")

        # The result may contain a nested "task" key, a nested "message" key,
        # or be the task directly.
        task_data = rpc_resp.result
        if "task" in task_data:
            task_data = task_data["task"]
            return Task(**task_data)
        if "message" in task_data:
            return Message(**task_data["message"])
        return Task(**task_data)

    # ── GetTask ───────────────────────────────────────────────────

    async def get_task(self, agent_url: str, task_id: str) -> Task:
        """Retrieve a task's current state via A2A GetTask.

        IMPORTANT: Posts directly to agent_url as per Execution Contract.

        Args:
            agent_url: The agent's JSON-RPC endpoint (AgentCard.url).
            task_id: The task ID to retrieve.

        Returns:
            The Task object with current state.
        """
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "GetTask",
            "params": {
                "taskId": task_id,
            },
        }

        logger.info("GetTask from %s: %s", agent_url, task_id)
        resp = await client.post(agent_url, json=payload)
        resp.raise_for_status()

        rpc_resp = JsonRpcResponse(**resp.json())

        if rpc_resp.error:
            raise A2AError(
                f"Agent returned error: [{rpc_resp.error.code}] {rpc_resp.error.message}"
            )

        if rpc_resp.result is None:
            raise A2AError("Agent returned empty result")

        return Task(**rpc_resp.result)


class A2AError(Exception):
    """Raised when an A2A protocol operation fails."""
