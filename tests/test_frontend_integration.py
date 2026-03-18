import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agents.orchestrator import server
from common.a2a_models import Artifact, Task, TaskStatus, TextPart, make_agent_message


class _FakeClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    async def discover_agent(self, _card_url):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self):
        return None


class FrontendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_root_and_frontend_mount_serve_ui(self):
        root_response = self.client.get("/")
        mounted_response = self.client.get("/frontend/index.html")

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(mounted_response.status_code, 200)
        self.assertIn("Federated Multi-Agent System", root_response.text)

    def test_frontend_is_self_contained_and_same_origin(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("cdnjs", response.text)
        self.assertNotIn("http://localhost:8000", response.text)
        self.assertIn('fetch("/query"', response.text)
        self.assertIn('fetch("/video/chat"', response.text)
        self.assertIn('fetch("/status"', response.text)

    def test_video_chat_endpoint_delegates_over_a2a(self):
        youtube_card = SimpleNamespace(
            name="YouTube Product Review Agent",
            url="http://localhost:5001/a2a/v1",
            description="Review search",
            skills=[SimpleNamespace(id="youtube-transcript-chat")],
        )
        response_task = Task(
            status=TaskStatus(state="completed", message=make_agent_message("chat done")),
            artifacts=[
                Artifact(
                    name="video-chat-response",
                    parts=[
                        TextPart(
                            text='{"youtube_url":"https://www.youtube.com/watch?v=abc123","answer":"It focuses on weight and clicks.","citations":["Chunk 1"],"confidence":0.8,"session_id":"session-123","history":[{"role":"user","content":"What matters?"},{"role":"assistant","content":"It focuses on weight and clicks."}]}'
                        )
                    ],
                )
            ],
        )
        fake_client = SimpleNamespace(
            discover_agent=AsyncMock(return_value=youtube_card),
            send_message=AsyncMock(return_value=response_task),
            get_task=AsyncMock(return_value=response_task),
            close=AsyncMock(return_value=None),
        )

        settings = SimpleNamespace(
            YOUTUBE_AGENT_CARD_URL="http://localhost:5001/.well-known/agent-card.json",
            A2A_CLIENT_TIMEOUT_SECONDS=1,
        )

        with patch("agents.orchestrator.server.get_settings", return_value=settings), patch(
            "agents.orchestrator.server.A2AClient",
            return_value=fake_client,
        ), patch(
            "agents.orchestrator.server.resolve_send_message_result",
            new=AsyncMock(return_value=response_task),
        ):
            response = self.client.post(
                "/video/chat",
                json={"session_id": "session-123", "chat_message": "What matters?"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], "session-123")
        self.assertEqual(payload["history"][0]["role"], "user")

    def test_status_endpoint_reports_degraded_when_agent_is_unavailable(self):
        settings = SimpleNamespace(
            PRODUCT_AGENT_CARD_URL="http://localhost:5002/.well-known/agent-card.json",
            YOUTUBE_AGENT_CARD_URL="http://localhost:5001/.well-known/agent-card.json",
        )
        product_card = SimpleNamespace(
            name="Product Discovery Agent",
            url="http://localhost:5002/a2a/v1",
            description="Product search",
            additionalInterfaces=[
                SimpleNamespace(url="http://localhost:5002/mcp", transport="MCP")
            ],
            skills=[],
        )

        with patch("agents.orchestrator.server.get_settings", return_value=settings), patch(
            "agents.orchestrator.server.A2AClient",
            return_value=_FakeClient([product_card, RuntimeError("boom")]),
        ), patch(
            "agents.orchestrator.server._verify_product_discovery_mcp",
            return_value=(True, None),
        ), patch(
            "agents.orchestrator.server._verify_youtube_review_runtime",
            return_value=(True, None),
        ):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["available_agents"], 1)
        self.assertEqual(payload["total_agents"], 2)

    def test_status_endpoint_degrades_when_product_mcp_is_unavailable(self):
        settings = SimpleNamespace(
            PRODUCT_AGENT_CARD_URL="http://localhost:5002/.well-known/agent-card.json",
            YOUTUBE_AGENT_CARD_URL="http://localhost:5001/.well-known/agent-card.json",
        )
        product_card = SimpleNamespace(
            name="Product Discovery Agent",
            url="http://localhost:5002/a2a/v1",
            description="Product search",
            additionalInterfaces=[
                SimpleNamespace(url="http://localhost:5002/mcp", transport="MCP")
            ],
            skills=[],
        )
        youtube_card = SimpleNamespace(
            name="YouTube Product Review Agent",
            url="http://localhost:5001/a2a/v1",
            description="Review search",
            skills=[],
        )

        with patch("agents.orchestrator.server.get_settings", return_value=settings), patch(
            "agents.orchestrator.server.A2AClient",
            return_value=_FakeClient([product_card, youtube_card]),
        ), patch(
            "agents.orchestrator.server._verify_product_discovery_mcp",
            return_value=(False, "search_products missing"),
        ), patch(
            "agents.orchestrator.server._verify_youtube_review_runtime",
            return_value=(True, None),
        ):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["available_agents"], 1)
        self.assertEqual(payload["agents"][0]["status"], "degraded")
        self.assertIn("MCP unavailable", payload["agents"][0]["description"])

    def test_status_endpoint_degrades_when_youtube_video_analysis_runtime_is_unavailable(self):
        settings = SimpleNamespace(
            PRODUCT_AGENT_CARD_URL="http://localhost:5002/.well-known/agent-card.json",
            YOUTUBE_AGENT_CARD_URL="http://localhost:5001/.well-known/agent-card.json",
        )
        product_card = SimpleNamespace(
            name="Product Discovery Agent",
            url="http://localhost:5002/a2a/v1",
            description="Product search",
            additionalInterfaces=[
                SimpleNamespace(url="http://localhost:5002/mcp", transport="MCP")
            ],
            skills=[],
        )
        youtube_card = SimpleNamespace(
            name="YouTube Product Review Agent",
            url="http://localhost:5001/a2a/v1",
            description="Review search",
            skills=[],
        )

        with patch("agents.orchestrator.server.get_settings", return_value=settings), patch(
            "agents.orchestrator.server.A2AClient",
            return_value=_FakeClient([product_card, youtube_card]),
        ), patch(
            "agents.orchestrator.server._verify_product_discovery_mcp",
            return_value=(True, None),
        ), patch(
            "agents.orchestrator.server._verify_youtube_review_runtime",
            return_value=(False, "transcript runtime unavailable"),
        ):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["available_agents"], 1)
        self.assertEqual(payload["agents"][1]["status"], "degraded")
        self.assertIn("video-analysis unavailable", payload["agents"][1]["description"])


if __name__ == "__main__":
    unittest.main()
