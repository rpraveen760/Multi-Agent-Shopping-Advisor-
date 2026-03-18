import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from agents.orchestrator import server


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
        trace_response = self.client.get("/trace")

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(mounted_response.status_code, 200)
        self.assertEqual(trace_response.status_code, 200)
        self.assertIn("Federated Multi-Agent System", root_response.text)
        self.assertIn("Live Trace", trace_response.text)

    def test_frontend_is_self_contained_and_same_origin(self):
        response = self.client.get("/")
        trace_response = self.client.get("/trace")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(trace_response.status_code, 200)
        self.assertNotIn("cdnjs", response.text)
        self.assertNotIn("cdnjs", trace_response.text)
        self.assertNotIn("http://localhost:8000", response.text)
        self.assertNotIn("http://localhost:8000", trace_response.text)
        self.assertIn('fetch("/query"', response.text)
        self.assertIn('fetch("/status"', response.text)
        self.assertIn('fetch("/query/trace"', trace_response.text)

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
        ):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["available_agents"], 1)
        self.assertEqual(payload["agents"][0]["status"], "degraded")
        self.assertIn("MCP unavailable", payload["agents"][0]["description"])


if __name__ == "__main__":
    unittest.main()
