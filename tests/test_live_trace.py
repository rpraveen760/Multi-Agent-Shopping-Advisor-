import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agents.orchestrator import server
from agents.orchestrator.trace import TraceStore
from common.a2a_models import RankedRecommendation, UnifiedResponse, UnifiedSourceLink


class TraceApiTests(unittest.TestCase):
    def setUp(self):
        self._original_store = server.TRACE_STORE
        server.TRACE_STORE = TraceStore()
        self.client = TestClient(server.app)

    def tearDown(self):
        server.TRACE_STORE = self._original_store

    def test_start_trace_query_returns_poll_handle(self):
        with patch("agents.orchestrator.server._launch_traced_query") as launch_mock:
            response = self.client.post("/query/trace", json={"query": "best earbuds"})

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["query"], "best earbuds")
        self.assertIn("/query/trace/", payload["status_url"])
        snapshot = server.TRACE_STORE.get_run(payload["run_id"])
        self.assertIsNotNone(snapshot)
        launch_mock.assert_called_once_with(payload["run_id"], "best earbuds")

    def test_get_trace_query_returns_not_found_for_unknown_run(self):
        response = self.client.get("/query/trace/missing-run")
        self.assertEqual(response.status_code, 404)


class TraceRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._original_store = server.TRACE_STORE
        server.TRACE_STORE = TraceStore()

    async def asyncTearDown(self):
        server.TRACE_STORE = self._original_store

    async def test_run_traced_query_records_final_response(self):
        snapshot = server.TRACE_STORE.create_run("best earbuds")
        fake_response = UnifiedResponse(
            query="best earbuds",
            recommendations=[
                RankedRecommendation(
                    rank=1,
                    product_name="Sony WF-1000XM5",
                    price="$99.99",
                    rating="4.5/5",
                    sentiment="positive",
                    score=0.91,
                    rationale="Best balance of value and performance.",
                    pros=["ANC"],
                    cons=["Price"],
                    confidence=0.9,
                )
            ],
            sources=[
                UnifiedSourceLink(
                    type="product",
                    title="Sony WF-1000XM5",
                    url="https://merchant.example/sony",
                    agent="product-discovery",
                )
            ],
            partial=False,
            notes=None,
        )

        with patch(
            "agents.orchestrator.server.run_query",
            new=AsyncMock(return_value=fake_response),
        ):
            await server._run_traced_query(snapshot.run_id, "best earbuds")

        updated = server.TRACE_STORE.get_run(snapshot.run_id)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.final_response["query"], "best earbuds")
        self.assertEqual(updated.final_response["recommendations"][0]["product_name"], "Sony WF-1000XM5")


if __name__ == "__main__":
    unittest.main()
