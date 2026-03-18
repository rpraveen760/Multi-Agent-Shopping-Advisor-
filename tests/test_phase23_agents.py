import asyncio
import json
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

if "bs4" not in sys.modules:
    bs4_module = types.ModuleType("bs4")

    class BeautifulSoup:  # pragma: no cover - test stub for optional dependency
        def __init__(self, html, parser):
            self.html = html

        def find(self, name):
            return None

        def get_text(self, *args, **kwargs):
            return ""

    bs4_module.BeautifulSoup = BeautifulSoup
    sys.modules["bs4"] = bs4_module

if "ddgs" not in sys.modules:
    ddgs_module = types.ModuleType("ddgs")

    class DDGS:  # pragma: no cover - test stub for optional dependency
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=10):
            return []

    ddgs_module.DDGS = DDGS
    sys.modules["ddgs"] = ddgs_module

if "googleapiclient.discovery" not in sys.modules:
    googleapiclient_module = sys.modules.setdefault(
        "googleapiclient",
        types.ModuleType("googleapiclient"),
    )
    discovery_module = types.ModuleType("googleapiclient.discovery")
    discovery_module.build = lambda *args, **kwargs: None
    googleapiclient_module.discovery = discovery_module
    sys.modules["googleapiclient.discovery"] = discovery_module

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class AsyncOpenAI:  # pragma: no cover - test stub for optional dependency
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

    openai_module.AsyncOpenAI = AsyncOpenAI
    sys.modules["openai"] = openai_module

if "youtube_transcript_api" not in sys.modules:
    transcript_module = types.ModuleType("youtube_transcript_api")

    class YouTubeTranscriptApi:  # pragma: no cover - test stub for optional dependency
        def fetch(self, video_id):
            raise RuntimeError("stub transcript API should be patched in tests")

    transcript_module.YouTubeTranscriptApi = YouTubeTranscriptApi
    sys.modules["youtube_transcript_api"] = transcript_module

from agents.product_discovery import server as product_server
from agents.youtube_review import agent as review_agent
from agents.youtube_review import server as review_server
from common.a2a_models import Product, ProductDiscoveryResult, ReviewSummary
from common.a2a_server import InMemoryTaskStore, create_a2a_routes


def build_app(agent_card, handler, store=None):
    app = FastAPI()
    app.include_router(
        create_a2a_routes(
            agent_card_dict=agent_card.model_dump(),
            handle_send_message=handler,
            task_store=store or InMemoryTaskStore(),
        )
    )
    return app


class ProductDiscoveryServerTests(unittest.TestCase):
    def test_agent_card_advertises_http_mcp_interface(self):
        interfaces = product_server.AGENT_CARD.additionalInterfaces or []
        mcp_interfaces = [iface for iface in interfaces if iface.transport == "MCP"]

        self.assertEqual(len(mcp_interfaces), 1)
        self.assertEqual(mcp_interfaces[0].url, "http://localhost:5002/mcp")

    def test_debug_progress_endpoint_returns_latest_snapshot(self):
        client = TestClient(product_server.app)
        trace_id = "trace-debug-1"
        payload = {"stage": "search-complete", "query": "best earbuds"}

        product_server.PROGRESS_STORE.publish(trace_id, payload)
        try:
            response = client.get(f"/debug/product-discovery/{trace_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["stage"], "search-complete")

            cleared = client.delete(f"/debug/product-discovery/{trace_id}")
            self.assertEqual(cleared.status_code, 200)

            missing = client.get(f"/debug/product-discovery/{trace_id}")
            self.assertEqual(missing.status_code, 404)
        finally:
            product_server.PROGRESS_STORE.clear(trace_id)

    def test_send_message_returns_discovery_artifact(self):
        client = TestClient(build_app(product_server.AGENT_CARD, product_server.handle_send_message))
        fake_result = ProductDiscoveryResult(
            query="best earbuds under 100",
            products=[
                Product(
                    name="Sony WF-1000XM5",
                    price="$99.99",
                    rating="4.5/5",
                    url="https://merchant.example/sony",
                    key_features=["ANC"],
                    source="Example Merchant",
                    evidence_urls=["https://merchant.example/sony"],
                    confidence=0.91,
                )
            ],
            summary="Found one grounded product candidate.",
        )

        with patch("agents.product_discovery.server.get_settings", return_value=SimpleNamespace()), patch(
            "agents.product_discovery.server.search_products",
            new=AsyncMock(return_value=fake_result),
        ):
            response = client.post(
                "/a2a/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": "product-1",
                    "method": "SendMessage",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"type": "text", "text": "best earbuds under 100"}],
                            "messageId": "msg-product-1",
                        }
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["result"]["task"]["status"]["state"], "completed")

        artifact_text = payload["result"]["task"]["artifacts"][0]["parts"][0]["text"]
        result = json.loads(artifact_text)
        self.assertEqual(result["query"], "best earbuds under 100")
        self.assertEqual(len(result["products"]), 1)


class YouTubeReviewServerTests(unittest.TestCase):
    def test_get_task_reports_working_state_during_review_pipeline(self):
        ready = threading.Event()
        release = threading.Event()
        store = InMemoryTaskStore()

        async def fake_get_review_summary(product_name, settings):
            ready.set()
            await asyncio.to_thread(release.wait, 2)
            return ReviewSummary(
                product_name=product_name,
                overall_sentiment="positive",
                score=8.7,
                pros=["Strong ANC"],
                cons=["Premium price"],
                key_quotes=["Sounds excellent"],
                recommendation="A strong buy if ANC matters most.",
                confidence=0.9,
                sources=[],
            )

        app = FastAPI()
        app.include_router(
            create_a2a_routes(
                agent_card_dict=review_server.AGENT_CARD.model_dump(),
                handle_send_message=review_server.handle_send_message,
                task_store=store,
            )
        )

        with patch("agents.youtube_review.server.get_settings", return_value=SimpleNamespace()), patch(
            "agents.youtube_review.server.get_review_summary",
            side_effect=fake_get_review_summary,
        ):
            send_client = TestClient(app)
            poll_client = TestClient(app)
            response_holder = {}

            send_payload = {
                "jsonrpc": "2.0",
                "id": "review-1",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Sony WF-1000XM5"}],
                        "messageId": "msg-review-1",
                    }
                },
            }

            def do_send():
                response_holder["response"] = send_client.post("/a2a/v1", json=send_payload)

            worker = threading.Thread(target=do_send)
            worker.start()

            self.assertTrue(ready.wait(1), "review pipeline did not start")
            task_id = next(iter(store._tasks))

            poll_response = poll_client.post(
                "/a2a/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": "review-poll-1",
                    "method": "GetTask",
                    "params": {"taskId": task_id},
                },
            )

            self.assertEqual(poll_response.status_code, 200)
            self.assertEqual(
                poll_response.json()["result"]["status"]["state"],
                "working",
            )

            release.set()
            worker.join(2)

            final_payload = response_holder["response"].json()
            self.assertEqual(final_payload["result"]["task"]["status"]["state"], "completed")
            self.assertEqual(
                final_payload["result"]["task"]["artifacts"][0]["metadata"]["schema"],
                "ReviewSummary",
            )


class YouTubeReviewAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_review_summary_offloads_blocking_steps(self):
        settings = SimpleNamespace(
            YOUTUBE_API_KEY="yt-key",
            OPENAI_API_KEY="oa-key",
            OPENAI_MODEL="gpt-4o-mini",
        )
        videos = [
            {
                "video_id": "vid-1",
                "title": "WF-1000XM5 Review",
                "channel": "Audio Lab",
                "description": "Detailed review",
                "published_at": "2026-01-01",
            }
        ]
        enriched = [{**videos[0], "view_count": "12345"}]
        evidence = [{**enriched[0], "transcript": "Battery life is great.", "evidence_source": "transcript"}]
        summary = ReviewSummary(
            product_name="Sony WF-1000XM5",
            overall_sentiment="positive",
            score=8.8,
            pros=["Great battery"],
            cons=["Expensive"],
            key_quotes=["Battery life is great."],
            recommendation="A polished flagship earbud overall.",
            confidence=0.85,
            sources=[],
        )
        to_thread_calls = []

        async def fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append(getattr(func, "__name__", func._mock_name))
            return func(*args, **kwargs)

        with patch.object(review_agent.asyncio, "to_thread", new=fake_to_thread), patch.object(
            review_agent,
            "_search_youtube",
            return_value=videos,
        ), patch.object(
            review_agent,
            "_fetch_video_metadata",
            return_value=enriched,
        ), patch.object(
            review_agent,
            "_extract_transcripts",
            return_value=evidence,
        ), patch.object(
            review_agent,
            "_summarize_reviews",
            new=AsyncMock(return_value=summary),
        ) as summarize_mock:
            result = await review_agent.get_review_summary("Sony WF-1000XM5", settings)

        self.assertEqual(
            to_thread_calls,
            ["_search_youtube", "_fetch_video_metadata", "_extract_transcripts"],
        )
        summarize_mock.assert_awaited_once()
        self.assertEqual(result.sources[0].url, "https://youtube.com/watch?v=vid-1")

    async def test_get_review_summary_returns_empty_review_when_no_usable_evidence(self):
        settings = SimpleNamespace(
            YOUTUBE_API_KEY="yt-key",
            OPENAI_API_KEY="oa-key",
            OPENAI_MODEL="gpt-4o-mini",
        )
        videos = [
            {
                "video_id": "vid-2",
                "title": "Mystery Earbuds Review",
                "channel": "Audio Lab",
                "description": "",
                "published_at": "2026-01-01",
            }
        ]
        enriched = [{**videos[0], "view_count": "50"}]
        empty_evidence = [{**enriched[0], "transcript": "", "evidence_source": "description"}]

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(review_agent.asyncio, "to_thread", new=fake_to_thread), patch.object(
            review_agent,
            "_search_youtube",
            return_value=videos,
        ), patch.object(
            review_agent,
            "_fetch_video_metadata",
            return_value=enriched,
        ), patch.object(
            review_agent,
            "_extract_transcripts",
            return_value=empty_evidence,
        ), patch.object(
            review_agent,
            "_summarize_reviews",
            new=AsyncMock(),
        ) as summarize_mock:
            result = await review_agent.get_review_summary("Mystery Earbuds", settings)

        summarize_mock.assert_not_awaited()
        self.assertEqual(result.product_name, "Mystery Earbuds")
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("No transcript or description evidence", result.recommendation)
        self.assertEqual(len(result.sources), 1)


if __name__ == "__main__":
    unittest.main()
