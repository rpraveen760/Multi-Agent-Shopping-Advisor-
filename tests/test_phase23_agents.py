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
from agents.youtube_review import sessions as review_sessions
from agents.youtube_review import server as review_server
from agents.youtube_review import video_analysis as video_analysis_agent
from common.a2a_models import (
    ExtractedProductDetails,
    Product,
    ProductDiscoveryResult,
    ReviewSummary,
    TranscriptChunk,
    UnifiedVideoAnalysisResponse,
    VideoMetadata,
    VideoChatResponse,
    YouTubeVideoRequest,
)
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
    def test_agent_card_advertises_video_analysis_skill(self):
        skill_ids = [skill.id for skill in review_server.AGENT_CARD.skills]
        self.assertIn("youtube-video-analysis", skill_ids)

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

    def test_send_message_accepts_structured_youtube_url_request(self):
        client = TestClient(build_app(review_server.AGENT_CARD, review_server.handle_send_message))
        fake_result = UnifiedVideoAnalysisResponse(
            youtube_url="https://www.youtube.com/watch?v=abc123",
            video=VideoMetadata(
                video_id="abc123",
                video_url="https://www.youtube.com/watch?v=abc123",
                title="Gaming Mouse Review",
                channel="Tech Lab",
                published_at="2026-03-01",
                view_count="1000",
            ),
            transcript_status="indexed",
            indexing_status="indexed",
            summary="Transcript-grounded product analysis.",
            partial=False,
            notes=None,
        )

        with patch("agents.youtube_review.server.get_settings", return_value=SimpleNamespace()), patch(
            "agents.youtube_review.server.analyze_video_review",
            new=AsyncMock(return_value=(fake_result, {"stage": "completed", "youtube_url": fake_result.youtube_url})),
        ):
            response = client.post(
                "/a2a/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": "video-1",
                    "method": "SendMessage",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "youtube_url": fake_result.youtube_url,
                                            "find_similar_products": True,
                                        }
                                    ),
                                }
                            ],
                            "messageId": "msg-video-1",
                        },
                        "metadata": {"product_mcp_url": "http://localhost:5002/mcp"},
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["result"]["task"]["status"]["state"], "completed")
        self.assertEqual(
            payload["result"]["task"]["artifacts"][0]["metadata"]["schema"],
            "UnifiedVideoAnalysisResponse",
        )

    def test_send_message_accepts_structured_transcript_chat_request(self):
        client = TestClient(build_app(review_server.AGENT_CARD, review_server.handle_send_message))
        fake_response = VideoChatResponse(
            youtube_url="https://www.youtube.com/watch?v=abc123",
            answer="The review recommends it for competitive gamers.",
            citations=["Chunk 1: It is very light."],
            confidence=0.82,
            session_id="session-123",
            history=[],
        )

        with patch("agents.youtube_review.server.get_settings", return_value=SimpleNamespace()), patch(
            "agents.youtube_review.server.chat_with_video_session",
            new=AsyncMock(return_value=(fake_response, {"stage": "chat-complete", "session_id": "session-123"})),
        ):
            response = client.post(
                "/a2a/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": "video-chat-1",
                    "method": "SendMessage",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "session_id": "session-123",
                                            "chat_message": "Should I buy it for esports?",
                                        }
                                    ),
                                }
                            ],
                            "messageId": "msg-video-chat-1",
                        }
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["result"]["task"]["status"]["state"], "completed")
        self.assertEqual(
            payload["result"]["task"]["artifacts"][0]["metadata"]["schema"],
            "VideoChatResponse",
        )

    def test_readiness_endpoint_reports_degraded_after_runtime_issue(self):
        client = TestClient(review_server.app)
        review_server._READINESS_STATE["last_runtime_status"] = "degraded"
        review_server._READINESS_STATE["last_runtime_issue"] = "IpBlocked"

        try:
            with patch(
                "agents.youtube_review.server.get_settings",
                return_value=SimpleNamespace(
                    OPENAI_API_KEY="oa-key",
                    YOUTUBE_API_KEY="yt-key",
                    ENABLE_PINECONE=True,
                    PINECONE_API_KEY="pc-key",
                ),
            ):
                response = client.get("/readiness")
        finally:
            review_server._READINESS_STATE["last_runtime_status"] = "idle"
            review_server._READINESS_STATE["last_runtime_issue"] = None

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertIn("IpBlocked", payload["issues"][0])


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

    async def test_video_analysis_keeps_partial_result_when_similar_product_lookup_fails(self):
        settings = SimpleNamespace(
            YOUTUBE_API_KEY="yt-key",
            OPENAI_API_KEY="oa-key",
            OPENAI_MODEL="gpt-4o-mini",
            OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
            OPENAI_EMBEDDING_DIMENSIONS=512,
            ENABLE_PINECONE=False,
            YOUTUBE_TRANSCRIPT_CHUNK_SIZE=900,
            YOUTUBE_TRANSCRIPT_CHUNK_OVERLAP=120,
        )
        video = VideoMetadata(
            video_id="abc123",
            video_url="https://www.youtube.com/watch?v=abc123",
            title="Gaming Mouse Review",
            channel="Tech Lab",
            published_at="2026-03-01",
            view_count="1000",
        )
        extracted = ExtractedProductDetails(
            product_name="Razer Viper V3 Pro",
            category="gaming mouse",
            features=["wireless", "lightweight"],
            price="$159.99",
            summary="A strong competitive gaming mouse.",
            evidence_quotes=["Lightweight shell"],
            confidence=0.88,
        )

        with patch(
            "agents.youtube_review.video_analysis._fetch_video_metadata",
            return_value=video,
        ), patch(
            "agents.youtube_review.video_analysis._fetch_transcript_text",
            return_value="This review covers the Razer Viper V3 Pro and its lightweight wireless design.",
        ), patch(
            "agents.youtube_review.video_analysis.index_transcript",
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        chunks=[TranscriptChunk(chunk_index=0, text="Transcript chunk")],
                    ),
                    True,
                )
            ),
        ), patch(
            "agents.youtube_review.video_analysis._extract_product_details",
            new=AsyncMock(return_value=extracted),
        ), patch(
            "agents.youtube_review.video_analysis._call_product_discovery_mcp",
            new=AsyncMock(side_effect=RuntimeError("mcp unavailable")),
        ):
            result, debug = await video_analysis_agent.analyze_video_review(
                YouTubeVideoRequest(
                    youtube_url="https://www.youtube.com/watch?v=abc123",
                    find_similar_products=True,
                    product_mcp_url="http://localhost:5002/mcp",
                ),
                settings,
            )

        self.assertTrue(result.partial)
        self.assertIsNotNone(result.extracted_product)
        self.assertIsNone(result.similar_products)
        self.assertIn("Similar-product lookup failed", result.notes)
        self.assertEqual(debug["stage"], "completed")
        self.assertIn("similar_products_error", debug)

    async def test_video_analysis_creates_session_and_persists_initial_chat_history(self):
        settings = SimpleNamespace(
            YOUTUBE_API_KEY="yt-key",
            OPENAI_API_KEY="oa-key",
            OPENAI_MODEL="gpt-4o-mini",
            OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
            OPENAI_EMBEDDING_DIMENSIONS=512,
            ENABLE_PINECONE=False,
            YOUTUBE_TRANSCRIPT_CHUNK_SIZE=900,
            YOUTUBE_TRANSCRIPT_CHUNK_OVERLAP=120,
        )
        video = VideoMetadata(
            video_id="session-vid",
            video_url="https://www.youtube.com/watch?v=session-vid",
            title="Gaming Mouse Review",
            channel="Tech Lab",
            published_at="2026-03-01",
            view_count="1000",
        )
        extracted = ExtractedProductDetails(
            product_name="Razer Viper V3 Pro",
            category="gaming mouse",
            features=["wireless", "lightweight"],
            price="$159.99",
            summary="A strong competitive gaming mouse.",
            evidence_quotes=["Lightweight shell"],
            confidence=0.88,
        )
        history_response = VideoChatResponse(
            youtube_url=video.video_url,
            answer="The transcript says it is extremely light and competition-focused.",
            citations=["Chunk 0: It weighs almost nothing."],
            confidence=0.84,
        )

        with patch(
            "agents.youtube_review.video_analysis._fetch_video_metadata",
            return_value=video,
        ), patch(
            "agents.youtube_review.video_analysis._fetch_transcript_text",
            return_value="This review covers the Razer Viper V3 Pro and its lightweight wireless design.",
        ), patch(
            "agents.youtube_review.video_analysis.index_transcript",
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        transcript_hash="hash-123",
                        chunks=[TranscriptChunk(chunk_index=0, text="Transcript chunk")],
                    ),
                    True,
                )
            ),
        ), patch(
            "agents.youtube_review.video_analysis._extract_product_details",
            new=AsyncMock(return_value=extracted),
        ), patch(
            "agents.youtube_review.video_analysis.answer_chat_over_transcript",
            new=AsyncMock(return_value=history_response),
        ):
            result, _debug = await video_analysis_agent.analyze_video_review(
                YouTubeVideoRequest(
                    youtube_url=video.video_url,
                    chat_message="What stands out in this review?",
                ),
                settings,
            )

        try:
            self.assertIsNotNone(result.session_id)
            self.assertIsNotNone(result.chat_response)
            self.assertEqual(result.chat_response.session_id, result.session_id)
            self.assertEqual(len(result.chat_response.history), 2)
            session = review_sessions.SESSION_STORE.get(result.session_id)
            self.assertIsNotNone(session)
            self.assertEqual(len(session.history), 2)
            self.assertEqual(session.history[0].role, "user")
            self.assertEqual(session.history[1].role, "assistant")
        finally:
            if result.session_id:
                review_sessions.SESSION_STORE.pop(result.session_id)

    async def test_chat_with_video_session_uses_persisted_history(self):
        settings = SimpleNamespace(
            OPENAI_API_KEY="oa-key",
            OPENAI_MODEL="gpt-4o-mini",
            OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
            OPENAI_EMBEDDING_DIMENSIONS=512,
        )
        video = VideoMetadata(
            video_id="history-vid",
            video_url="https://www.youtube.com/watch?v=history-vid",
            title="Drawing Tablet Review",
            channel="Art Tech",
            published_at="2026-03-02",
            view_count="500",
        )
        extracted = ExtractedProductDetails(
            product_name="Wacom Cintiq 16",
            category="drawing tablet",
            features=["pen display"],
            price="$649.99",
            summary="Popular display tablet.",
            evidence_quotes=["The pen feels natural."],
            confidence=0.91,
        )
        session = review_sessions.SESSION_STORE.create(
            video=video,
            transcript_hash="hash-history",
            extracted_product=extracted,
        )
        review_sessions.SESSION_STORE.append_turn(session.session_id, role="user", content="What tablet is reviewed?")
        review_sessions.SESSION_STORE.append_turn(session.session_id, role="assistant", content="The video reviews the Wacom Cintiq 16.")
        response_model = VideoChatResponse(
            youtube_url=video.video_url,
            answer="The reviewer says the pen feels natural for sketching.",
            citations=["Chunk 2: The pen feels natural."],
            confidence=0.8,
        )

        try:
            with patch(
                "agents.youtube_review.video_analysis.answer_chat_over_transcript",
                new=AsyncMock(return_value=response_model),
            ) as chat_mock:
                response, _debug = await video_analysis_agent.chat_with_video_session(
                    SimpleNamespace(session_id=session.session_id, chat_message="What do they say about the pen?"),
                    settings,
                )

            history = chat_mock.await_args.kwargs["conversation_history"]
            self.assertEqual(len(history), 2)
            self.assertEqual(response.session_id, session.session_id)
            self.assertEqual(len(response.history), 4)
            self.assertEqual(response.history[-1].content, "The reviewer says the pen feels natural for sketching.")
        finally:
            review_sessions.SESSION_STORE.pop(session.session_id)

    async def test_video_analysis_refresh_reuses_session_and_cleans_prior_video_namespace(self):
        settings = SimpleNamespace(
            YOUTUBE_API_KEY="yt-key",
            OPENAI_API_KEY="oa-key",
            OPENAI_MODEL="gpt-4o-mini",
            OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
            OPENAI_EMBEDDING_DIMENSIONS=512,
            ENABLE_PINECONE=True,
            PINECONE_API_KEY="pc-key",
            YOUTUBE_TRANSCRIPT_CHUNK_SIZE=900,
            YOUTUBE_TRANSCRIPT_CHUNK_OVERLAP=120,
        )
        prior_video = VideoMetadata(
            video_id="prior-vid",
            video_url="https://www.youtube.com/watch?v=prior-vid",
            title="Prior Mouse Review",
            channel="Tech Lab",
            published_at="2026-03-01",
            view_count="1000",
        )
        refreshed_video = VideoMetadata(
            video_id="fresh-vid",
            video_url="https://www.youtube.com/watch?v=fresh-vid",
            title="Fresh Mouse Review",
            channel="Tech Lab",
            published_at="2026-03-02",
            view_count="1200",
        )
        extracted = ExtractedProductDetails(
            product_name="Razer Viper V3 Pro",
            category="gaming mouse",
            features=["wireless", "lightweight"],
            price="$159.99",
            summary="A strong competitive gaming mouse.",
            evidence_quotes=["Lightweight shell"],
            confidence=0.88,
        )
        prior_session = review_sessions.SESSION_STORE.create(
            video=prior_video,
            transcript_hash="hash-prior",
            extracted_product=extracted,
            session_id="session-refresh-1",
        )
        review_sessions.SESSION_STORE.append_turn(
            prior_session.session_id,
            role="user",
            content="What did the old review say?",
        )

        try:
            with patch(
                "agents.youtube_review.video_analysis._fetch_video_metadata",
                return_value=refreshed_video,
            ), patch(
                "agents.youtube_review.video_analysis._fetch_transcript_text",
                return_value="Fresh transcript for a different mouse review.",
            ), patch(
                "agents.youtube_review.video_analysis.index_transcript",
                new=AsyncMock(
                    return_value=(
                        SimpleNamespace(
                            transcript_hash="hash-fresh",
                            chunks=[TranscriptChunk(chunk_index=0, text="Fresh transcript chunk")],
                        ),
                        True,
                    )
                ),
            ), patch(
                "agents.youtube_review.video_analysis._extract_product_details",
                new=AsyncMock(return_value=extracted),
            ), patch(
                "agents.youtube_review.video_analysis.clear_transcript_index",
                new=AsyncMock(return_value=True),
            ) as clear_mock:
                result, debug = await video_analysis_agent.analyze_video_review(
                    YouTubeVideoRequest(
                        youtube_url=refreshed_video.video_url,
                        session_id=prior_session.session_id,
                    ),
                    settings,
                )

            clear_mock.assert_awaited_once_with(video_id="prior-vid", settings=settings)
            self.assertEqual(result.session_id, prior_session.session_id)
            refreshed_session = review_sessions.SESSION_STORE.get(prior_session.session_id)
            self.assertIsNotNone(refreshed_session)
            self.assertEqual(refreshed_session.video.video_id, "fresh-vid")
            self.assertEqual(refreshed_session.history, [])
            self.assertEqual(debug["refreshed_video_id"], "prior-vid")
        finally:
            review_sessions.SESSION_STORE.pop(prior_session.session_id)


if __name__ == "__main__":
    unittest.main()
