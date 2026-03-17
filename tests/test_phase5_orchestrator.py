import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class AsyncOpenAI:  # pragma: no cover - test stub for optional dependency
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

        async def close(self):
            return None

    openai_module.AsyncOpenAI = AsyncOpenAI
    sys.modules["openai"] = openai_module

if "langgraph.graph" not in sys.modules:
    langgraph_module = sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
    graph_module = types.ModuleType("langgraph.graph")

    class StateGraph:  # pragma: no cover - test stub for optional dependency
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, *args, **kwargs):
            return None

        def set_entry_point(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def compile(self):
            class _Compiled:
                async def ainvoke(self, state):
                    return state

            return _Compiled()

    graph_module.END = "END"
    graph_module.StateGraph = StateGraph
    langgraph_module.graph = graph_module
    sys.modules["langgraph.graph"] = graph_module

from agents.orchestrator import graph as orchestrator_graph
from agents.orchestrator.routing import route_query
from common.a2a_models import AgentCard
from common.a2a_models import (
    Artifact,
    Product,
    ProductDiscoveryResult,
    ReviewSummary,
    Task,
    TaskStatus,
    TextPart,
    UnifiedResponse,
    VideoSource,
    make_agent_message,
)


class OrchestratorHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_route_query_enables_reviews_for_product_lookups(self):
        decision = route_query(
            query="LG UltraGear GX9",
            discovered_agents={
                "product-discovery": AgentCard(
                    name="Product Discovery Agent",
                    description="Product agent",
                    url="http://localhost:5002/a2a/v1",
                    skills=[],
                ),
                "youtube-review": AgentCard(
                    name="YouTube Product Review Agent",
                    description="Review agent",
                    url="http://localhost:5001/a2a/v1",
                    skills=[],
                ),
            },
            card_urls={
                "product-discovery": "http://localhost:5002/.well-known/agent-card.json",
                "youtube-review": "http://localhost:5001/.well-known/agent-card.json",
            },
        )

        self.assertTrue(decision.needs_product_discovery)
        self.assertTrue(decision.needs_youtube_reviews)
        self.assertEqual(
            [route.agent_name for route in decision.routes],
            ["product-discovery", "youtube-review"],
        )

    async def test_resolve_send_message_result_polls_until_terminal(self):
        initial_task = Task(status=TaskStatus(state="working", message=make_agent_message("working")))
        completed_task = Task(
            id=initial_task.id,
            contextId=initial_task.contextId,
            status=TaskStatus(state="completed", message=make_agent_message("done")),
        )
        fake_client = SimpleNamespace(get_task=AsyncMock(return_value=completed_task))
        settings = SimpleNamespace(A2A_CLIENT_TIMEOUT_SECONDS=1)

        with patch("agents.orchestrator.graph.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            result = await orchestrator_graph._resolve_send_message_result(
                result=initial_task,
                client=fake_client,
                agent_url="http://agent.example/a2a/v1",
                settings=settings,
            )

        fake_client.get_task.assert_awaited_once_with(
            "http://agent.example/a2a/v1",
            initial_task.id,
        )
        sleep_mock.assert_awaited()
        self.assertEqual(result.status.state, "completed")

    def test_build_source_links_includes_evidence_urls_and_dedupes(self):
        product_result = ProductDiscoveryResult(
            query="best earbuds",
            products=[
                Product(
                    name="Sony WF-1000XM5",
                    price="$99.99",
                    rating="4.5/5",
                    url="https://merchant.example/sony",
                    key_features=["ANC"],
                    source="Example Merchant",
                    evidence_urls=[
                        "https://merchant.example/sony",
                        "https://reviews.example/sony-proof",
                    ],
                    confidence=0.9,
                )
            ],
            summary="One result",
        )
        review_results = {
            "Sony WF-1000XM5": ReviewSummary(
                product_name="Sony WF-1000XM5",
                overall_sentiment="positive",
                score=8.8,
                pros=["Great ANC"],
                cons=["Expensive"],
                key_quotes=[],
                recommendation="Solid pick.",
                confidence=0.85,
                sources=[
                    VideoSource(
                        title="Sony Review",
                        channel="Audio Lab",
                        url="https://youtube.com/watch?v=abc123",
                        published_at="2026-01-01",
                    )
                ],
            )
        }

        sources = orchestrator_graph._build_source_links(product_result, review_results)
        urls = [source.url for source in sources]

        self.assertEqual(
            urls,
            [
                "https://merchant.example/sony",
                "https://reviews.example/sony-proof",
                "https://youtube.com/watch?v=abc123",
            ],
        )

    def test_extract_named_debug_artifacts(self):
        task = Task(
            status=TaskStatus(state="working", message=make_agent_message("working")),
            artifacts=[
                Artifact(
                    name="product-discovery-debug",
                    parts=[TextPart(text='{"stage":"enrichment-complete","candidates":[{"url":"https://example.com"}]}')],
                ),
                Artifact(
                    name="review-debug",
                    parts=[TextPart(text='{"product_name":"Sony WF-1000XM5","stage":"summary-complete"}')],
                ),
            ],
        )

        product_trace = orchestrator_graph._extract_product_trace(task)
        review_trace = orchestrator_graph._extract_review_trace(task)

        self.assertEqual(product_trace["stage"], "enrichment-complete")
        self.assertEqual(review_trace["product_name"], "Sony WF-1000XM5")

    def test_fallback_synthesis_supports_review_only_results(self):
        review_results = {
            "Sony WF-1000XM5": ReviewSummary(
                product_name="Sony WF-1000XM5",
                overall_sentiment="positive",
                score=8.7,
                pros=["Strong ANC"],
                cons=["Pricey"],
                key_quotes=[],
                recommendation="Worth it for ANC-focused buyers.",
                confidence=0.81,
                sources=[
                    VideoSource(
                        title="Sony Review",
                        channel="Audio Lab",
                        url="https://youtube.com/watch?v=abc123",
                        published_at="2026-01-01",
                    )
                ],
            )
        }

        unified = orchestrator_graph._fallback_synthesis(
            query="sony xm5 review",
            product_result=None,
            review_results=review_results,
            errors=["LLM failed"],
        )

        self.assertEqual(len(unified.recommendations), 1)
        self.assertEqual(unified.recommendations[0].product_name, "Sony WF-1000XM5")
        self.assertEqual(unified.recommendations[0].sentiment, "positive")
        self.assertEqual(unified.sources[0].url, "https://youtube.com/watch?v=abc123")

    async def test_synthesize_does_not_mark_missing_reviews_when_not_requested(self):
        product_result = ProductDiscoveryResult(
            query="lg monitor",
            products=[
                Product(
                    name="LG UltraGear GX9",
                    price="$1,699",
                    rating=None,
                    url="https://merchant.example/lg",
                    key_features=["OLED"],
                    source="Example Merchant",
                    evidence_urls=["https://merchant.example/lg"],
                    confidence=0.88,
                )
            ],
            summary="One product",
        )
        llm_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"recommendations":[{"rank":1,"product_name":"LG UltraGear GX9",'
                            '"price":"$1,699","rating":null,"sentiment":null,"score":0.88,'
                            '"rationale":"Great monitor.","pros":["OLED"],"cons":[],"confidence":0.88}],'
                            '"notes":null}'
                        )
                    )
                )
            ]
        )
        fake_llm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=llm_response))
            )
        )

        result = await orchestrator_graph.synthesize(
            {
                "query": "lg monitor",
                "settings": SimpleNamespace(OPENAI_MODEL="gpt-4o-mini", OPENAI_API_KEY="oa-key"),
                "product_result": product_result,
                "review_results": {},
                "routing_decision": SimpleNamespace(
                    needs_product_discovery=True,
                    needs_youtube_reviews=False,
                ),
                "errors": [],
                "llm_client": fake_llm,
            }
        )

        unified = result["unified_response"]
        self.assertFalse(unified.partial)
        self.assertIsNone(unified.notes)


class RunQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_query_closes_owned_clients(self):
        closed = {"a2a": False, "llm": False}

        class FakeA2AClient:
            async def close(self):
                closed["a2a"] = True

        class FakeLLMClient:
            def __init__(self, *args, **kwargs):
                pass

            async def close(self):
                closed["llm"] = True

        class FakeCompiledGraph:
            async def ainvoke(self, state):
                return {
                    "unified_response": UnifiedResponse(
                        query=state["query"],
                        recommendations=[],
                        sources=[],
                        partial=False,
                        notes=None,
                    )
                }

        class FakeGraph:
            def compile(self):
                return FakeCompiledGraph()

        settings = SimpleNamespace(OPENAI_API_KEY="oa-key")

        with patch("agents.orchestrator.graph.A2AClient", return_value=FakeA2AClient()), patch(
            "agents.orchestrator.graph.AsyncOpenAI",
            FakeLLMClient,
        ), patch(
            "agents.orchestrator.graph.build_graph",
            return_value=FakeGraph(),
        ):
            response = await orchestrator_graph.run_query(
                user_query="best earbuds",
                settings=settings,
            )

        self.assertEqual(response.query, "best earbuds")
        self.assertTrue(closed["a2a"])
        self.assertTrue(closed["llm"])


if __name__ == "__main__":
    unittest.main()
