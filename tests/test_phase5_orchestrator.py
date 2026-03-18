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
from agents.orchestrator.routing import QueryUnderstanding, route_query
from common.a2a_models import AgentCard, AgentCardInterface
from common.a2a_models import (
    Artifact,
    Product,
    ProductDiscoveryResult,
    ReviewSummary,
    Task,
    TaskStatus,
    TextPart,
    UnifiedResponse,
    UnifiedVideoAnalysisResponse,
    VideoSource,
    VideoMetadata,
    make_agent_message,
)


class OrchestratorHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_youtube_video_analysis_passes_product_mcp_url(self):
        video_result = UnifiedVideoAnalysisResponse(
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
        task = Task(
            status=TaskStatus(state="completed", message=make_agent_message("done")),
            artifacts=[
                Artifact(
                    name="video-analysis-result",
                    parts=[TextPart(text=video_result.model_dump_json())],
                )
            ],
        )
        fake_client = SimpleNamespace(send_message=AsyncMock(return_value=task), get_task=AsyncMock())
        state = {
            "query": "https://www.youtube.com/watch?v=abc123",
            "youtube_url": "https://www.youtube.com/watch?v=abc123",
            "session_id": "session-123",
            "chat_message": "What product is being reviewed?",
            "find_similar_products": True,
            "settings": SimpleNamespace(A2A_CLIENT_TIMEOUT_SECONDS=1),
            "routing_decision": SimpleNamespace(
                routes=[
                    SimpleNamespace(
                        agent_name="youtube-review",
                        agent_url="http://localhost:5001/a2a/v1",
                    )
                ]
            ),
            "discovered_agents": {
                "product-discovery": AgentCard(
                    name="Product Discovery Agent",
                    description="Product agent",
                    url="http://localhost:5002/a2a/v1",
                    additionalInterfaces=[
                        AgentCardInterface(url="http://localhost:5002/mcp", transport="MCP")
                    ],
                    skills=[],
                )
            },
            "a2a_client": fake_client,
            "errors": [],
        }

        result = await orchestrator_graph.execute_youtube_video_analysis(state)

        self.assertEqual(result["video_analysis"].youtube_url, "https://www.youtube.com/watch?v=abc123")
        fake_client.send_message.assert_awaited_once()
        _, payload = fake_client.send_message.await_args.args[:2]
        self.assertIn("\"youtube_url\": \"https://www.youtube.com/watch?v=abc123\"", payload)
        self.assertIn("\"session_id\": \"session-123\"", payload)
        self.assertEqual(
            fake_client.send_message.await_args.kwargs["metadata"]["product_mcp_url"],
            "http://localhost:5002/mcp",
        )

    async def test_synthesize_video_response_preserves_session_id(self):
        video_result = UnifiedVideoAnalysisResponse(
            youtube_url="https://www.youtube.com/watch?v=abc123",
            session_id="session-123",
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

        result = await orchestrator_graph.synthesize(
            {
                "query": video_result.youtube_url,
                "settings": SimpleNamespace(OPENAI_MODEL="gpt-4o-mini", OPENAI_API_KEY="oa-key"),
                "product_result": None,
                "review_results": {},
                "video_analysis": video_result,
                "routing_decision": None,
                "errors": [],
            }
        )

        self.assertEqual(result["unified_response"].session_id, "session-123")

    async def test_route_query_enables_reviews_for_product_lookups(self):
        llm_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"needs_product_discovery":true,"needs_youtube_reviews":false,"reasoning":"Product data should be enough."}'
                    )
                )
            ]
        )
        fake_llm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=llm_response))
            )
        )

        decision = await route_query(
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
            settings=SimpleNamespace(OPENAI_MODEL="gpt-4o-mini"),
            llm_client=fake_llm,
            query_understanding=QueryUnderstanding(
                original_query="LG UltraGear GX9",
                reformulated_query="LG UltraGear GX9 gaming monitor",
                product_category="monitor",
                budget=None,
                intent="product_search",
                key_terms=["LG UltraGear GX9"],
                reasoning="Concrete product lookup.",
            ),
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

    async def test_execute_product_discovery_prefers_mcp(self):
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
                    evidence_urls=["https://merchant.example/sony"],
                    confidence=0.91,
                )
            ],
            summary="One result",
        )
        fake_client = SimpleNamespace(send_message=AsyncMock(), get_task=AsyncMock())
        state = {
            "settings": SimpleNamespace(A2A_CLIENT_TIMEOUT_SECONDS=1),
            "routing_decision": SimpleNamespace(
                needs_product_discovery=True,
                routes=[
                    SimpleNamespace(
                        agent_name="product-discovery",
                        agent_url="http://localhost:5002/a2a/v1",
                        query_text="best earbuds",
                    )
                ],
            ),
            "discovered_agents": {
                "product-discovery": AgentCard(
                    name="Product Discovery Agent",
                    description="Product agent",
                    url="http://localhost:5002/a2a/v1",
                    additionalInterfaces=[
                        AgentCardInterface(
                            url="http://localhost:5002/mcp",
                            transport="MCP",
                        )
                    ],
                    skills=[],
                )
            },
            "a2a_client": fake_client,
            "errors": [],
        }

        with patch(
            "agents.orchestrator.graph._call_product_discovery_via_mcp",
            new=AsyncMock(return_value=product_result),
        ) as mcp_mock:
            result = await orchestrator_graph.execute_product_discovery(state)

        mcp_mock.assert_awaited_once()
        fake_client.send_message.assert_not_called()
        self.assertIs(result["product_result"], product_result)

    async def test_call_product_discovery_via_mcp_requests_three_results(self):
        call_arguments = {}

        class FakeToolResult:
            isError = False
            structuredContent = {
                "query": "best earbuds",
                "products": [],
                "summary": "No products",
            }
            content = []

        class FakeSession:
            def __init__(self, *args, **kwargs):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                return SimpleNamespace(tools=[SimpleNamespace(name="search_products")])

            async def call_tool(self, name, arguments):
                call_arguments["name"] = name
                call_arguments["arguments"] = arguments
                return FakeToolResult()

        class FakeHTTPClient:
            async def __aenter__(self):
                return (object(), object(), lambda: "session-id")

            async def __aexit__(self, exc_type, exc, tb):
                return False

        fake_mcp_module = types.ModuleType("mcp")
        fake_mcp_module.ClientSession = FakeSession
        fake_mcp_client_module = types.ModuleType("mcp.client")
        fake_mcp_streamable_module = types.ModuleType("mcp.client.streamable_http")
        fake_mcp_streamable_module.streamable_http_client = lambda url: FakeHTTPClient()

        with patch.dict(
            sys.modules,
            {
                "mcp": fake_mcp_module,
                "mcp.client": fake_mcp_client_module,
                "mcp.client.streamable_http": fake_mcp_streamable_module,
            },
        ):
            result = await orchestrator_graph._call_product_discovery_via_mcp(
                query="best earbuds",
                settings=SimpleNamespace(A2A_CLIENT_TIMEOUT_SECONDS=1),
                mcp_url="http://localhost:5002/mcp",
            )

        self.assertEqual(call_arguments["name"], "search_products")
        self.assertEqual(call_arguments["arguments"]["max_results"], 3)
        self.assertEqual(result.query, "best earbuds")

    async def test_execute_product_discovery_falls_back_to_a2a(self):
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
                    evidence_urls=["https://merchant.example/sony"],
                    confidence=0.91,
                )
            ],
            summary="One result",
        )
        task = Task(
            status=TaskStatus(state="completed", message=make_agent_message("done")),
            artifacts=[
                Artifact(
                    name="product-discovery-result",
                    parts=[TextPart(text=product_result.model_dump_json())],
                )
            ],
        )
        fake_client = SimpleNamespace(
            send_message=AsyncMock(return_value=task),
            get_task=AsyncMock(),
        )
        state = {
            "settings": SimpleNamespace(A2A_CLIENT_TIMEOUT_SECONDS=1),
            "routing_decision": SimpleNamespace(
                needs_product_discovery=True,
                routes=[
                    SimpleNamespace(
                        agent_name="product-discovery",
                        agent_url="http://localhost:5002/a2a/v1",
                        query_text="best earbuds",
                    )
                ],
            ),
            "discovered_agents": {
                "product-discovery": AgentCard(
                    name="Product Discovery Agent",
                    description="Product agent",
                    url="http://localhost:5002/a2a/v1",
                    additionalInterfaces=[
                        AgentCardInterface(
                            url="http://localhost:5002/mcp",
                            transport="MCP",
                        )
                    ],
                    skills=[],
                )
            },
            "a2a_client": fake_client,
            "errors": [],
        }

        with patch(
            "agents.orchestrator.graph._call_product_discovery_via_mcp",
            new=AsyncMock(side_effect=RuntimeError("mcp boom")),
        ):
            result = await orchestrator_graph.execute_product_discovery(state)

        fake_client.send_message.assert_awaited_once_with(
            "http://localhost:5002/a2a/v1",
            "best earbuds",
        )
        self.assertEqual(result["product_result"].products[0].name, "Sony WF-1000XM5")
        self.assertIn("falling back to A2A", result["errors"][0])

    def test_resolve_product_mcp_url_uses_http_interface_from_agent_card(self):
        agent_card = AgentCard(
            name="Product Discovery Agent",
            description="Product agent",
            url="http://localhost:5002/a2a/v1",
            additionalInterfaces=[
                AgentCardInterface(url="http://localhost:5002/a2a/v1", transport="JSONRPC"),
                AgentCardInterface(url="http://localhost:5002/mcp", transport="MCP"),
            ],
            skills=[],
        )

        url = orchestrator_graph._resolve_product_mcp_url(agent_card)

        self.assertEqual(url, "http://localhost:5002/mcp")

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

    def test_select_review_targets_limits_to_top_three_products(self):
        product_result = ProductDiscoveryResult(
            query="best drawing tablet with screen",
            products=[
                Product(
                    name="XP-Pen Artist 13.3 Pro",
                    url="https://merchant.example/xppen",
                    source="Example Merchant",
                    confidence=0.85,
                    evidence_urls=["https://merchant.example/xppen"],
                ),
                Product(
                    name="Wacom Cintiq 16",
                    url="https://merchant.example/wacom",
                    source="Example Merchant",
                    confidence=0.95,
                    evidence_urls=["https://merchant.example/wacom", "https://reviews.example/wacom"],
                ),
                Product(
                    name="Huion Kamvas 13",
                    url="https://merchant.example/huion",
                    source="Example Merchant",
                    confidence=0.9,
                    evidence_urls=["https://merchant.example/huion"],
                ),
                Product(
                    name="Gaomon PD1161",
                    url="https://merchant.example/gaomon",
                    source="Example Merchant",
                    confidence=0.8,
                    evidence_urls=["https://merchant.example/gaomon"],
                ),
            ],
            summary="Three good candidates and one weaker fallback.",
        )

        targets = orchestrator_graph._select_review_targets(product_result)

        self.assertEqual(
            targets,
            ["Wacom Cintiq 16", "Huion Kamvas 13", "XP-Pen Artist 13.3 Pro"],
        )

    async def test_execute_youtube_reviews_skips_when_no_finalized_products(self):
        fake_client = SimpleNamespace(
            send_message=AsyncMock(),
            get_task=AsyncMock(),
        )
        state = {
            "query": "best gaming chair under 150",
            "settings": SimpleNamespace(
                YOUTUBE_REVIEW_CONCURRENCY=2,
                A2A_CLIENT_TIMEOUT_SECONDS=1,
            ),
            "routing_decision": SimpleNamespace(
                needs_youtube_reviews=True,
                routes=[
                    SimpleNamespace(
                        agent_name="youtube-review",
                        agent_url="http://localhost:5001/a2a/v1",
                    )
                ],
            ),
            "product_result": ProductDiscoveryResult(
                query="best gaming chair under 150",
                products=[],
                summary="No concrete shortlist could be finalized.",
            ),
            "a2a_client": fake_client,
            "errors": [],
        }

        result = await orchestrator_graph.execute_youtube_reviews(state)

        self.assertEqual(result["review_results"], {})
        fake_client.send_message.assert_not_called()

    async def test_execute_youtube_reviews_reports_nonempty_failure_detail(self):
        fake_client = SimpleNamespace(
            send_message=AsyncMock(side_effect=RuntimeError()),
            get_task=AsyncMock(),
        )
        state = {
            "query": "gaming mouse",
            "settings": SimpleNamespace(
                YOUTUBE_REVIEW_CONCURRENCY=2,
                A2A_CLIENT_TIMEOUT_SECONDS=1,
            ),
            "routing_decision": SimpleNamespace(
                needs_youtube_reviews=True,
                routes=[
                    SimpleNamespace(
                        agent_name="youtube-review",
                        agent_url="http://localhost:5001/a2a/v1",
                    )
                ],
            ),
            "product_result": ProductDiscoveryResult(
                query="gaming mouse",
                products=[
                    Product(
                        name="Razer Viper V3 Pro",
                        url="https://merchant.example/razer-viper-v3-pro",
                        source="Example Merchant",
                        confidence=0.95,
                        evidence_urls=["https://merchant.example/razer-viper-v3-pro"],
                    )
                ],
                summary="One concrete gaming mouse.",
            ),
            "a2a_client": fake_client,
            "errors": [],
        }

        result = await orchestrator_graph.execute_youtube_reviews(state)

        self.assertEqual(result["review_results"], {})
        self.assertIn("RuntimeError", result["errors"][0])

    def test_should_fetch_reviews_skips_when_no_finalized_products(self):
        state = {
            "routing_decision": SimpleNamespace(needs_youtube_reviews=True),
            "product_result": ProductDiscoveryResult(
                query="best gaming chair under 150",
                products=[],
                summary="No shortlist",
            ),
        }

        self.assertEqual(orchestrator_graph._should_fetch_reviews(state), "synthesize")

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

    async def test_synthesize_matches_recommendations_back_to_discovered_products(self):
        product_result = ProductDiscoveryResult(
            query="best drawing tablet with screen",
            products=[
                Product(
                    name="Wacom Cintiq 16",
                    price="₹49,999",
                    rating=None,
                    url="https://merchant.example/wacom",
                    key_features=["15.6-inch pen display"],
                    source="Example Merchant",
                    evidence_urls=["https://merchant.example/wacom"],
                    confidence=0.93,
                )
            ],
            summary="One strong candidate",
        )
        llm_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"recommendations":[{"rank":1,"product_name":"Cintiq 16",'
                            '"price":"₹49,999","rating":null,"sentiment":"positive","score":0.91,'
                            '"rationale":"Strong beginner pen display.","pros":["Great drawing feel"],'
                            '"cons":["Expensive"],"confidence":0.9}],"notes":null}'
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
                "query": "best drawing tablet with screen",
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
        self.assertEqual(len(unified.recommendations), 1)
        self.assertEqual(unified.recommendations[0].product_name, "Wacom Cintiq 16")

    async def test_synthesize_returns_partial_when_no_concrete_shortlist_is_finalized(self):
        result = await orchestrator_graph.synthesize(
            {
                "query": "best gaming chair under 150",
                "settings": SimpleNamespace(OPENAI_MODEL="gpt-4o-mini", OPENAI_API_KEY="oa-key"),
                "product_result": ProductDiscoveryResult(
                    query="best gaming chair under 150",
                    products=[],
                    summary="No concrete shortlist could be finalized.",
                ),
                "review_results": {},
                "routing_decision": SimpleNamespace(
                    needs_product_discovery=True,
                    needs_youtube_reviews=True,
                ),
                "errors": [],
            }
        )

        unified = result["unified_response"]
        self.assertTrue(unified.partial)
        self.assertEqual(unified.recommendations, [])
        self.assertIn("No concrete products could be finalized", unified.notes)


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
