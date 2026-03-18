import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class AsyncOpenAI:  # pragma: no cover - test stub for optional dependency
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

    openai_module.AsyncOpenAI = AsyncOpenAI
    sys.modules["openai"] = openai_module

if "mcp.server" not in sys.modules:
    mcp_module = sys.modules.setdefault("mcp", types.ModuleType("mcp"))
    server_module = types.ModuleType("mcp.server")
    stdio_module = types.ModuleType("mcp.server.stdio")
    types_module = types.ModuleType("mcp.types")

    class Server:  # pragma: no cover - test stub for optional dependency
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            def decorator(func):
                return func

            return decorator

        def call_tool(self):
            def decorator(func):
                return func

            return decorator

        def create_initialization_options(self):
            return {}

        async def run(self, *args, **kwargs):
            return None

    class TextContent:  # pragma: no cover - test stub for optional dependency
        def __init__(self, type, text):
            self.type = type
            self.text = text

    class Tool:  # pragma: no cover - test stub for optional dependency
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    async def _noop_async_context():
        yield (None, None)

    class _StdioContext:  # pragma: no cover - test stub for optional dependency
        async def __aenter__(self):
            return (None, None)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def stdio_server():
        return _StdioContext()

    server_module.Server = Server
    stdio_module.stdio_server = stdio_server
    types_module.TextContent = TextContent
    types_module.Tool = Tool
    mcp_module.server = server_module
    mcp_module.types = types_module
    sys.modules["mcp.server"] = server_module
    sys.modules["mcp.server.stdio"] = stdio_module
    sys.modules["mcp.types"] = types_module

from agents.product_discovery import agent as product_agent
from agents.product_discovery import mcp_server


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "test_mcp_stdio.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location("test_mcp_stdio_script", SCRIPT_PATH)
mcp_stdio_script = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(mcp_stdio_script)


class ProductDiscoveryAgentTests(unittest.IsolatedAsyncioTestCase):
    def _legacy_test_extract_price_supports_real_rupee_symbol(self):
        self.assertEqual(product_agent._extract_price("Deal price: ₹4,999 today"), "₹4,999")

    def _legacy_test_build_search_queries_adds_brand_hints_for_drawing_tablets(self):
        queries = product_agent._build_search_queries("best drawing tablet with screen for beginners")

        self.assertGreaterEqual(len(queries), 2)
        self.assertIn("Wacom", queries[-1])
        self.assertIn("XP-Pen", queries[-1])

    def test_extract_price_supports_real_rupee_symbol(self):
        self.assertEqual(product_agent._extract_price("Deal price: \u20b94,999 today"), "\u20b94,999")

    def test_build_search_queries_are_generic_and_category_agnostic(self):
        queries = product_agent._build_search_queries("best drawing tablet with screen for beginners")

        self.assertEqual(
            queries,
            [
                "best drawing tablet with screen for beginners best models price",
                "best drawing tablet with screen for beginners buy review",
                "best drawing tablet with screen for beginners top picks",
            ],
        )
        self.assertFalse(hasattr(product_agent, "_CATEGORY_SEARCH_HINTS"))

    async def test_normalize_products_filters_generic_category_names(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "Wacom Cintiq 16 pen display",
                "snippet": "Wacom Cintiq 16 now ₹49,999",
                "enriched_title": "Wacom Cintiq 16 product page",
                "url": "https://merchant.example/products/wacom-cintiq-16",
                "merchant": "Example Merchant",
            },
            {
                "title": "XP-Pen Artist 13.3 Pro display tablet",
                "snippet": "XP-Pen Artist 13.3 Pro pen display",
                "enriched_title": "XP-Pen Artist 13.3 Pro page",
                "url": "https://merchant.example/xppen-artist-13-pro",
                "merchant": "Example Merchant",
            },
        ]
        payload = {
            "products": [
                {
                    "name": "Drawing Tablet with Screen",
                    "price": None,
                    "rating": None,
                    "url": "https://merchant.example/generic-tablet",
                    "key_features": ["Screen"],
                    "source": "Example Merchant",
                    "confidence": 0.8,
                },
                {
                    "name": "Wacom Cintiq 16",
                    "price": "₹49,999",
                    "rating": None,
                    "url": "https://merchant.example/products/wacom-cintiq-16",
                    "key_features": ["15.6-inch pen display"],
                    "source": "Example Merchant",
                    "confidence": 0.92,
                },
            ],
            "summary": "Found one concrete pen display option.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=json.dumps(payload))
                        )
                    ]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient):
            result = await product_agent._normalize_products(
                "best drawing tablet with screen for beginners",
                candidates,
                settings,
                3,
            )

        self.assertEqual([product.name for product in result.products], ["Wacom Cintiq 16"])

    async def test_normalize_products_rejects_generic_budget_bucket_names(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "BestOffice Gaming Chair product page",
                "snippet": "BestOffice gaming chair now $129.99",
                "enriched_title": "BestOffice Gaming Chair product page",
                "url": "https://merchant.example/products/bestoffice-gaming-chair",
                "merchant": "Example Merchant",
                "candidate_entity": "BestOffice Gaming Chair",
                "shortlist_reasons": ["product page cues", "price detected"],
            },
            {
                "title": "Homall Gaming Chair product page",
                "snippet": "Homall gaming chair for budget gaming setups",
                "enriched_title": "Homall Gaming Chair product page",
                "url": "https://merchant.example/products/homall-gaming-chair",
                "merchant": "Example Merchant",
                "candidate_entity": "Homall Gaming Chair",
                "shortlist_reasons": ["product page cues"],
            },
        ]
        payload = {
            "products": [
                {
                    "name": "Gaming Chair Under $150",
                    "price": "$150",
                    "rating": None,
                    "url": "https://merchant.example/budget-gaming-chair-roundup",
                    "key_features": ["Budget pick"],
                    "source": "Example Merchant",
                    "confidence": 0.82,
                }
            ],
            "summary": "No concrete shortlist.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient):
            result, debug = await product_agent._normalize_products_with_debug(
                "best gaming chair under 150",
                candidates,
                settings,
                3,
            )

        self.assertEqual(result.products, [])
        self.assertEqual(debug["rejected_entities"][0]["name"], "Gaming Chair Under $150")

    async def test_normalize_products_accepts_repeated_non_model_entity_with_grounded_url(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "BestOffice Gaming Chair",
                "snippet": "BestOffice Gaming Chair now $129.99",
                "enriched_title": "BestOffice Gaming Chair product page",
                "url": "https://merchant.example/products/bestoffice-gaming-chair",
                "merchant": "Example Merchant",
                "candidate_entity": "BestOffice Gaming Chair",
                "shortlist_score": 4.2,
                "shortlist_reasons": ["product page cues", "price detected"],
                "matched_queries": ["best gaming chair under 150 best models price"],
            },
            {
                "title": "BestOffice Gaming Chair review",
                "snippet": "BestOffice Gaming Chair budget pick review",
                "enriched_title": "BestOffice Gaming Chair product page",
                "url": "https://merchant.example/products/bestoffice-gaming-chair?ref=review",
                "merchant": "Example Merchant",
                "candidate_entity": "BestOffice Gaming Chair",
                "shortlist_score": 3.9,
                "shortlist_reasons": ["cross-source entity support"],
                "matched_queries": ["best gaming chair under 150 buy review"],
            },
        ]
        payload = {
            "products": [
                {
                    "name": "BestOffice Gaming Chair",
                    "price": "$129.99",
                    "rating": "4.2/5",
                    "url": "https://merchant.example/products/bestoffice-gaming-chair",
                    "key_features": ["PU leather", "lumbar support"],
                    "source": "Example Merchant",
                    "confidence": 0.87,
                }
            ],
            "summary": "One concrete chair.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient):
            result = await product_agent._normalize_products(
                "best gaming chair under 150",
                candidates,
                settings,
                3,
            )

        self.assertEqual([product.name for product in result.products], ["BestOffice Gaming Chair"])
        self.assertEqual(result.products[0].url, "https://merchant.example/products/bestoffice-gaming-chair")

    def test_specific_product_with_review_snippet_still_counts_as_product_detail(self):
        candidate = {
            "title": "Razer Viper V3 Pro",
            "snippet": "Razer Viper V3 Pro review and specs",
            "enriched_title": "Razer Viper V3 Pro product page",
            "url": "https://merchant.example/products/razer-viper-v3-pro",
            "merchant": "Example Merchant",
            "candidate_entity": "Razer Viper V3 Pro",
        }

        self.assertTrue(product_agent._is_product_detail_candidate(candidate))
        self.assertTrue(
            product_agent._looks_like_specific_product_name(
                "Razer Viper V3 Pro",
                [candidate],
                "best gaming mouse",
            )
        )

    def test_extract_candidate_entity_name_prefers_real_product_over_roundup_label(self):
        candidate = {
            "title": "The Best Gaming Mouse of 2026: Mice Reviews - RTINGS.com",
            "snippet": "The Razer Viper V3 Pro is the best gaming mouse we've tested.",
            "enriched_title": "The Best Gaming Mouse of 2026: Mice Reviews - RTINGS.com",
            "url": "https://www.rtings.com/mouse/reviews/best/by-usage/gaming",
        }

        self.assertEqual(
            product_agent._extract_candidate_entity_name(candidate),
            "Razer Viper V3 Pro",
        )

    async def test_normalize_products_accepts_real_products_from_editorial_evidence(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "The Best Gaming Mouse of 2026",
                "snippet": "The Razer Basilisk V3 is the best mouse for gaming that we've tested.",
                "enriched_title": "The Best Gaming Mouse of 2026: Mice Reviews - RTINGS.com",
                "url": "https://www.rtings.com/mouse/reviews/best/by-usage/gaming",
                "merchant": "rtings.com",
                "candidate_entity": "Razer Basilisk V3",
                "shortlist_score": 3.9,
                "shortlist_reasons": ["specific product entity", "cross-source entity support"],
            },
            {
                "title": "Best Gaming Mouse for 2026 | Tom's Hardware",
                "snippet": "The Logitech G305 LIGHTSPEED remains a standout budget gaming mouse.",
                "enriched_title": "Best Gaming Mouse for 2026 | Tom's Hardware",
                "url": "https://www.tomshardware.com/best-picks/best-gaming-mouse",
                "merchant": "tomshardware.com",
                "candidate_entity": "Logitech G305 LIGHTSPEED",
                "shortlist_score": 3.7,
                "shortlist_reasons": ["specific product entity", "cross-source entity support"],
            },
        ]
        payload = {
            "products": [
                {
                    "name": "Razer Basilisk V3",
                    "price": None,
                    "rating": None,
                    "url": "https://hallucinated.example/basilisk",
                    "key_features": ["Ergonomic gaming mouse"],
                    "source": "RTINGS",
                    "confidence": 0.86,
                },
                {
                    "name": "Logitech G305 LIGHTSPEED",
                    "price": None,
                    "rating": None,
                    "url": "https://hallucinated.example/g305",
                    "key_features": ["Wireless budget pick"],
                    "source": "Tom's Hardware",
                    "confidence": 0.82,
                },
            ],
            "summary": "Two real gaming mouse entities were identified from roundup coverage.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient), patch.object(
            product_agent,
            "_resolve_product_detail_candidates",
            new=AsyncMock(return_value=[]),
        ):
            result, debug = await product_agent._normalize_products_with_debug(
                "best gaming mouse",
                candidates,
                settings,
                3,
            )

        self.assertEqual(
            [product.name for product in result.products],
            ["Razer Basilisk V3", "Logitech G305 LIGHTSPEED"],
        )
        self.assertTrue(
            any("grounded evidence page" in item["reasons"] for item in debug["shortlist_reasons"])
        )

    async def test_normalize_products_upgrades_to_resolved_detail_page_when_available(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "The Best Gaming Mouse of 2026",
                "snippet": "The Razer Viper V3 Pro is the best gaming mouse we've tested.",
                "enriched_title": "The Best Gaming Mouse of 2026: Mice Reviews - RTINGS.com",
                "url": "https://www.rtings.com/mouse/reviews/best/by-usage/gaming",
                "merchant": "rtings.com",
                "candidate_entity": "Razer Viper V3 Pro",
                "shortlist_score": 3.9,
                "shortlist_reasons": ["specific product entity", "cross-source entity support"],
            }
        ]
        resolved_detail_candidates = [
            {
                "title": "Razer Viper V3 Pro",
                "snippet": "Buy the Razer Viper V3 Pro today",
                "enriched_title": "Razer Viper V3 Pro product page",
                "url": "https://merchant.example/products/razer-viper-v3-pro",
                "merchant": "Example Merchant",
                "candidate_entity": "Razer Viper V3 Pro",
                "extracted_price": "$159.99",
            }
        ]
        payload = {
            "products": [
                {
                    "name": "Razer Viper V3 Pro",
                    "price": "$159.99",
                    "rating": None,
                    "url": "https://hallucinated.example/viper-v3-pro",
                    "key_features": ["Lightweight", "Pro-grade sensor"],
                    "source": "RTINGS",
                    "confidence": 0.9,
                }
            ],
            "summary": "One concrete gaming mouse.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient), patch.object(
            product_agent,
            "_resolve_product_detail_candidates",
            new=AsyncMock(return_value=resolved_detail_candidates),
        ):
            result, debug = await product_agent._normalize_products_with_debug(
                "best gaming mouse",
                candidates,
                settings,
                3,
            )

        self.assertEqual(
            result.products[0].url,
            "https://merchant.example/products/razer-viper-v3-pro",
        )
        self.assertEqual(result.products[0].source, "Example Merchant")
        self.assertTrue(
            any("resolved product detail page" in item["reasons"] for item in debug["shortlist_reasons"])
        )

    def test_filter_candidates_prioritizes_concrete_product_pages(self):
        candidates = [
            {
                "title": "Best drawing tablets for beginners in 2026",
                "snippet": "Review guide comparing top picks",
                "enriched_title": "Best drawing tablets buying guide",
                "url": "https://www.techradar.com/best-drawing-tablets",
                "merchant": "techradar",
                "page_text_excerpt": "Best drawing tablet review guide for beginners",
            },
            {
                "title": "Wacom Cintiq 16",
                "snippet": "Buy Wacom Cintiq 16 pen display today",
                "enriched_title": "Wacom Cintiq 16 product page",
                "url": "https://merchant.example/products/wacom-cintiq-16",
                "merchant": "Amazon India",
                "page_text_excerpt": "Wacom Cintiq 16 15.6-inch pen display with bundled pen",
                "extracted_price": "\u20b949,999",
            },
            {
                "title": "XP-Pen Artist 13.3 Pro",
                "snippet": "XP-Pen Artist 13.3 Pro pen display",
                "enriched_title": "XP-Pen Artist 13.3 Pro product page",
                "url": "https://merchant.example/products/xppen-artist-13-3-pro",
                "merchant": "Flipkart",
                "page_text_excerpt": "XP-Pen Artist 13.3 Pro pen display with laminated screen",
                "extracted_price": "\u20b925,999",
            },
        ]

        filtered, candidate_entities = product_agent._filter_candidates_for_shortlist(
            "best drawing tablet with screen for beginners",
            candidates,
            3,
        )

        self.assertEqual(
            [item["title"] for item in filtered[:2]],
            ["Wacom Cintiq 16", "XP-Pen Artist 13.3 Pro"],
        )
        self.assertTrue(
            all(
                item["shortlist_score"] >= product_agent.SHORTLIST_SCORE_THRESHOLD
                for item in filtered
            )
        )
        self.assertGreaterEqual(len(candidate_entities), 2)

    async def test_search_products_with_debug_reports_filtered_candidates(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        raw_candidates = [
            {
                "title": "Wacom Cintiq 16 product",
                "snippet": "Wacom Cintiq 16 deal",
                "url": "https://merchant.example/wacom-cintiq-16",
                "matched_query": "best drawing tablet with screen for beginners best models review price",
                "matched_queries": ["best drawing tablet with screen for beginners best models review price"],
            }
        ]
        enriched_candidates = [
            {
                "title": "Wacom Cintiq 16 product",
                "snippet": "Wacom Cintiq 16 deal",
                "url": "https://merchant.example/wacom-cintiq-16",
                "matched_query": "best drawing tablet with screen for beginners best models review price",
                "matched_queries": ["best drawing tablet with screen for beginners best models review price"],
                "merchant": "Amazon India",
                "enriched_title": "Wacom Cintiq 16 product page",
                "page_text_excerpt": "Wacom Cintiq 16 pen display for creators",
                "extracted_price": "\u20b949,999",
            }
        ]
        result = product_agent.ProductDiscoveryResult(
            query="best drawing tablet with screen for beginners",
            products=[
                product_agent.Product(
                    name="Wacom Cintiq 16",
                    price="\u20b949,999",
                    rating=None,
                    url="https://merchant.example/wacom-cintiq-16",
                    key_features=["15.6-inch pen display"],
                    source="Amazon India",
                    evidence_urls=["https://merchant.example/wacom-cintiq-16"],
                    confidence=0.93,
                )
            ],
            summary="One concrete pen display candidate.",
        )
        normalized_debug = {
            "normalized_products": [
                {
                    "name": "Wacom Cintiq 16",
                    "price": "\u20b949,999",
                    "rating": None,
                    "source": "Amazon India",
                    "url": "https://merchant.example/wacom-cintiq-16",
                    "confidence": 0.93,
                    "key_features": ["15.6-inch pen display"],
                    "evidence_urls": ["https://merchant.example/wacom-cintiq-16"],
                }
            ],
            "finalized_products": [
                {
                    "name": "Wacom Cintiq 16",
                    "price": "\u20b949,999",
                    "rating": None,
                    "source": "Amazon India",
                    "url": "https://merchant.example/wacom-cintiq-16",
                    "confidence": 0.93,
                    "key_features": ["15.6-inch pen display"],
                    "evidence_urls": ["https://merchant.example/wacom-cintiq-16"],
                }
            ],
            "shortlist_reasons": [
                {
                    "product_name": "Wacom Cintiq 16",
                    "reasons": ["model-like title", "price detected"],
                    "confidence": 0.93,
                    "evidence_urls": ["https://merchant.example/wacom-cintiq-16"],
                }
            ],
            "rejected_generic_products": [
                {
                    "name": "Drawing Tablet with Screen",
                    "reason": "generic-or-unreviewable product label",
                }
            ],
            "rejected_entities": [
                {
                    "name": "Drawing Tablet with Screen",
                    "reason": "generic-or-unreviewable product label",
                }
            ],
            "summary": "One concrete pen display candidate.",
        }

        with patch.object(product_agent, "_search_duckduckgo", return_value=raw_candidates), patch.object(
            product_agent,
            "_enrich_candidates",
            new=AsyncMock(return_value=enriched_candidates),
        ), patch.object(
            product_agent,
            "_normalize_products_with_debug",
            new=AsyncMock(return_value=(result, normalized_debug)),
        ):
            _, debug = await product_agent.search_products_with_debug(
                "best drawing tablet with screen for beginners",
                settings,
                3,
            )

        self.assertEqual(debug["candidate_count"], 1)
        self.assertEqual(len(debug["filtered_candidates"]), 1)
        self.assertEqual(debug["rejected_generic_products"][0]["name"], "Drawing Tablet with Screen")

    async def test_normalize_products_grounds_product_urls_to_candidates(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "Sony WF-1000XM5 earbuds",
                "snippet": "Sony WF-1000XM5 now $99.99",
                "enriched_title": "Sony WF-1000XM5 product page",
                "url": "https://merchant.example/sony-wf-1000xm5",
                "merchant": "Example Merchant",
            }
        ]
        payload = {
            "products": [
                {
                    "name": "Sony WF-1000XM5",
                    "price": "$99.99",
                    "rating": "4.5/5",
                    "url": "https://hallucinated.example/not-grounded",
                    "key_features": ["ANC", "LDAC"],
                    "source": "Example Merchant",
                    "confidence": 0.93,
                }
            ],
            "summary": "Found one strong candidate.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=json.dumps(payload))
                        )
                    ]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient):
            result = await product_agent._normalize_products(
                "best earbuds under 100",
                candidates,
                settings,
                5,
            )

        self.assertEqual(len(result.products), 1)
        self.assertEqual(
            result.products[0].url,
            "https://merchant.example/sony-wf-1000xm5",
        )
        self.assertEqual(
            result.products[0].evidence_urls,
            ["https://merchant.example/sony-wf-1000xm5"],
        )

    async def test_normalize_products_keeps_editorial_urls_out_of_canonical_product_url(self):
        settings = SimpleNamespace(OPENAI_API_KEY="oa-key", OPENAI_MODEL="gpt-4o-mini")
        candidates = [
            {
                "title": "BestOffice Gaming Chair review",
                "snippet": "Budget gaming chair roundup",
                "enriched_title": "Best gaming chair roundup",
                "url": "https://reviews.example/best-gaming-chairs",
                "merchant": "reviews.example",
                "candidate_entity": "BestOffice Gaming Chair",
                "shortlist_score": 2.8,
                "shortlist_reasons": ["editorial roundup"],
            },
            {
                "title": "BestOffice Gaming Chair",
                "snippet": "BestOffice Gaming Chair product page",
                "enriched_title": "BestOffice Gaming Chair product page",
                "url": "https://merchant.example/products/bestoffice-gaming-chair",
                "merchant": "Example Merchant",
                "candidate_entity": "BestOffice Gaming Chair",
                "shortlist_score": 4.0,
                "shortlist_reasons": ["product page cues"],
            },
        ]
        payload = {
            "products": [
                {
                    "name": "BestOffice Gaming Chair",
                    "price": "$129.99",
                    "rating": None,
                    "url": "https://reviews.example/best-gaming-chairs",
                    "key_features": ["Budget friendly"],
                    "source": "Example Merchant",
                    "confidence": 0.84,
                }
            ],
            "summary": "One chair candidate.",
        }

        class FakeCompletions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
                )

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.object(product_agent, "AsyncOpenAI", FakeClient):
            result = await product_agent._normalize_products(
                "best gaming chair under 150",
                candidates,
                settings,
                3,
            )

        self.assertEqual(result.products[0].url, "https://merchant.example/products/bestoffice-gaming-chair")

    def test_find_evidence_urls_does_not_match_everything_for_short_names(self):
        candidates = [
            {
                "title": "Laptop deal",
                "snippet": "Best sale on ultrabooks",
                "enriched_title": "Laptop page",
                "url": "https://merchant.example/laptop",
            },
            {
                "title": "Phone deal",
                "snippet": "Android sale",
                "enriched_title": "Phone page",
                "url": "https://merchant.example/phone",
            },
        ]

        self.assertEqual(product_agent._find_evidence_urls("TV", candidates), [])


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_tool_coerces_numeric_string_max_results(self):
        fake_result = SimpleNamespace(model_dump_json=lambda indent=2: '{"ok": true}')

        with patch("agents.product_discovery.mcp_server.get_settings", return_value=SimpleNamespace()), patch(
            "agents.product_discovery.mcp_server.search_products",
            new=AsyncMock(return_value=fake_result),
        ) as search_mock:
            result = await mcp_server.call_tool(
                "search_products",
                {"query": "best earbuds", "max_results": "3"},
            )

        search_mock.assert_awaited_once()
        self.assertEqual(search_mock.await_args.kwargs["max_results"], 3)
        self.assertEqual(result[0].text, '{"ok": true}')

    async def test_call_tool_rejects_out_of_range_max_results(self):
        with self.assertRaises(ValueError):
            await mcp_server.call_tool(
                "search_products",
                {"query": "best earbuds", "max_results": 0},
            )

    async def test_call_tool_supports_structured_similar_product_lookup(self):
        result = await mcp_server.call_tool(
            "find_similar_products",
            {
                "product_name": "Razer Viper V3 Pro",
                "category": "gaming mouse",
                "features": ["wireless", "lightweight", "esports"],
                "price": "$159.99",
                "source_video_url": "https://www.youtube.com/watch?v=abc123",
            },
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["request"]["product_name"], "Razer Viper V3 Pro")
        self.assertGreaterEqual(len(payload["products"]), 1)
        self.assertNotEqual(payload["products"][0]["name"], "Razer Viper V3 Pro")


class MCPStdioHelperTests(unittest.TestCase):
    def test_content_length_helpers_round_trip_messages(self):
        messages = [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        ]

        encoded = mcp_stdio_script._build_stdin_data(messages)
        decoded = list(mcp_stdio_script._iter_framed_messages(encoded))

        self.assertEqual(decoded, messages)


if __name__ == "__main__":
    unittest.main()
