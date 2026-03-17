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
