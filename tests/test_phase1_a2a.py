import asyncio
import threading
import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.a2a_client import A2AClient
from common.a2a_models import INVALID_REQUEST, Message, make_task
from common.a2a_server import InMemoryTaskStore, create_a2a_routes


def build_app(handler):
    app = FastAPI()
    agent_card = {
        "protocolVersion": "0.3.0",
        "name": "Test Agent",
        "description": "Test agent card",
        "url": "http://localhost:9999/a2a/v1",
        "skills": [],
    }
    app.include_router(create_a2a_routes(agent_card, handler, InMemoryTaskStore()))
    return app


class A2AServerTests(unittest.TestCase):
    def test_invalid_request_returns_error_for_non_object_json(self):
        async def handler(send_req):
            return make_task(send_req.message, state="completed")

        client = TestClient(build_app(handler))
        response = client.post("/a2a/v1", json=["not", "an", "object"])

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], INVALID_REQUEST)
        self.assertIn("must be an object", payload["error"]["message"])

    def test_invalid_request_returns_error_for_bad_envelope(self):
        async def handler(send_req):
            return make_task(send_req.message, state="completed")

        client = TestClient(build_app(handler))
        response = client.post(
            "/a2a/v1",
            json={"jsonrpc": "2.0", "id": "bad-1", "params": {}},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], INVALID_REQUEST)
        self.assertEqual(payload["id"], "bad-1")

    def test_invalid_request_rejects_non_2_0_jsonrpc_version(self):
        async def handler(send_req):
            return make_task(send_req.message, state="completed")

        client = TestClient(build_app(handler))
        response = client.post(
            "/a2a/v1",
            json={
                "jsonrpc": "1.0",
                "id": "bad-2",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hello"}],
                        "messageId": "msg-2",
                    }
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], INVALID_REQUEST)
        self.assertEqual(payload["id"], "bad-2")

    def test_get_task_can_observe_in_flight_task(self):
        ready = threading.Event()
        release = threading.Event()
        shared = {}

        async def handler(send_req, task_store, task):
            shared["task_id"] = task.id
            task_store.update_status(task.id, "working")
            ready.set()
            await asyncio.to_thread(release.wait, 2)
            return make_task(send_req.message, state="completed")

        app = build_app(handler)
        send_client = TestClient(app)
        poll_client = TestClient(app)

        send_payload = {
            "jsonrpc": "2.0",
            "id": "send-1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "hello"}],
                    "messageId": "msg-1",
                }
            },
        }
        response_holder = {}

        def do_send():
            response_holder["response"] = send_client.post("/a2a/v1", json=send_payload)

        worker = threading.Thread(target=do_send)
        worker.start()

        self.assertTrue(ready.wait(1), "handler did not reach working state")

        poll_response = poll_client.post(
            "/a2a/v1",
            json={
                "jsonrpc": "2.0",
                "id": "poll-1",
                "method": "GetTask",
                "params": {"taskId": shared["task_id"]},
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
        self.assertEqual(final_payload["result"]["task"]["id"], shared["task_id"])
        self.assertEqual(final_payload["result"]["task"]["status"]["state"], "completed")


class A2AClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_parses_message_only_response(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": "rpc-1",
                    "result": {
                        "message": {
                            "role": "agent",
                            "parts": [{"type": "text", "text": "pong"}],
                            "messageId": "msg-agent-1",
                        }
                    },
                },
            )

        client = A2AClient(timeout=5)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await client.send_message("http://testserver/a2a/v1", "ping")
        finally:
            await client.close()

        self.assertIsInstance(result, Message)
        self.assertEqual(result.parts[0].text, "pong")


if __name__ == "__main__":
    unittest.main()
