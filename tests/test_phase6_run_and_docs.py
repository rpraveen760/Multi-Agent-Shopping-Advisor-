import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.returncode = None
        self.signal_sent = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def send_signal(self, sig):
        self.signal_sent = sig
        self.returncode = 0


class RunScriptTests(unittest.TestCase):
    def setUp(self):
        run._processes.clear()
        run._shutting_down = False

    def tearDown(self):
        run._shutdown_all()

    def test_merge_pythonpath_preserves_existing_entries(self):
        with patch.dict(os.environ, {"PYTHONPATH": "existing-path"}, clear=False):
            merged = run._merge_pythonpath(Path("C:/project"))

        self.assertTrue(merged.endswith(f"{os.pathsep}existing-path"))
        self.assertIn("project", merged)

    def test_graceful_terminate_prefers_ctrl_break_on_windows(self):
        proc = _FakePopen()

        with patch.object(run.os, "name", "nt"), patch.object(
            run.signal,
            "CTRL_BREAK_EVENT",
            1234,
            create=True,
        ):
            run._graceful_terminate(proc)

        self.assertEqual(proc.signal_sent, 1234)
        self.assertFalse(proc.terminated)

    def test_start_service_tracks_log_handle_and_extends_pythonpath(self):
        service = {
            "name": "Test Service",
            "module": "agents.test.server:app",
            "port": 9999,
            "health": "http://localhost:9999/health",
        }

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            run,
            "PROJECT_ROOT",
            Path(tmpdir),
        ), patch.dict(
            os.environ,
            {"PYTHONPATH": "existing-path"},
            clear=False,
        ), patch.object(
            run.subprocess,
            "Popen",
            _FakePopen,
        ):
            proc = run._start_service(service)
            self.assertIsInstance(proc, _FakePopen)
            self.assertEqual(len(run._processes), 1)
            self.assertIn(str(Path(tmpdir)), proc.kwargs["env"]["PYTHONPATH"])
            self.assertIn("existing-path", proc.kwargs["env"]["PYTHONPATH"])
            self.assertFalse(run._processes[0].log_handle.closed)

            run._shutdown_all()

            self.assertEqual(run._processes, [])

    def test_validate_required_configuration_rejects_placeholder_keys(self):
        with patch.object(
            run,
            "get_settings",
            return_value=SimpleNamespace(
                OPENAI_API_KEY="sk-your-openai-key",
                YOUTUBE_API_KEY="your-youtube-data-api-key",
            ),
        ):
            ok, detail = run._validate_required_configuration()

        self.assertFalse(ok)
        self.assertIn("OPENAI_API_KEY", detail)
        self.assertIn("YOUTUBE_API_KEY", detail)

    def test_verify_orchestrator_status_requires_ok_payload(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"status": "degraded"},
        )

        with patch.object(run.httpx, "get", return_value=fake_response):
            ok, detail = run._verify_orchestrator_status()

        self.assertFalse(ok)
        self.assertIn("degraded", detail)


class DocumentationTests(unittest.TestCase):
    def test_readme_matches_phase6_contract(self):
        readme = Path(run.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("Copy-Item .env.example .env", readme)
        self.assertIn("curl.exe", readme)
        self.assertIn("../ARCHITECTURE.md", readme)
        self.assertIn("python scripts/test_mcp_stdio.py list", readme)
        self.assertIn("YouTube product review URL", readme)
        self.assertIn("/video/chat", readme)
        self.assertIn('"sources"', readme)
        self.assertIn("http://localhost:8000/", readme)
        self.assertIn("GET http://localhost:8000/status", readme)
        self.assertNotIn("Live Trace", readme)
        self.assertNotIn("cp .env.example .env", readme)

    def test_env_example_has_clear_required_placeholders(self):
        env_example = Path(run.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("# Required API keys", env_example)
        self.assertIn("OPENAI_API_KEY=sk-your-openai-key", env_example)
        self.assertIn("YOUTUBE_API_KEY=your-youtube-data-api-key", env_example)


if __name__ == "__main__":
    unittest.main()
