#!/usr/bin/env python3
"""Unified launcher for the Federated Multi-Agent System.

Starts all three services as managed subprocesses, waits for health checks,
and shuts them down cleanly on Ctrl+C.

Usage:
    python run.py
    python run.py --orchestrator-only
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import httpx

from common.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parent

SERVICES = [
    {
        "name": "YouTube Review Agent",
        "module": "agents.youtube_review.server:app",
        "port": 5001,
        "health": "http://localhost:5001/health",
    },
    {
        "name": "Product Discovery Agent",
        "module": "agents.product_discovery.server:app",
        "port": 5002,
        "health": "http://localhost:5002/health",
    },
    {
        "name": "Central Orchestrator",
        "module": "agents.orchestrator.server:app",
        "port": 8000,
        "health": "http://localhost:8000/health",
    },
]

HEALTH_CHECK_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 1


class _Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


C = _Colors()


@dataclass
class ManagedProcess:
    service_name: str
    process: subprocess.Popen
    log_handle: TextIO


_processes: list[ManagedProcess] = []
_shutting_down = False
_PLACEHOLDER_SECRETS = {
    "",
    "sk-your-openai-key",
    "your-youtube-data-api-key",
}


def _merge_pythonpath(project_root: Path) -> str:
    """Prepend the project root without discarding an existing PYTHONPATH."""
    existing = os.environ.get("PYTHONPATH")
    if not existing:
        return str(project_root)
    return os.pathsep.join([str(project_root), existing])


def _close_log_handle(log_handle: TextIO) -> None:
    """Best-effort close for a managed log file."""
    if log_handle.closed:
        return
    try:
        log_handle.flush()
    finally:
        log_handle.close()


def _graceful_terminate(proc: subprocess.Popen) -> None:
    """Request shutdown, preferring a console signal on Windows."""
    if proc.poll() is not None:
        return

    if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except Exception:
            pass

    proc.terminate()


def _shutdown_all() -> None:
    """Gracefully terminate all managed subprocesses."""
    global _shutting_down

    if _shutting_down:
        return
    _shutting_down = True

    try:
        for managed in reversed(_processes):
            _graceful_terminate(managed.process)

        deadline = time.time() + 5
        for managed in _processes:
            remaining = max(0, deadline - time.time())
            try:
                managed.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                managed.process.kill()
            finally:
                _close_log_handle(managed.log_handle)
    finally:
        _processes.clear()
        _shutting_down = False


def _signal_handler(_signum: int, _frame) -> None:
    """Handle Ctrl+C and SIGTERM."""
    print(f"\n{C.YELLOW}Shutting down services...{C.RESET}")
    _shutdown_all()
    sys.exit(0)


def _start_service(service: dict) -> subprocess.Popen:
    """Start one uvicorn service as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        service["module"],
        "--host",
        "0.0.0.0",
        "--port",
        str(service["port"]),
        "--log-level",
        "info",
    ]

    log_file = PROJECT_ROOT / ".logs" / f"{service['name'].lower().replace(' ', '_')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_handle = open(log_file, "w", encoding="utf-8", buffering=1)
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": _merge_pythonpath(PROJECT_ROOT)},
            creationflags=creationflags,
        )
    except Exception:
        _close_log_handle(log_handle)
        raise

    _processes.append(
        ManagedProcess(
            service_name=service["name"],
            process=proc,
            log_handle=log_handle,
        )
    )
    return proc


def _wait_for_health(service: dict) -> bool:
    """Poll a health endpoint until the service is ready."""
    deadline = time.time() + HEALTH_CHECK_TIMEOUT

    while time.time() < deadline:
        try:
            resp = httpx.get(service["health"], timeout=2)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            pass

        time.sleep(HEALTH_CHECK_INTERVAL)

    return False


def _validate_required_configuration() -> tuple[bool, str | None]:
    """Validate required credentials before claiming the system is usable."""
    settings = get_settings()
    missing: list[str] = []

    openai_key = settings.OPENAI_API_KEY.strip()
    youtube_key = settings.YOUTUBE_API_KEY.strip()

    if openai_key.lower() in _PLACEHOLDER_SECRETS:
        missing.append("OPENAI_API_KEY")
    if youtube_key.lower() in _PLACEHOLDER_SECRETS:
        missing.append("YOUTUBE_API_KEY")

    if missing:
        joined = ", ".join(missing)
        return False, f"Missing or placeholder API keys: {joined}"

    return True, None


def _verify_orchestrator_status() -> tuple[bool, str | None]:
    """Check the orchestrator's aggregated readiness before printing success."""
    try:
        resp = httpx.get("http://localhost:8000/status", timeout=5)
    except httpx.HTTPError as exc:
        return False, f"Failed to query orchestrator status: {exc}"

    if resp.status_code != 200:
        return False, f"Orchestrator status endpoint returned HTTP {resp.status_code}"

    try:
        payload = resp.json()
    except ValueError as exc:
        return False, f"Orchestrator status endpoint returned invalid JSON: {exc}"

    if payload.get("status") != "ok":
        return False, f"Orchestrator reported degraded readiness: {payload.get('status')}"

    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Federated Multi-Agent System")
    parser.add_argument(
        "--orchestrator-only",
        action="store_true",
        help="Start only the orchestrator (assumes agents are already running)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_shutdown_all)

    services = SERVICES
    if args.orchestrator_only:
        services = [service for service in SERVICES if service["port"] == 8000]

    print(f"\n{C.BOLD}{C.CYAN}Federated Multi-Agent System{C.RESET}")
    print(f"{C.CYAN}{'=' * 50}{C.RESET}\n")

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print(f"{C.RED}ERROR: .env file not found at {env_file}{C.RESET}")
        print("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    config_ok, config_error = _validate_required_configuration()
    if not config_ok:
        print(f"{C.RED}ERROR: {config_error}{C.RESET}")
        print("Update .env with real API keys before starting the stack.")
        sys.exit(1)

    for service in services:
        print(f"  Starting {C.BOLD}{service['name']}{C.RESET} on port {service['port']}...")
        proc = _start_service(service)
        if proc.poll() is not None:
            print(f"  {C.RED}FAILED to start {service['name']} (exit code {proc.returncode}){C.RESET}")
            _shutdown_all()
            sys.exit(1)

    print()

    all_healthy = True
    for service in services:
        print(f"  Waiting for {service['name']}...", end=" ", flush=True)
        if _wait_for_health(service):
            print(f"{C.GREEN}OK{C.RESET}")
        else:
            print(f"{C.RED}TIMEOUT{C.RESET}")
            all_healthy = False

    if not all_healthy:
        print(f"\n{C.RED}Some services failed to start. Check logs in .logs/{C.RESET}")
        _shutdown_all()
        sys.exit(1)

    status_ok, status_error = _verify_orchestrator_status()
    if not status_ok:
        print(f"\n{C.RED}Stack started in a degraded state: {status_error}{C.RESET}")
        print("Check logs in .logs/ and GET http://localhost:8000/status for details.")
        _shutdown_all()
        sys.exit(1)

    print(f"\n{C.GREEN}All services ready!{C.RESET}\n")
    print(f"  {C.BOLD}Web UI:{C.RESET}              http://localhost:8000/")
    print(f"  {C.BOLD}Orchestrator:{C.RESET}        http://localhost:8000/query")
    print(f"  {C.BOLD}Agent discovery:{C.RESET}     http://localhost:8000/agents")

    if not args.orchestrator_only:
        print(f"  {C.BOLD}YouTube Agent:{C.RESET}       http://localhost:5001/.well-known/agent-card.json")
        print(f"  {C.BOLD}Product Agent:{C.RESET}       http://localhost:5002/.well-known/agent-card.json")

    print(f"\n  {C.BOLD}Example:{C.RESET}")
    print(
        '    curl.exe -s -X POST http://localhost:8000/query '
        '-H "Content-Type: application/json" '
        '-d "{\\"query\\": \\"best wireless earbuds under 100\\"}"'
    )
    print(f"\n  {C.YELLOW}Press Ctrl+C to stop all services{C.RESET}\n")

    try:
        while True:
            for managed in list(_processes):
                if managed.process.poll() is not None:
                    print(
                        f"\n{C.RED}{managed.service_name} exited unexpectedly "
                        f"(code {managed.process.returncode}){C.RESET}"
                    )
                    print("Check logs in .logs/ for details.")
                    _shutdown_all()
                    sys.exit(1)
            time.sleep(2)
    except KeyboardInterrupt:
        _signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
