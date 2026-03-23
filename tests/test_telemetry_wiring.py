"""Phase 0 – Verify automatic telemetry wiring for all tools,
percentile calculations, lock-wait metrics, and stat accumulation.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


@pytest.fixture(autouse=True)
def _enable_telemetry_and_reset():
    """Enable telemetry and reset stats for every test in this module."""
    import mathematica_mcp.telemetry as mod

    original_features = mod.FEATURES
    mod.FEATURES = SimpleNamespace(telemetry=True)
    mod.reset_stats()
    yield
    mod.FEATURES = original_features


# ---------------------------------------------------------------------------
# Telemetry decorator unit tests
# ---------------------------------------------------------------------------


class TestTelemetryTool:
    """Test the telemetry_tool decorator in isolation."""

    def test_records_timing_on_success(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_usage_stats

        @telemetry_tool("test_success")
        async def my_tool():
            return "ok"

        asyncio.run(my_tool())
        stats = get_usage_stats()
        assert "test_success" in stats
        assert stats["test_success"]["calls"] == 1
        assert stats["test_success"]["errors"] == 0
        assert stats["test_success"]["total_time_ms"] >= 0

    def test_records_timing_on_error(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_usage_stats

        @telemetry_tool("test_error")
        async def my_tool():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            asyncio.run(my_tool())

        stats = get_usage_stats()
        assert stats["test_error"]["calls"] == 1
        assert stats["test_error"]["errors"] == 1

    def test_accumulates_multiple_calls(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_usage_stats

        @telemetry_tool("test_multi")
        async def my_tool():
            return "ok"

        for _ in range(5):
            asyncio.run(my_tool())

        stats = get_usage_stats()
        assert stats["test_multi"]["calls"] == 5
        assert stats["test_multi"]["errors"] == 0
        assert stats["test_multi"]["total_time_ms"] >= 0

    def test_percentiles_present_after_calls(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_usage_stats

        @telemetry_tool("test_pct")
        async def my_tool():
            return "ok"

        for _ in range(10):
            asyncio.run(my_tool())

        stats = get_usage_stats()
        entry = stats["test_pct"]
        assert "p50_ms" in entry
        assert "p95_ms" in entry
        assert "min_ms" in entry
        assert "max_ms" in entry
        assert entry["p50_ms"] >= entry["min_ms"]
        assert entry["p95_ms"] >= entry["p50_ms"]
        assert entry["max_ms"] >= entry["p95_ms"]

    def test_reset_clears_everything(self):
        from mathematica_mcp.telemetry import (
            telemetry_tool,
            get_usage_stats,
            reset_stats,
        )

        @telemetry_tool("test_reset")
        async def my_tool():
            return "ok"

        asyncio.run(my_tool())
        assert "test_reset" in get_usage_stats()

        reset_stats()
        assert get_usage_stats() == {}

    def test_marker_attributes_set(self):
        from mathematica_mcp.telemetry import telemetry_tool

        @telemetry_tool("test_marker")
        async def my_tool():
            return "ok"

        assert getattr(my_tool, "_telemetry_instrumented", False) is True
        assert getattr(my_tool, "_telemetry_name", None) == "test_marker"

    def test_instrumented_tools_tracking(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_instrumented_tools

        @telemetry_tool("test_tracking_a")
        async def tool_a():
            return "a"

        @telemetry_tool("test_tracking_b")
        async def tool_b():
            return "b"

        instrumented = get_instrumented_tools()
        assert "test_tracking_a" in instrumented
        assert "test_tracking_b" in instrumented

    def test_functools_wraps_preserves_name(self):
        from mathematica_mcp.telemetry import telemetry_tool

        @telemetry_tool("original_name")
        async def my_specific_tool():
            """My docstring."""
            return "ok"

        assert my_specific_tool.__name__ == "my_specific_tool"
        assert my_specific_tool.__doc__ == "My docstring."


# ---------------------------------------------------------------------------
# Percentile calculation tests
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list_returns_zero(self):
        from mathematica_mcp.telemetry import _percentile

        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        from mathematica_mcp.telemetry import _percentile

        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 95) == 42.0

    def test_even_distribution(self):
        from mathematica_mcp.telemetry import _percentile

        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = _percentile(values, 50)
        assert p50 == 30.0

    def test_p95_near_max(self):
        from mathematica_mcp.telemetry import _percentile

        values = list(range(1, 101))  # 1..100
        p95 = _percentile(values, 95)
        assert p95 >= 95

    def test_unsorted_input_handled(self):
        from mathematica_mcp.telemetry import _percentile

        values = [50.0, 10.0, 30.0, 20.0, 40.0]
        assert _percentile(values, 50) == 30.0


# ---------------------------------------------------------------------------
# Connection lock metrics tests
# ---------------------------------------------------------------------------


class TestConnectionLockMetrics:
    def test_lock_metrics_on_successful_command(self):
        from mathematica_mcp.connection import MathematicaConnection

        conn = MathematicaConnection()
        # Simulate a socket that returns a valid JSON frame.
        response = json.dumps({"status": "ok", "result": {"pong": True}}) + "\n"

        class FakeSocket:
            def getpeername(self):
                return ("localhost", 9881)

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return response.encode("utf-8")

        conn._socket = FakeSocket()  # type: ignore[assignment]

        result = conn.send_command("ping")
        assert result == {"pong": True}

        metrics = conn.get_lock_metrics()
        assert metrics["acquisitions"] == 1
        assert metrics["wait_p50_ms"] >= 0
        assert metrics["hold_p50_ms"] >= 0

    def test_lock_metrics_accumulate(self):
        from mathematica_mcp.connection import MathematicaConnection

        conn = MathematicaConnection()
        response = json.dumps({"status": "ok", "result": {"ok": True}}) + "\n"

        class FakeSocket:
            def getpeername(self):
                return ("localhost", 9881)

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return response.encode("utf-8")

        conn._socket = FakeSocket()  # type: ignore[assignment]

        for _ in range(5):
            conn.send_command("ping")

        metrics = conn.get_lock_metrics()
        assert metrics["acquisitions"] == 5
        assert len(conn._lock_wait_times) == 5
        assert len(conn._lock_hold_times) == 5

    def test_lock_metrics_on_error(self):
        from mathematica_mcp.connection import MathematicaConnection

        conn = MathematicaConnection()

        class TimeoutSocket:
            def getpeername(self):
                return ("localhost", 9881)

            def sendall(self, data):
                import socket

                raise socket.timeout("timed out")

            def recv(self, bufsize):
                return b""

            def close(self):
                pass

        conn._socket = TimeoutSocket()  # type: ignore[assignment]

        with pytest.raises(TimeoutError):
            conn.send_command("slow_command")

        metrics = conn.get_lock_metrics()
        assert metrics["acquisitions"] == 1
        # Hold time should still be recorded (via finally block).
        assert len(conn._lock_hold_times) == 1

    def test_lock_metrics_capped(self):
        from mathematica_mcp.connection import MathematicaConnection

        conn = MathematicaConnection()
        response = json.dumps({"status": "ok", "result": {}}) + "\n"

        class FakeSocket:
            def getpeername(self):
                return ("localhost", 9881)

            def sendall(self, data):
                pass

            def recv(self, bufsize):
                return response.encode("utf-8")

        conn._socket = FakeSocket()  # type: ignore[assignment]

        for _ in range(250):
            conn.send_command("ping")

        assert len(conn._lock_wait_times) <= conn._MAX_METRIC_SAMPLES
        assert len(conn._lock_hold_times) <= conn._MAX_METRIC_SAMPLES
        assert conn._lock_acquisitions == 250


# ---------------------------------------------------------------------------
# All tools are instrumented (subprocess-based integration test)
# ---------------------------------------------------------------------------

# Script that lists all registered tool names AND all instrumented tool names
# in a single subprocess to get an isolated import.
_CHECK_SCRIPT = """
import asyncio, json
from mathematica_mcp.server import mcp
from mathematica_mcp.telemetry import get_instrumented_tools

tools = asyncio.run(mcp.list_tools())
registered = sorted(tool.name for tool in tools)
instrumented = sorted(get_instrumented_tools())
print(json.dumps({"registered": registered, "instrumented": instrumented}))
"""


def _run_check(env_overrides: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    for key in list(env):
        if key == "MATHEMATICA_PROFILE" or key.startswith("MATHEMATICA_ENABLE_"):
            env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return json.loads(result.stdout)


def test_all_full_profile_tools_instrumented():
    """Every tool registered under the full profile must be telemetry-instrumented."""
    data = _run_check({"MATHEMATICA_ENABLE_TELEMETRY": "true"})
    registered = set(data["registered"])
    instrumented = set(data["instrumented"])

    missing = registered - instrumented
    assert not missing, (
        f"Tools registered but NOT instrumented: {sorted(missing)}"
    )


def test_all_math_profile_tools_instrumented():
    """Every tool in the math profile must be instrumented."""
    data = _run_check({"MATHEMATICA_PROFILE": "math", "MATHEMATICA_ENABLE_TELEMETRY": "true"})
    registered = set(data["registered"])
    instrumented = set(data["instrumented"])

    missing = registered - instrumented
    assert not missing, (
        f"Math-profile tools registered but NOT instrumented: {sorted(missing)}"
    )


def test_all_notebook_profile_tools_instrumented():
    """Every tool in the notebook profile must be instrumented."""
    data = _run_check({"MATHEMATICA_PROFILE": "notebook", "MATHEMATICA_ENABLE_TELEMETRY": "true"})
    registered = set(data["registered"])
    instrumented = set(data["instrumented"])

    missing = registered - instrumented
    assert not missing, (
        f"Notebook-profile tools registered but NOT instrumented: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# TelemetryMcpWrapper unit test
# ---------------------------------------------------------------------------


class TestTelemetryMcpWrapper:
    def test_wrapper_instruments_tool(self):
        """The wrapper intercepts .tool() and applies telemetry."""
        from mathematica_mcp.server import _TelemetryMcpWrapper
        from mathematica_mcp.telemetry import get_instrumented_tools, reset_stats

        reset_stats()

        class FakeMcp:
            registered = []

            def tool(self):
                def decorator(func):
                    self.registered.append(func)
                    return func
                return decorator

        fake = FakeMcp()
        wrapper = _TelemetryMcpWrapper(fake)

        @wrapper.tool()
        async def my_new_tool():
            return "hello"

        # The function registered with fake mcp should be the telemetry wrapper.
        assert len(fake.registered) == 1
        registered_func = fake.registered[0]
        assert getattr(registered_func, "_telemetry_instrumented", False) is True
        assert "my_new_tool" in get_instrumented_tools()

    def test_wrapper_delegates_other_attrs(self):
        from mathematica_mcp.server import _TelemetryMcpWrapper

        class FakeMcp:
            some_attr = 42

        wrapper = _TelemetryMcpWrapper(FakeMcp())
        assert wrapper.some_attr == 42


# ---------------------------------------------------------------------------
# Benchmark JSON schema stability
# ---------------------------------------------------------------------------


class TestBenchmarkSchema:
    """Verify the stats output format is stable for benchmark tooling."""

    def test_stats_schema_with_calls(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_usage_stats

        @telemetry_tool("schema_test")
        async def my_tool():
            return "ok"

        for _ in range(3):
            asyncio.run(my_tool())

        stats = get_usage_stats()
        entry = stats["schema_test"]

        # Required keys
        required = {"calls", "total_time_ms", "errors", "p50_ms", "p95_ms", "min_ms", "max_ms"}
        assert required <= set(entry.keys()), f"Missing keys: {required - set(entry.keys())}"

        # Types
        assert isinstance(entry["calls"], int)
        assert isinstance(entry["total_time_ms"], int)
        assert isinstance(entry["errors"], int)
        assert isinstance(entry["p50_ms"], (int, float))
        assert isinstance(entry["p95_ms"], (int, float))
        assert isinstance(entry["min_ms"], (int, float))
        assert isinstance(entry["max_ms"], (int, float))

    def test_stats_schema_empty(self):
        from mathematica_mcp.telemetry import get_usage_stats

        stats = get_usage_stats()
        assert stats == {}

    def test_stats_serializable_to_json(self):
        from mathematica_mcp.telemetry import telemetry_tool, get_usage_stats

        @telemetry_tool("json_test")
        async def my_tool():
            return "ok"

        asyncio.run(my_tool())

        stats = get_usage_stats()
        # Must be JSON-serializable for benchmark artifact storage.
        serialized = json.dumps(stats)
        roundtripped = json.loads(serialized)
        assert roundtripped["json_test"]["calls"] == 1
