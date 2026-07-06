"""Live addon smoke test (P0: verify addon/SocketListen on the running kernel).

Requires a Mathematica frontend running with the MCP addon listening
(``StartMCPServer[]``). Skipped automatically when the socket is unreachable,
so it is a no-op in CI. Read-only: only ``ping``/``get_status`` — never touches
open notebooks.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

from mathematica_mcp.connection import DEFAULT_HOST, DEFAULT_PORT, MathematicaConnection

# The live marks are applied per-test (below) rather than at module level so the
# pure-logic regression test can still run under a "not wolfram_runtime" filter.
_live = pytest.mark.needs_live_addon
_runtime = pytest.mark.wolfram_runtime


def _kernel_version_ok(version: object) -> bool:
    """A live kernel reports a positive version number; 14.x stays supported.

    Replaces the old ``>= 15.0`` gate, which failed against every supported 14.x
    kernel (docs/roadmaps/v1.0-lean-plan-v2.md). A ``bool`` (int subclass) is
    rejected by the threshold, and non-numbers by the isinstance check.
    """
    return isinstance(version, (int, float)) and version >= 13.0


def _addon_reachable() -> bool:
    try:
        with socket.create_connection((DEFAULT_HOST, DEFAULT_PORT), timeout=2.0):
            return True
    except OSError:
        return False


@pytest.fixture
def conn() -> Iterator[MathematicaConnection]:
    if not _addon_reachable():
        pytest.skip(f"no MCP addon listening on {DEFAULT_HOST}:{DEFAULT_PORT}")
    c = MathematicaConnection()
    if not c.connect():
        pytest.skip(f"could not connect to addon: {c.last_error}")
    yield c
    c.disconnect()


def test_kernel_version_ok_accepts_14x():
    """Regression for the version gate below: 14.x must be accepted (the old
    ``>= 15.0`` gate failed against every supported 14.x kernel)."""
    assert _kernel_version_ok(15.1)
    assert _kernel_version_ok(14.2)
    assert _kernel_version_ok(13)
    assert not _kernel_version_ok(12.9)
    assert not _kernel_version_ok(True)  # bool rejected by the threshold
    assert not _kernel_version_ok("15")
    assert not _kernel_version_ok(None)


@_live
@_runtime
def test_ping(conn: MathematicaConnection) -> None:
    result = conn.send_command("ping", timeout=10)
    assert result.get("pong") is True
    assert "version" in result


@_live
@_runtime
def test_status_reports_live_kernel(conn: MathematicaConnection) -> None:
    """The addon must answer get_status and report a live kernel carrying its own
    real version number (>= 13; 14.x stays supported per the v1.0 plan)."""
    result = conn.send_command("get_status", timeout=10)
    assert result.get("mcp_server_running") is True
    assert result.get("mcp_port") == DEFAULT_PORT
    kernel_version = result.get("kernel_version")
    assert _kernel_version_ok(kernel_version), f"expected a live kernel version >= 13, addon reports {kernel_version!r}"
