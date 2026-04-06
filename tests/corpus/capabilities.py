"""Session-scoped capability probes for corpus test gating.

Probes are cached and run once per session. Environment variables
control which optional capabilities are tested:

    MCP_CORPUS_ENABLE_NETWORK=1
    MCP_CORPUS_ENABLE_RESOURCE=1
    MCP_CORPUS_ENABLE_SUBKERNELS=1
"""

from __future__ import annotations

import functools
import os
import socket


@functools.lru_cache(maxsize=1)
def probe_capabilities() -> dict[str, bool]:
    """Return a snapshot of available capabilities."""
    from tests.conftest import wolfram_runtime_available

    caps: dict[str, bool] = {
        "wolfram_runtime": wolfram_runtime_available(),
        "live_addon": _probe_addon_socket(),
        "frontend": False,
        "network": False,
        "resource": False,
        "subkernels": False,
    }

    if caps["live_addon"]:
        caps["frontend"] = caps["live_addon"]  # conservative: addon implies frontend possible

    if os.environ.get("MCP_CORPUS_ENABLE_NETWORK"):
        caps["network"] = _probe_network()

    if os.environ.get("MCP_CORPUS_ENABLE_RESOURCE"):
        caps["resource"] = caps.get("network", False)

    if os.environ.get("MCP_CORPUS_ENABLE_SUBKERNELS"):
        caps["subkernels"] = True

    return caps


def _probe_addon_socket(timeout: float = 1.0) -> bool:
    """Check if the Mathematica addon is reachable on its default port."""
    host = os.environ.get("MATHEMATICA_HOST", "localhost")
    port = int(os.environ.get("MATHEMATICA_PORT", "9881"))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


def _probe_network(timeout: float = 2.0) -> bool:
    """Check network connectivity via DNS resolution (no external HTTP calls)."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo("api.wolframcloud.com", 443)
        return True
    except (socket.gaierror, OSError):
        return False
