"""P4: V15 features behind $VersionNumber guards + addon protocol handshake.

Mocked tests cover the Python-side protocol-skew detection and scan the addon
source for the guards. The ``wolfram_runtime`` tests exercise the actual
$VersionNumber >= 15 guard and its MMCP_FORCE_V14 override headlessly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mathematica_mcp import server as srv

ADDON = Path(__file__).resolve().parents[1] / "addon" / "MathematicaMCP.wl"


# ---- Python-side protocol skew detection -----------------------------------


def test_protocol_flags_missing():
    r: dict = {}
    srv._check_addon_protocol(r)
    assert r["addon_outdated"] is True
    assert "reinstall" in r["addon_hint"].lower()


def test_protocol_flags_old():
    r = {"protocol_version": 1}
    srv._check_addon_protocol(r)
    assert r["addon_outdated"] is True


def test_protocol_ok_when_current():
    from mathematica_mcp.connection import ADDON_PROTOCOL_VERSION

    r = {"protocol_version": ADDON_PROTOCOL_VERSION}
    srv._check_addon_protocol(r)
    assert "addon_outdated" not in r


# ---- addon source guards ---------------------------------------------------


def test_addon_source_has_protocol_and_guards():
    import re

    from mathematica_mcp.connection import ADDON_PROTOCOL_VERSION

    wl = ADDON.read_text()
    # Lockstep, not a hardcoded literal: the addon's protocol version must equal
    # the Python client's expectation, whatever it currently is.
    m = re.search(r"\$MCPProtocolVersion = (\d+)", wl)
    assert m is not None, "addon must declare $MCPProtocolVersion"
    assert int(m.group(1)) == ADDON_PROTOCOL_VERSION
    assert "mcpVersionAtLeast15" in wl
    assert "MMCP_FORCE_V14" in wl
    assert "ShowChatbar -> False" in wl
    assert '"protocol_version" -> $MCPProtocolVersion' in wl


# ---- live guard behavior (headless, no frontend needed) --------------------

_PROBE = f"""
Get["{ADDON}"];
opts = MathematicaMCP`Private`mcpNotebookOptions[<|"title" -> "T"|>];
Print["HAS_CHATBAR=", MemberQ[opts, HoldPattern[ShowChatbar -> False]]];
Print["PROTOCOL=", MathematicaMCP`Private`$MCPProtocolVersion];
Print["VERSION=", $VersionNumber];
"""


def _probe_version(out: str) -> float | None:
    """Parse the kernel ``$VersionNumber`` line printed by ``_PROBE`` (``VERSION=15.1``).

    ``$VersionNumber`` is the true kernel version and is unaffected by
    ``MMCP_FORCE_V14`` (that only gates the addon's ShowChatbar branch), so it is
    the right value to decide whether a v15-only feature can exist.
    """
    for line in out.splitlines():
        if line.startswith("VERSION="):
            try:
                return float(line[len("VERSION=") :].strip())
            except ValueError:
                return None
    return None


def _run_probe(force_v14: bool) -> str:
    env = dict(os.environ)
    if force_v14:
        env["MMCP_FORCE_V14"] = "1"
    else:
        env.pop("MMCP_FORCE_V14", None)
    out = subprocess.run(
        ["wolframscript", "-code", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return out.stdout


def test_probe_version_parsing():
    """Regression for the version-gating logic below: a 14.x kernel reports 14.x
    (< 15), which must route the v15-only test to SKIP rather than fail."""
    assert _probe_version("HAS_CHATBAR=False\nPROTOCOL=2\nVERSION=14.2\n") == 14.2
    assert _probe_version("VERSION=15.1\n") == 15.1
    assert _probe_version("PROTOCOL=2\n") is None  # no VERSION line
    assert _probe_version("VERSION=oops\n") is None


@pytest.mark.wolfram_runtime
def test_showchatbar_present_on_v15(require_wolfram_runtime):
    from mathematica_mcp.connection import ADDON_PROTOCOL_VERSION

    out = _run_probe(force_v14=False)
    assert f"PROTOCOL={ADDON_PROTOCOL_VERSION}" in out
    version = _probe_version(out)
    if version is None or version < 15.0:
        # ShowChatbar is a V15+ notebook option; a 14.x kernel cannot carry it,
        # so this feature check is inapplicable there (not a failure).
        pytest.skip(f"kernel version {version} < 15 has no ShowChatbar")
    assert "HAS_CHATBAR=True" in out


@pytest.mark.wolfram_runtime
def test_showchatbar_suppressed_when_forced_v14(require_wolfram_runtime):
    out = _run_probe(force_v14=True)
    assert "HAS_CHATBAR=False" in out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
