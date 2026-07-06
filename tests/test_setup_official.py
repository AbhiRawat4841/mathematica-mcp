"""P5: `setup --with-official` writes the official Wolfram MCP server alongside
ours, snapshot-tested for the JSON clients (Claude Desktop / Claude Code / Cursor)
and the Codex TOML client."""

from __future__ import annotations

import json

import pytest

from mathematica_mcp import cli

JSON_CLIENTS = ["claude-desktop", "claude-code", "cursor"]


@pytest.mark.parametrize("client", JSON_CLIENTS)
def test_with_official_writes_both_servers(client, tmp_path):
    info = cli.CLIENT_CONFIGS[client]
    cfg = tmp_path / "config.json"
    assert cli.update_json_config(cfg, info, use_uvx=True, profile="lean", with_official=True)

    data = json.loads(cfg.read_text())
    servers = data[info["key"]]
    assert info["server_name"] in servers
    assert cli.OFFICIAL_WOLFRAM_SERVER_NAME in servers
    assert servers[info["server_name"]]["env"]["MATHEMATICA_PROFILE"] == "lean"
    assert servers[cli.OFFICIAL_WOLFRAM_SERVER_NAME]["command"] == "wolframscript"


@pytest.mark.parametrize("client", JSON_CLIENTS)
def test_without_official_omits_wolfram(client, tmp_path):
    info = cli.CLIENT_CONFIGS[client]
    cfg = tmp_path / "config.json"
    assert cli.update_json_config(cfg, info, use_uvx=True, profile="lean", with_official=False)

    data = json.loads(cfg.read_text())
    assert cli.OFFICIAL_WOLFRAM_SERVER_NAME not in data[info["key"]]


def test_with_official_preserves_existing_servers(tmp_path):
    info = cli.CLIENT_CONFIGS["claude-desktop"]
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))

    assert cli.update_json_config(cfg, info, use_uvx=True, profile=None, with_official=True)
    servers = json.loads(cfg.read_text())["mcpServers"]
    assert "other" in servers  # not clobbered
    assert "mathematica" in servers
    assert "wolfram" in servers


def test_codex_toml_with_official(tmp_path):
    cfg = tmp_path / "config.toml"
    assert cli.update_toml_config(cfg, "mathematica", use_uvx=True, profile="lean", with_official=True)
    text = cfg.read_text()
    assert "[mcp_servers.mathematica]" in text
    assert "[mcp_servers.wolfram]" in text
    assert "wolframscript" in text


def test_codex_toml_with_official_idempotent(tmp_path):
    """Second run must not duplicate the [mcp_servers.wolfram] table.

    Pre-fix, the second run left the old wolfram table in place and appended a
    new one, yielding a duplicate table that tomllib refuses to parse.
    """
    tomllib = pytest.importorskip("tomllib")
    cfg = tmp_path / "config.toml"
    assert cli.update_toml_config(cfg, "mathematica", use_uvx=True, profile="lean", with_official=True)
    assert cli.update_toml_config(cfg, "mathematica", use_uvx=True, profile="lean", with_official=True)

    text = cfg.read_text()
    assert text.count("[mcp_servers.wolfram]") == 1
    assert text.count("[mcp_servers.mathematica]") == 1
    # Still valid TOML (a duplicate table would raise here).
    parsed = tomllib.loads(text)
    assert "mathematica" in parsed["mcp_servers"]
    assert "wolfram" in parsed["mcp_servers"]


def test_json_with_official_idempotent(tmp_path):
    """Second run keeps a single wolfram entry and re-parses cleanly."""
    info = cli.CLIENT_CONFIGS["claude-desktop"]
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))

    assert cli.update_json_config(cfg, info, use_uvx=True, profile="lean", with_official=True)
    assert cli.update_json_config(cfg, info, use_uvx=True, profile="lean", with_official=True)

    servers = json.loads(cfg.read_text())["mcpServers"]
    assert list(servers).count("wolfram") == 1
    assert "mathematica" in servers
    assert "other" in servers  # unrelated server preserved across re-runs
