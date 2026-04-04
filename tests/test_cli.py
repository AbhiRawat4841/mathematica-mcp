from __future__ import annotations

import json
from pathlib import Path

from mathematica_mcp import cli


def test_get_addon_init_paths_prefers_wolfram_on_macos(tmp_path: Path):
    paths = cli.get_addon_init_paths(tmp_path, "Darwin")

    assert paths[0] == tmp_path / "Library/Wolfram/Kernel/init.m"
    assert paths[1] == tmp_path / "Library/Mathematica/Kernel/init.m"


def test_check_mathematica_addon_finds_wolfram_init(monkeypatch, tmp_path: Path):
    init_path = tmp_path / "Library/Wolfram/Kernel/init.m"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text("(* MathematicaMCP *)\nMathematicaMCP`StartMCPServer[];\n")

    monkeypatch.setattr(cli, "get_system", lambda: "Darwin")
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)

    ok, msg = cli.check_mathematica_addon()

    assert ok is True
    assert str(init_path) in msg


def test_generate_mcp_config_includes_profile_env(monkeypatch):
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: f"/resolved/{command}")

    config = cli.generate_mcp_config(use_uvx=True, profile="math")

    assert config["command"] == "/resolved/uvx"
    assert config["args"] == ["mathematica-mcp-full"]
    assert config["env"] == {"MATHEMATICA_PROFILE": "math"}


def test_generate_mcp_config_falls_back_to_bare_command_when_unresolved(monkeypatch):
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: command)

    config = cli.generate_mcp_config(use_uvx=True, profile=None)

    assert config["command"] == "uvx"
    assert config["args"] == ["mathematica-mcp-full"]


def test_generate_mcp_config_local_uses_resolved_uv(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: f"/resolved/{command}")
    monkeypatch.setattr(cli, "get_package_dir", lambda: tmp_path)

    config = cli.generate_mcp_config(use_uvx=False, profile=None)

    assert config["command"] == "/resolved/uv"
    assert config["args"] == ["--directory", str(tmp_path), "run", "mathematica-mcp-full"]


def test_generate_toml_config_uses_resolved_launcher(monkeypatch):
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: f"/resolved/{command}")

    config = cli.generate_toml_config("mathematica", use_uvx=True, profile="notebook")

    assert 'command = "/resolved/uvx"' in config
    assert 'args = ["mathematica-mcp-full"]' in config
    assert 'env = { MATHEMATICA_PROFILE = "notebook" }' in config


def test_update_json_config_uses_resolved_launcher_and_merges_env(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mathematica": {
                        "command": "old",
                        "args": ["stale"],
                        "env": {"KEEP_ME": "1"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: f"/resolved/{command}")

    ok = cli.update_json_config(
        config_path,
        cli.CLIENT_CONFIGS["claude-desktop"],
        use_uvx=True,
        profile="math",
    )

    assert ok is True
    updated = json.loads(config_path.read_text())
    server = updated["mcpServers"]["mathematica"]
    assert server["command"] == "/resolved/uvx"
    assert server["args"] == ["mathematica-mcp-full"]
    assert server["env"] == {"KEEP_ME": "1", "MATHEMATICA_PROFILE": "math"}


def test_update_toml_config_rewrites_stale_mathematica_section(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                "",
                "[mcp_servers.mathematica]",
                'command = "uvx"',
                'args = ["mathematica-mcp"]',
                "",
                "[features]",
                "multi_agent = true",
                "",
            ]
        )
    )
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: f"/resolved/{command}")
    monkeypatch.setattr(cli, "get_package_dir", lambda: tmp_path / "repo")

    ok = cli.update_toml_config(config_path, "mathematica", use_uvx=False, profile=None)

    assert ok is True
    updated = config_path.read_text()
    assert 'command = "uvx"' not in updated
    assert 'args = ["mathematica-mcp"]' not in updated
    assert "[mcp_servers.mathematica]" in updated
    assert 'command = "/resolved/uv"' in updated
    assert 'args = ["--directory",' in updated
    assert updated.count("[mcp_servers.mathematica]") == 1
    assert "[features]" in updated


def test_update_toml_config_appends_when_section_missing(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-5.4"\n')
    monkeypatch.setattr(cli, "resolve_launcher", lambda command: f"/resolved/{command}")

    ok = cli.update_toml_config(config_path, "mathematica", use_uvx=True, profile="notebook")

    assert ok is True
    updated = config_path.read_text()
    assert "[mcp_servers.mathematica]" in updated
    assert 'command = "/resolved/uvx"' in updated
    assert 'args = ["mathematica-mcp-full"]' in updated
    assert 'env = { MATHEMATICA_PROFILE = "notebook" }' in updated


def test_install_claude_code_guidance_writes_command_and_marked_block(tmp_path: Path):
    written = cli.install_claude_code_guidance(tmp_path, profile="notebook")

    command_path = tmp_path / ".claude" / "commands" / "mathematica.md"
    claude_md_path = tmp_path / "CLAUDE.md"

    assert command_path in written
    assert claude_md_path in written
    assert "Profile: `notebook`" in command_path.read_text()

    claude_md = claude_md_path.read_text()
    assert "<!-- mathematica-mcp:start -->" in claude_md
    assert "<!-- mathematica-mcp:end -->" in claude_md
    assert "Primary execution tool: `execute_code()`" in claude_md


def test_install_claude_code_guidance_updates_existing_marked_block(tmp_path: Path):
    claude_md_path = tmp_path / "CLAUDE.md"
    claude_md_path.write_text("Project notes\n\n<!-- mathematica-mcp:start -->\nold\n<!-- mathematica-mcp:end -->\n")

    cli.install_claude_code_guidance(tmp_path, profile="math")
    updated = claude_md_path.read_text()

    assert "Project notes" in updated
    assert "old" not in updated
    assert "Current profile default output target: `cli`" in updated
