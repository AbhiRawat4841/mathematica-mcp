from __future__ import annotations

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


def test_generate_mcp_config_includes_profile_env():
    config = cli.generate_mcp_config(use_uvx=True, profile="math")

    assert config["command"] == "uvx"
    assert config["args"] == ["mathematica-mcp-full"]
    assert config["env"] == {"MATHEMATICA_PROFILE": "math"}


def test_update_toml_config_rewrites_stale_mathematica_section(tmp_path: Path):
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

    ok = cli.update_toml_config(config_path, "mathematica", use_uvx=False, profile=None)

    assert ok is True
    updated = config_path.read_text()
    assert 'command = "uvx"' not in updated
    assert 'args = ["mathematica-mcp"]' not in updated
    assert '[mcp_servers.mathematica]' in updated
    assert 'args = ["--directory",' in updated
    assert updated.count("[mcp_servers.mathematica]") == 1
    assert "[features]" in updated


def test_update_toml_config_appends_when_section_missing(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-5.4"\n')

    ok = cli.update_toml_config(config_path, "mathematica", use_uvx=True, profile="notebook")

    assert ok is True
    updated = config_path.read_text()
    assert '[mcp_servers.mathematica]' in updated
    assert 'command = "uvx"' in updated
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
