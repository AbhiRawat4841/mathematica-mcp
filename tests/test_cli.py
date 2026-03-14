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
