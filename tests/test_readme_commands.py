import asyncio
import codeop
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


README_PATH = Path(__file__).resolve().parents[1] / "README.md"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    kind: str  # bash | wolfram | python
    source: str
    code: str
    ok: bool
    detail: str = ""


def _extract_fenced_blocks(text: str):
    """Return list of (lang, body) for fenced code blocks."""
    pattern = re.compile(r"```(?P<lang>\w+)?\n(?P<body>.*?)\n```", re.S)
    return [(m.group("lang") or "", m.group("body")) for m in pattern.finditer(text)]


def _make_isolated_home(tmp_path: Path) -> Path:
    """Create an isolated HOME with a symlinked ~/mcp/mathematica-mcp."""
    home = tmp_path / "home"
    (home / "mcp").mkdir(parents=True, exist_ok=True)

    target = home / "mcp" / "mathematica-mcp"
    # symlink to repo root so README paths work unchanged
    target.symlink_to(REPO_ROOT, target_is_directory=True)
    return home


def _run_bash(script: str, *, env: dict, timeout_s: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
    )


def _wolframscript_exists() -> bool:
    from shutil import which

    return which("wolframscript") is not None


def _run_wolfram(code: str, *, env: dict, timeout_s: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wolframscript", "-code", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
    )


def _extract_python_statements(block: str) -> list[str]:
    """Split a README python block into runnable statements.

    We intentionally skip comment lines. If a statement is truncated with `...`,
    we emit a special marker statement so the report can call it out.
    """
    compiler = codeop.CommandCompiler()
    statements: list[str] = []
    buf: list[str] = []

    def flush_incomplete(reason: str):
        nonlocal buf
        if buf:
            statements.append(f"__README_INCOMPLETE__:{reason}:{''.join(buf)}")
            buf = []

    for line in block.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # The README uses `...` to omit content; treat it as not runnable.
        if stripped == "..." or stripped.endswith("..."):
            flush_incomplete("ellipsis")
            statements.append("__README_ELLIPSIS__")
            continue

        buf.append(line)
        src = "".join(buf)
        try:
            compiled = compiler(src)
        except SyntaxError as e:
            statements.append(f"__README_SYNTAX_ERROR__:{e.msg}:{src}")
            buf = []
            continue

        if compiled is not None:
            statements.append(src)
            buf = []

    if buf:
        statements.append(f"__README_INCOMPLETE__:eof:{''.join(buf)}")

    return statements


def _await_if_needed(value):
    if asyncio.iscoroutine(value):
        return asyncio.run(value)
    return value


def _execute_python_statement(stmt: str):
    """Execute a single statement, awaiting tool calls when needed."""
    import ast
    import importlib

    server = importlib.import_module("mathematica_mcp.server")

    # Execute in a restricted-ish namespace: only the server module tools.
    env: dict = {name: getattr(server, name) for name in dir(server)}

    tree = ast.parse(stmt, mode="exec")
    if len(tree.body) != 1:
        exec(compile(tree, "<readme>", "exec"), env, env)
        return None

    node = tree.body[0]
    if isinstance(node, ast.Expr):
        expr = ast.Expression(node.value)
        value = eval(compile(expr, "<readme>", "eval"), env, env)
        return _await_if_needed(value)

    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        # Support `x = tool_call(...)` patterns.
        expr = ast.Expression(node.value)
        value = eval(compile(expr, "<readme>", "eval"), env, env)
        value = _await_if_needed(value)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            env[target.id] = value
        return value

    exec(compile(tree, "<readme>", "exec"), env, env)
    return None


def _looks_like_error_json(s: str) -> tuple[bool, str]:
    try:
        data = json.loads(s)
    except Exception:
        return False, ""

    if isinstance(data, dict):
        if data.get("success") is False:
            return True, data.get("error") or "success=false"
        if "error" in data and data.get("error"):
            return True, str(data.get("error"))

    return False, ""


def test_readme_commands_smoke(tmp_path: Path):
    readme = README_PATH.read_text(encoding="utf-8")
    blocks = _extract_fenced_blocks(readme)

    isolated_home = _make_isolated_home(tmp_path)
    env = os.environ.copy()
    env["HOME"] = str(isolated_home)
    env.setdefault("UV_CACHE_DIR", str(isolated_home / ".cache" / "uv"))

    # Ensure Python tool calls that use `os.path.expanduser("~")` also stay isolated.
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(isolated_home)

    results: list[CommandResult] = []

    try:
        # 1) Bash blocks
        for lang, body in blocks:
            if lang != "bash":
                continue

            placeholder_lines = [
                ln
                for ln in body.splitlines()
                if "<PID>" in ln or re.search(r"\bkill\b.*<", ln)
            ]
            for ln in placeholder_lines:
                results.append(
                    CommandResult(
                        kind="bash",
                        source="README bash",
                        code=ln.strip(),
                        ok=False,
                        detail="placeholder value; not runnable as-is",
                    )
                )

            script_lines = [ln for ln in body.splitlines() if "<PID>" not in ln]
            script = "\n".join(script_lines).strip()
            if not script:
                continue

            # Avoid installing into the user's global Python environment.
            # If README suggests `pip install -e ...`, run it inside an isolated venv.
            if re.search(r"\bpip\s+install\s+-e\b", script):
                venv_dir = tmp_path / "pip-venv"
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                venv_bin = venv_dir / ("Scripts" if os.name == "nt" else "bin")
                venv_env = env.copy()
                venv_env["PATH"] = (
                    str(venv_bin) + os.pathsep + venv_env.get("PATH", "")
                )
                proc = _run_bash(f"set -e\n{script}", env=venv_env, timeout_s=600)
            else:
                run_env = env.copy()
                # Prevent `uv sync` from mutating the repo's `.venv` (which can
                # break the currently running pytest process on reruns).
                if re.search(r"\buv\s+sync\b", script):
                    run_env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "uv-project-env")
                proc = _run_bash(f"set -e\n{script}", env=run_env)

            ok = proc.returncode == 0
            if re.search(r"\blsof\b.*:9881", script) and proc.returncode in (0, 1):
                ok = True

            detail = (proc.stderr or proc.stdout).strip()
            results.append(
                CommandResult(
                    kind="bash",
                    source="README bash",
                    code=script,
                    ok=ok,
                    detail=detail,
                )
            )

        # 2) Mathematica block: run via wolframscript if available
        for lang, body in blocks:
            if lang != "mathematica":
                continue

            if not _wolframscript_exists():
                results.append(
                    CommandResult(
                        kind="wolfram",
                        source="README mathematica",
                        code=body.strip(),
                        ok=False,
                        detail="wolframscript not found",
                    )
                )
                continue

            # Run the README commands as-is, but ensure we stop the server.
            code = (
                body.strip()
                + "\n\n"
                + "Quiet[Check[StopMCPServer[], Null]];\n"
                + "ExportString[MCPServerStatus[], \"RawJSON\"]\n"
            )
            proc = _run_wolfram(code, env=env)
            ok = proc.returncode == 0
            detail = (proc.stderr or proc.stdout).strip()
            results.append(
                CommandResult(
                    kind="wolfram",
                    source="README mathematica",
                    code=body.strip(),
                    ok=ok,
                    detail=detail,
                )
            )

        # 3) Python blocks: execute statements when possible
        for lang, body in blocks:
            if lang != "python":
                continue

            statements = _extract_python_statements(body)
            for stmt in statements:
                if stmt == "__README_ELLIPSIS__" or stmt.startswith(
                    "__README_INCOMPLETE__"
                ):
                    results.append(
                        CommandResult(
                            kind="python",
                            source="README python",
                            code=stmt,
                            ok=False,
                            detail="example truncated with ellipsis; not runnable as-is",
                        )
                    )
                    continue
                if stmt.startswith("__README_SYNTAX_ERROR__"):
                    results.append(
                        CommandResult(
                            kind="python",
                            source="README python",
                            code=stmt,
                            ok=False,
                            detail="syntax error",
                        )
                    )
                    continue

                try:
                    out = _execute_python_statement(stmt)
                except Exception as e:
                    results.append(
                        CommandResult(
                            kind="python",
                            source="README python",
                            code=stmt.strip(),
                            ok=False,
                            detail=f"exception: {type(e).__name__}: {e}",
                        )
                    )
                    continue

                if out is None:
                    results.append(
                        CommandResult(
                            kind="python",
                            source="README python",
                            code=stmt.strip(),
                            ok=True,
                            detail="",
                        )
                    )
                    continue

                if isinstance(out, str):
                    is_err, err_detail = _looks_like_error_json(out)
                    results.append(
                        CommandResult(
                            kind="python",
                            source="README python",
                            code=stmt.strip(),
                            ok=not is_err,
                            detail=err_detail,
                        )
                    )
                else:
                    results.append(
                        CommandResult(
                            kind="python",
                            source="README python",
                            code=stmt.strip(),
                            ok=True,
                            detail="",
                        )
                    )

        # Always write a report file for inspection.
        report_path = REPO_ROOT / "readme_command_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "results": [r.__dict__ for r in results],
                    "failing": [r.__dict__ for r in results if not r.ok],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        failing = [r for r in results if not r.ok]
        if failing:
            lines = ["README command failures (non-exhaustive details):"]
            for r in failing:
                code_preview = r.code.strip().replace("\n", "\\n")
                if len(code_preview) > 160:
                    code_preview = code_preview[:160] + "..."
                detail = r.detail.strip().replace("\n", " ")
                if len(detail) > 200:
                    detail = detail[:200] + "..."
                lines.append(f"- [{r.kind}] {code_preview} :: {detail}")
            raise AssertionError("\n".join(lines))
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
