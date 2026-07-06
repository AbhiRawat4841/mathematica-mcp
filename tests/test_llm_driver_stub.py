"""Offline end-to-end test for benchmarks/llm_driver.py stub mode.

Runs the scripted stub policy with an injected fake executor (no MCP server,
no kernel, no network) and asserts the emitted JSONL scores cleanly through
benchmarks/score_trace_corpus.py --profile lean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import llm_driver  # noqa: E402


class RecordingExecutor:
    """Injected fake: records calls, returns canned success payloads."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, name: str, args: dict) -> str:
        self.calls.append(name)
        return json.dumps({"success": True, "tool": name})


def test_stub_replays_lean_preferred_tools():
    scenarios = llm_driver.load_scenarios()
    assert len(scenarios) >= 12  # 10 original + 3 error-recovery, minus none
    executor = RecordingExecutor()
    rows = llm_driver.run_stub(scenarios, "lean", executor=executor)

    assert len(rows) == len(scenarios)
    lean_vocab = {
        "status",
        "notebooks",
        "cells",
        "edit_cells",
        "evaluate",
        "screenshot",
        "kernel",
        "vars",
        "read_notebook_file",
        "guide",
        "batch",
        "verify_derivation",
    }
    for row, scenario in zip(rows, scenarios, strict=True):
        assert row["scenario"] == scenario["id"]
        assert row["success"] is True
        assert row["tools"] == scenario["lean_preferred_tools"]
        assert set(row["tools"]) <= lean_vocab
        assert row["first_useful_ms"] is not None
        assert row["total_ms"] >= 0
    assert executor.calls  # the injected executor actually ran


def test_error_recovery_scenarios_present():
    ids = {s["id"] for s in llm_driver.load_scenarios()}
    assert {"error_recovery_bad_syntax", "error_recovery_unknown_cursor", "error_recovery_addon_down"} <= ids


def test_stub_jsonl_scores_with_profile_lean(tmp_path):
    rows = llm_driver.run_stub(llm_driver.load_scenarios(), "lean", executor=RecordingExecutor())
    trace = tmp_path / "stub_lean.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "score_trace_corpus.py"),
            str(trace),
            str(trace),
            "--profile",
            "lean",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["baseline"]["runs"] == len(rows)
    assert report["baseline"]["success_rate"] == 1.0
    # Preferred sequences must never trip the lean anti-pattern detector.
    assert report["baseline"]["anti_pattern_rate"] == 0.0
    assert report["delta"]["success_rate"] == 0.0


def test_scorer_backward_compatible_without_profile_flag(tmp_path):
    rows = llm_driver.run_stub(llm_driver.load_scenarios(), "classic", executor=RecordingExecutor())
    trace = tmp_path / "stub_classic.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "score_trace_corpus.py"), str(trace), str(trace)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["baseline"]["runs"] == len(rows)
