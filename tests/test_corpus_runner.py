"""Corpus-driven MCP test runner.

Two parametrized entry points:
- test_corpus_case: atomic single-tool tests
- test_corpus_workflow: multi-step stateful workflows

Loads tests from tests/corpus/mathematica_mcp_corpus.json.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from corpus.adapters import ADAPTERS
from corpus.models import CorpusCase, CorpusWorkflow, Oracle
from corpus.normalize import NormalizedResult, normalize
from corpus.verifiers import extract_dot_path, verify, verify_workflow_context

# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).parent / "corpus" / "mathematica_mcp_corpus.json"


def _load_manifest() -> list[dict]:
    if not CORPUS_PATH.exists():
        pytest.skip(f"Corpus manifest not found: {CORPUS_PATH}")
    raw = json.loads(CORPUS_PATH.read_text())
    # Validate each item against Pydantic models
    items = []
    for item in raw:
        kind = item.get("kind", "case")
        if kind == "case":
            CorpusCase(**item)  # validates; raises on bad data
        elif kind == "workflow":
            CorpusWorkflow(**item)
        items.append(item)
    return items


MANIFEST = _load_manifest()
CASES = [c for c in MANIFEST if c.get("kind", "case") == "case"]
WORKFLOWS = [w for w in MANIFEST if w.get("kind") == "workflow"]


# ---------------------------------------------------------------------------
# Active-profile gate
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _active_tools() -> set[str]:
    """Get tools registered in the current server profile."""
    try:
        from mathematica_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        return {tool.name for tool in tools}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _template_params(params: Any, tmp_path: Path) -> Any:
    """Recursively substitute {tmp_path} in params."""
    if isinstance(params, dict):
        return {k: _template_params(v, tmp_path) for k, v in params.items()}
    if isinstance(params, list):
        return [_template_params(v, tmp_path) for v in params]
    if isinstance(params, str) and "{tmp_path}" in params:
        return params.replace("{tmp_path}", str(tmp_path))
    return params


def _resolve_params(step: dict, state: dict) -> dict:
    """Resolve params_from_state into concrete params."""
    params = dict(step.get("params", {}))
    for param_key, state_key in step.get("params_from_state", {}).items():
        if state_key in state:
            params[param_key] = state[state_key]
    return params


def _check_poll_condition(result: NormalizedResult, poll: dict) -> bool:
    """Evaluate structured poll condition: {path, op, value}."""
    actual = extract_dot_path(result, poll["path"])
    op = poll["op"]
    expected = poll["value"]
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    return False


def _marks_for(item: dict) -> list:
    """Build pytest marks from item metadata."""
    marks = [
        pytest.mark.corpus,
        getattr(pytest.mark, f"tier_{item['tier']}"),
        getattr(pytest.mark, f"profile_{item['min_profile']}"),
    ]
    for cap in item.get("required_capabilities", []):
        marks.append(getattr(pytest.mark, f"needs_{cap}"))
    return marks


def _skip_if_missing(item: dict, capabilities: dict[str, bool]) -> None:
    """Skip test if required capabilities are not available."""
    for cap in item.get("required_capabilities", []):
        if not capabilities.get(cap, False):
            pytest.skip(f"Capability not available: {cap}")


def _skip_if_tool_unavailable(tool_name: str) -> None:
    """Skip if the tool is not registered in the current server profile."""
    tools = _active_tools()
    if tools and tool_name not in tools:
        # Only skip if we successfully enumerated tools
        pytest.skip(f"Tool {tool_name} not in active profile")


def _format_failure(item: dict, result: NormalizedResult, explanation: str) -> str:
    return (
        f"[{item['id']}] {item['title']}\n"
        f"  Tool: {item.get('tool', 'workflow')}\n"
        f"  Oracle: {item.get('oracle', item.get('final_oracle', {}))}\n"
        f"  Output: {result.output_text!r}\n"
        f"  Detail: {explanation}"
    )


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


@dataclass
class WorkflowContext:
    step_results: list[tuple[str, NormalizedResult, bool]] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    cleanup_errors: list[str] = field(default_factory=list)
    final_result: NormalizedResult | None = None


# ---------------------------------------------------------------------------
# Capability fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus_capabilities() -> dict[str, bool]:
    """Session-scoped capability snapshot."""
    try:
        from corpus.capabilities import probe_capabilities

        return probe_capabilities()
    except Exception:
        # Graceful fallback if probing fails
        return {"wolfram_runtime": False}


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["id"], marks=_marks_for(c)) for c in CASES],
)
def test_corpus_case(case: dict, corpus_capabilities: dict, tmp_path: Path) -> None:
    _skip_if_missing(case, corpus_capabilities)
    _skip_if_tool_unavailable(case["tool"])

    params = _template_params(case["params"], tmp_path)
    adapter = ADAPTERS[case["backend"]]
    raw = adapter(case["tool"], params, timeout=case.get("timeout_s", 30))
    result = normalize(raw, tool_name=case["tool"])

    oracle = Oracle(**case["oracle"])
    passed, explanation = verify(oracle, result)
    if not passed:
        pytest.fail(_format_failure(case, result, explanation))


@pytest.mark.parametrize(
    "wf",
    [pytest.param(w, id=w["id"], marks=_marks_for(w)) for w in WORKFLOWS],
)
def test_corpus_workflow(wf: dict, corpus_capabilities: dict, tmp_path: Path) -> None:
    _skip_if_missing(wf, corpus_capabilities)

    state: dict[str, Any] = {}
    step_results: list[tuple[str, NormalizedResult, bool]] = []
    cleanup_errors: list[str] = []
    result: NormalizedResult | None = None

    try:
        for step in wf["steps"]:
            # Active-profile gate for each workflow step
            _skip_if_tool_unavailable(step["tool"])

            step_backend = step.get("backend") or wf["backend"]
            adapter = ADAPTERS[step_backend]
            params = _resolve_params(step, state)
            params = _template_params(params, tmp_path)

            # Polling support — fail if condition never satisfied
            poll = step.get("poll")
            poll_satisfied = poll is None  # no poll = always satisfied
            for attempt in range(step.get("max_attempts", 1)):
                raw = adapter(step["tool"], params, timeout=step.get("timeout_s", 30))
                result = normalize(raw, tool_name=step["tool"])

                if poll and _check_poll_condition(result, poll):
                    poll_satisfied = True
                    break
                if step.get("max_attempts", 1) > 1 and attempt < step["max_attempts"] - 1:
                    time.sleep(step.get("delay_ms", 500) / 1000)

            if not poll_satisfied:
                step_results.append((step["tool"], result, False))
                pytest.fail(
                    f"Workflow {wf['id']} step {step['tool']}: "
                    f"poll condition not satisfied after {step.get('max_attempts', 1)} attempts"
                )

            # Extract and save
            for save_spec in step.get("save", []):
                val = extract_dot_path(result, save_spec["path"])
                state[save_spec["state_key"]] = val

            # Per-step assertion
            if step.get("oracle"):
                oracle = Oracle(**step["oracle"])
                passed, msg = verify(oracle, result)
                step_results.append((step["tool"], result, passed))
                if not passed:
                    pytest.fail(f"Workflow {wf['id']} failed at step {step['tool']}: {msg}")
            else:
                step_results.append((step["tool"], result, True))

    finally:
        for cs in wf.get("cleanup", []):
            try:
                step_backend = cs.get("backend") or wf["backend"]
                adapter = ADAPTERS[step_backend]
                cleanup_params = _resolve_params(cs, state)
                cleanup_params = _template_params(cleanup_params, tmp_path)
                adapter(cs["tool"], cleanup_params, timeout=cs.get("timeout_s", 30))
            except Exception as e:
                cleanup_errors.append(f"{cs['tool']}: {e}")

    # Final workflow-level assertion — uses full context, not just last result
    assert result is not None, f"Workflow {wf['id']} produced no result"
    final_oracle = Oracle(**wf["final_oracle"])

    passed, explanation = verify_workflow_context(final_oracle, step_results, state, cleanup_errors, result)
    if not passed:
        detail = (
            f"  Steps: {len(step_results)} executed\n"
            f"  Failed steps: {[t for t, _, p in step_results if not p]}\n"
            f"  State keys: {list(state.keys())}\n"
            f"  Cleanup errors: {cleanup_errors}\n"
            f"  Detail: {explanation}"
        )
        pytest.fail(f"[{wf['id']}] {wf['title']}\n{detail}")
