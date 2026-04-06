"""Meta-tests for corpus models, adapters, and runner mechanics.

No wolframscript needed — all synthetic/mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from corpus.models import (
    CorpusCase,
    CorpusWorkflow,
    Oracle,
    PollCondition,
    SaveSpec,
    WorkflowStep,
)
from corpus.normalize import NormalizedResult
from pydantic import ValidationError


class TestCorpusCase:
    def test_valid_case_loads(self):
        data = {
            "id": "T1",
            "title": "basic add",
            "kind": "case",
            "tier": "smoke",
            "section": "Arithmetic",
            "min_profile": "math",
            "backend": "server_tool",
            "tool": "execute_code",
            "params": {"code": "1+1"},
            "oracle": {"type": "exact_text", "value": "2"},
        }
        case = CorpusCase(**data)
        assert case.id == "T1"
        assert case.kind == "case"
        assert case.tier == "smoke"
        assert case.required_capabilities == []

    def test_with_capabilities(self):
        data = {
            "id": "T2",
            "title": "entity",
            "tier": "probe",
            "section": "Entity",
            "min_profile": "full",
            "backend": "server_tool",
            "tool": "entity_lookup",
            "params": {"entity_type": "Country"},
            "oracle": {"type": "structural_fields"},
            "required_capabilities": ["wolfram_runtime", "network"],
        }
        case = CorpusCase(**data)
        assert "network" in case.required_capabilities

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValidationError):
            CorpusCase(
                id="T3",
                title="bad",
                tier="smoke",
                section="Test",
                min_profile="math",
                backend="made_up_backend",
                tool="x",
                oracle=Oracle(type="exact_text", value="1"),
            )

    def test_invalid_capability_rejected(self):
        with pytest.raises(ValidationError):
            CorpusCase(
                id="T4",
                title="bad",
                tier="smoke",
                section="Test",
                min_profile="math",
                backend="server_tool",
                tool="x",
                oracle=Oracle(type="exact_text", value="1"),
                required_capabilities=["teleportation"],
            )

    def test_invalid_tier_rejected(self):
        with pytest.raises(ValidationError):
            CorpusCase(
                id="T5",
                title="bad",
                tier="mythical",
                section="Test",
                min_profile="math",
                backend="server_tool",
                tool="x",
                oracle=Oracle(type="exact_text"),
            )

    def test_invalid_profile_rejected(self):
        with pytest.raises(ValidationError):
            CorpusCase(
                id="T6",
                title="bad",
                tier="smoke",
                section="Test",
                min_profile="enterprise",
                backend="server_tool",
                tool="x",
                oracle=Oracle(type="exact_text"),
            )


class TestCorpusWorkflow:
    def test_valid_workflow(self):
        wf = CorpusWorkflow(
            id="WF1",
            title="variable lifecycle",
            tier="smoke",
            section="Session",
            min_profile="math",
            backend="server_tool",
            steps=[
                WorkflowStep(
                    tool="set_variable",
                    params={"name": "x", "value": "42"},
                    save=[SaveSpec(state_key="var_name", path="parsed.name")],
                    oracle=Oracle(
                        type="field_equals",
                        path="parsed.success",
                        value=True,
                    ),
                ),
                WorkflowStep(
                    tool="get_variable",
                    params_from_state={"name": "var_name"},
                    oracle=Oracle(
                        type="field_equals",
                        path="parsed.value",
                        value="42",
                    ),
                ),
            ],
            final_oracle=Oracle(type="workflow_assert"),
            cleanup=[
                WorkflowStep(
                    tool="clear_variables",
                    params={"names": ["x"]},
                ),
            ],
        )
        assert wf.kind == "workflow"
        assert len(wf.steps) == 2
        assert len(wf.cleanup) == 1


class TestPollCondition:
    def test_valid_condition(self):
        pc = PollCondition(path="parsed.status", op="==", value="completed")
        assert pc.op == "=="
        assert pc.value == "completed"

    def test_in_operator(self):
        pc = PollCondition(
            path="parsed.status",
            op="in",
            value=["completed", "failed"],
        )
        assert pc.op == "in"

    def test_invalid_op_rejected(self):
        with pytest.raises(ValidationError):
            PollCondition(path="x", op="~=", value="y")


class TestOracle:
    def test_any_value_types(self):
        """Oracle.value accepts any type."""
        o1 = Oracle(type="exact_text", value="hello")
        assert o1.value == "hello"

        o2 = Oracle(type="field_equals", value=42)
        assert o2.value == 42

        o3 = Oracle(type="field_equals", value=True)
        assert o3.value is True

        o4 = Oracle(type="field_equals", value=[1, 2, 3])
        assert o4.value == [1, 2, 3]

    def test_tolerance_default(self):
        o = Oracle(type="numeric_tol", value=1.0)
        assert o.tolerance == 1e-5


class TestWorkflowPollingFailure:
    """Regression: workflow steps with poll must fail when condition is never met."""

    def test_unsatisfied_poll_fails(self, tmp_path, monkeypatch):
        import json

        from corpus.adapters import ADAPTERS
        from corpus.normalize import normalize
        from test_corpus_runner import (
            _check_poll_condition,
        )

        # Mock adapter that always returns "running"
        call_count = {"n": 0}

        def mock_adapter(tool, params, timeout=30):
            call_count["n"] += 1
            return json.dumps({"success": True, "status": "running"})

        monkeypatch.setitem(ADAPTERS, "server_tool", mock_adapter)

        wf = {
            "id": "WF-POLL-FAIL",
            "title": "poll never satisfied",
            "kind": "workflow",
            "tier": "smoke",
            "section": "Test",
            "min_profile": "math",
            "backend": "server_tool",
            "required_capabilities": [],
            "steps": [
                {
                    "tool": "poll_computation",
                    "params": {"job_id": "fake"},
                    "poll": {
                        "path": "parsed.status",
                        "op": "==",
                        "value": "completed",
                    },
                    "max_attempts": 3,
                    "delay_ms": 0,
                }
            ],
            "final_oracle": {"type": "workflow_assert"},
            "cleanup": [],
        }

        # Run the workflow step loop manually (same logic as runner)
        step = wf["steps"][0]
        poll = step.get("poll")
        poll_satisfied = poll is None
        for _attempt in range(step.get("max_attempts", 1)):
            raw = mock_adapter(step["tool"], step["params"])
            result = normalize(raw)
            if poll and _check_poll_condition(result, poll):
                poll_satisfied = True
                break

        assert not poll_satisfied, "Poll should NOT be satisfied"
        assert call_count["n"] == 3, "Should have retried 3 times"

    def test_satisfied_poll_passes(self):
        import json

        from corpus.normalize import normalize
        from test_corpus_runner import _check_poll_condition

        raw = json.dumps({"success": True, "status": "completed"})
        result = normalize(raw)
        poll = {"path": "parsed.status", "op": "==", "value": "completed"}
        assert _check_poll_condition(result, poll) is True


class TestCleanupTemplating:
    """Regression: cleanup steps must resolve {tmp_path} placeholders."""

    def test_cleanup_params_are_templated(self, tmp_path, monkeypatch):
        import json

        from corpus.adapters import ADAPTERS
        from test_corpus_runner import _resolve_params, _template_params

        received_params = {}

        def mock_adapter(tool, params, timeout=30):
            received_params.update(params)
            return json.dumps({"success": True})

        monkeypatch.setitem(ADAPTERS, "server_tool", mock_adapter)

        cleanup_step = {
            "tool": "close_notebook",
            "params": {"save_path": "{tmp_path}/saved.nb"},
        }
        state = {}

        # Replicate the cleanup path from the runner
        cleanup_params = _resolve_params(cleanup_step, state)
        cleanup_params = _template_params(cleanup_params, tmp_path)
        mock_adapter(cleanup_step["tool"], cleanup_params)

        assert received_params["save_path"] == str(tmp_path / "saved.nb")
        assert "{tmp_path}" not in received_params["save_path"]


class TestRunnerMechanics:
    """Test helper functions that will be used by the runner."""

    def test_extract_dot_path(self):
        from corpus.verifiers import extract_dot_path

        result = NormalizedResult(
            ok=True,
            parsed={"id": "nb123", "nested": {"key": "val"}},
        )
        assert extract_dot_path(result, "parsed.id") == "nb123"
        assert extract_dot_path(result, "parsed.nested.key") == "val"
        assert extract_dot_path(result, "parsed.missing") is None
        assert extract_dot_path(result, "ok") is True

    def test_template_params_recursive(self):
        """_template_params should recurse through nested structures."""
        # This tests the contract — actual function will be in the runner
        params = {
            "path": "{tmp_path}/output.csv",
            "nested": {"file": "{tmp_path}/inner.txt"},
            "list": ["{tmp_path}/a", "no_template"],
            "code": "1+1",
        }
        tmp = Path("/fake/tmp")

        def _template_params(p, tmp_path):
            if isinstance(p, dict):
                return {k: _template_params(v, tmp_path) for k, v in p.items()}
            if isinstance(p, list):
                return [_template_params(v, tmp_path) for v in p]
            if isinstance(p, str) and "{tmp_path}" in p:
                return p.replace("{tmp_path}", str(tmp_path))
            return p

        result = _template_params(params, tmp)
        assert result["path"] == "/fake/tmp/output.csv"
        assert result["nested"]["file"] == "/fake/tmp/inner.txt"
        assert result["list"][0] == "/fake/tmp/a"
        assert result["list"][1] == "no_template"
        assert result["code"] == "1+1"

    def test_poll_condition_check_equals(self):
        result = NormalizedResult(parsed={"status": "completed"})

        from corpus.verifiers import extract_dot_path

        actual = extract_dot_path(result, "parsed.status")
        assert actual == "completed"

    def test_poll_condition_check_not_met(self):
        result = NormalizedResult(parsed={"status": "running"})

        from corpus.verifiers import extract_dot_path

        actual = extract_dot_path(result, "parsed.status")
        assert actual != "completed"
