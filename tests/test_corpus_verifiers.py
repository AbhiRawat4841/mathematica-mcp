"""Meta-tests for the corpus verification strategies.

Uses synthetic NormalizedResult objects — no wolframscript needed
(except TestSymbolicEquiv which requires the kernel as oracle engine).
"""

from __future__ import annotations

import pytest
from corpus.models import Oracle
from corpus.normalize import Artifact, NormalizedResult
from corpus.verifiers import verify, verify_workflow_context


class TestFieldEquals:
    def test_match(self):
        result = NormalizedResult(ok=True, parsed={"status": "ok", "count": 5})
        oracle = Oracle(type="field_equals", path="parsed.count", value=5)
        passed, _ = verify(oracle, result)
        assert passed

    def test_mismatch(self):
        result = NormalizedResult(ok=True, parsed={"count": 3})
        oracle = Oracle(type="field_equals", path="parsed.count", value=5)
        passed, _ = verify(oracle, result)
        assert not passed

    def test_nested_path(self):
        result = NormalizedResult(ok=True, parsed={"data": {"id": "abc123"}})
        oracle = Oracle(type="field_equals", path="parsed.data.id", value="abc123")
        passed, _ = verify(oracle, result)
        assert passed

    def test_missing_path(self):
        result = NormalizedResult(ok=True, parsed={"x": 1})
        oracle = Oracle(type="field_equals", path="parsed.nonexistent", value=1)
        passed, _ = verify(oracle, result)
        assert not passed

    def test_bool_value(self):
        result = NormalizedResult(ok=True, parsed={"created": True})
        oracle = Oracle(type="field_equals", path="parsed.created", value=True)
        passed, _ = verify(oracle, result)
        assert passed


class TestFieldContains:
    def test_substring_present(self):
        result = NormalizedResult(ok=True, parsed={"description": "This is a test function"})
        oracle = Oracle(
            type="field_contains",
            path="parsed.description",
            value="test",
        )
        passed, _ = verify(oracle, result)
        assert passed

    def test_substring_missing(self):
        result = NormalizedResult(ok=True, parsed={"description": "hello"})
        oracle = Oracle(type="field_contains", path="parsed.description", value="xyz")
        passed, _ = verify(oracle, result)
        assert not passed


class TestExactText:
    def test_exact_match(self):
        result = NormalizedResult(ok=True, output_text="42")
        oracle = Oracle(type="exact_text", value="42")
        passed, _ = verify(oracle, result)
        assert passed

    def test_whitespace_normalized(self):
        result = NormalizedResult(ok=True, output_text="  42  ")
        oracle = Oracle(type="exact_text", value="42")
        passed, _ = verify(oracle, result)
        assert passed

    def test_mismatch(self):
        result = NormalizedResult(ok=True, output_text="43")
        oracle = Oracle(type="exact_text", value="42")
        passed, _ = verify(oracle, result)
        assert not passed


class TestNumericTol:
    def test_within_tolerance(self):
        result = NormalizedResult(ok=True, output_text="0.746824")
        oracle = Oracle(type="numeric_tol", value=0.746824133, tolerance=1e-4)
        passed, _ = verify(oracle, result)
        assert passed

    def test_outside_tolerance(self):
        result = NormalizedResult(ok=True, output_text="0.5")
        oracle = Oracle(type="numeric_tol", value=0.746824133, tolerance=1e-4)
        passed, _ = verify(oracle, result)
        assert not passed

    def test_scientific_notation(self):
        result = NormalizedResult(ok=True, output_text="1.602e-19")
        oracle = Oracle(type="numeric_tol", value=1.602e-19, tolerance=1e-22)
        passed, _ = verify(oracle, result)
        assert passed

    def test_mathematica_scientific(self):
        result = NormalizedResult(ok=True, output_text="1.602*^-19")
        oracle = Oracle(type="numeric_tol", value=1.602e-19, tolerance=1e-22)
        passed, _ = verify(oracle, result)
        assert passed

    def test_non_numeric(self):
        result = NormalizedResult(ok=True, output_text="not a number")
        oracle = Oracle(type="numeric_tol", value=1.0, tolerance=0.1)
        passed, _ = verify(oracle, result)
        assert not passed


class TestBoolean:
    def test_true(self):
        result = NormalizedResult(ok=True, output_text="True")
        oracle = Oracle(type="boolean", value="True")
        passed, _ = verify(oracle, result)
        assert passed

    def test_false(self):
        result = NormalizedResult(ok=True, output_text="False")
        oracle = Oracle(type="boolean", value="False")
        passed, _ = verify(oracle, result)
        assert passed

    def test_mismatch(self):
        result = NormalizedResult(ok=True, output_text="True")
        oracle = Oracle(type="boolean", value="False")
        passed, _ = verify(oracle, result)
        assert not passed


class TestStructuralFields:
    def test_default_checks(self):
        result = NormalizedResult(ok=True, output_text="something")
        oracle = Oracle(type="structural_fields")
        passed, _ = verify(oracle, result)
        assert passed

    def test_default_fails_on_empty(self):
        result = NormalizedResult(ok=True, output_text="")
        oracle = Oracle(type="structural_fields")
        passed, _ = verify(oracle, result)
        assert not passed

    def test_custom_check_equals(self):
        result = NormalizedResult(ok=True, parsed={"cell_count": 5})
        oracle = Oracle(
            type="structural_fields",
            checks=["parsed.cell_count == 5"],
        )
        passed, _ = verify(oracle, result)
        assert passed

    def test_custom_check_not_equals(self):
        result = NormalizedResult(ok=True, parsed={"status": "ok"})
        oracle = Oracle(
            type="structural_fields",
            checks=["parsed.status != None"],
        )
        passed, _ = verify(oracle, result)
        assert passed


class TestArtifactExists:
    def test_with_file_artifact(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 10)
        result = NormalizedResult(ok=True, artifacts=[Artifact(kind="file", path=str(f))])
        oracle = Oracle(type="artifact_exists")
        passed, _ = verify(oracle, result)
        assert passed

    def test_with_image_artifact(self):
        result = NormalizedResult(
            ok=True,
            artifacts=[Artifact(kind="image", data=b"\x89PNG\r\n")],
        )
        oracle = Oracle(type="artifact_exists")
        passed, _ = verify(oracle, result)
        assert passed

    def test_empty_artifacts(self):
        result = NormalizedResult(ok=True, artifacts=[])
        oracle = Oracle(type="artifact_exists")
        passed, _ = verify(oracle, result)
        assert not passed

    def test_empty_image_data(self):
        result = NormalizedResult(ok=True, artifacts=[Artifact(kind="image", data=b"")])
        oracle = Oracle(type="artifact_exists")
        passed, _ = verify(oracle, result)
        assert not passed


class TestWarningTag:
    def test_tag_present(self):
        result = NormalizedResult(
            ok=True,
            warnings=["Power::infy: infinite expression encountered"],
        )
        oracle = Oracle(type="warning_tag", value="Power::infy")
        passed, _ = verify(oracle, result)
        assert passed

    def test_tag_missing(self):
        result = NormalizedResult(ok=True, warnings=["General::stop"])
        oracle = Oracle(type="warning_tag", value="Power::infy")
        passed, _ = verify(oracle, result)
        assert not passed

    def test_empty_warnings(self):
        result = NormalizedResult(ok=True, warnings=[])
        oracle = Oracle(type="warning_tag", value="any")
        passed, _ = verify(oracle, result)
        assert not passed


class TestRawContains:
    def test_all_present(self):
        result = NormalizedResult(ok=True, raw='{"output": "x -> 2, x -> 3"}')
        oracle = Oracle(type="raw_contains", contains=["x -> 2", "x -> 3"])
        passed, _ = verify(oracle, result)
        assert passed

    def test_partial_missing(self):
        result = NormalizedResult(ok=True, raw='{"output": "x -> 2"}')
        oracle = Oracle(type="raw_contains", contains=["x -> 2", "x -> 3"])
        passed, _ = verify(oracle, result)
        assert not passed

    def test_empty_contains(self):
        result = NormalizedResult(ok=True, raw="anything")
        oracle = Oracle(type="raw_contains", contains=None)
        passed, _ = verify(oracle, result)
        assert not passed


class TestWorkflowAssert:
    def test_ok_result(self):
        result = NormalizedResult(ok=True)
        oracle = Oracle(type="workflow_assert")
        passed, _ = verify(oracle, result)
        assert passed

    def test_failed_result(self):
        result = NormalizedResult(ok=False, error_text="step 3 failed")
        oracle = Oracle(type="workflow_assert")
        passed, _ = verify(oracle, result)
        assert not passed


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestSymbolicEquiv:
    def test_trig_identity(self):
        result = NormalizedResult(ok=True, output_text="1")
        oracle = Oracle(type="symbolic_equiv", value="Sin[x]^2 + Cos[x]^2")
        passed, _ = verify(oracle, result)
        assert passed

    def test_simple_equivalence(self):
        result = NormalizedResult(ok=True, output_text="2 + 3")
        oracle = Oracle(type="symbolic_equiv", value="5")
        passed, _ = verify(oracle, result)
        assert passed


class TestWorkflowContext:
    def test_all_steps_passed(self):
        oracle = Oracle(type="workflow_assert")
        r1 = NormalizedResult(ok=True, output_text="42")
        r2 = NormalizedResult(ok=True, output_text="done")
        passed, _ = verify_workflow_context(
            oracle,
            step_results=[("set_variable", r1, True), ("get_variable", r2, True)],
            state={"var": "42"},
            cleanup_errors=[],
            final_result=r2,
        )
        assert passed

    def test_step_failure_detected(self):
        oracle = Oracle(type="workflow_assert")
        r1 = NormalizedResult(ok=True)
        r2 = NormalizedResult(ok=False, error_text="boom")
        passed, msg = verify_workflow_context(
            oracle,
            step_results=[("step1", r1, True), ("step2", r2, False)],
            state={},
            cleanup_errors=[],
            final_result=r2,
        )
        assert not passed
        assert "step2" in msg

    def test_cleanup_errors_reported(self):
        oracle = Oracle(type="workflow_assert")
        r1 = NormalizedResult(ok=True)
        passed, msg = verify_workflow_context(
            oracle,
            step_results=[("step1", r1, True)],
            state={},
            cleanup_errors=["close_notebook: timeout"],
            final_result=r1,
        )
        assert not passed
        assert "Cleanup errors" in msg

    def test_final_result_not_ok(self):
        oracle = Oracle(type="workflow_assert")
        r1 = NormalizedResult(ok=False, error_text="kernel crash")
        passed, msg = verify_workflow_context(
            oracle,
            step_results=[("step1", r1, True)],
            state={},
            cleanup_errors=[],
            final_result=r1,
        )
        assert not passed
        assert "not ok" in msg


class TestVerifyDispatch:
    def test_unknown_type_fails(self):
        result = NormalizedResult(ok=True, output_text="42")
        # Bypass Pydantic validation to create an oracle with invalid type
        oracle = Oracle.model_construct(type="nonexistent_type")
        passed, msg = verify(oracle, result)
        assert not passed
        assert "Unknown oracle type" in msg
