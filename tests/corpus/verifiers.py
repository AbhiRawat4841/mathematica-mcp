"""Verification strategies for corpus test assertions.

Each verifier: (oracle, result) -> (passed: bool, explanation: str)

Uses execute_in_kernel() as the oracle engine for symbolic_equiv and
numeric_tol — never as the primary execution path.
"""

from __future__ import annotations

import re
from typing import Any

from .models import Oracle
from .normalize import NormalizedResult


def verify(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    """Dispatch to the appropriate verifier based on oracle type."""
    handler = _VERIFIERS.get(oracle.type)
    if handler is None:
        return False, f"Unknown oracle type: {oracle.type}"
    try:
        return handler(oracle, result)
    except Exception as e:
        return False, f"Verifier {oracle.type} raised: {e}"


# ---------------------------------------------------------------------------
# Individual verifiers
# ---------------------------------------------------------------------------


def _verify_field_equals(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    actual = _extract_dot_path(result, oracle.path)
    if actual is None:
        return False, f"Path {oracle.path!r} not found in result"
    passed = actual == oracle.value
    return passed, f"field[{oracle.path}] = {actual!r}, expected {oracle.value!r}"


def _verify_field_contains(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    actual = _extract_dot_path(result, oracle.path)
    if actual is None:
        return False, f"Path {oracle.path!r} not found"
    needle = str(oracle.value)
    actual_str = str(actual)
    passed = needle in actual_str
    return passed, f"{oracle.path!r} contains {needle!r}: {passed}"


def _verify_exact_text(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    actual = (result.output_text or "").strip()
    expected = str(oracle.value).strip()
    passed = actual == expected
    return passed, f"exact: actual={actual!r}, expected={expected!r}"


def _verify_symbolic_equiv(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    actual = (result.output_text or "").strip()
    expected = str(oracle.value).strip()

    if not actual:
        return False, "No output_text to compare"

    # Type-aware dispatch
    # 1. Rule lists: sort and compare textually
    if _looks_like_rules(actual) and _looks_like_rules(expected):
        return _compare_sorted_rules(actual, expected)

    # 2. Booleans: exact match
    if actual in ("True", "False") or expected in ("True", "False"):
        passed = actual == expected
        return passed, f"Boolean: {actual} vs {expected}"

    # 3. Standard case: Simplify[a - b] === 0 via oracle kernel
    try:
        from mathematica_mcp.session import execute_in_kernel

        code = f"Simplify[({actual}) - ({expected})] === 0"
        check = execute_in_kernel(code, timeout=10)
        if check.get("output_inputform", "").strip() == "True":
            return True, "Simplify confirms equivalence"

        # Fallback: FullSimplify
        code2 = f"FullSimplify[({actual}) - ({expected})] === 0"
        check2 = execute_in_kernel(code2, timeout=15)
        if check2.get("output_inputform", "").strip() == "True":
            return True, "FullSimplify confirms equivalence"
    except Exception as e:
        # If kernel unavailable, fall back to text comparison
        return actual == expected, f"Kernel unavailable ({e}), fell back to text match"

    return False, f"Symbolic mismatch: actual={actual!r}, expected={expected!r}"


def _verify_numeric_tol(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    actual_text = (result.output_text or "").strip()
    expected = oracle.value
    tolerance = oracle.tolerance

    # Type-aware: handle Quantity
    if "Quantity" in actual_text:
        try:
            from mathematica_mcp.session import execute_in_kernel

            mag = execute_in_kernel(f"QuantityMagnitude[{actual_text}]", timeout=5)
            actual_text = mag.get("output_inputform", actual_text)
        except Exception:
            pass

    actual_num = _parse_float(actual_text)
    expected_num = _parse_float(str(expected))

    if actual_num is None or expected_num is None:
        return False, f"Cannot parse numbers: actual={actual_text!r}, expected={expected!r}"

    diff = abs(actual_num - expected_num)
    passed = diff < tolerance
    return passed, f"diff={diff:.2e}, tolerance={tolerance:.2e}"


def _verify_boolean(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    actual = (result.output_text or "").strip()
    expected = str(oracle.value).strip()
    passed = actual == expected
    return passed, f"boolean: actual={actual!r}, expected={expected!r}"


def _verify_structural_fields(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    if not oracle.checks:
        # Default: just check ok=True and output_text is non-empty
        passed = result.ok and bool(result.output_text)
        return passed, f"ok={result.ok}, has_output={bool(result.output_text)}"

    failures = []
    for check in oracle.checks:
        # Checks like "parsed.success == True", "ok == True", "output_text != None"
        if "==" in check:
            path, expected = check.split("==", 1)
            actual = _extract_dot_path(result, path.strip())
            expected_val = _coerce_check_value(expected.strip())
            if actual != expected_val:
                failures.append(f"{path.strip()}={actual!r}, expected {expected_val!r}")
        elif "!=" in check:
            path, expected = check.split("!=", 1)
            actual = _extract_dot_path(result, path.strip())
            expected_val = _coerce_check_value(expected.strip())
            if actual == expected_val:
                failures.append(f"{path.strip()} should not be {expected_val!r}")

    if failures:
        return False, "; ".join(failures)
    return True, "all structural checks passed"


def _verify_artifact_exists(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    if not result.artifacts:
        return False, "No artifacts in result"
    for art in result.artifacts:
        if art.kind == "file" and art.path:
            import os

            if not os.path.exists(art.path):
                return False, f"Artifact file missing: {art.path}"
            if os.path.getsize(art.path) == 0:
                return False, f"Artifact file empty: {art.path}"
        elif art.kind == "image" and art.data is not None:
            if len(art.data) == 0:
                return False, "Image artifact has empty data"
    return True, f"{len(result.artifacts)} artifact(s) present"


def _verify_warning_tag(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    needle = str(oracle.value)
    for w in result.warnings:
        if needle in w:
            return True, f"Found {needle!r} in warning: {w!r}"
    return False, f"{needle!r} not found in warnings: {result.warnings}"


def _verify_raw_contains(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    raw = result.raw or ""
    if not oracle.contains:
        return False, "No 'contains' list in oracle"
    missing = [s for s in oracle.contains if s not in raw]
    if missing:
        return False, f"Missing in raw output: {missing}"
    return True, f"All {len(oracle.contains)} substrings found"


def _verify_workflow_assert(oracle: Oracle, result: NormalizedResult) -> tuple[bool, str]:
    # This is the fallback for NormalizedResult-only calls.
    # The full workflow_assert path goes through verify_workflow_context().
    if result.ok:
        return True, "Workflow completed successfully"
    return False, f"Workflow failed: {result.error_text}"


def verify_workflow_context(
    oracle: Oracle,
    step_results: list[tuple[str, NormalizedResult, bool]],
    state: dict[str, Any],
    cleanup_errors: list[str],
    final_result: NormalizedResult | None,
) -> tuple[bool, str]:
    """Full workflow assertion that inspects step history, state, and cleanup."""
    failures: list[str] = []

    # Check all steps passed
    for tool, _result, passed in step_results:
        if not passed:
            failures.append(f"Step {tool} failed")

    # Check cleanup errors
    if cleanup_errors:
        failures.append(f"Cleanup errors: {cleanup_errors}")

    # Check final result
    if final_result is not None and not final_result.ok:
        failures.append(f"Final result not ok: {final_result.error_text}")

    # Check oracle-specific conditions
    if oracle.checks:
        for check in oracle.checks:
            if check not in [tool for tool, _, _ in step_results]:
                failures.append(f"Expected step not found: {check}")

    if failures:
        return False, "; ".join(failures)
    return True, f"Workflow passed: {len(step_results)} steps, {len(state)} state keys"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_float(s: str) -> float | None:
    s = s.strip()
    # Handle Mathematica's *^ notation for scientific (e.g. 1.602*^-19)
    s = s.replace("*^", "e")
    m = _FLOAT_RE.search(s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _looks_like_rules(s: str) -> bool:
    return "->" in s and s.strip().startswith("{")


def _compare_sorted_rules(actual: str, expected: str) -> tuple[bool, str]:
    """Compare rule lists by sorting individual rules."""
    # Extract rule strings, sort, compare
    actual_rules = sorted(re.findall(r"\w+\s*->\s*[^,}]+", actual))
    expected_rules = sorted(re.findall(r"\w+\s*->\s*[^,}]+", expected))
    # Normalize whitespace in each rule
    actual_norm = [r.replace(" ", "") for r in actual_rules]
    expected_norm = [r.replace(" ", "") for r in expected_rules]
    passed = actual_norm == expected_norm
    return passed, f"rules: {actual_norm} vs {expected_norm}"


def extract_dot_path(result: NormalizedResult, path: str) -> Any:
    """Public API for dot-path extraction — used by runner."""
    return _extract_dot_path(result, path)


def _extract_dot_path(obj: Any, path: str | None) -> Any:
    """Navigate a dot-path like 'parsed.status' or 'parsed.nested.field'."""
    if path is None:
        return None
    parts = path.split(".")
    current: Any = obj
    for part in parts:
        if isinstance(current, NormalizedResult):
            current = getattr(current, part, None)
        elif isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _coerce_check_value(s: str) -> Any:
    """Coerce a string from a structural check into its Python type."""
    if s == "True":
        return True
    if s == "False":
        return False
    if s == "None":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip("'\"")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_VERIFIERS: dict[str, Any] = {
    "field_equals": _verify_field_equals,
    "field_contains": _verify_field_contains,
    "exact_text": _verify_exact_text,
    "symbolic_equiv": _verify_symbolic_equiv,
    "numeric_tol": _verify_numeric_tol,
    "boolean": _verify_boolean,
    "structural_fields": _verify_structural_fields,
    "artifact_exists": _verify_artifact_exists,
    "warning_tag": _verify_warning_tag,
    "raw_contains": _verify_raw_contains,
    "workflow_assert": _verify_workflow_assert,
}
