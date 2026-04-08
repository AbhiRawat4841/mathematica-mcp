"""Tests for transport_classification.py — shared classification logic."""

from __future__ import annotations

from mathematica_mcp.constants import AttemptOutcome
from mathematica_mcp.transport_classification import (
    TransportStatus,
    classify_attempt_outcome,
    classify_final_transport,
    extract_error_families,
)


# ---------------------------------------------------------------------------
# extract_error_families (migrated from test_execute_code_metadata.py)
# ---------------------------------------------------------------------------


class TestExtractErrorFamilies:
    def test_notebook_structured_messages(self):
        response = {
            "messages": [
                {"type": "error", "tag": "Syntax::sntxi", "text": "Incomplete expression."},
                {"type": "warning", "tag": "General::stop", "text": "Further output suppressed."},
            ]
        }
        families = extract_error_families(response)
        assert families == ["Syntax"]

    def test_notebook_multiple_families(self):
        response = {
            "messages": [
                {"type": "error", "tag": "Part::partw", "text": "..."},
                {"type": "error", "tag": "Syntax::sntxi", "text": "..."},
            ]
        }
        families = extract_error_families(response)
        assert families == ["Part", "Syntax"]

    def test_kernel_raw_warnings(self):
        response = {
            "warnings": ["{HoldForm[Syntax::sntxi], HoldForm[Part::partw]}"],
        }
        families = extract_error_families(response)
        assert families == ["Part", "Syntax"]

    def test_unknown_tag_becomes_other(self):
        response = {
            "messages": [
                {"type": "error", "tag": "CustomModule::badarg", "text": "..."},
            ]
        }
        families = extract_error_families(response)
        assert families == ["other"]

    def test_general_stop_excluded(self):
        response = {
            "warnings": ["General::stop: Further output suppressed."],
        }
        families = extract_error_families(response)
        assert families == []

    def test_fallback_to_error_field(self):
        response = {
            "success": False,
            "error": "Syntax::sntxi: Incomplete expression",
            "warnings": [],
        }
        families = extract_error_families(response)
        assert families == ["Syntax"]

    def test_output_scan_only_on_failure_with_empty_error(self):
        response = {
            "success": True,
            "output": "The error Syntax::sntxi is documented here",
        }
        families = extract_error_families(response)
        assert families == []


# ---------------------------------------------------------------------------
# classify_final_transport
# ---------------------------------------------------------------------------


class TestClassifyFinalTransport:
    def test_timeout(self):
        response = {"timed_out": True, "success": False}
        assert classify_final_transport(response, fell_back=False) == TransportStatus.TIMEOUT

    def test_success_no_fallback(self):
        response = {"success": True}
        assert classify_final_transport(response, fell_back=False) == TransportStatus.OK

    def test_success_with_fallback(self):
        response = {"success": True}
        assert classify_final_transport(response, fell_back=True) == TransportStatus.DEGRADED_FALLBACK

    def test_failure_with_error_families_no_fallback(self):
        response = {"success": False, "error_families": ["Syntax"]}
        assert classify_final_transport(response, fell_back=False) == TransportStatus.OK

    def test_failure_with_error_families_and_fallback(self):
        response = {"success": False, "error_families": ["Syntax"]}
        assert classify_final_transport(response, fell_back=True) == TransportStatus.DEGRADED_FALLBACK

    def test_failure_no_families(self):
        response = {"success": False}
        assert classify_final_transport(response, fell_back=False) == TransportStatus.INFRA_ERROR


# ---------------------------------------------------------------------------
# classify_attempt_outcome
# ---------------------------------------------------------------------------


class TestClassifyAttemptOutcome:
    def test_ok_result(self):
        result = {"success": True, "output": "2"}
        assert classify_attempt_outcome(result) == AttemptOutcome.OK

    def test_infra_error_from_exception(self):
        assert classify_attempt_outcome(None, exc=ConnectionError("refused")) == AttemptOutcome.INFRA_ERROR

    def test_infra_error_from_none_result(self):
        assert classify_attempt_outcome(None) == AttemptOutcome.INFRA_ERROR

    def test_timeout(self):
        result = {"success": False, "timed_out": True}
        assert classify_attempt_outcome(result) == AttemptOutcome.TIMEOUT

    def test_semantic_error(self):
        result = {
            "success": False,
            "messages": [{"type": "error", "tag": "Syntax::sntxi", "text": "..."}],
        }
        assert classify_attempt_outcome(result) == AttemptOutcome.SEMANTIC_ERROR

    def test_infra_error_no_families(self):
        result = {"success": False, "error": "Connection reset"}
        assert classify_attempt_outcome(result) == AttemptOutcome.INFRA_ERROR

    def test_exception_takes_precedence_over_result(self):
        result = {"success": True, "output": "ok"}
        assert classify_attempt_outcome(result, exc=RuntimeError("boom")) == AttemptOutcome.INFRA_ERROR

    def test_success_with_warnings_is_ok(self):
        result = {"success": True, "warnings": ["General::stop: ..."]}
        assert classify_attempt_outcome(result) == AttemptOutcome.OK

    def test_error_key_without_success_false_is_infra_error(self):
        """Result with 'error' key but no success=False must be classified as failure."""
        result = {"error": "socket closed"}
        assert classify_attempt_outcome(result) == AttemptOutcome.INFRA_ERROR

    def test_error_key_with_success_true_is_infra_error(self):
        """Even success=True with error key → infra error (mirrors server fallback predicate)."""
        result = {"success": True, "error": "partial failure"}
        assert classify_attempt_outcome(result) == AttemptOutcome.INFRA_ERROR

    def test_non_dict_result_is_infra_error(self):
        """Non-dict results (rare string returns) must not crash."""
        assert classify_attempt_outcome("some string result") == AttemptOutcome.INFRA_ERROR

    def test_non_dict_int_result_is_infra_error(self):
        assert classify_attempt_outcome(42) == AttemptOutcome.INFRA_ERROR
