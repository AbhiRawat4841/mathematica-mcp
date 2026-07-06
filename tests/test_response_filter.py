"""Tests for response_filter.py — payload shaping."""

from __future__ import annotations

from mathematica_mcp.response_filter import _filter_response, _summarize_large_output

# ---------------------------------------------------------------------------
# _summarize_large_output
# ---------------------------------------------------------------------------


class TestSummarizeLargeOutput:
    def test_below_threshold_returns_none(self):
        assert _summarize_large_output("short") is None

    def test_above_threshold_returns_summary(self):
        big = "x" * 5000
        result = _summarize_large_output(big, threshold=4000)
        assert result is not None
        assert result["original_length"] == 5000

    def test_custom_threshold(self):
        text = "a" * 100
        assert _summarize_large_output(text, threshold=50) is not None
        assert _summarize_large_output(text, threshold=200) is None

    def test_truncated_preview_length(self):
        big = "x" * 5000
        result = _summarize_large_output(big, threshold=4000)
        assert len(result["truncated_preview"]) == 500

    def test_tail_preview_length(self):
        big = "x" * 5000
        result = _summarize_large_output(big, threshold=4000)
        assert len(result["tail_preview"]) == 200

    def test_list_structure_detected(self):
        # Build a large list
        big_list = "{" + ", ".join(str(i) for i in range(500)) + "}"
        result = _summarize_large_output(big_list, threshold=100)
        assert result is not None
        assert "element_count" in result
        assert result["element_count"] == 500


# ---------------------------------------------------------------------------
# _filter_response — standard
# ---------------------------------------------------------------------------


class TestFilterResponseStandard:
    def test_standard_is_exact_identity(self):
        response = {"success": True, "output": "2", "timing_ms": 5, "messages": []}
        filtered = _filter_response(response, "standard")
        assert filtered is response  # same object, not a copy

    def test_unknown_treated_as_standard(self):
        response = {"success": True, "output": "2"}
        filtered = _filter_response(response, "some_unknown_value")
        assert filtered is response

    def test_standard_preserves_frozen_keys(self):
        """Verify REQUIRED_SUCCESS_KEYS from test_session.py are preserved."""
        required = {
            "success",
            "output",
            "output_inputform",
            "output_fullform",
            "warnings",
            "timing_ms",
            "execution_method",
        }
        response = {k: "test" for k in required}
        filtered = _filter_response(response, "standard")
        missing = required - set(filtered.keys())
        assert not missing, f"Missing keys: {missing}"


# ---------------------------------------------------------------------------
# _filter_response — compact
# ---------------------------------------------------------------------------


class TestFilterResponseCompact:
    def test_compact_strips_verbose_fields(self):
        response = {
            "success": True,
            "output": "2",
            "timing_ms": 5,
            "messages": [{"type": "error"}],
            "output_fullform": "2",
            "output_tex": "2",
            "detailed_analyses": [],
            "llm_error_report": "...",
            "route_variant": "compute",
            "execution_path": "addon_cli",
            "overall_timing_ms": 10,
            "error_summary": "...",
            "code": "1+1",
            "requested_style": "compute",
            "error_analysis": {"severity": "warning", "retry_with": "N[...]"},
        }
        filtered = _filter_response(response, "compact")
        for field in [
            "messages",
            "output_fullform",
            "output_tex",
            "detailed_analyses",
            "llm_error_report",
            "route_variant",
            "execution_path",
            "overall_timing_ms",
            "error_summary",
            "code",
            "requested_style",
        ]:
            assert field not in filtered, f"{field} should be stripped"
        # error_analysis is intentionally KEPT in compact when non-empty (guidance
        # + retry_with must reach the caller) - see _COMPACT_KEEP.
        assert "error_analysis" in filtered

    def test_compact_keeps_essential_fields(self):
        response = {
            "success": True,
            "output": "2",
            "timing_ms": 5,
            "status": "executed_in_notebook",
            "message": "Executed.",
        }
        filtered = _filter_response(response, "compact")
        assert filtered["success"] is True
        assert filtered["output"] == "2"
        assert filtered["timing_ms"] == 5
        assert filtered["status"] == "executed_in_notebook"
        assert filtered["message"] == "Executed."

    def test_compact_keeps_notebook_ids(self):
        response = {
            "success": True,
            "output": "2",
            "timing_ms": 5,
            "notebook_id": "nb-123",
            "cell_id": "cell-456",
        }
        filtered = _filter_response(response, "compact")
        assert filtered["notebook_id"] == "nb-123"
        assert filtered["cell_id"] == "cell-456"

    def test_compact_preserves_error_on_failure(self):
        response = {"success": False, "output": "", "error": "Syntax error", "timing_ms": 5}
        filtered = _filter_response(response, "compact")
        assert filtered["error"] == "Syntax error"

    def test_compact_preserves_transport_status(self):
        response = {"success": False, "output": "", "timing_ms": 5, "transport_status": "infra_error"}
        filtered = _filter_response(response, "compact")
        assert filtered["transport_status"] == "infra_error"

    def test_compact_preserves_error_families(self):
        response = {"success": False, "output": "", "timing_ms": 5, "error_families": ["Syntax"]}
        filtered = _filter_response(response, "compact")
        assert filtered["error_families"] == ["Syntax"]

    def test_compact_preserves_timed_out_and_from_cache(self):
        response = {"success": True, "output": "2", "timing_ms": 0, "timed_out": False, "from_cache": True}
        filtered = _filter_response(response, "compact")
        assert filtered["from_cache"] is True
        assert filtered["timed_out"] is False

    def test_compact_preserves_rendered_image_and_is_graphics(self):
        response = {
            "success": True,
            "output": "[Graphics rendered to image: /tmp/plot.png]",
            "output_inputform": "Graphics[...]",
            "timing_ms": 100,
            "rendered_image": "/tmp/plot.png",
            "is_graphics": True,
        }
        filtered = _filter_response(response, "compact")
        assert filtered["rendered_image"] == "/tmp/plot.png"
        assert filtered["is_graphics"] is True

    def test_compact_graphics_swaps_to_inputform(self):
        response = {
            "success": True,
            "output": "[Graphics rendered to image: /tmp/plot.png]",
            "output_inputform": "Graphics[Line[{{0,0},{1,1}}]]",
            "timing_ms": 100,
            "is_graphics": True,
        }
        filtered = _filter_response(response, "compact")
        assert filtered["output"] == "Graphics[Line[{{0,0},{1,1}}]]"

    def test_compact_non_graphics_keeps_output(self):
        response = {"success": True, "output": "42", "timing_ms": 5}
        filtered = _filter_response(response, "compact")
        assert filtered["output"] == "42"

    def test_compact_summarizes_large_output(self):
        response = {"success": True, "output": "x" * 5000, "timing_ms": 5}
        filtered = _filter_response(response, "compact")
        assert "output_summary" in filtered
        assert len(filtered["output"]) == 500  # truncated

    def test_compact_does_not_summarize_small_output(self):
        response = {"success": True, "output": "42", "timing_ms": 5}
        filtered = _filter_response(response, "compact")
        assert "output_summary" not in filtered


# ---------------------------------------------------------------------------
# _filter_response — verbose
# ---------------------------------------------------------------------------


class TestFilterResponseVerbose:
    def test_verbose_adds_detail_level_only(self):
        response = {"success": True, "output": "2", "timing_ms": 5}
        filtered = _filter_response(response, "verbose")
        assert filtered["detail_level"] == "verbose"
        # Original keys preserved
        assert filtered["success"] is True
        assert filtered["output"] == "2"


# ---------------------------------------------------------------------------
# _filter_response — diagnostic
# ---------------------------------------------------------------------------


class TestFilterResponseDiagnostic:
    def test_diagnostic_adds_cache_epoch(self):
        response = {"success": True, "output": "2"}
        filtered = _filter_response(response, "diagnostic", cache_epoch=5)
        assert filtered["detail_level"] == "diagnostic"
        assert filtered["cache_epoch"] == 5

    def test_diagnostic_adds_routing_hints(self):
        response = {"success": True, "output": "2"}
        hints = ["Plot timeout 40%"]
        filtered = _filter_response(response, "diagnostic", routing_hints=hints)
        assert filtered["routing_hints"] == hints

    def test_diagnostic_omits_hints_when_none(self):
        response = {"success": True, "output": "2"}
        filtered = _filter_response(response, "diagnostic")
        assert "routing_hints" not in filtered
        assert "cache_epoch" not in filtered


# ---------------------------------------------------------------------------
# _filter_response - compact failure rule (never slim a failing response)
# ---------------------------------------------------------------------------


class TestFilterResponseFailureRule:
    def test_success_false_returns_full_identity(self):
        response = {
            "success": False,
            "output": "",
            "transport_status": "infra_error",
            "error_families": ["Syntax"],
            "messages": [{"type": "error"}],
            "warnings": ["Syntax::sntx"],
            "error_analysis": {"severity": "error"},
            "code": "1+",  # a non-keep field must survive too
        }
        filtered = _filter_response(response, "compact")
        assert filtered is response  # full identity, nothing slimmed
        for key in ("transport_status", "error_families", "messages", "warnings", "error_analysis", "code"):
            assert key in filtered

    def test_truthy_error_key_returns_full_identity(self):
        response = {"output": "", "error": "boom", "transport_status": "infra_error"}
        filtered = _filter_response(response, "compact")
        assert filtered is response
        assert filtered["error"] == "boom"

    def test_success_true_with_empty_error_is_slimmed(self):
        response = {"success": True, "output": "2", "error": "", "code": "1+1"}
        filtered = _filter_response(response, "compact")
        assert filtered is not response  # slimmed, not identity
        assert "code" not in filtered
        assert "error" not in filtered  # empty string stripped

    def test_no_success_and_falsy_error_is_slimmed(self):
        response = {"output": "2", "error": None, "code": "1+1"}
        filtered = _filter_response(response, "compact")
        assert filtered is not response
        assert "code" not in filtered


class TestFilterResponseRealNotebookFailureShapes:
    """The notebook timeout/error responses server.py builds carry NO success or
    error key, so failure is inferred from timed_out/has_errors/status so that
    compact does not strip output_preview/messages/has_errors/error_summary."""

    def test_notebook_timeout_shape_passes_through_identity(self):
        # Mirror of server.py:949-964 (the addon_notebook timeout branch).
        response = {
            "status": "timeout",
            "code": "Pause[999]",
            "notebook_id": "nb-1",
            "cell_id": "cell-1",
            "evaluated": False,
            "timed_out": True,
            "timing_ms": 300000,
            "message": "Computation exceeded 300s timeout. Increase timeout or use submit_computation() for long tasks.",
            "output_preview": "partial output so far...",
            "executed_output_target": "notebook",
            "executed_mode": "kernel",
        }
        filtered = _filter_response(response, "compact")
        assert filtered is response  # full identity, nothing slimmed
        assert filtered["output_preview"] == "partial output so far..."
        assert filtered["timed_out"] is True

    def test_notebook_executed_with_errors_shape_passes_through_identity(self):
        # Mirror of server.py:979-1032 (the addon_notebook has_errors branch).
        response = {
            "status": "executed_with_errors",
            "code": "1/0",
            "notebook_id": "nb-2",
            "cell_id": "cell-2",
            "evaluated": True,
            "message": "Code executed in notebook but produced errors. See 'error_analysis' field for detailed suggestions.",
            "messages": [{"type": "error", "tag": "Power::infy"}],
            "has_errors": True,
            "has_warnings": False,
            "error_summary": "1 error(s): Power::infy",
            "error_analysis": {"total_errors": 1, "severity": "error", "retry_with": None},
        }
        filtered = _filter_response(response, "compact")
        assert filtered is response  # full identity, nothing slimmed
        for key in ("messages", "has_errors", "error_summary", "error_analysis"):
            assert key in filtered


class TestFilterResponseKeepsPendingGuidance:
    def test_compact_keeps_next_step(self):
        # The evaluation_pending contract carries recovery guidance in next_step;
        # compact must not strip it.
        response = {
            "success": True,
            "status": "evaluation_pending",
            "evaluation_complete": False,
            "waited_seconds": 0.2,
            "next_step": "Re-check the notebook with get_cells to read the output once it appears.",
        }
        filtered = _filter_response(response, "compact")
        assert filtered["next_step"] == response["next_step"]
        assert filtered["status"] == "evaluation_pending"


# ---------------------------------------------------------------------------
# _filter_response - summarize_large flag (lean paginator serves full output)
# ---------------------------------------------------------------------------


class TestFilterResponseSummarizeLargeFlag:
    def test_summarize_large_true_summarizes(self):
        response = {"success": True, "output": "x" * 5000, "timing_ms": 5}
        filtered = _filter_response(response, "compact", summarize_large=True)
        assert "output_summary" in filtered
        assert len(filtered["output"]) == 500

    def test_summarize_large_false_keeps_full_output(self):
        # Lean path: the cursor paginator serves the full output downstream, so
        # compact must NOT pre-summarise it.
        response = {"success": True, "output": "x" * 5000, "timing_ms": 5}
        filtered = _filter_response(response, "compact", summarize_large=False)
        assert "output_summary" not in filtered
        assert filtered["output"] == "x" * 5000


# ---------------------------------------------------------------------------
# _filter_response - compact empty-field stripping (success path)
# ---------------------------------------------------------------------------


class TestFilterResponseEmptyStripping:
    def test_strips_empty_string_list_dict_none(self):
        response = {
            "success": True,
            "output": "2",
            "message": "",
            "error_families": [],
            "error_analysis": None,
            "timing_ms": 5,
        }
        filtered = _filter_response(response, "compact")
        for empty in ("message", "error_families", "error_analysis"):
            assert empty not in filtered, f"{empty} should be stripped when empty"
        assert filtered["timing_ms"] == 5

    def test_never_strips_empty_state_delta(self):
        # state_delta is always kept even when empty: callers index
        # response["state_delta"] directly, so stripping it would KeyError.
        response = {"success": True, "output": "2", "state_delta": {}}
        filtered = _filter_response(response, "compact")
        assert filtered["state_delta"] == {}

    def test_always_keeps_success_and_empty_output(self):
        response = {"success": True, "output": ""}
        filtered = _filter_response(response, "compact")
        assert filtered["success"] is True
        assert filtered["output"] == ""  # empty output is itself a result

    def test_keeps_error_analysis_when_non_empty(self):
        response = {"success": True, "output": "2", "error_analysis": {"severity": "warning"}}
        filtered = _filter_response(response, "compact")
        assert filtered["error_analysis"] == {"severity": "warning"}

    def test_keeps_zero_and_false_values(self):
        # 0 / False carry information - they are not "empty".
        response = {"success": True, "output": "2", "timing_ms": 0, "timed_out": False}
        filtered = _filter_response(response, "compact")
        assert filtered["timing_ms"] == 0
        assert filtered["timed_out"] is False


# ---------------------------------------------------------------------------
# Lean evaluate default: MATHEMATICA_RESPONSE_DETAIL env override
# ---------------------------------------------------------------------------


class TestLeanResponseDetailEnv:
    def test_default_is_compact(self, monkeypatch):
        monkeypatch.delenv("MATHEMATICA_RESPONSE_DETAIL", raising=False)
        from mathematica_mcp.server import _resolve_lean_response_detail

        assert _resolve_lean_response_detail() == "compact"

    def test_env_standard_restores_old_default(self, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_RESPONSE_DETAIL", "standard")
        from mathematica_mcp.server import _resolve_lean_response_detail

        assert _resolve_lean_response_detail() == "standard"

    def test_env_alias_is_normalized(self, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_RESPONSE_DETAIL", "medium")
        from mathematica_mcp.server import _resolve_lean_response_detail

        assert _resolve_lean_response_detail() == "standard"

    def test_invalid_env_falls_back_to_compact_with_warning(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("MATHEMATICA_RESPONSE_DETAIL", "bogus")
        from mathematica_mcp.server import _resolve_lean_response_detail

        with caplog.at_level(logging.WARNING, logger="mathematica_mcp"):
            assert _resolve_lean_response_detail() == "compact"
        assert any("bogus" in r.getMessage() for r in caplog.records)

    def test_module_constant_is_import_time_not_dynamic(self, monkeypatch):
        # _LEAN_RESPONSE_DETAIL is resolved ONCE at import. Changing the env after
        # import must NOT retroactively change the module constant. This pins the
        # deliberate import-time semantics so a later env tweak is a no-op by design.
        from mathematica_mcp import server as srv

        original = srv._LEAN_RESPONSE_DETAIL
        monkeypatch.setenv("MATHEMATICA_RESPONSE_DETAIL", "standard")
        assert original == srv._LEAN_RESPONSE_DETAIL  # constant unaffected by post-import env
        # The resolver, called fresh, DOES see the new env, proving the constant's
        # staleness is by design, not the resolver being broken.
        assert srv._resolve_lean_response_detail() == "standard"
