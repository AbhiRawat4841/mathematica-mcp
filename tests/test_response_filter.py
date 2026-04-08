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
            "error_analysis": {},
            "route_variant": "compute",
            "execution_path": "addon_cli",
            "overall_timing_ms": 10,
            "error_summary": "...",
            "code": "1+1",
            "requested_style": "compute",
        }
        filtered = _filter_response(response, "compact")
        for field in [
            "messages",
            "output_fullform",
            "output_tex",
            "detailed_analyses",
            "llm_error_report",
            "error_analysis",
            "route_variant",
            "execution_path",
            "overall_timing_ms",
            "error_summary",
            "code",
            "requested_style",
        ]:
            assert field not in filtered, f"{field} should be stripped"

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
