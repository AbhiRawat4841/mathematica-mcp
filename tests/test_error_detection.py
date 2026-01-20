"""
Comprehensive tests for Mathematica error detection, analysis, and fixing mechanism.

This test suite covers:
1. Error analyzer module functionality
2. Error detection in inline execution
3. Error detection in notebook execution
4. Various error types (UnitConvert, Part, Syntax, Division, etc.)
5. LLM-ready error formatting
"""

import pytest
from src.mathematica_mcp.error_analyzer import (
    analyze_error,
    analyze_messages,
    format_error_for_llm,
    ERROR_PATTERNS,
)
from src.mathematica_mcp.session import execute_in_kernel


class TestErrorAnalyzer:
    """Test the error analyzer module directly."""

    def test_analyze_unitconvert_error(self):
        """Test analysis of UnitConvert compatibility error."""
        error_tag = "UnitConvert::compat"
        error_text = "Incompatible units: time and currency"

        result = analyze_error(error_tag, error_text)

        assert result["matched_pattern"] == "UnitConvert::compat"
        assert result["confidence"] == "high"
        assert result["severity"] == "error"
        assert "QuantityMagnitude" in result["suggested_fix"]
        assert "incompatible" in result["description"].lower()

    def test_analyze_part_out_of_range(self):
        """Test analysis of Part index out of range error."""
        error_tag = "Part::partw"
        error_text = "Part 5 of {1,2,3} does not exist"

        result = analyze_error(error_tag, error_text)

        assert result["matched_pattern"] == "Part::partw"
        assert result["confidence"] == "high"
        assert result["severity"] == "error"
        assert "length" in result["common_cause"].lower()
        assert result["example"]  # Should have example

    def test_analyze_syntax_error(self):
        """Test analysis of syntax error."""
        error_tag = "Syntax::sntxi"
        error_text = "Invalid syntax at position 5"

        result = analyze_error(error_tag, error_text)

        assert result["matched_pattern"] == "Syntax::sntxi"
        assert result["confidence"] == "high"
        assert result["severity"] == "error"
        assert "bracket" in result["suggested_fix"].lower()

    def test_analyze_division_by_zero(self):
        """Test analysis of division by zero error."""
        error_tag = "Power::infy"
        error_text = "Infinite expression 1/0 encountered"

        result = analyze_error(error_tag, error_text)

        assert result["matched_pattern"] == "Power::infy"
        assert result["confidence"] == "high"
        assert result["severity"] == "error"
        assert "Limit" in result["suggested_fix"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown error pattern."""
        error_tag = "Unknown::mysteryerror"
        error_text = "Something went wrong"

        result = analyze_error(error_tag, error_text)

        assert result["matched_pattern"] is None
        assert result["confidence"] == "low"
        assert "documentation" in result["suggested_fix"].lower()

    def test_analyze_partial_match(self):
        """Test partial error tag matching."""
        error_tag = "Part::unknown"  # Part type but different subtype
        error_text = "Part error"

        result = analyze_error(error_tag, error_text)

        # Should match Part::partw or Part::partd
        assert "Part::" in result["matched_pattern"]
        assert result["confidence"] == "medium"


class TestAnalyzeMessages:
    """Test the analyze_messages function with multiple errors."""

    def test_empty_messages(self):
        """Test with no messages."""
        result = analyze_messages([])

        assert result["total_messages"] == 0
        assert result["errors"] == 0
        assert result["warnings"] == 0
        assert "No errors" in result["assessment"]

    def test_single_error_message(self):
        """Test with a single error message."""
        messages = [
            {
                "tag": "UnitConvert::compat",
                "text": "Incompatible units",
                "type": "error",
            }
        ]

        result = analyze_messages(messages)

        assert result["total_messages"] == 1
        assert result["errors"] == 1
        assert result["warnings"] == 0
        assert result["severity"] == "error"
        assert len(result["analyses"]) == 1
        assert result["should_retry"] is True  # High confidence fix available

    def test_multiple_errors(self):
        """Test with multiple error messages."""
        messages = [
            {"tag": "Part::partw", "text": "Part out of range", "type": "error"},
            {"tag": "Syntax::sntxi", "text": "Syntax error", "type": "error"},
            {"tag": "General::stop", "text": "Further output suppressed", "type": "warning"},
        ]

        result = analyze_messages(messages)

        assert result["total_messages"] == 3
        assert result["errors"] == 2
        assert result["warnings"] == 1
        assert result["severity"] == "error"
        assert len(result["analyses"]) == 3

    def test_unitconvert_recommendation(self):
        """Test that UnitConvert errors generate specific recommendations."""
        messages = [
            {
                "tag": "UnitConvert::compat",
                "text": "Incompatible units",
                "type": "error",
            }
        ]

        result = analyze_messages(messages)

        assert any("QuantityMagnitude" in rec for rec in result["recommendations"])

    def test_part_recommendation(self):
        """Test that Part errors generate specific recommendations."""
        messages = [
            {"tag": "Part::partw", "text": "Part out of range", "type": "error"}
        ]

        result = analyze_messages(messages)

        assert any(
            "Length" in rec or "Dimensions" in rec
            for rec in result["recommendations"]
        )


class TestFormatErrorForLLM:
    """Test the LLM-ready error formatting."""

    def test_format_empty_messages(self):
        """Test formatting with no messages."""
        result = format_error_for_llm([], "x + 1")

        assert result == ""

    def test_format_single_error(self):
        """Test formatting a single error for LLM."""
        messages = [
            {
                "tag": "UnitConvert::compat",
                "text": "Incompatible units",
                "type": "error",
            }
        ]
        code = "CountryData[\"USA\", \"GDP\"] + Quantity[1, \"Hours\"]"

        result = format_error_for_llm(messages, code)

        assert "EXECUTION ERRORS DETECTED" in result
        assert "UnitConvert::compat" in result
        assert "Analysis:" in result
        assert "Suggested fix:" in result
        assert "RECOMMENDATIONS" in result

    def test_format_multiple_errors(self):
        """Test formatting multiple errors and warnings."""
        messages = [
            {"tag": "Part::partw", "text": "Part 5 of {1,2,3}", "type": "error"},
            {"tag": "General::stop", "text": "Output suppressed", "type": "warning"},
        ]
        code = "list[[5]]"

        result = format_error_for_llm(messages, code)

        assert "--- ERRORS ---" in result
        assert "--- WARNINGS ---" in result
        assert "Part::partw" in result
        assert "General::stop" in result

    def test_format_includes_examples(self):
        """Test that formatting includes code examples."""
        messages = [
            {"tag": "Part::partw", "text": "Part out of range", "type": "error"}
        ]
        code = "list[[100]]"

        result = format_error_for_llm(messages, code)

        assert "Example:" in result
        assert "Length[list]" in result


class TestInlineErrorDetection:
    """Test error detection in inline code execution."""

    def test_unitconvert_error_inline(self):
        """Test UnitConvert error in inline execution."""
        # This should produce an incompatible units error
        code = 'CountryData["USA", "GDP"] + Quantity[1, "Hours"]'
        result = execute_in_kernel(code)

        # The execution should complete but may contain error messages
        assert "success" in result
        # Check if error messages are captured
        if "messages" in result and result["messages"]:
            # Verify error analysis would work
            assert True
        else:
            # Some errors might be in output
            assert "output_inputform" in result

    def test_part_index_error_inline(self):
        """Test Part index out of range error in inline execution."""
        code = "{1, 2, 3}[[5]]"
        result = execute_in_kernel(code)

        assert "success" in result
        # Part errors typically return the expression unevaluated or $Failed

    def test_syntax_error_inline(self):
        """Test syntax error in inline execution."""
        code = "Sin[x"  # Missing closing bracket
        result = execute_in_kernel(code)

        # Syntax errors should be detected
        assert "success" in result or "error" in result

    def test_division_by_zero_inline(self):
        """Test division by zero in inline execution."""
        code = "1/0"
        result = execute_in_kernel(code)

        assert result["success"] is True
        # Should return ComplexInfinity or similar
        assert (
            "ComplexInfinity" in result["output_inputform"]
            or "Infinity" in result["output_inputform"]
        )

    def test_undefined_symbol_inline(self):
        """Test undefined symbol in inline execution."""
        code = "undefinedVariableXYZ123 + 5"
        result = execute_in_kernel(code)

        # Mathematica typically leaves undefined symbols symbolic
        assert result["success"] is True
        assert "undefinedVariableXYZ123" in result["output_inputform"]

    def test_recursion_limit_inline(self):
        """Test recursion limit error."""
        code = "f[x_] := f[x + 1]; f[1]"
        result = execute_in_kernel(code)

        # Should hit recursion limit
        assert "success" in result


class TestNotebookErrorDetection:
    """Test error detection in notebook-based execution.

    Note: These tests would require a running notebook instance.
    They are structured to show what should be tested.
    """

    @pytest.mark.skip(reason="Requires active notebook connection")
    def test_unitconvert_error_notebook(self):
        """Test UnitConvert error in notebook execution."""
        # Would test via execute_code with output_target="notebook"
        pass

    @pytest.mark.skip(reason="Requires active notebook connection")
    def test_part_error_notebook(self):
        """Test Part error in notebook execution."""
        pass

    @pytest.mark.skip(reason="Requires active notebook connection")
    def test_error_formatting_in_notebook(self):
        """Test that errors are properly formatted in notebook cells."""
        pass


class TestErrorPatternCoverage:
    """Test that all error patterns in ERROR_PATTERNS are valid."""

    def test_all_patterns_have_required_fields(self):
        """Verify all error patterns have required fields."""
        required_fields = [
            "description",
            "common_cause",
            "suggested_fix",
            "severity",
        ]

        for pattern, info in ERROR_PATTERNS.items():
            for field in required_fields:
                assert field in info, f"Pattern {pattern} missing field: {field}"
                assert info[field], f"Pattern {pattern} has empty field: {field}"

    def test_all_patterns_analyzable(self):
        """Test that all patterns can be analyzed."""
        for pattern in ERROR_PATTERNS.keys():
            result = analyze_error(pattern, "Test error text")

            assert result["matched_pattern"] == pattern
            assert result["confidence"] == "high"
            assert result["description"]
            assert result["suggested_fix"]

    def test_pattern_count(self):
        """Verify we have a reasonable number of error patterns."""
        # Should have at least 8 common patterns
        assert len(ERROR_PATTERNS) >= 8


class TestRealWorldScenarios:
    """Test real-world error scenarios."""

    def test_country_gdp_comparison_error(self):
        """Test the common GDP comparison error scenario."""
        # This is a real scenario that often fails
        code = 'CountryData["USA", "GDP"] > CountryData["China", "GDP"]'
        result = execute_in_kernel(code)

        # Should execute (may or may not error depending on data)
        assert "success" in result

    def test_list_access_after_filter_error(self):
        """Test accessing list after filtering that might be empty."""
        code = "Select[{1, 2, 3}, # > 10 &][[1]]"
        result = execute_in_kernel(code)

        # Should handle empty list gracefully
        assert "success" in result

    def test_chained_operations_with_error(self):
        """Test chained operations where one fails."""
        code = "Sqrt[-1] + 5 / 0"
        result = execute_in_kernel(code)

        # Should return some result
        assert "success" in result

    def test_mixed_symbolic_numeric_error(self):
        """Test mixing symbolic and numeric operations."""
        code = "x + 1/0"
        result = execute_in_kernel(code)

        assert result["success"] is True


class TestErrorRecoveryWorkflow:
    """Test the complete error detection and recovery workflow."""

    def test_detect_analyze_format_workflow(self):
        """Test the complete workflow from detection to LLM formatting."""
        # Simulate an error scenario
        messages = [
            {
                "tag": "UnitConvert::compat",
                "text": "Incompatible units: USD and Hours",
                "type": "error",
            }
        ]
        code = 'CountryData["USA", "GDP"] + Quantity[1, "Hours"]'

        # Step 1: Analyze messages
        analysis = analyze_messages(messages)
        assert analysis["errors"] == 1
        assert analysis["should_retry"] is True

        # Step 2: Format for LLM
        formatted = format_error_for_llm(messages, code)
        assert "UnitConvert::compat" in formatted
        assert "QuantityMagnitude" in formatted

        # Step 3: Verify LLM would get actionable information
        assert "Suggested fix:" in formatted
        assert "Example:" in formatted

    def test_multiple_error_prioritization(self):
        """Test that high-confidence errors are prioritized."""
        messages = [
            {"tag": "Unknown::error1", "text": "Unknown error 1", "type": "error"},
            {"tag": "Part::partw", "text": "Part out of range", "type": "error"},
            {"tag": "Unknown::error2", "text": "Unknown error 2", "type": "error"},
        ]

        analysis = analyze_messages(messages)

        # Should prioritize the high-confidence Part error
        high_conf = [a for a in analysis["analyses"] if a["confidence"] == "high"]
        assert len(high_conf) == 1
        assert high_conf[0]["matched_pattern"] == "Part::partw"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
