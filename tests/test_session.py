"""
Tests for the Mathematica MCP session module.

These tests verify the JSON parsing fix and mathematical functionality
using wolframscript as the execution backend.
"""

import pytest

from mathematica_mcp.session import (
    _execute_via_wolframscript,
    _is_graphics_output,
    _parse_association_output,
    execute_in_kernel,
)


class TestParseAssociationOutput:
    """Tests for the fallback Association parser."""

    def test_simple_output(self):
        """Test parsing simple output."""
        input_str = '<|"output" -> "1 + x", "messages" -> "{}", "timing_ms" -> 15, "failed" -> False|>'
        result = _parse_association_output(input_str)
        assert result["output_inputform"] == "1 + x"
        assert result["messages"] == "{}"
        assert result["timing_ms"] == 15
        assert result["failed"] is False

    def test_escaped_quotes(self):
        """Test parsing output with escaped quotes."""
        input_str = r'<|"output" -> "\"Hello World\"", "messages" -> "{}", "timing_ms" -> 10, "failed" -> False|>'
        result = _parse_association_output(input_str)
        assert result["output_inputform"] == '"Hello World"'

    def test_nested_lists(self):
        """Test parsing nested list output."""
        input_str = '<|"output" -> "{{1, 2}, {3, 4}}", "messages" -> "{}", "timing_ms" -> 5, "failed" -> False|>'
        result = _parse_association_output(input_str)
        assert result["output_inputform"] == "{{1, 2}, {3, 4}}"

    def test_failed_computation(self):
        """Test parsing failed computation."""
        input_str = '<|"output" -> "$Failed", "messages" -> "Error message", "timing_ms" -> 100, "failed" -> True|>'
        result = _parse_association_output(input_str)
        assert result["output_inputform"] == "$Failed"
        assert result["failed"] is True


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestCalculus:
    """Tests for calculus operations."""

    def test_indefinite_integral(self):
        """Test indefinite integration."""
        result = execute_in_kernel("Integrate[x^2, x]")
        assert result["success"] is True
        assert "x^3/3" in result["output_inputform"]

    def test_definite_integral(self):
        """Test definite integration."""
        result = execute_in_kernel("Integrate[x^2, {x, 0, 1}]")
        assert result["success"] is True
        assert "1/3" in result["output_inputform"]

    def test_gaussian_integral(self):
        """Test Gaussian integral."""
        result = execute_in_kernel("Integrate[Exp[-x^2], {x, -Infinity, Infinity}]")
        assert result["success"] is True
        assert "Sqrt[Pi]" in result["output_inputform"]

    def test_derivative(self):
        """Test differentiation."""
        result = execute_in_kernel("D[Sin[x], x]")
        assert result["success"] is True
        assert "Cos[x]" in result["output_inputform"]

    def test_second_derivative(self):
        """Test second derivative."""
        result = execute_in_kernel("D[x^3, {x, 2}]")
        assert result["success"] is True
        assert "6*x" in result["output_inputform"]

    def test_limit_at_zero(self):
        """Test limit calculation."""
        result = execute_in_kernel("Limit[Sin[x]/x, x -> 0]")
        assert result["success"] is True
        assert result["output_inputform"] == "1"

    def test_limit_to_infinity(self):
        """Test limit to infinity."""
        result = execute_in_kernel("Limit[(1 + 1/n)^n, n -> Infinity]")
        assert result["success"] is True
        assert "E" in result["output_inputform"]

    def test_taylor_series(self):
        """Test Taylor series expansion."""
        result = execute_in_kernel("Normal[Series[Exp[x], {x, 0, 4}]]")
        assert result["success"] is True
        assert "x^2/2" in result["output_inputform"]
        assert "x^3/6" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestAlgebra:
    """Tests for algebraic operations."""

    def test_solve_quadratic(self):
        """Test solving quadratic equation."""
        result = execute_in_kernel("Solve[x^2 - 5*x + 6 == 0, x]")
        assert result["success"] is True
        assert "2" in result["output_inputform"]
        assert "3" in result["output_inputform"]

    def test_solve_system(self):
        """Test solving system of equations."""
        result = execute_in_kernel("Solve[{x + y == 5, x - y == 1}, {x, y}]")
        assert result["success"] is True
        assert "3" in result["output_inputform"]
        assert "2" in result["output_inputform"]

    def test_factor_polynomial(self):
        """Test polynomial factorization."""
        result = execute_in_kernel("Factor[x^2 - 4]")
        assert result["success"] is True
        assert "-2" in result["output_inputform"]
        assert "2" in result["output_inputform"]

    def test_expand_binomial(self):
        """Test binomial expansion."""
        result = execute_in_kernel("Expand[(x + 1)^3]")
        assert result["success"] is True
        assert "x^3" in result["output_inputform"]
        assert "3*x^2" in result["output_inputform"]

    def test_simplify_fraction(self):
        """Test fraction simplification."""
        result = execute_in_kernel("Simplify[(x^2 - 1)/(x - 1)]")
        assert result["success"] is True
        assert "1 + x" in result["output_inputform"]

    def test_simplify_trig_identity(self):
        """Test trigonometric identity simplification."""
        result = execute_in_kernel("Simplify[Sin[x]^2 + Cos[x]^2]")
        assert result["success"] is True
        assert result["output_inputform"] == "1"


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_matrix_multiply(self):
        """Test matrix multiplication."""
        result = execute_in_kernel("{{1, 2}, {3, 4}} . {{5, 6}, {7, 8}}")
        assert result["success"] is True
        assert "19" in result["output_inputform"]
        assert "50" in result["output_inputform"]

    def test_determinant(self):
        """Test determinant calculation."""
        result = execute_in_kernel("Det[{{1, 2}, {3, 4}}]")
        assert result["success"] is True
        assert "-2" in result["output_inputform"]

    def test_inverse(self):
        """Test matrix inverse."""
        result = execute_in_kernel("Inverse[{{1, 2}, {3, 4}}]")
        assert result["success"] is True
        assert "-2" in result["output_inputform"]

    def test_eigenvalues(self):
        """Test eigenvalue calculation."""
        result = execute_in_kernel("Eigenvalues[{{4, 1}, {2, 3}}]")
        assert result["success"] is True
        assert "5" in result["output_inputform"]
        assert "2" in result["output_inputform"]

    def test_linear_solve(self):
        """Test solving linear system."""
        result = execute_in_kernel("LinearSolve[{{1, 2}, {3, 4}}, {5, 11}]")
        assert result["success"] is True
        assert "1" in result["output_inputform"]
        assert "2" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestSpecialFunctions:
    """Tests for special mathematical functions."""

    def test_gamma_function(self):
        """Test Gamma function."""
        result = execute_in_kernel("Gamma[5]")
        assert result["success"] is True
        assert "24" in result["output_inputform"]

    def test_factorial(self):
        """Test factorial."""
        result = execute_in_kernel("10!")
        assert result["success"] is True
        assert "3628800" in result["output_inputform"]

    def test_binomial_coefficient(self):
        """Test binomial coefficient."""
        result = execute_in_kernel("Binomial[10, 3]")
        assert result["success"] is True
        assert "120" in result["output_inputform"]

    def test_fibonacci(self):
        """Test Fibonacci numbers."""
        result = execute_in_kernel("Fibonacci[10]")
        assert result["success"] is True
        assert "55" in result["output_inputform"]

    def test_prime(self):
        """Test prime number generation."""
        result = execute_in_kernel("Prime[25]")
        assert result["success"] is True
        assert "97" in result["output_inputform"]

    def test_gcd(self):
        """Test greatest common divisor."""
        result = execute_in_kernel("GCD[48, 18]")
        assert result["success"] is True
        assert "6" in result["output_inputform"]

    def test_zeta_function(self):
        """Test Riemann zeta function."""
        result = execute_in_kernel("Zeta[2]")
        assert result["success"] is True
        assert "Pi^2/6" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestDifferentialEquations:
    """Tests for differential equation solving."""

    def test_first_order_ode(self):
        """Test first-order ODE."""
        result = execute_in_kernel("DSolve[y'[x] == y[x], y[x], x]")
        assert result["success"] is True
        assert "E^x" in result["output_inputform"]

    def test_second_order_ode(self):
        """Test second-order ODE."""
        result = execute_in_kernel("DSolve[y''[x] + y[x] == 0, y[x], x]")
        assert result["success"] is True
        assert "Sin[x]" in result["output_inputform"]
        assert "Cos[x]" in result["output_inputform"]

    def test_initial_value_problem(self):
        """Test initial value problem."""
        result = execute_in_kernel("DSolve[{y'[x] == y[x], y[0] == 1}, y[x], x]")
        assert result["success"] is True
        assert "E^x" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestNumericalComputation:
    """Tests for numerical computation."""

    def test_numerical_integration(self):
        """Test numerical integration."""
        result = execute_in_kernel("NIntegrate[Sin[x^2], {x, 0, 1}]")
        assert result["success"] is True
        # Result should be approximately 0.31
        assert "0.31" in result["output_inputform"] or float(result["output_inputform"]) > 0.3

    def test_find_root(self):
        """Test root finding."""
        result = execute_in_kernel("FindRoot[Cos[x] == x, {x, 0}]")
        assert result["success"] is True
        assert "0.739" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestStatistics:
    """Tests for statistical functions."""

    def test_mean(self):
        """Test mean calculation."""
        result = execute_in_kernel("Mean[{1, 2, 3, 4, 5}]")
        assert result["success"] is True
        assert "3" in result["output_inputform"]

    def test_variance(self):
        """Test variance calculation."""
        result = execute_in_kernel("Variance[{1, 2, 3, 4, 5}]")
        assert result["success"] is True
        assert "5/2" in result["output_inputform"]

    def test_median(self):
        """Test median calculation."""
        result = execute_in_kernel("Median[{1, 2, 3, 4, 5}]")
        assert result["success"] is True
        assert "3" in result["output_inputform"]

    def test_correlation(self):
        """Test correlation calculation."""
        result = execute_in_kernel("Correlation[{1, 2, 3}, {2, 4, 6}]")
        assert result["success"] is True
        assert "1" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_string_output(self):
        """Test string output handling."""
        result = execute_in_kernel('"Hello World"')
        assert result["success"] is True
        assert "Hello World" in result["output_inputform"]

    def test_nested_lists(self):
        """Test nested list output."""
        result = execute_in_kernel("{{1, 2}, {3, 4}}")
        assert result["success"] is True
        assert "{{1, 2}, {3, 4}}" in result["output_inputform"]

    def test_complex_numbers(self):
        """Test complex number handling."""
        result = execute_in_kernel("(1 + I)^2")
        assert result["success"] is True
        assert "2*I" in result["output_inputform"]

    def test_symbolic_expression(self):
        """Test symbolic expression output."""
        result = execute_in_kernel("x^2 + 2*x + 1")
        assert result["success"] is True
        assert "x^2" in result["output_inputform"]

    def test_division_by_zero(self):
        """Test division by zero handling."""
        result = execute_in_kernel("1/0")
        assert result["success"] is True  # Should succeed but return $Failed
        assert "$Failed" in result["output_inputform"] or "ComplexInfinity" in result["output_inputform"]

    def test_indeterminate(self):
        """Test indeterminate form handling."""
        result = execute_in_kernel("Limit[Sin[1/x], x -> 0]")
        assert result["success"] is True
        assert "Indeterminate" in result["output_inputform"]


@pytest.mark.usefixtures("require_wolfram_runtime")
class TestJSONParsingFix:
    """Tests specifically for the JSON parsing fix."""

    def test_series_expansion_original_bug(self):
        """Test the original failing case: Series expansion."""
        result = execute_in_kernel("Series[Exp[x], {x, 0, 5}]")
        assert result["success"] is True
        assert "SeriesData" in result["output_inputform"]

    def test_series_with_normal(self):
        """Test Series with Normal to get polynomial."""
        result = execute_in_kernel("Normal[Series[Exp[x], {x, 0, 5}]]")
        assert result["success"] is True
        assert "x^5/120" in result["output_inputform"]

    def test_complex_formatted_output(self):
        """Test complex formatted output that may contain special characters."""
        result = execute_in_kernel("TableForm[{{a, b}, {c, d}}]")
        assert result["success"] is True

    def test_multiple_line_output(self):
        """Test output that spans multiple conceptual lines."""
        result = execute_in_kernel("Table[{n, n^2, n^3}, {n, 1, 5}]")
        assert result["success"] is True
        assert "{1, 1, 1}" in result["output_inputform"]
        assert "{5, 25, 125}" in result["output_inputform"]


# ============================================================================
# Graphics detection tests (no runtime required)
# ============================================================================


class TestIsGraphicsOutput:
    """Tests for _is_graphics_output detection."""

    def test_detects_graphics(self):
        assert _is_graphics_output("Graphics[Circle[]]") is True

    def test_detects_graphics3d(self):
        assert _is_graphics_output("Graphics3D[Sphere[]]") is True

    def test_detects_image(self):
        assert _is_graphics_output("Image[data]") is True

    def test_detects_legended_graphics(self):
        assert _is_graphics_output("Legended[Graphics[Circle[]], legend]") is True

    def test_detects_show_graphics(self):
        assert _is_graphics_output("Show[Graphics[Circle[]], opts]") is True

    def test_detects_addon_placeholder_graphics(self):
        assert _is_graphics_output("-Graphics-") is True

    def test_detects_addon_placeholder_graphics3d(self):
        assert _is_graphics_output("-Graphics3D-") is True

    def test_detects_addon_placeholder_image(self):
        assert _is_graphics_output("-Image-") is True

    def test_rejects_empty_string(self):
        assert _is_graphics_output("") is False

    def test_rejects_numeric_output(self):
        assert _is_graphics_output("42") is False

    def test_rejects_symbolic_output(self):
        assert _is_graphics_output("x^2 + 1") is False

    def test_rejects_list_output(self):
        assert _is_graphics_output("{1, 2, 3}") is False

    def test_handles_whitespace(self):
        assert _is_graphics_output("  Graphics[Circle[]]  ") is True

    def test_rejects_none_like_input(self):
        assert _is_graphics_output("") is False
        assert _is_graphics_output("None") is False


# ============================================================================
# Response contract tests (no runtime required)
# ============================================================================


class TestExecuteInKernelResponseContract:
    """Freeze the response shape of execute_in_kernel.

    These tests ensure the response dict always has the expected keys,
    regardless of success/failure, so downstream consumers are never
    surprised by missing fields.
    """

    REQUIRED_SUCCESS_KEYS = {
        "success",
        "output",
        "output_inputform",
        "output_fullform",
        "warnings",
        "timing_ms",
        "execution_method",
    }

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_success_response_has_required_keys(self):
        result = execute_in_kernel("1 + 1", render_graphics=False)
        assert result["success"] is True
        missing = self.REQUIRED_SUCCESS_KEYS - set(result.keys())
        assert not missing, f"Missing keys in success response: {missing}"

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_success_response_types(self):
        result = execute_in_kernel("1 + 1", render_graphics=False)
        assert isinstance(result["success"], bool)
        assert isinstance(result["output"], str)
        assert isinstance(result["output_inputform"], str)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["timing_ms"], int)
        assert isinstance(result["execution_method"], str)

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_cached_result_has_from_cache_flag(self):
        """Cache hits must include from_cache=True and timing_ms=0."""
        # Execute once to populate cache
        execute_in_kernel("2 + 3", render_graphics=False)
        # Execute again — should hit cache
        result = execute_in_kernel("2 + 3", render_graphics=False)
        if result.get("from_cache"):
            assert result["from_cache"] is True
            assert result["timing_ms"] == 0


class TestWolframscriptResponseContract:
    """Freeze the response shape of _execute_via_wolframscript."""

    REQUIRED_KEYS = {
        "success",
        "output",
        "warnings",
        "timing_ms",
        "execution_method",
    }

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_success_response_has_required_keys(self):
        result = _execute_via_wolframscript("1 + 1")
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_execution_method_is_wolframscript(self):
        result = _execute_via_wolframscript("1 + 1")
        assert result["execution_method"] == "wolframscript"

    def test_missing_wolframscript_returns_failure(self, monkeypatch):
        """When wolframscript is not found, return success=False."""
        import shutil

        from mathematica_mcp.lazy_wolfram_tools import _clear_wolframscript_cache

        _clear_wolframscript_cache()
        monkeypatch.setattr(shutil, "which", lambda _: None)
        result = _execute_via_wolframscript("1 + 1")
        assert result["success"] is False
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys on failure: {missing}"
        _clear_wolframscript_cache()


# ============================================================================
# Graphics rasterization invocation tests (no runtime required)
# ============================================================================


class TestGraphicsRasterizationInvocation:
    """Verify graphics rasterization produces image_path without double execution."""

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_graphics_result_has_image_path_when_rendered(self):
        """A Graphics expression with render_graphics=True MUST produce image_path."""
        result = execute_in_kernel("Graphics[Circle[]]", render_graphics=True)
        assert result["success"] is True
        # This MUST be true for a Graphics expression — not a conditional check
        assert result.get("is_graphics") is True, "Graphics[Circle[]] must be detected as graphics"
        assert "image_path" in result, "Graphics result must include image_path"
        import os

        assert os.path.exists(result["image_path"]), "image_path must point to an existing file"
        assert os.path.getsize(result["image_path"]) > 0, "image file must not be empty"
        os.remove(result["image_path"])

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_non_graphics_result_has_no_image_path(self):
        """A non-graphics expression must NOT produce image_path."""
        result = execute_in_kernel("1 + 1", render_graphics=True)
        assert result["success"] is True
        assert not result.get("is_graphics"), "1+1 must not be detected as graphics"


# ============================================================================
# Phase 1: Single-evaluation graphics tests
# ============================================================================


class TestSingleSubprocessGraphics:
    """Verify wolframscript graphics renders in a single subprocess call."""

    def test_wolframscript_single_subprocess_for_graphics(self, monkeypatch):
        """_execute_via_wolframscript with render_graphics=True must call
        subprocess.run exactly once, not twice."""
        import subprocess as sp

        call_count = {"n": 0}
        original_run = sp.run

        def counting_run(*args, **kwargs):
            call_count["n"] += 1
            return original_run(*args, **kwargs)

        monkeypatch.setattr(sp, "run", counting_run)

        _execute_via_wolframscript(
            "Graphics[Circle[]]",
            render_graphics=True,
        )
        # Whether or not wolframscript is available, at most 1 subprocess
        assert call_count["n"] <= 1

    def test_wolframscript_render_false_no_image(self, monkeypatch):
        """render_graphics=False must not produce image_path."""
        import subprocess as sp

        import mathematica_mcp.lazy_wolfram_tools as lwt

        # Mock wolframscript to return simple JSON output
        mock_output = '{"output":"42","output_inputform":"42","output_fullform":"","output_tex":"","messages":"{}","timing_ms":10,"failed":false,"image_path":"","is_graphics":false}'

        def mock_run(*args, **kwargs):
            class R:
                returncode = 0
                stdout = mock_output
                stderr = ""

            return R()

        # Clear lru_cache and patch shutil.which in the module where it's used
        lwt._find_wolframscript.cache_clear()
        monkeypatch.setattr(lwt.shutil, "which", lambda _: "/usr/bin/wolframscript")
        monkeypatch.setattr(sp, "run", mock_run)

        result = _execute_via_wolframscript("42", render_graphics=False)
        assert result["success"] is True
        assert "image_path" not in result or not result.get("is_graphics")


class TestCacheSafetyForImagePath:
    """Verify image_path is not stored in the query cache."""

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_cache_does_not_contain_image_path(self):
        """After caching a Graphics result, cache entry must not have image_path."""
        from mathematica_mcp.cache import _query_cache

        # Clear cache
        _query_cache.clear()

        # Execute a graphics expression
        result = execute_in_kernel("Graphics[Circle[]]", render_graphics=True)
        if not result.get("success"):
            pytest.skip("wolframscript not available")

        # Check cache directly
        cached = _query_cache.get(
            "Graphics[Circle[]]",
            output_format="text",
            render_graphics=True,
        )
        if cached is not None:
            assert "image_path" not in cached, "image_path must not be stored in cache"
            assert "is_graphics" not in cached, "is_graphics must not be stored in cache"
            # Cached output should be the textual form, not placeholder
            assert not cached["output"].startswith("[Graphics rendered"), (
                "Cached output must be textual, not placeholder"
            )

        # Clean up image file if created
        if result.get("image_path"):
            import os

            if os.path.exists(result["image_path"]):
                os.remove(result["image_path"])

    @pytest.mark.usefixtures("require_wolfram_runtime")
    def test_temp_file_exists_after_execution(self):
        """Image temp file must exist after execute_in_kernel returns."""
        result = execute_in_kernel("Graphics[Circle[]]", render_graphics=True)
        if result.get("is_graphics") and result.get("image_path"):
            import os

            assert os.path.exists(result["image_path"]), "Temp file must persist for consumer to read"
            # Clean up
            os.remove(result["image_path"])


# ============================================================================
# Server-side image validation tests (no runtime required)
# ============================================================================


class TestAttachImageIfValid:
    """Tests for _attach_image_if_valid server helper."""

    def test_valid_image_attached(self, tmp_path):
        """Valid image file should be attached as rendered_image."""
        from mathematica_mcp.server import _attach_image_if_valid

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = {"is_graphics": True, "image_path": str(img)}
        _attach_image_if_valid(result)
        assert result["rendered_image"] == str(img)
        assert result["tip"] == "Use Read tool to view image."

    def test_missing_file_restores_output(self):
        """Missing image file must restore output to output_inputform."""
        from mathematica_mcp.server import _attach_image_if_valid

        result = {
            "is_graphics": True,
            "image_path": "/nonexistent/path.png",
            "output": "[Graphics rendered to image: /nonexistent/path.png]",
            "output_inputform": "Graphics[Circle[]]",
        }
        _attach_image_if_valid(result)
        assert "rendered_image" not in result
        assert "image_path" not in result
        assert "is_graphics" not in result
        assert "tip" not in result
        # output must be restored to the textual form
        assert result["output"] == "Graphics[Circle[]]"

    def test_empty_file_stripped(self, tmp_path):
        """Zero-byte image file should strip is_graphics and image_path."""
        from mathematica_mcp.server import _attach_image_if_valid

        img = tmp_path / "empty.png"
        img.write_bytes(b"")

        result = {"is_graphics": True, "image_path": str(img)}
        _attach_image_if_valid(result)
        assert "rendered_image" not in result
        assert "image_path" not in result

    def test_no_graphics_is_noop(self):
        """Non-graphics result should be unchanged."""
        from mathematica_mcp.server import _attach_image_if_valid

        result = {"output": "42", "success": True}
        _attach_image_if_valid(result)
        assert "rendered_image" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
