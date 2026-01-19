"""
Tests for the Mathematica MCP session module.

These tests verify the JSON parsing fix and mathematical functionality
using wolframscript as the execution backend.
"""

import pytest
from src.mathematica_mcp.session import (
    execute_in_kernel,
    _parse_association_output,
    _execute_via_wolframscript,
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


class TestNumericalComputation:
    """Tests for numerical computation."""

    def test_numerical_integration(self):
        """Test numerical integration."""
        result = execute_in_kernel("NIntegrate[Sin[x^2], {x, 0, 1}]")
        assert result["success"] is True
        # Result should be approximately 0.31
        assert (
            "0.31" in result["output_inputform"]
            or float(result["output_inputform"]) > 0.3
        )

    def test_find_root(self):
        """Test root finding."""
        result = execute_in_kernel("FindRoot[Cos[x] == x, {x, 0}]")
        assert result["success"] is True
        assert "0.739" in result["output_inputform"]


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
        assert (
            "$Failed" in result["output_inputform"]
            or "ComplexInfinity" in result["output_inputform"]
        )

    def test_indeterminate(self):
        """Test indeterminate form handling."""
        result = execute_in_kernel("Limit[Sin[1/x], x -> 0]")
        assert result["success"] is True
        assert "Indeterminate" in result["output_inputform"]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
