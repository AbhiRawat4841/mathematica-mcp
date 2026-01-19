"""
Tests for the derivation verification functionality.

These tests verify that step-by-step mathematical derivations
can be validated using Mathematica's Simplify function.
"""

import pytest
from src.mathematica_mcp.session import execute_in_kernel


class TestDerivationVerification:
    """Tests for verifying mathematical derivations step by step."""

    def test_algebraic_identity_valid(self):
        """Test that valid algebraic identity steps are recognized."""
        # (x+y)^2 = x^2 + 2xy + y^2
        step1 = "(x + y)^2"
        step2 = "x^2 + 2*x*y + y^2"

        result = execute_in_kernel(f"Simplify[{step1} - ({step2})]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_trigonometric_identity_valid(self):
        """Test that valid trig identity steps are recognized."""
        # sin^2(x) + cos^2(x) = 1
        step1 = "Sin[x]^2 + Cos[x]^2"
        step2 = "1"

        result = execute_in_kernel(f"Simplify[{step1} - ({step2})]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_factorization_valid(self):
        """Test that valid factorization is recognized."""
        # x^2 - 1 = (x-1)(x+1)
        step1 = "x^2 - 1"
        step2 = "(x - 1)*(x + 1)"

        result = execute_in_kernel(f"Simplify[{step1} - ({step2})]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_invalid_step_detected(self):
        """Test that invalid derivation steps are detected."""
        # x^2 + 1 != x^2 - 1 (invalid step)
        step1 = "x^2 + 1"
        step2 = "x^2 - 1"

        result = execute_in_kernel(f"Simplify[{step1} - ({step2})]")
        assert result["success"] is True
        assert result["output_inputform"] != "0"
        assert "2" in result["output_inputform"]  # Difference should be 2

    def test_double_angle_formula(self):
        """Test double angle formula derivation."""
        # sin(2x) = 2*sin(x)*cos(x)
        step1 = "Sin[2*x]"
        step2 = "2*Sin[x]*Cos[x]"

        result = execute_in_kernel(f"Simplify[{step1} - ({step2})]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_logarithm_laws(self):
        """Test logarithm laws."""
        # log(a*b) = log(a) + log(b)
        result = execute_in_kernel(
            "Simplify[Log[a*b] - (Log[a] + Log[b]), Assumptions -> {a > 0, b > 0}]"
        )
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_exponential_laws(self):
        """Test exponential laws."""
        # e^(a+b) = e^a * e^b
        step1 = "Exp[a + b]"
        step2 = "Exp[a]*Exp[b]"

        result = execute_in_kernel(f"Simplify[{step1} - ({step2})]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_derivative_verification(self):
        """Test that derivative calculations are correct."""
        # d/dx(x^3) = 3x^2
        result = execute_in_kernel("Simplify[D[x^3, x] - 3*x^2]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_integral_verification(self):
        """Test that integral calculations are correct."""
        # integral of 3x^2 = x^3 (ignoring constant)
        result = execute_in_kernel("Simplify[D[Integrate[3*x^2, x], x] - 3*x^2]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_multi_step_derivation(self):
        """Test a multi-step derivation."""
        steps = [
            "(x + 1)^3",
            "(x + 1)^2 * (x + 1)",
            "(x^2 + 2*x + 1) * (x + 1)",
            "x^3 + 3*x^2 + 3*x + 1",
        ]

        for i in range(len(steps) - 1):
            result = execute_in_kernel(f"Simplify[({steps[i]}) - ({steps[i + 1]})]")
            assert result["success"] is True, f"Step {i + 1} failed"
            assert result["output_inputform"] == "0", (
                f"Step {i + 1} -> {i + 2} is not valid: {result['output']}"
            )

    def test_euler_identity(self):
        """Test Euler's identity: e^(i*pi) + 1 = 0."""
        result = execute_in_kernel("Simplify[Exp[I*Pi] + 1]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"

    def test_de_moivre_theorem(self):
        """Test De Moivre's theorem."""
        # (cos(x) + i*sin(x))^n = cos(n*x) + i*sin(n*x)
        # Test for n=3
        step1 = "(Cos[x] + I*Sin[x])^3"
        step2 = "Cos[3*x] + I*Sin[3*x]"

        result = execute_in_kernel(f"ComplexExpand[Simplify[{step1} - ({step2})]]")
        assert result["success"] is True
        assert result["output_inputform"] == "0"


class TestSymbolicEquivalence:
    """Tests for checking symbolic equivalence."""

    def test_equivalent_expressions(self):
        """Test that equivalent expressions are recognized."""
        pairs = [
            ("(a + b)^2", "a^2 + 2*a*b + b^2"),
            ("Sin[2*x]", "2*Sin[x]*Cos[x]"),
            ("Cos[2*x]", "Cos[x]^2 - Sin[x]^2"),
            ("Tan[x]", "Sin[x]/Cos[x]"),
            ("Sec[x]", "1/Cos[x]"),
        ]

        for expr1, expr2 in pairs:
            result = execute_in_kernel(f"Simplify[{expr1} - ({expr2})]")
            assert result["success"] is True
            assert result["output_inputform"] == "0", (
                f"{expr1} != {expr2}: got {result['output']}"
            )

    def test_non_equivalent_expressions(self):
        """Test that non-equivalent expressions are not equal."""
        pairs = [
            ("Sin[x]", "Cos[x]"),
            ("x^2", "x^3"),
            ("Log[x]", "Exp[x]"),
        ]

        for expr1, expr2 in pairs:
            result = execute_in_kernel(f"Simplify[{expr1} - ({expr2})]")
            assert result["success"] is True
            assert result["output_inputform"] != "0", (
                f"{expr1} should not equal {expr2}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
