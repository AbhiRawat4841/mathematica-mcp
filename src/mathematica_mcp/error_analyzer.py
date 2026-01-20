"""
Error Analysis and Auto-Fix Suggestions for Mathematica Errors

This module provides pattern matching and suggestion generation for common
Mathematica errors encountered during code execution.
"""

from typing import Dict, List, Optional, Any
import re


# Knowledge base of common error patterns and fixes
ERROR_PATTERNS = {
    "UnitConvert::compat": {
        "description": "Incompatible units error",
        "common_cause": "Attempting to convert between incompatible unit types (e.g., time and currency)",
        "suggested_fix": "Use QuantityMagnitude[] to extract numeric values before performing arithmetic operations",
        "example": "QuantityMagnitude[CountryData[\"USA\", \"GDP\"], \"USDollars\"] instead of CountryData[\"USA\", \"GDP\"]",
        "severity": "error",
    },
    "Part::partw": {
        "description": "Part specification out of range",
        "common_cause": "Index exceeds the length of the list",
        "suggested_fix": "Check list length before accessing, or use Part with a default value",
        "example": "list[[idx, default]] or use Length[list] to verify bounds",
        "severity": "error",
    },
    "Part::partd": {
        "description": "Part specification is longer than depth",
        "common_cause": "Trying to access nested elements that don't exist",
        "suggested_fix": "Verify the structure depth with Depth[] or Dimensions[]",
        "example": "Use Dimensions[data] to understand structure before accessing",
        "severity": "error",
    },
    "Syntax::sntxi": {
        "description": "Syntax error in input",
        "common_cause": "Malformed expression, missing brackets, or invalid syntax",
        "suggested_fix": "Check for matching brackets [], {}, (), and proper operator usage",
        "example": "Use SyntaxQ[] to validate expressions before evaluation",
        "severity": "error",
    },
    "General::stop": {
        "description": "Further output suppressed",
        "common_cause": "Too many similar messages generated",
        "suggested_fix": "Fix the underlying issue causing repeated messages",
        "example": "Previous errors may indicate the root cause",
        "severity": "warning",
    },
    "Divide::infy": {
        "description": "Infinite expression encountered",
        "common_cause": "Division by zero or undefined mathematical operation",
        "suggested_fix": "Add checks for zero denominators or use Limit[] for indeterminate forms",
        "example": "If[denominator != 0, numerator/denominator, Infinity]",
        "severity": "error",
    },
    "Power::infy": {
        "description": "Infinite expression in power",
        "common_cause": "Invalid power operation (e.g., 0^0 or infinity^0)",
        "suggested_fix": "Add domain constraints or use Limit[] for boundary cases",
        "example": "Use Assuming[x > 0, expression] to constrain domain",
        "severity": "error",
    },
    "Set::write": {
        "description": "Tag is protected",
        "common_cause": "Attempting to modify a protected symbol or built-in function",
        "suggested_fix": "Use a different variable name or Unprotect[] (not recommended for built-ins)",
        "example": "Choose descriptive variable names that don't conflict with built-ins",
        "severity": "error",
    },
    "Recursion::reclim": {
        "description": "Recursion depth exceeded",
        "common_cause": "Infinite recursion or deeply nested recursive calls",
        "suggested_fix": "Add base case to recursion or increase $RecursionLimit",
        "example": "Verify termination conditions in recursive functions",
        "severity": "error",
    },
    "Syntax::tsntxi": {
        "description": "Syntax error: extra input",
        "common_cause": "Unexpected characters or tokens after valid expression",
        "suggested_fix": "Remove extra characters or properly terminate the expression",
        "example": "Check for stray commas, brackets, or operators",
        "severity": "error",
    },
}


def analyze_error(error_tag: str, error_text: str) -> Dict[str, Any]:
    """
    Analyze a Mathematica error and provide suggestions.

    Args:
        error_tag: The error tag (e.g., "UnitConvert::compat")
        error_text: The full error message text

    Returns:
        Dictionary with analysis and suggestions
    """
    # Check for exact pattern match
    for pattern, info in ERROR_PATTERNS.items():
        if pattern in error_tag:
            return {
                "matched_pattern": pattern,
                "description": info["description"],
                "common_cause": info["common_cause"],
                "suggested_fix": info["suggested_fix"],
                "example": info.get("example", ""),
                "severity": info.get("severity", "error"),
                "confidence": "high",
            }

    # Check for partial matches (e.g., just the error type)
    error_type = error_tag.split("::")[0] if "::" in error_tag else error_tag
    for pattern, info in ERROR_PATTERNS.items():
        if error_type in pattern:
            return {
                "matched_pattern": pattern,
                "description": info["description"],
                "common_cause": info["common_cause"],
                "suggested_fix": info["suggested_fix"],
                "example": info.get("example", ""),
                "severity": info.get("severity", "error"),
                "confidence": "medium",
            }

    # No match found - return generic analysis
    return {
        "matched_pattern": None,
        "description": f"Unknown error: {error_tag}",
        "common_cause": "Unrecognized error pattern",
        "suggested_fix": (
            "Consult Wolfram Language documentation for this error. "
            f"Search for '{error_tag}' in the Mathematica documentation."
        ),
        "example": "",
        "severity": "error" if "error" in error_tag.lower() else "warning",
        "confidence": "low",
    }


def analyze_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a list of error/warning messages and provide overall assessment.

    Args:
        messages: List of message dictionaries with 'tag', 'text', 'type' fields

    Returns:
        Dictionary with overall analysis and recommendations
    """
    if not messages:
        return {
            "total_messages": 0,
            "errors": 0,
            "warnings": 0,
            "assessment": "No errors or warnings detected",
            "recommendations": [],
        }

    errors = [m for m in messages if m.get("type") == "error"]
    warnings = [m for m in messages if m.get("type") == "warning"]

    # Analyze each message
    analyses = []
    for msg in messages:
        analysis = analyze_error(msg.get("tag", ""), msg.get("text", ""))
        analysis["original_message"] = msg
        analyses.append(analysis)

    # Generate recommendations
    recommendations = []
    high_confidence_fixes = [
        a for a in analyses if a.get("confidence") == "high"
    ]

    if high_confidence_fixes:
        recommendations.append(
            "High-confidence fixes available for the following errors:"
        )
        for fix in high_confidence_fixes[:3]:  # Top 3
            recommendations.append(
                f"  • {fix['matched_pattern']}: {fix['suggested_fix']}"
            )

    # Check for common patterns
    if any(
        a.get("matched_pattern") and "UnitConvert" in a.get("matched_pattern", "")
        for a in analyses
    ):
        recommendations.append(
            "TIP: When working with Entity data (countries, cities, etc.), "
            "use QuantityMagnitude[] to extract numeric values."
        )

    if any(
        a.get("matched_pattern") and "Part::" in a.get("matched_pattern", "")
        for a in analyses
    ):
        recommendations.append(
            "TIP: Use Length[], Dimensions[], or Depth[] to inspect data "
            "structure before accessing elements."
        )

    return {
        "total_messages": len(messages),
        "errors": len(errors),
        "warnings": len(warnings),
        "severity": "error" if errors else "warning" if warnings else "info",
        "assessment": (
            f"Found {len(errors)} error(s) and {len(warnings)} warning(s)"
        ),
        "analyses": analyses,
        "recommendations": recommendations,
        "should_retry": len(high_confidence_fixes) > 0,
    }


def format_error_for_llm(
    messages: List[Dict[str, Any]], code: str
) -> str:
    """
    Format error messages and analysis for LLM consumption.

    Args:
        messages: List of error/warning messages
        code: The code that was executed

    Returns:
        Formatted string for LLM
    """
    if not messages:
        return ""

    analysis = analyze_messages(messages)

    output = []
    output.append("=" * 60)
    output.append("EXECUTION ERRORS DETECTED")
    output.append("=" * 60)
    output.append(
        f"\nSummary: {analysis['assessment']}"
    )

    if analysis["errors"] > 0:
        output.append("\n--- ERRORS ---")
        for msg in messages:
            if msg.get("type") == "error":
                output.append(f"\n{msg.get('tag', 'Unknown')}")
                output.append(f"  {msg.get('text', '')}")

                # Add analysis
                err_analysis = analyze_error(
                    msg.get("tag", ""), msg.get("text", "")
                )
                if err_analysis.get("matched_pattern"):
                    output.append(f"\n  Analysis: {err_analysis['description']}")
                    output.append(f"  Likely cause: {err_analysis['common_cause']}")
                    output.append(f"  Suggested fix: {err_analysis['suggested_fix']}")
                    if err_analysis.get("example"):
                        output.append(f"  Example: {err_analysis['example']}")

    if analysis["warnings"] > 0:
        output.append("\n--- WARNINGS ---")
        for msg in messages:
            if msg.get("type") == "warning":
                output.append(f"\n{msg.get('tag', 'Unknown')}")
                output.append(f"  {msg.get('text', '')}")

    if analysis["recommendations"]:
        output.append("\n--- RECOMMENDATIONS ---")
        for rec in analysis["recommendations"]:
            output.append(rec)

    output.append("\n" + "=" * 60)

    return "\n".join(output)


# Export main functions
__all__ = [
    "analyze_error",
    "analyze_messages",
    "format_error_for_llm",
    "ERROR_PATTERNS",
]
