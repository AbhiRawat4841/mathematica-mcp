#!/usr/bin/env python
"""
Demonstration script for the error detection and analysis system.

This script shows how the error detection system works in various scenarios,
both for inline and notebook execution modes.
"""

from src.mathematica_mcp.error_analyzer import (
    analyze_error,
    analyze_messages,
    format_error_for_llm,
)
from src.mathematica_mcp.session import execute_in_kernel


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_error_analyzer():
    """Demonstrate the error analyzer on different error types."""
    print_section("DEMO 1: Error Analyzer Module")

    # Test different error types
    error_scenarios = [
        {
            "tag": "UnitConvert::compat",
            "text": "Incompatible units: USD and Hours",
            "description": "Unit conversion error",
        },
        {
            "tag": "Part::partw",
            "text": "Part 5 of {1,2,3} does not exist",
            "description": "List index out of range",
        },
        {
            "tag": "Syntax::sntxi",
            "text": "Invalid syntax at position 10",
            "description": "Syntax error",
        },
        {
            "tag": "Unknown::custom",
            "text": "Unknown error occurred",
            "description": "Unknown error type",
        },
    ]

    for scenario in error_scenarios:
        print(f"\n{scenario['description']}:")
        print(f"  Error tag: {scenario['tag']}")

        result = analyze_error(scenario["tag"], scenario["text"])

        print(f"  Matched: {result['matched_pattern']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Fix: {result['suggested_fix'][:60]}...")


def demo_message_analysis():
    """Demonstrate analyzing multiple error messages."""
    print_section("DEMO 2: Multiple Error Message Analysis")

    messages = [
        {"tag": "UnitConvert::compat", "text": "Incompatible units", "type": "error"},
        {"tag": "Part::partw", "text": "Part out of range", "type": "error"},
        {"tag": "General::stop", "text": "Output suppressed", "type": "warning"},
    ]

    print("\nAnalyzing 3 messages (2 errors, 1 warning)...")
    analysis = analyze_messages(messages)

    print(f"\nTotal messages: {analysis['total_messages']}")
    print(f"Errors: {analysis['errors']}")
    print(f"Warnings: {analysis['warnings']}")
    print(f"Should retry: {analysis['should_retry']}")

    print("\nRecommendations:")
    for rec in analysis["recommendations"]:
        print(f"  - {rec}")


def demo_llm_formatting():
    """Demonstrate LLM-ready error formatting."""
    print_section("DEMO 3: LLM-Ready Error Formatting")

    messages = [
        {
            "tag": "UnitConvert::compat",
            "text": "Incompatible units: USD and Hours",
            "type": "error",
        }
    ]
    code = 'CountryData["USA", "GDP"] + Quantity[1, "Hours"]'

    print(f"\nCode that produced error:\n  {code}\n")

    formatted = format_error_for_llm(messages, code)
    print(formatted)


def demo_inline_execution():
    """Demonstrate error detection in inline execution."""
    print_section("DEMO 4: Inline Execution Error Detection")

    test_cases = [
        {
            "name": "UnitConvert Error",
            "code": 'CountryData["USA", "GDP"] + Quantity[1, "Hours"]',
        },
        {"name": "Part Index Error", "code": "{1, 2, 3}[[10]]"},
        {"name": "Division by Zero", "code": "1/0"},
        {"name": "Undefined Symbol", "code": "unknownVar + 5"},
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  Code: {case['code']}")

        result = execute_in_kernel(case["code"])

        print(f"  Success: {result.get('success', 'N/A')}")
        output = result.get("output_inputform", "")[:60]
        print(f"  Output: {output}...")

        # Check for error messages
        if "messages" in result and result["messages"]:
            print(f"  Messages detected: Yes")
        else:
            print(f"  Messages detected: No")


def demo_real_world_workflow():
    """Demonstrate a complete error detection and recovery workflow."""
    print_section("DEMO 5: Real-World Error Recovery Workflow")

    # Simulate a common mistake: comparing GDP values directly
    code = 'CountryData["USA", "GDP"] > CountryData["China", "GDP"]'

    print(f"User code:\n  {code}\n")
    print("Executing...")

    result = execute_in_kernel(code)

    print(f"\nExecution result: {result.get('success', False)}")
    print(f"Output: {result.get('output_inputform', 'N/A')[:80]}...")

    # Simulate error message capture
    if "UnitConvert" in str(result) or "Incompatible" in str(result):
        print("\nError detected! Analyzing...")

        simulated_messages = [
            {
                "tag": "UnitConvert::compat",
                "text": "Incompatible units when comparing",
                "type": "error",
            }
        ]

        analysis = analyze_messages(simulated_messages)
        formatted = format_error_for_llm(simulated_messages, code)

        print(formatted)

        # Suggest corrected code
        corrected_code = """QuantityMagnitude[CountryData["USA", "GDP"], "USDollars"] >
QuantityMagnitude[CountryData["China", "GDP"], "USDollars"]"""

        print("\nSuggested correction:")
        print(f"  {corrected_code}")


def demo_error_pattern_coverage():
    """Show all available error patterns."""
    print_section("DEMO 6: Available Error Patterns")

    from src.mathematica_mcp.error_analyzer import ERROR_PATTERNS

    print(f"\nTotal error patterns: {len(ERROR_PATTERNS)}\n")

    for i, (pattern, info) in enumerate(ERROR_PATTERNS.items(), 1):
        print(f"{i}. {pattern}")
        print(f"   Description: {info['description']}")
        print(f"   Severity: {info['severity']}")
        print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#  ERROR DETECTION AND ANALYSIS SYSTEM - COMPREHENSIVE DEMO" + " " * 9 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    try:
        demo_error_analyzer()
        demo_message_analysis()
        demo_llm_formatting()
        demo_inline_execution()
        demo_real_world_workflow()
        demo_error_pattern_coverage()

        print("\n" + "=" * 70)
        print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n\nERROR during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
