# Mathematica MCP Test Suite

This directory contains comprehensive tests for the Mathematica MCP server.

## Test Files

### `test_session.py`
Tests for the core Mathematica session module:
- JSON parsing and Association output handling
- Calculus operations (integration, differentiation, limits, series)
- Algebra (solving equations, factoring, expanding, simplifying)
- Linear algebra (matrices, determinants, eigenvalues)
- Special functions (Gamma, Fibonacci, Prime, Zeta)
- Differential equations
- Numerical computation
- Statistics
- Edge cases and the JSON parsing fix

### `test_error_detection.py` ⭐ NEW
Comprehensive tests for error detection, analysis, and fixing:
- **Error Analyzer Module** (6 tests) - Core analysis functionality
- **Message Analysis** (5 tests) - Multi-error handling and recommendations
- **LLM Formatting** (4 tests) - Error formatting for AI consumption
- **Inline Execution** (6 tests) - Error detection in code execution
- **Notebook Detection** (3 tests) - Notebook-based errors (skipped without connection)
- **Pattern Coverage** (3 tests) - Error pattern database validation
- **Real-World Scenarios** (4 tests) - Practical error cases
- **Recovery Workflow** (2 tests) - End-to-end error handling

**Total: 33 test cases covering 10+ error types**

### `test_readme_commands.py`
Tests for README examples and command validation

### `test_derivation_verification.py`
Tests for mathematical derivation verification

## Running Tests

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_error_detection.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_error_detection.py::TestErrorAnalyzer -v
```

### Run with coverage:
```bash
python -m pytest tests/test_error_detection.py --cov=src/mathematica_mcp/error_analyzer -v
```

### Run with detailed output:
```bash
python -m pytest tests/test_error_detection.py -vv --tb=long
```

## Test Results

Latest test run (as of 2026-01-20):

```
test_error_detection.py:
  ✅ 30 passed
  ⏭️ 3 skipped (require notebook connection)
  ⏱️ ~5.7 seconds runtime
```

## Error Types Covered

The error detection system handles:

1. **UnitConvert::compat** - Incompatible unit operations
2. **Part::partw** - List index out of range
3. **Part::partd** - Structure depth issues
4. **Syntax::sntxi** - Syntax errors
5. **Syntax::tsntxi** - Extra input syntax errors
6. **Divide::infy** - Division by zero
7. **Power::infy** - Invalid power operations
8. **Set::write** - Protected symbol modification
9. **Recursion::reclim** - Recursion limit exceeded
10. **General::stop** - Output suppression warnings

## Documentation

- `TEST_SUMMARY.md` - Detailed test coverage and results
- `demo_error_detection.py` - Demonstration scripts for error detection features

## Requirements

- Python 3.11+
- pytest 9.0+
- Wolfram Language / Mathematica (for actual execution tests)
- wolframscript (must be in PATH)

## Quick Examples

### Test a specific error type:
```python
from src.mathematica_mcp.error_analyzer import analyze_error

result = analyze_error("UnitConvert::compat", "Incompatible units")
print(result['suggested_fix'])
# Output: Use QuantityMagnitude[] to extract numeric values...
```

### Format errors for LLM:
```python
from src.mathematica_mcp.error_analyzer import format_error_for_llm

messages = [{'tag': 'Part::partw', 'text': 'Part out of range', 'type': 'error'}]
code = '{1, 2, 3}[[5]]'

formatted = format_error_for_llm(messages, code)
print(formatted)
# Outputs structured error report with analysis and fixes
```

## Contributing

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Group related tests in classes (e.g., `TestErrorAnalyzer`)
3. Use descriptive test names (e.g., `test_analyze_unitconvert_error`)
4. Include docstrings explaining what each test validates
5. Update TEST_SUMMARY.md with new coverage

## CI/CD Integration

These tests are designed to run in CI/CD pipelines. For best results:
- Ensure wolframscript is available in the CI environment
- Tests that require notebook connections are automatically skipped
- Use `pytest --tb=short` for concise CI output
