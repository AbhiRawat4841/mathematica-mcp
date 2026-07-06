# Mathematica MCP Comprehensive Test Corpus

> **Purpose**: A single, exhaustive test corpus for validating the `mathematica-mcp` server across all major Wolfram Language domains. Designed for consumption by AI agents running automated evaluation loops.
>
> **Source**: Consolidated from 3 independent deep-research analyses, deduplicated and mapped to actual MCP tool names and profiles.
>
> **Coverage target**: 90%+ of real-world Mathematica usage patterns across 30+ domains, 150+ unique test cases.

---

## How to Use This Corpus (Agent Instructions)

1. **Select a profile** (`math`, `notebook`, or `full`) based on your server configuration.
2. **For each test**, call the specified **MCP Tool** with the given **Input**.
3. **Evaluate success** using the **Verification Type** and **Success Criteria** columns.
4. **Skip** tests tagged `internet-required` or `frontend-required` if those capabilities are unavailable; mark as `skipped`, not `failed`.
5. **Report** results as: `PASS`, `FAIL` (with actual output), or `SKIP` (with reason).

---

## Verification Strategies

| Code | Strategy | How to Apply |
|------|----------|--------------|
| `exact` | Direct equality (`===`) | Output must match expected value character-for-character (modulo whitespace/ordering for sets of rules) |
| `symbolic` | Algebraic equivalence | `Simplify[result - expected] === 0` or `FullSimplify` - accounts for different but equivalent forms |
| `numeric` | Tolerance-based | `Abs[result - expected] < tolerance` - for all `N`/`NIntegrate`/`FindRoot`/`NMinimize` outputs |
| `structural` | Head/Length/MatchQ check | Check `Head`, `Length`, `Dimensions`, `MatchQ`, or `FreeQ[$Failed]` - for Graphics, InterpolatingFunction, Image, Association |
| `contains` | Subset/membership check | Output contains specific elements (e.g., solution rules contain `x -> value`) |
| `error` | Expected error/message | Output includes specific message name or returns `$Failed`/`$Aborted` |
| `boolean` | True/False match | Output is exactly `True` or `False` |
| `string` | String equality | Output string matches expected string exactly |
| `roundtrip` | Export then Import | Imported data equals original exported data |
| `non-empty` | Non-trivial output | Output is not `Null`, `$Failed`, or empty; has expected Head/type |

---

## MCP Tool Reference (Quick Map)

| Profile | Tool Group | Tools |
|---------|-----------|-------|
| `math` | core | `execute_code`, `check_syntax`, `get_mathematica_status`, `get_feature_status`, `get_kernel_state`, `get_messages`, `get_session_brief`, `restart_kernel` |
| `math` | session | `set_variable`, `get_variable`, `list_variables`, `clear_variables`, `get_expression_info` |
| `math` | knowledge | `convert_units`, `entity_lookup`, `get_constant`, `interpret_natural_language`, `wolfram_alpha` |
| `math` | debug | `time_expression`, `trace_evaluation`, `verify_derivation`, `get_computation_journal`, `clear_computation_journal` |
| `math` | kernel_tools | `load_package`, `list_loaded_packages` |
| `math` | symbol_lookup | `resolve_function`, `get_symbol_info`, `suggest_similar_functions` |
| `notebook` | notebook_primary | `create_notebook`, `open_notebook_file`, `save_notebook`, `close_notebook`, `export_notebook`, `get_cell_content`, `get_cells`, `get_notebook_info`, `get_notebooks`, `rasterize_expression`, `read_notebook`, `screenshot_cell`, `screenshot_notebook` |
| `notebook` | data | `import_data`, `export_data`, `list_supported_formats` |
| `notebook` | graphics | `export_graphics`, `inspect_graphics`, `compare_plots`, `create_animation` |
| `full` | math_aliases | `mathematica_integrate`, `mathematica_solve`, `mathematica_simplify`, `mathematica_differentiate`, `mathematica_expand`, `mathematica_factor`, `mathematica_limit`, `mathematica_series` |
| `full` | function_repository | `search_function_repository`, `get_function_repository_info`, `load_resource_function` |
| `full` | data_repository | `search_data_repository`, `load_dataset`, `get_dataset_info` |
| `full` | async | `submit_computation`, `poll_computation`, `get_computation_result` |
| `full` | cache | `cache_expression`, `get_cached`, `list_cache`, `clear_expression_cache` |
| `full` | notebook_advanced | `write_cell`, `delete_cell`, `evaluate_cell`, `evaluate_selection`, `scroll_to_cell`, `select_cell` |
| `full` | file_legacy | `convert_notebook`, `get_notebook_cell`, `get_notebook_outline`, `parse_notebook_python`, `read_notebook_content`, `run_script` |

---

## Section 1: System & Infrastructure (8 tests)

Tests that the MCP server itself is operational and its meta-tools work correctly.

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria |
|----|----------|---------|---------------|-----------------|--------------|------------------|
| SYS-01 | `get_mathematica_status` | math | Call with no arguments | Status object with version info | structural | Returns non-empty response with Mathematica version string |
| SYS-02 | `get_kernel_state` | math | Call with no arguments | Kernel state info | structural | Returns kernel state including memory usage and uptime |
| SYS-03 | `get_feature_status` | math | Call with no arguments | Feature availability map | structural | Returns which features are enabled/disabled |
| SYS-04 | `check_syntax` | math | `code: "Plot[Sin[x], {x, 0, 2Pi}]"` | Valid syntax | boolean | Reports syntax is valid (no errors) |
| SYS-05 | `check_syntax` | math | `code: "Plot[Sin[x], {x, 0, 2Pi"` | Invalid syntax | boolean | Reports syntax error (missing bracket) |
| SYS-06 | `get_messages` | math | Execute `1/0` then call `get_messages` | Message list containing `Power::infy` | contains | Response includes the infinity warning message |
| SYS-07 | `list_loaded_packages` | math | Call with no arguments | List of loaded packages | structural | Returns a list (may be empty); no error |
| SYS-08 | `list_supported_formats` | notebook | Call with no arguments | List of import/export formats | structural | Returns non-empty list of format strings |

---

## Section 2: Code Intelligence & Symbol Lookup (8 tests)

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria |
|----|----------|---------|---------------|-----------------|--------------|------------------|
| INT-01 | `get_symbol_info` | math | `symbol: "Integrate"` | Info about Integrate function | structural | Returns description, usage, attributes for Integrate |
| INT-02 | `get_symbol_info` | math | `symbol: "DSolve"` | Info about DSolve | structural | Returns description mentioning differential equations |
| INT-03 | `resolve_function` | math | `description: "solve equations"` | Suggested functions | contains | Response includes `Solve` or `Reduce` |
| INT-04 | `suggest_similar_functions` | math | `function: "Intgrate"` (typo) | Suggested corrections | contains | Response includes `Integrate` as a suggestion |
| INT-05 | `get_expression_info` | math | First `execute_code`: `expr = x^2 + 3x + 1`, then `get_expression_info` on it | Expression structure | structural | Returns Head (`Plus`), depth, leaf count |
| INT-06 | `get_constant` | math | `constant: "Pi"` | Value of Pi | numeric | Returns `3.14159265...` or symbolic `Pi` |
| INT-07 | `get_constant` | math | `constant: "E"` | Euler's number | numeric | Returns `2.71828182...` or symbolic `E` |
| INT-08 | `get_constant` | math | `constant: "SpeedOfLight"` | Speed of light | structural | Returns a Quantity with value ~299792458 m/s |

---

## Section 3: Variable & State Management (7 tests)

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria |
|----|----------|---------|---------------|-----------------|--------------|------------------|
| VAR-01 | `set_variable` | math | `name: "testVar", value: "42"` | Variable set confirmation | exact | Variable `testVar` is set to `42` |
| VAR-02 | `get_variable` | math | `name: "testVar"` (after VAR-01) | `42` | exact | Returns `42` |
| VAR-03 | `list_variables` | math | Call after VAR-01 | List containing `testVar` | contains | `testVar` appears in the variable list |
| VAR-04 | `set_variable` | math | `name: "myList", value: "{1, 2, 3}"` | Variable set | exact | Variable stores list `{1, 2, 3}` |
| VAR-05 | `get_variable` | math | `name: "myList"` (after VAR-04) | `{1, 2, 3}` | exact | Returns the stored list |
| VAR-06 | `clear_variables` | math | Call to clear all variables | Cleared confirmation | structural | After clearing, `list_variables` returns empty or no `testVar` |
| VAR-07 | `execute_code` | math | `x = 42; x` then separate call `x` | `42` in both calls (if stateful) | exact | Validates kernel state persistence across calls |

---

## Section 4: Arithmetic & Number Theory (18 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| ARITH-01 | `execute_code` | math | `2 + 3` | `5` | exact | Integer addition |
| ARITH-02 | `execute_code` | math | `2^100` | `1267650600228229401496703205376` | exact | Big-integer arithmetic - no overflow |
| ARITH-03 | `execute_code` | math | `1/3 + 1/7` | `10/21` | exact | Exact rational arithmetic |
| ARITH-04 | `execute_code` | math | `N[Pi, 50]` | `3.14159265358979323846264338327950288419716939937510` | numeric | 50-digit Pi; tolerance `< 10^-49` |
| ARITH-05 | `execute_code` | math | `Sqrt[2] + Sqrt[3]` | `Sqrt[2] + Sqrt[3]` (stays symbolic) | exact | Does NOT return a decimal - remains symbolic |
| ARITH-06 | `execute_code` | math | `Sqrt[-1]` | `I` | exact | Returns imaginary unit |
| ARITH-07 | `execute_code` | math | `1/0` | `ComplexInfinity` | exact | Returns `ComplexInfinity` with `Power::infy` message |
| ARITH-08 | `execute_code` | math | `0/0` | `Indeterminate` | exact | Returns `Indeterminate` with message |
| ARITH-09 | `execute_code` | math | `PrimeQ[104729]` | `True` | boolean | 104729 is the 10000th prime |
| ARITH-10 | `execute_code` | math | `Prime[100]` | `541` | exact | The 100th prime number |
| ARITH-11 | `execute_code` | math | `FactorInteger[1023]` | `{{3, 1}, {11, 1}, {31, 1}}` | exact | 1023 = 3 x 11 x 31 |
| ARITH-12 | `execute_code` | math | `EulerPhi[12]` | `4` | exact | Euler's totient function |
| ARITH-13 | `execute_code` | math | `GCD[48, 18]` | `6` | exact | Greatest common divisor |
| ARITH-14 | `execute_code` | math | `PowerMod[2, 10, 1000]` | `24` | exact | Modular exponentiation |
| ARITH-15 | `execute_code` | math | `Binomial[10, 3]` | `120` | exact | Binomial coefficient C(10,3) |
| ARITH-16 | `execute_code` | math | `PartitionsP[10]` | `42` | exact | Integer partition count |
| ARITH-17 | `execute_code` | math | `CatalanNumber[5]` | `42` | exact | 5th Catalan number |
| ARITH-18 | `execute_code` | math | `IntegerLength[2^100000]` | `30103` | exact | Big-integer precision - no floating error |

---

## Section 5: Symbolic Algebra (12 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| ALG-01 | `mathematica_expand` | full | `expression: "(x + y)^3"` | `x^3 + 3*x^2*y + 3*x*y^2 + y^3` | symbolic | Binomial expansion correct |
| ALG-02 | `mathematica_factor` | full | `expression: "x^2 - 5*x + 6"` | `(x - 2)*(x - 3)` | symbolic | Correct linear factors |
| ALG-03 | `mathematica_simplify` | full | `expression: "(x^2 - 1)/(x - 1)"` | `1 + x` | symbolic | Common factor cancelled |
| ALG-04 | `execute_code` | math | `Apart[1/(x^2 - 1)]` | `1/(2*(-1 + x)) - 1/(2*(1 + x))` | symbolic | Partial fraction decomposition; verify via `Together` returning original |
| ALG-05 | `execute_code` | math | `Together[1/x + 1/(x + 1)]` | `(1 + 2*x)/(x*(1 + x))` | symbolic | Combined fraction correct |
| ALG-06 | `execute_code` | math | `PolynomialQuotientRemainder[x^3 + 2*x + 1, x - 1, x]` | `{1 + x + x^2, 4}` | exact | Quotient and remainder |
| ALG-07 | `mathematica_factor` | full | `expression: "x^2 + 1"` | `1 + x^2` | exact | Unchanged - irreducible over rationals |
| ALG-08 | `execute_code` | math | `Factor[x^2 + 1, GaussianIntegers -> True]` | `(x - I)(x + I)` | symbolic | Factors over Gaussian integers |
| ALG-09 | `mathematica_simplify` | full | `expression: "Sin[x]^2 + Cos[x]^2"` | `1` | exact | Pythagorean identity |
| ALG-10 | `execute_code` | math | `FullSimplify[Gamma[z + 1]/Gamma[z]]` | `z` | symbolic | Gamma function recurrence |
| ALG-11 | `execute_code` | math | `ComplexExpand[Abs[x + I y]]` | `Sqrt[x^2 + y^2]` | symbolic | Complex absolute value expansion |
| ALG-12 | `mathematica_expand` | full | `expression: "(x + 1)^4"` | `1 + 4x + 6x^2 + 4x^3 + x^4` | symbolic | Binomial theorem for degree 4 |

---

## Section 6: Equation Solving (12 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| SOLVE-01 | `mathematica_solve` | full | `equation: "x^2 - 5*x + 6 == 0", variable: "x"` | `{{x -> 2}, {x -> 3}}` | contains | Solutions contain `x -> 2` and `x -> 3` |
| SOLVE-02 | `mathematica_solve` | full | `equation: "x^2 + 1 == 0", variable: "x"` | `{{x -> -I}, {x -> I}}` | contains | Complex roots found |
| SOLVE-03 | `execute_code` | math | `Solve[x^2 + 1 == 0, x, Reals]` | `{}` | exact | No real solutions - empty list |
| SOLVE-04 | `execute_code` | math | `Solve[{x + y == 3, 2*x - y == 0}, {x, y}]` | `{{x -> 1, y -> 2}}` | contains | System of linear equations |
| SOLVE-05 | `execute_code` | math | `Solve[x^3 - 6x^2 + 11x - 6 == 0, x]` | `{{x -> 1}, {x -> 2}, {x -> 3}}` | contains | Cubic with 3 real roots |
| SOLVE-06 | `execute_code` | math | `Reduce[x^2 - 4 < 0, x, Reals]` | `-2 < x < 2` | symbolic | Inequality reduction to interval |
| SOLVE-07 | `execute_code` | math | `FindRoot[Cos[x] == x, {x, 0.5}]` | `{x -> 0.739085}` | numeric | Dottie number; tolerance `< 1e-5` |
| SOLVE-08 | `execute_code` | math | `NSolve[x^5 - x - 1 == 0, x]` | 5 complex roots | structural | Returns 5 rules; one real root ~1.16730 |
| SOLVE-09 | `execute_code` | math | `LinearSolve[{{1, 2}, {3, 5}}, {1, 2}]` | `{-1, 1}` | exact | Verify `M.x == b` |
| SOLVE-10 | `execute_code` | math | `LinearSolve[{{1, 1}, {1, -1}}, {a, b}]` | `{(a+b)/2, (a-b)/2}` | symbolic | Symbolic linear solve |
| SOLVE-11 | `execute_code` | math | `Reduce[a x == b, x]` | `(a != 0 && x == b/a) \|\| (a == 0 && b == 0)` | symbolic | Parametric reduction with conditions |
| SOLVE-12 | `execute_code` | math | `Minimize[x^2 - 4*x + 5, x]` | `{1, {x -> 2}}` | exact | Minimum value 1 at x = 2 |

---

## Section 7: Calculus (16 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| CALC-01 | `mathematica_differentiate` | full | `expression: "x^3 + 2*x", variable: "x"` | `2 + 3*x^2` | symbolic | First derivative |
| CALC-02 | `mathematica_differentiate` | full | `expression: "Sin[x^2]", variable: "x"` | `2*x*Cos[x^2]` | symbolic | Chain rule |
| CALC-03 | `execute_code` | math | `D[x^4, {x, 3}]` | `24*x` | symbolic | Third derivative |
| CALC-04 | `execute_code` | math | `D[Exp[-x^2], x]` | `-2 E^(-x^2) x` | symbolic | Gaussian derivative (chain rule) |
| CALC-05 | `mathematica_integrate` | full | `expression: "Sin[x]", variable: "x"` | `-Cos[x]` | symbolic | Basic indefinite integral |
| CALC-06 | `mathematica_integrate` | full | `expression: "x * Log[x]", variable: "x"` | `-x^2/4 + (1/2) x^2 Log[x]` | symbolic | Integration by parts |
| CALC-07 | `execute_code` | math | `Integrate[x^2, {x, 0, 1}]` | `1/3` | exact | Definite integral - exact rational |
| CALC-08 | `execute_code` | math | `Integrate[1/(1 + x^2), {x, 0, Infinity}]` | `Pi/2` | exact | Improper integral |
| CALC-09 | `execute_code` | math | `Integrate[Exp[-x^2], {x, -Infinity, Infinity}]` | `Sqrt[Pi]` | exact | Gaussian integral |
| CALC-10 | `execute_code` | math | `NIntegrate[Exp[-x^2], {x, 0, 1}]` | `0.746824` | numeric | Numerical quadrature; tolerance `< 1e-5` |
| CALC-11 | `mathematica_limit` | full | `expression: "Sin[x]/x", variable: "x", point: "0"` | `1` | exact | Classic limit |
| CALC-12 | `mathematica_limit` | full | `expression: "(1 + 1/n)^n", variable: "n", point: "Infinity"` | `E` | exact | Definition of Euler's number |
| CALC-13 | `mathematica_series` | full | `expression: "Sin[x]", variable: "x", point: "0", order: "5"` | `x - x^3/6 + x^5/120 + O[x]^6` | symbolic | Maclaurin series |
| CALC-14 | `execute_code` | math | `Sum[1/k^2, {k, 1, Infinity}]` | `Pi^2/6` | exact | Basel problem |
| CALC-15 | `execute_code` | math | `Sum[1/k^4, {k, 1, Infinity}]` | `Pi^4/90` | exact | Convergent infinite series |
| CALC-16 | `execute_code` | math | `LaplaceTransform[t * Exp[-a t], t, s]` | `1/(a + s)^2` | symbolic | Laplace transform (control theory) |

---

## Section 8: Differential Equations (8 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| DE-01 | `execute_code` | math | `DSolve[y'[x] == y[x], y[x], x]` | `{{y[x] -> C[1]*E^x}}` | structural | General solution with constant C[1] |
| DE-02 | `execute_code` | math | `DSolve[{y'[x] == -2*y[x], y[0] == 3}, y[x], x]` | `{{y[x] -> 3*E^(-2*x)}}` | symbolic | IVP particular solution |
| DE-03 | `execute_code` | math | `DSolve[{y''[x] + y[x] == 0, y[0] == 1, y'[0] == 0}, y[x], x]` | `{{y[x] -> Cos[x]}}` | symbolic | Second-order IVP - harmonic oscillator |
| DE-04 | `execute_code` | math | `NDSolve[{y'[x] == -y[x]^2, y[0] == 1}, y, {x, 0, 5}]` | `{{y -> InterpolatingFunction[...]}}` | structural | Head is InterpolatingFunction; `y[1]/.%[[1]]` ~ 0.5 |
| DE-05 | `execute_code` | math | `sol = NDSolveValue[{y'[t] == y[t] (1 - y[t]), y[0] == 0.1}, y, {t, 0, 10}]; sol[10]` | `~0.9999` | numeric | Logistic equation saturates; tolerance `< 1e-3` |
| DE-06 | `execute_code` | math | `NDSolveValue[{y'[t] == t, y[0] == 0}, y[3], {t, 0, 3}]` | `4.5` | numeric | Simple ODE: y = t^2/2, y(3) = 4.5; tolerance `< 0.01` |
| DE-07 | `execute_code` | math | `u = NDSolveValue[{D[v[t,x],t] == D[v[t,x],{x,2}], v[0,x] == Sin[Pi x], v[t,0] == 0, v[t,1] == 0}, v, {t,0,0.1}, {x,0,1}]; u[0.1, 0.5]` | `~0.3727` | numeric | 1D heat equation PDE; tolerance `< 1e-2` |
| DE-08 | `execute_code` | math | `DSolve[y'[x] - a*y[x] == 0, y[x], x]` | `{{y[x] -> E^(a x) C[1]}}` | structural | Parametric ODE with integration constant |

---

## Section 9: Linear Algebra (12 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| LA-01 | `execute_code` | math | `Det[{{1, 2}, {3, 4}}]` | `-2` | exact | 2x2 determinant |
| LA-02 | `execute_code` | math | `Det[{{a, b}, {c, d}}]` | `-b c + a d` | symbolic | Symbolic determinant |
| LA-03 | `execute_code` | math | `Inverse[{{1, 2}, {3, 4}}]` | `{{-2, 1}, {3/2, -1/2}}` | exact | Numeric inverse; verify `M.Inv == IdentityMatrix[2]` |
| LA-04 | `execute_code` | math | `Eigenvalues[{{2, 1}, {1, 2}}]` | `{3, 1}` | exact | Eigenvalues of symmetric matrix |
| LA-05 | `execute_code` | math | `Eigenvectors[{{2, 1}, {1, 2}}]` | `{{1, 1}, {-1, 1}}` | exact | Eigenvectors; verify `A.v == lambda*v` |
| LA-06 | `execute_code` | math | `MatrixRank[{{1,2,3},{4,5,6},{7,8,9}}]` | `2` | exact | Rank-deficient matrix |
| LA-07 | `execute_code` | math | `NullSpace[{{1,2,3},{4,5,6},{7,8,9}}]` | `{{1, -2, 1}}` | exact | Kernel of singular matrix; verify `M.v == {0,0,0}` |
| LA-08 | `execute_code` | math | `Transpose[{{1, 2}, {3, 4}}]` | `{{1, 3}, {2, 4}}` | exact | Matrix transpose |
| LA-09 | `execute_code` | math | `IdentityMatrix[3]` | `{{1,0,0},{0,1,0},{0,0,1}}` | exact | Identity matrix generation |
| LA-10 | `execute_code` | math | `MatrixPower[{{1, 1}, {0, 1}}, n]` | `{{1, n}, {0, 1}}` | symbolic | Symbolic matrix exponentiation |
| LA-11 | `execute_code` | math | `Normal[SparseArray[{{1,1}->1, {2,3}->5}, {2,3}]]` | `{{1,0,0},{0,0,5}}` | exact | Sparse to dense conversion |
| LA-12 | `execute_code` | math | `SingularValueList[{{1, 0}, {0, 2}}]` | `{2, 1}` | exact | SVD - descending order |

---

## Section 10: Optimization (6 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| OPT-01 | `execute_code` | math | `Minimize[x^2 - 4*x + 5, x]` | `{1, {x -> 2}}` | exact | Min value 1 at x=2 |
| OPT-02 | `execute_code` | math | `Maximize[{x + y, x^2 + y^2 <= 1}, {x, y}]` | `{Sqrt[2], {x -> 1/Sqrt[2], y -> 1/Sqrt[2]}}` | symbolic | Constrained max on unit disk |
| OPT-03 | `execute_code` | math | `NMinimize[{x^2 + y^2, x + y >= 1}, {x, y}]` | `~{0.5, {x -> 0.5, y -> 0.5}}` | numeric | Numeric tolerance `1e-4` |
| OPT-04 | `execute_code` | math | `FindMinimum[(x - 2)^2 + 1, {x, 0}]` | `{1., {x -> 2.}}` | numeric | Local minimum finding |
| OPT-05 | `execute_code` | math | `Minimize[{x + 2*y, x + y >= 10, x >= 0, y >= 0}, {x, y}, Integers]` | `{10, {x -> 10, y -> 0}}` | exact | Integer linear programming |
| OPT-06 | `execute_code` | math | `FindMinimum[x^4 - 3*x^2 + 2, {x, 0.5}]` | `~{-0.25, {x -> 1.22474}}` | numeric | Local minimum of quartic; tolerance `< 0.01` |

---

## Section 11: Statistics & Probability (10 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| STAT-01 | `execute_code` | math | `Mean[{1, 2, 3, 4, 5}]` | `3` | exact | Arithmetic mean |
| STAT-02 | `execute_code` | math | `Median[{1, 3, 5, 7, 9, 11}]` | `6` | exact | (5+7)/2 |
| STAT-03 | `execute_code` | math | `Variance[{1, 2, 3, 4, 5}]` | `5/2` | exact | Exact rational sample variance |
| STAT-04 | `execute_code` | math | `StandardDeviation[{2, 4, 4, 4, 5, 5, 7, 9}]` | `2` | exact | Sample standard deviation |
| STAT-05 | `execute_code` | math | `Correlation[{1,2,3,4,5}, {2,4,6,8,10}]` | `1` | exact | Perfect positive correlation |
| STAT-06 | `execute_code` | math | `PDF[NormalDistribution[0, 1], 0]` | `1/Sqrt[2*Pi]` | symbolic | Symbolic PDF at center |
| STAT-07 | `execute_code` | math | `CDF[NormalDistribution[0, 1], 0]` | `1/2` | exact | CDF at mean |
| STAT-08 | `execute_code` | math | `Length[RandomVariate[NormalDistribution[0,1], 100]]` | `100` | exact | Structural: list of length 100 |
| STAT-09 | `execute_code` | math | `SeedRandom[42]; RandomVariate[NormalDistribution[]]` | Deterministic float | numeric | Seeded random is reproducible across calls |
| STAT-10 | `execute_code` | math | `Probability[x > 0, x \[Distributed] NormalDistribution[]]` | `1/2` | exact | Symbolic probability evaluation |

---

## Section 12: List Manipulation & Functional Programming (18 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| LIST-01 | `execute_code` | math | `Table[i^2, {i, 1, 5}]` | `{1, 4, 9, 16, 25}` | exact | List generation |
| LIST-02 | `execute_code` | math | `Map[f, {a, b, c}]` | `{f[a], f[b], f[c]}` | exact | Map function application |
| LIST-03 | `execute_code` | math | `Map[# + 1 &, {1, 2, 3}]` | `{2, 3, 4}` | exact | Pure function with Map |
| LIST-04 | `execute_code` | math | `Select[{1,2,3,4,5,6}, EvenQ]` | `{2, 4, 6}` | exact | Predicate-based filtering |
| LIST-05 | `execute_code` | math | `Cases[{1, "a", 2, "b", 3}, _Integer]` | `{1, 2, 3}` | exact | Pattern-based extraction |
| LIST-06 | `execute_code` | math | `Apply[Plus, {1, 2, 3, 4, 5}]` | `15` | exact | Head replacement (@@) |
| LIST-07 | `execute_code` | math | `FoldList[Plus, 0, {1, 2, 3, 4}]` | `{0, 1, 3, 6, 10}` | exact | Cumulative sums |
| LIST-08 | `execute_code` | math | `Flatten[{{1,2},{3,{4,5}}}]` | `{1, 2, 3, 4, 5}` | exact | Full flattening |
| LIST-09 | `execute_code` | math | `Flatten[{{a, {b}}, c}, 1]` | `{a, {b}, c}` | exact | Level-specified flattening |
| LIST-10 | `execute_code` | math | `Sort[{3,1,4,1,5,9}, Greater]` | `{9, 5, 4, 3, 1, 1}` | exact | Custom sort (descending) |
| LIST-11 | `execute_code` | math | `Partition[{1,2,3,4,5,6}, 2]` | `{{1,2},{3,4},{5,6}}` | exact | Partitioning into pairs |
| LIST-12 | `execute_code` | math | `Table[i * j, {i, 1, 2}, {j, 1, 3}]` | `{{1, 2, 3}, {2, 4, 6}}` | exact | Multi-dimensional table |
| LIST-13 | `execute_code` | math | `Nest[# + 1 &, 0, 5]` | `5` | exact | Repeated application |
| LIST-14 | `execute_code` | math | `NestList[2*# &, 1, 4]` | `{1, 2, 4, 8, 16}` | exact | Doubling sequence |
| LIST-15 | `execute_code` | math | `FixedPoint[Floor[#/2] &, 100]` | `0` | exact | Convergence to fixed point |
| LIST-16 | `execute_code` | math | `Through[{Min, Max}[{3,1,4,1,5}]]` | `{1, 5}` | exact | Simultaneous function application |
| LIST-17 | `execute_code` | math | `Composition[Sqrt, Abs][-9]` | `3` | exact | Function composition |
| LIST-18 | `execute_code` | math | `MapThread[Plus, {{1,2,3},{10,20,30}}]` | `{11, 22, 33}` | exact | Element-wise parallel mapping |

---

## Section 13: Data Structures - Associations & Datasets (6 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| DS-01 | `execute_code` | math | `Association["a" -> 1, "b" -> 2, "c" -> 3]` | `<\|"a"->1,"b"->2,"c"->3\|>` | structural | Valid Association head |
| DS-02 | `execute_code` | math | `assoc = <\|"a" -> 1, "b" -> 2\|>; assoc["b"]` | `2` | exact | Key lookup |
| DS-03 | `execute_code` | math | `GroupBy[Range[10], Mod[#, 3] &, Total]` | `<\|0 -> 18, 1 -> 22, 2 -> 15\|>` | exact | GroupBy with reduction |
| DS-04 | `execute_code` | math | `Merge[{<\|"a"->1,"b"->2\|>, <\|"a"->3,"c"->4\|>}, Total]` | `<\|"a"->4,"b"->2,"c"->4\|>` | exact | Merge with sum |
| DS-05 | `execute_code` | math | `Lookup[<\|"a" -> 1, "b" -> 2\|>, "c", 0]` | `0` | exact | Lookup with default |
| DS-06 | `execute_code` | math | `ds = Dataset[{<\|"id"->1,"x"->10\|>,<\|"id"->2,"x"->20\|>}]; Normal[ds[All,"x"]]` | `{10, 20}` | exact | Dataset column extraction |

---

## Section 14: String Processing (8 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| STR-01 | `execute_code` | math | `StringJoin["Hello", " ", "World"]` | `"Hello World"` | string | String concatenation |
| STR-02 | `execute_code` | math | `StringReplace["the cat sat on the mat", "cat" -> "dog"]` | `"the dog sat on the mat"` | string | String replacement |
| STR-03 | `execute_code` | math | `StringCases["the year 2025 and 2026", DigitCharacter..]` | `{"2025", "2026"}` | exact | Digit extraction |
| STR-04 | `execute_code` | math | `StringSplit["a-b-c-d", "-"]` | `{"a", "b", "c", "d"}` | exact | String splitting |
| STR-05 | `execute_code` | math | `StringLength["Wolfram"]` | `7` | exact | Character count |
| STR-06 | `execute_code` | math | `StringCases["abc123def456", RegularExpression["[0-9]+"]]` | `{"123", "456"}` | exact | Regex extraction |
| STR-07 | `execute_code` | math | `StringMatchQ["Hello123", LetterCharacter.. ~~ DigitCharacter..]` | `True` | boolean | Pattern-based string matching |
| STR-08 | `execute_code` | math | `StringReplace["BEGINIGNORE\nabc\nENDIGNORE", RegularExpression["(?s)BEGINIGNORE.*ENDIGNORE"] -> ""]` | `""` | string | Multiline regex replace (PCRE dotall) |

---

## Section 15: Pattern Matching & Rules (8 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| PAT-01 | `execute_code` | math | `Cases[{1, 2, 3, 4, 5}, x_ /; x > 3]` | `{4, 5}` | exact | Condition filter |
| PAT-02 | `execute_code` | math | `{1, 2, 3} /. x_Integer :> x^2` | `{1, 4, 9}` | exact | RuleDelayed transformation |
| PAT-03 | `execute_code` | math | `Cases[{1, 2, "x", 3.5, 4}, _?NumericQ]` | `{1, 2, 3.5, 4}` | exact | PatternTest filter |
| PAT-04 | `execute_code` | math | `MatchQ[x^2 + y^2, _Plus]` | `True` | boolean | Structural pattern matching |
| PAT-05 | `execute_code` | math | `ReplaceAll[f[1,2] + f[3,4], f[x_,y_] :> x*y]` | `14` | exact | Rule application: f[1,2]->2, f[3,4]->12 |
| PAT-06 | `execute_code` | math | `Sin[Sin[Sin[x]]] //. Sin[z_] :> z` | `x` | exact | ReplaceRepeated until fixed point |
| PAT-07 | `execute_code` | math | `{-1, 4, 9} /. (x_ /; x > 0) :> Sqrt[x]` | `{-1, 2, 3}` | exact | Conditional rule with guard |
| PAT-08 | `execute_code` | math | `Cases[{1, "a", 2.0, "b"}, _Integer \| _String]` | `{1, "a", "b"}` | exact | Alternatives pattern |

---

## Section 16: Core Language & Scoping (7 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| LANG-01 | `execute_code` | math | `Module[{x = 2}, x^2] + 1` | `5` | exact | Lexical scoping isolates local x |
| LANG-02 | `execute_code` | math | `Block[{x = 2}, x^3]` | `8` | exact | Dynamic scoping |
| LANG-03 | `execute_code` | math | `With[{c = 3}, c * x]` | `3 x` | symbolic | Constant replacement before evaluation |
| LANG-04 | `execute_code` | math | `Head[123.45]` | `Real` | exact | Atomic type identification |
| LANG-05 | `execute_code` | math | `Head[123]` | `Integer` | exact | Integer head |
| LANG-06 | `execute_code` | math | `expr = HoldComplete[(1 + 1)^2]; ReleaseHold[expr]` | `4` | exact | Evaluation control: Hold then release |
| LANG-07 | `execute_code` | math | `Clear[g]; g[x_?PrimeQ] := Prime[x] - x; {g[10], g[11]}` | `{g[10], 20}` | exact | Guarded definitions: 10 not prime so stays unevaluated; Prime[11]=31, 31-11=20 |

---

## Section 17: Visualization (13 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria | Notes |
|----|----------|---------|-------|-----------------|--------------|------------------|-------|
| VIZ-01 | `execute_code` | math | `Head[Plot[Sin[x], {x, 0, 2 Pi}]]` | `Graphics` | exact | 2D plot produces Graphics object | |
| VIZ-02 | `execute_code` | math | `Head[Plot3D[Sin[x]*Cos[y], {x,-Pi,Pi}, {y,-Pi,Pi}]]` | `Graphics3D` | exact | 3D surface plot | |
| VIZ-03 | `execute_code` | math | `Head[ListPlot[{1.2, 2.5, 3.1, 2.8, 4.0}]]` | `Graphics` | exact | Scatter plot from data | |
| VIZ-04 | `execute_code` | math | `Head[ContourPlot[x^2+y^2, {x,-3,3}, {y,-3,3}]]` | `Graphics` | exact | Contour map | |
| VIZ-05 | `execute_code` | math | `Head[ParametricPlot[{Sin[2t],Cos[3t]}, {t,0,2Pi}]]` | `Graphics` | exact | Lissajous curve | |
| VIZ-06 | `execute_code` | math | `Head[PolarPlot[1 + 2 Cos[t], {t, 0, 2 Pi}]]` | `Graphics` | exact | Limacon polar plot | |
| VIZ-07 | `execute_code` | math | `Head[BarChart[{3, 7, 2, 5, 9}]]` | `Graphics` | exact | Bar chart | |
| VIZ-08 | `execute_code` | math | `Head[RegionPlot[x^2 + y^2 < 1, {x,-1.5,1.5}, {y,-1.5,1.5}]]` | `Graphics` | exact | Filled region | |
| VIZ-09 | `execute_code` | math | `Head[Plot[1/x, {x, -1, 1}]]` | `Graphics` | exact | Singularity at 0 handled gracefully | |
| VIZ-10 | `export_graphics` | notebook | Export a `Plot[Sin[x], {x,0,2Pi}]` to PNG | Valid file path or base64 image | structural | File exists or base64 has valid PNG header | |
| VIZ-11 | `inspect_graphics` | notebook | Inspect `Plot[Sin[x], {x,0,2Pi}]` | Graphics metadata | structural | Returns plot range, primitives, or dimensions | |
| VIZ-12 | `rasterize_expression` | notebook | Rasterize `x^2 + Sqrt[y]` | Image data | structural | Returns image (base64 or file path) | |
| VIZ-13 | `execute_code` | math | `Head[Plot[{}, {x, 0, 1}]]` | `Graphics` | exact | Empty plot returns axes-only Graphics, not $Failed | Edge case |

---

## Section 18: Data Import/Export (7 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| IO-01 | `execute_code` | math | `ExportString[{{1,2,3},{4,5,6}}, "CSV"]` | `"1,2,3\n4,5,6\n"` | string | CSV formatted string |
| IO-02 | `execute_code` | math | `ImportString["1,2,3\n4,5,6", "CSV"]` | `{{1,2,3},{4,5,6}}` | exact | CSV parsed to nested list |
| IO-03 | `execute_code` | math | `ImportString["{\"x\":10,\"y\":20}", "RawJSON"]` | `<\|"x"->10,"y"->20\|>` | exact | JSON to Association |
| IO-04 | `execute_code` | math | `tmp = FileNameJoin[{$TemporaryDirectory, "mcp_test.csv"}]; Export[tmp, {{"a","b"},{1,2},{3,4}}, "CSV"]; Import[tmp, "CSV"]` | `{{"a","b"},{1,2},{3,4}}` | roundtrip | CSV file round-trip |
| IO-05 | `import_data` | notebook | Import a CSV string | Parsed data structure | structural | Returns list of lists |
| IO-06 | `export_data` | notebook | Export `{{1,2},{3,4}}` to CSV format | CSV formatted output | structural | Valid CSV string or file |
| IO-07 | `execute_code` | math | `sa = SparseArray[{{1,1}->1,{100,100}->2},{100,100}]; tmp = FileNameJoin[{$TemporaryDirectory,"sa.mx"}]; Export[tmp,sa,"MX"]; Import[tmp,"MX"] === sa` | `True` | boolean | MX round-trip for sparse arrays |

---

## Section 19: Image Processing (6 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| IMG-01 | `execute_code` | math | `img = Image[Table[Mod[i+j,2], {i,100}, {j,100}]]; {Head[img], ImageDimensions[img]}` | `{Image, {100, 100}}` | exact | Checkerboard image created |
| IMG-02 | `execute_code` | math | `ImageDimensions[ImageResize[Image[RandomReal[1,{100,100}]], {50,50}]]` | `{50, 50}` | exact | Image resized correctly |
| IMG-03 | `execute_code` | math | `ImageData[ColorNegate[Image[{{0, 1}, {1, 0}}]]]` | `{{1., 0.}, {0., 1.}}` | numeric | Pixel values inverted; tolerance per element `< 0.01` |
| IMG-04 | `execute_code` | math | `Head[EdgeDetect[Image[Table[If[30<i<70&&30<j<70,1.,0.],{i,100},{j,100}]]]]` | `Image` | exact | Edge detection produces Image |
| IMG-05 | `execute_code` | math | `ImageChannels[ColorConvert[Image[RandomReal[1,{64,64,3}]], "Grayscale"]]` | `1` | exact | RGB to grayscale conversion |
| IMG-06 | `execute_code` | math | `img = Image[ConstantArray[0.25, {10, 10}]]; ImageMeasurements[img, "MeanIntensity"]` | `0.25` | numeric | Mean intensity of constant image; tolerance `< 1e-10` |

---

## Section 20: Graph Theory (6 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| GRAPH-01 | `execute_code` | math | `GraphQ[Graph[{1->2, 2->3, 3->1}]]` | `True` | boolean | Valid Graph construction |
| GRAPH-02 | `execute_code` | math | `{VertexCount[CompleteGraph[5]], EdgeCount[CompleteGraph[5]]}` | `{5, 10}` | exact | K5 has 5 vertices, 10 edges |
| GRAPH-03 | `execute_code` | math | `FindShortestPath[Graph[{1<->2,2<->3,3<->4,1<->4}], 1, 3]` | Path from 1 to 3 | structural | `First[%] === 1 && Last[%] === 3` and length <= 3 |
| GRAPH-04 | `execute_code` | math | `Normal[AdjacencyMatrix[CompleteGraph[3]]]` | `{{0,1,1},{1,0,1},{1,1,0}}` | exact | Adjacency matrix of K3 |
| GRAPH-05 | `execute_code` | math | `Module[{g=CycleGraph[6]}, {GraphDistance[g,1,4], ConnectedGraphQ[g], VertexDegree[g,1]}]` | `{3, True, 2}` | exact | Cycle graph properties |
| GRAPH-06 | `execute_code` | math | `g = Graph[{1 <-> 2, 2 <-> 3}]; Normal[AdjacencyMatrix[g]]` | `{{0,1,0},{1,0,1},{0,1,0}}` | exact | Simple path graph adjacency |

---

## Section 21: Geometry & Signal Processing (8 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| GEO-01 | `execute_code` | math | `RegionMeasure[Disk[{0,0}, 1]]` | `Pi` | exact | Area of unit disk |
| GEO-02 | `execute_code` | math | `Volume[Ball[{0,0,0}, 1]]` | `4 Pi/3` | exact | Volume of unit sphere |
| GEO-03 | `execute_code` | math | `Head[ConvexHullMesh[RandomReal[1, {20, 2}]]]` | `BoundaryMeshRegion` | exact | Convex hull mesh |
| GEO-04 | `execute_code` | math | `CoordinateTransform["Cartesian"->"Polar", {3,4}]` | `{5, ArcTan[4/3]}` | symbolic | Cartesian to polar |
| GEO-05 | `execute_code` | math | `Length[Fourier[Table[Sin[2 Pi k/32], {k, 0, 127}]]]` | `128` | exact | FFT produces 128 complex values |
| GEO-06 | `execute_code` | math | `Chop[InverseFourier[Fourier[{1,2,3,4,5,6,7,8}]] - {1,2,3,4,5,6,7,8}]` | `{0,0,0,0,0,0,0,0}` | exact | FFT round-trip identity |
| GEO-07 | `execute_code` | math | `Length[LowpassFilter[Table[Sin[2Pi*0.05*k]+0.5Sin[2Pi*0.4*k],{k,0,255}],0.2]]` | `256` | exact | Low-pass filter preserves length |
| GEO-08 | `execute_code` | math | `CoordinateTransform["Cartesian"->"Spherical", {1,1,1}]` | `{Sqrt[3], ArcCos[1/Sqrt[3]], Pi/4}` | symbolic | 3D coordinate transformation |

---

## Section 22: Units & Physical Constants (6 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| UNIT-01 | `convert_units` | math | Convert `100 Kilometers` to `Miles` | `~62.1371 Miles` | numeric | Magnitude ~62.14; tolerance `< 0.01` |
| UNIT-02 | `convert_units` | math | Convert `1 Hours` to `Seconds` | `3600 Seconds` | exact | Exact conversion |
| UNIT-03 | `execute_code` | math | `CompatibleUnitQ[Quantity[1,"Miles"], Quantity[1,"Kilometers"]]` | `True` | boolean | Same dimension check |
| UNIT-04 | `execute_code` | math | `UnitConvert[Quantity[1, "ElementaryCharge"], "Coulombs"]` | `~1.602e-19 Coulombs` | numeric | Physical constant conversion; tolerance `< 1e-25` |
| UNIT-05 | `get_constant` | math | `constant: "SpeedOfLight"` | `~299792458 m/s` | numeric | Speed of light value |
| UNIT-06 | `get_constant` | math | `constant: "PlanckConstant"` | `~6.626e-34 J*s` | numeric | Planck's constant |

---

## Section 23: Entity & Knowledge Base (5 tests)

> **Note**: These tests require internet connectivity. Mark as `SKIP` if offline.

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria | Notes |
|----|----------|---------|-------|-----------------|--------------|------------------|-------|
| ENT-01 | `entity_lookup` | math | `entity: "Country", name: "France", property: "Capital"` | Entity for Paris | contains | Response includes "Paris" | internet-required |
| ENT-02 | `entity_lookup` | math | `entity: "Element", name: "Oxygen", property: "AtomicNumber"` | `8` | exact | Atomic number of oxygen | internet-required |
| ENT-03 | `entity_lookup` | math | `entity: "Planet", name: "Mars", property: "Mass"` | `~6.39e23 kg` | numeric | Planetary mass | internet-required |
| ENT-04 | `wolfram_alpha` | math | `query: "population of Tokyo"` | Population number | structural | Returns a numeric quantity > 10 million | internet-required |
| ENT-05 | `interpret_natural_language` | math | `input: "convert 100 km to miles"` | `~62.14 miles` | numeric | Natural language computation | internet-required |

---

## Section 24: Machine Learning & NLP (8 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| ML-01 | `execute_code` | math | `FindClusters[{1,2,3,100,101,102}]` | `{{1,2,3},{100,101,102}}` | exact | Two correctly separated clusters |
| ML-02 | `execute_code` | math | `ClusteringComponents[{1,2,3,100,101,102}]` | `{1,1,1,2,2,2}` or equivalent | structural | Two distinct label groups |
| ML-03 | `execute_code` | math | `Nearest[{1,2,5,10,15}, 6, 3]` | `{5, 10, 2}` | exact | Three nearest neighbors |
| ML-04 | `execute_code` | math | `Predict[{1->1, 2->4, 3->9}, 4]` | `~16` | numeric | Quadratic prediction; tolerance `< 5` |
| ML-05 | `execute_code` | math | `TextWords["The quick brown fox jumps over the lazy dog"]` | 9 words | exact | `Length[%] === 9` |
| ML-06 | `execute_code` | math | `WordCount["The quick brown fox"]` | `4` | exact | Word count |
| ML-07 | `execute_code` | math | `TextSentences["Hello world. How are you? I am fine."]` | 3 sentences | exact | `Length[%] === 3` |
| ML-08 | `execute_code` | math | `training = {{0,0}->"A",{0,1}->"A",{1,0}->"B",{1,1}->"B"}; clf = Classify[training, Method->"NearestNeighbors"]; clf[{0.1, 0.2}]` | `"A"` | string | Nearest-neighbor classification |

---

## Section 25: Domain-Specific - Physics & Engineering (8 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| PHYS-01 | `execute_code` | math | `L=1; f[i_, x_] := Sqrt[2/L] Sin[i*Pi*(x + L/2)/L]; Integrate[f[1,x]*f[2,x], {x, -L/2, L/2}]` | `0` | exact | Quantum orthogonality of wavefunctions |
| PHYS-02 | `execute_code` | math | `metric = DiagonalMatrix[{-1, 1, r^2, r^2 Sin[theta]^2}]; Inverse[metric]` | `{{-1,0,0,0},{0,1,0,0},{0,0,1/r^2,0},{0,0,0,Csc[theta]^2/r^2}}` | symbolic | Inverse Schwarzschild metric |
| PHYS-03 | `execute_code` | math | `Expand[(p1 + p2 - p3)^2 + m^2]` | `m^2 + p1^2 + 2 p1 p2 + p2^2 - 2 p1 p3 - 2 p2 p3 + p3^2` | symbolic | 4-momentum expansion (Feynman integrals) |
| PHYS-04 | `execute_code` | math | `u = {x^2, x*y}; Grad[u, {x, y}]` | `{{2x, 0}, {y, x}}` | symbolic | Displacement gradient tensor (FEM) |
| PHYS-05 | `execute_code` | math | `Assuming[a > 0, Integrate[Exp[-a x], {x, 0, Infinity}]]` | `1/a` | symbolic | Parametric integral with assumption |
| PHYS-06 | `verify_derivation` | math | Verify that `D[Sin[x]^2, x]` equals `2 Sin[x] Cos[x]` | Verification result | boolean | Derivation confirmed correct |
| PHYS-07 | `execute_code` | math | `z = 1 + Exp[-beta*eps]; FullSimplify[-D[Log[z], beta]]` | `eps / (1 + E^(beta eps))` | symbolic | Partition function average energy |
| PHYS-08 | `execute_code` | math | `FourierTransform[Exp[-t^2], t, w]` | `Sqrt[Pi] Exp[-w^2/4]` or equivalent | symbolic | Fourier transform of Gaussian |

---

## Section 26: Specialized Domains (10 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| SPEC-01 | `execute_code` | math | `TimeValue[1000, 0.05, 10]` | `~1628.89` | numeric | Future value; tolerance `< 0.01` |
| SPEC-02 | `execute_code` | math | `Head[TransferFunctionModel[1/(s^2+s+1), s]]` | `TransferFunctionModel` | exact | Control system model |
| SPEC-03 | `execute_code` | math | `TensorProduct[{a,b},{c,d}]` | `{{a*c,a*d},{b*c,b*d}}` | exact | Outer product |
| SPEC-04 | `execute_code` | math | `TensorContract[{{a,b},{c,d}}, {{1,2}}]` | `a + d` | symbolic | Matrix trace via contraction |
| SPEC-05 | `execute_code` | math | `ArrayReshape[Range[12], {3,4}]` | `{{1,2,3,4},{5,6,7,8},{9,10,11,12}}` | exact | 3x4 matrix reshape |
| SPEC-06 | `execute_code` | math | `HammingDistance[{0,1,0,1},{1,1,0,0}]` | `2` | exact | Coding theory distance |
| SPEC-07 | `execute_code` | math | `HammingDistance["karolin","kathrin"]` | `3` | exact | String Hamming distance |
| SPEC-08 | `execute_code` | math | `Head[AudioGenerator[{"Sin",440}, 1]]` | `Audio` | exact | Audio synthesis |
| SPEC-09 | `execute_code` | math | `BooleanMinimize[a && b \|\| a && !b]` | `a` | exact | Boolean simplification |
| SPEC-10 | `execute_code` | math | `BooleanConvert[Implies[a, b], "CNF"]` | `!a \|\| b` | symbolic | Implication to CNF |

---

## Section 27: Date/Time & Boolean (5 tests)

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria |
|----|----------|---------|-------|-----------------|--------------|------------------|
| DT-01 | `execute_code` | math | `Head[DateObject[{2025, 1, 1}]]` | `DateObject` | exact | Date construction |
| DT-02 | `execute_code` | math | `DateDifference[{2025,1,1},{2025,12,31}]` | `Quantity[364, "Days"]` | exact | Day count between dates |
| DT-03 | `execute_code` | math | `DateString[{2025,7,4}, {"Year","-","Month","-","Day"}]` | `"2025-07-04"` | string | ISO date formatting |
| DT-04 | `execute_code` | math | `SatisfiableQ[a && !a]` | `False` | boolean | Contradiction detection |
| DT-05 | `execute_code` | math | `SatisfiableQ[a \|\| b]` | `True` | boolean | Satisfiability check |

---

## Section 28: Notebook Operations (8 tests)

> **Note**: Tests marked `frontend-required` need the Mathematica frontend. Mark as `SKIP` if running kernel-only.

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria | Notes |
|----|----------|---------|---------------|-----------------|--------------|------------------|-------|
| NB-01 | `create_notebook` | notebook | Create a new empty notebook | NotebookObject or notebook ID | structural | Returns a valid notebook reference | frontend-required |
| NB-02 | `write_cell` | full | Write `"1 + 1"` as an Input cell to notebook from NB-01 | Cell written confirmation | structural | Cell appears in notebook | frontend-required |
| NB-03 | `evaluate_cell` | full | Evaluate the cell from NB-02 | `2` | exact | Cell evaluates to 2 | frontend-required |
| NB-04 | `get_cells` | notebook | Get cells from notebook NB-01 | List of cells | structural | Returns list with at least 1 cell | frontend-required |
| NB-05 | `get_cell_content` | notebook | Get content of cell from NB-02 | Cell content string | contains | Contains `"1 + 1"` or the evaluated result | frontend-required |
| NB-06 | `get_notebooks` | notebook | List all open notebooks | List of notebooks | structural | Returns a list (may contain NB-01's notebook) | frontend-required |
| NB-07 | `save_notebook` | notebook | Save notebook from NB-01 to temp path | Save confirmation | structural | File written to disk | frontend-required |
| NB-08 | `close_notebook` | notebook | Close notebook from NB-01 | Close confirmation | structural | Notebook no longer in open list | frontend-required |

---

## Section 29: Debugging & Performance (8 tests)

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria |
|----|----------|---------|---------------|-----------------|--------------|------------------|
| DBG-01 | `time_expression` | math | `expression: "Integrate[Sin[x]^10, x]"` | Timing + result | structural | Returns both a numeric time and the integral result |
| DBG-02 | `trace_evaluation` | math | `expression: "1 + 2 + 3"` | Evaluation trace | structural | Shows step-by-step evaluation including Plus |
| DBG-03 | `execute_code` | math | `f = Compile[{{x, _Real}}, Sin[x]^2 + Cos[x]^2]; f[1.0]` | `1.0` | numeric | Compiled function correctness; tolerance `< 1e-10` |
| DBG-04 | `execute_code` | math | `Table[Prime[n], {n, 20}] === ParallelTable[Prime[n], {n, 20}]` | `True` | boolean | Parallel execution equivalence |
| DBG-05 | `execute_code` | math | `AbsoluteTiming[Det[RandomReal[1, {100, 100}]]][[1]]` | Numeric time value | numeric | Returns a positive float (benchmark; typically `< 1.0s`) |
| DBG-06 | `execute_code` | math | `$MachinePrecision` | `~15.9546` | numeric | Machine precision query |
| DBG-07 | `execute_code` | math | `N[1/3, 100]` | 100-digit decimal | structural | String length > 100 characters |
| DBG-08 | `execute_code` | math | `Directory[]` | String path | string | Returns a valid filesystem path |

---

## Section 30: Repository Integration (6 tests)

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria | Notes |
|----|----------|---------|---------------|-----------------|--------------|------------------|-------|
| REPO-01 | `search_function_repository` | full | `query: "integer partition"` | List of matching functions | structural | Returns results with names and descriptions | internet-required |
| REPO-02 | `get_function_repository_info` | full | `function: "IntegerPartitionFrequency"` | Function details | structural | Returns description, usage info | internet-required |
| REPO-03 | `load_resource_function` | full | `function: "IntegerPartitionFrequency"` | Loaded confirmation | structural | Function becomes callable | internet-required |
| REPO-04 | `search_data_repository` | full | `query: "sample data"` | List of datasets | structural | Returns dataset names | internet-required |
| REPO-05 | `load_package` | math | `package: "Developer\`"` | Package loaded | structural | No error; package functions available |
| REPO-06 | `list_loaded_packages` | math | After REPO-05 | Package list | contains | List includes "Developer\`" |

---

## Section 31: Async Computation & Caching (8 tests)

| ID | MCP Tool | Profile | Input / Action | Expected Output | Verification | Success Criteria |
|----|----------|---------|---------------|-----------------|--------------|------------------|
| ASYNC-01 | `submit_computation` | full | Submit `Factor[x^100 - 1]` | Computation ID | structural | Returns a valid computation ID |
| ASYNC-02 | `poll_computation` | full | Poll computation from ASYNC-01 | Status (running/completed) | structural | Returns status string |
| ASYNC-03 | `get_computation_result` | full | Get result from ASYNC-01 (after completion) | Factored polynomial | structural | Non-empty mathematical result |
| ASYNC-04 | `cache_expression` | full | Cache `Integrate[Sin[x]^10, x]` with key `"sin10"` | Cache confirmation | structural | Expression cached |
| ASYNC-05 | `get_cached` | full | Retrieve key `"sin10"` | Cached integral result | structural | Returns the previously cached result |
| ASYNC-06 | `list_cache` | full | List all cache entries | Cache list | contains | Contains `"sin10"` |
| ASYNC-07 | `clear_expression_cache` | full | Clear all cache | Clear confirmation | structural | `list_cache` returns empty after this |
| ASYNC-08 | `run_script` | full | Run a multi-line script: `x = 2; y = 3; x + y` | `5` | exact | Multi-line script execution |

---

## Section 32: Edge Cases & Error Handling (17 tests)

These tricky expressions expose common MCP server failures around assumptions, precision, output types, error handling, and resource limits.

| ID | MCP Tool | Profile | Input | Expected Output | Verification | Success Criteria | What It Tests |
|----|----------|---------|-------|-----------------|--------------|------------------|---------------|
| EDGE-01 | `execute_code` | math | `Sqrt[2] // N` | `~1.41421` | numeric | Forcing numeric evaluation of symbolic | Symbolic-to-numeric conversion |
| EDGE-02 | `execute_code` | math | `Integrate[Sin[Sin[x]], {x, 0, Pi}]` | Unevaluated integral | structural | Returns the integral form (no closed form exists) | Handling non-closed-form results |
| EDGE-03 | `execute_code` | math | `Assuming[x > 0, Simplify[Sqrt[x^2]]]` | `x` | exact | Assumption-dependent simplification | Assumptions propagation |
| EDGE-04 | `execute_code` | math | `Inverse[{{1, 2}, {2, 4}}]` | Error message | error | Returns `Inverse::sing` message or $Failed | Singular matrix error handling |
| EDGE-05 | `execute_code` | math | `Limit[1/x, x -> 0, Direction -> "FromAbove"]` | `Infinity` | exact | Directional limit (right) | One-sided limits |
| EDGE-06 | `execute_code` | math | `Limit[1/x, x -> 0, Direction -> "FromBelow"]` | `-Infinity` | exact | Directional limit (left) | Opposite direction |
| EDGE-07 | `execute_code` | math | `TimeConstrained[Integrate[Sin[Sin[Sin[x]]], x], 5]` | `$Aborted` | exact | Timeout aborts intractable computation | Timeout enforcement |
| EDGE-08 | `execute_code` | math | `MemoryConstrained[Range[10^8], 10^7]` | `$Aborted` | exact | Memory limit prevents RAM exhaustion | Memory limit enforcement |
| EDGE-09 | `execute_code` | math | `Block[{$RecursionLimit = 20}, f[x_] := f[x+1]; f[0]]` | `$Aborted` or recursion error | error | Infinite recursion halted | Recursion limit |
| EDGE-10 | `execute_code` | math | `Series[1/(1-x), {x, 0, 5}]` | `1+x+x^2+x^3+x^4+x^5+O[x]^6` | symbolic | SeriesData object handling | Non-standard output type |
| EDGE-11 | `execute_code` | math | `NMinimize[{x, x > 1 && x < 0}, x]` | Infeasible result | structural | Returns message about contradictory constraints or Infinity | Contradictory constraints |
| EDGE-12 | `execute_code` | math | `Head[Manipulate[Plot[Sin[n x],{x,0,2Pi}],{n,1,10}]]` | `Manipulate` or `DynamicModule` | exact | Frontend-absent graceful handling | Frontend-absent behavior |
| EDGE-13 | `execute_code` | math | `Assuming[a > 0, Integrate[Exp[-a x], {x, 0, Infinity}]]` | `1/a` | symbolic | Assumption-gated improper integral | Assuming blocks |
| EDGE-14 | `execute_code` | math | `NIntegrate[Exp[-x^2], {x, 0, 1}, WorkingPrecision -> 50, PrecisionGoal -> 20]` | `~0.746824133` | numeric | High-precision stability; tolerance `< 1e-20` | Arbitrary precision numerics |
| EDGE-15 | `execute_code` | math | `FileExistsQ[$InstallationDirectory]` | `True` | boolean | Installation directory exists | Filesystem access |
| EDGE-16 | `execute_code` | math | `expr = Expand[(x + y + z)^20]; StringLength[ToString[expr, InputForm]]` | Large integer (> 5000) | structural | Result is a positive integer > 5000 | Large output handling / truncation |
| EDGE-17 | `restart_kernel` | math | Restart the kernel | Restart confirmation | structural | Kernel restarts; subsequent execute_code works | Kernel restart resilience |

---

## Coverage Summary

| Section | Domain | Test Count | Profile Required |
|---------|--------|-----------|------------------|
| 1 | System & Infrastructure | 8 | math / notebook |
| 2 | Code Intelligence | 8 | math |
| 3 | Variable & State | 7 | math |
| 4 | Arithmetic & Number Theory | 18 | math |
| 5 | Symbolic Algebra | 12 | math / full |
| 6 | Equation Solving | 12 | math / full |
| 7 | Calculus | 16 | math / full |
| 8 | Differential Equations | 8 | math |
| 9 | Linear Algebra | 12 | math |
| 10 | Optimization | 6 | math |
| 11 | Statistics & Probability | 10 | math |
| 12 | List & Functional Programming | 18 | math |
| 13 | Data Structures | 6 | math |
| 14 | String Processing | 8 | math |
| 15 | Pattern Matching | 8 | math |
| 16 | Core Language & Scoping | 7 | math |
| 17 | Visualization | 13 | math / notebook |
| 18 | Data Import/Export | 7 | math / notebook |
| 19 | Image Processing | 6 | math |
| 20 | Graph Theory | 6 | math |
| 21 | Geometry & Signals | 8 | math |
| 22 | Units & Constants | 6 | math |
| 23 | Entity & Knowledge Base | 5 | math |
| 24 | Machine Learning & NLP | 8 | math |
| 25 | Physics & Engineering | 8 | math |
| 26 | Specialized Domains | 10 | math |
| 27 | Date/Time & Boolean | 5 | math |
| 28 | Notebook Operations | 8 | notebook / full |
| 29 | Debugging & Performance | 8 | math |
| 30 | Repository Integration | 6 | math / full |
| 31 | Async & Caching | 8 | full |
| 32 | Edge Cases & Errors | 17 | math |
| **TOTAL** | **32 domains** | **283 tests** | |

### Profile Distribution

| Profile | Tests Runnable | % of Corpus |
|---------|---------------|-------------|
| `math` (minimum) | ~210 | ~74% |
| `notebook` | ~240 | ~85% |
| `full` (all tools) | 283 | 100% |

### Environment Dependencies

| Dependency | Affected Tests | Count |
|------------|---------------|-------|
| Internet required | ENT-01..05, REPO-01..04 | 9 |
| Frontend required | NB-01..08, VIZ-12 | 9 |
| Subkernels (parallel) | DBG-04 | 1 |
| System clock | (none critical) | 0 |

---

## Recommended Execution Order

1. **System checks** (SYS-*) - verify MCP server is alive
2. **State management** (VAR-*) - verify kernel persistence
3. **Arithmetic** (ARITH-*) - validate basic computation
4. **Core language** (LANG-*) - validate scoping/evaluation
5. **Algebra + Calculus + Solving** (ALG-*, CALC-*, SOLVE-*) - core math
6. **Linear algebra + DE + Optimization** (LA-*, DE-*, OPT-*) - advanced math
7. **Lists + Patterns + Strings** (LIST-*, PAT-*, STR-*) - data manipulation
8. **Statistics + ML** (STAT-*, ML-*) - data science
9. **Visualization** (VIZ-*) - graphics pipeline
10. **Import/Export** (IO-*) - data I/O
11. **Domain-specific** (PHYS-*, SPEC-*, DS-*, GRAPH-*, GEO-*, UNIT-*, ENT-*) - specialized
12. **Notebooks** (NB-*) - frontend integration
13. **Repository + Async + Cache** (REPO-*, ASYNC-*) - full-profile features
14. **Debugging** (DBG-*) - performance
15. **Edge cases** (EDGE-*) - stress and error handling (run last)

---

## Agent Reporting Template

For each test, report results in this format:

```
TEST_ID: <ID>
STATUS: PASS | FAIL | SKIP
MCP_TOOL: <tool_name>
INPUT: <what was sent>
EXPECTED: <expected output>
ACTUAL: <actual output>
VERIFICATION: <verification type used>
NOTES: <any relevant details, error messages, timing>
```

### Aggregate Summary Format

```
TOTAL: <N>
PASSED: <N> (<percent>%)
FAILED: <N> (<percent>%)
SKIPPED: <N> (<percent>%) [reasons: internet=X, frontend=Y, ...]
PASS_RATE (excluding skips): <percent>%
```
