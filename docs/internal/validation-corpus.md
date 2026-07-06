# Mathematica MCP Evaluation Corpus

> **Note:** This is documentation only. The executable source of truth is `tests/corpus/mathematica_mcp_corpus.json`. See `tests/README.md` for the corpus test architecture.

> **Profile note:** All tool names below are **classic-profile** vocabulary (`MATHEMATICA_PROFILE=classic`), and the profile column gives the minimum classic-family profile (`math`/`notebook`/`full`). The default `lean` profile exposes 12 consolidated tools instead (`evaluate`, `notebooks`, `cells`, `edit_cells`, `screenshot`, `kernel`, `vars`, `read_notebook_file`, `verify_derivation`, `status`, `guide`, `batch`).

A comprehensive test corpus for validating Mathematica MCP server tools and Wolfram Language coverage. Optimized for AI-agent use:

- every test has a stable ID,
- every test states the preferred MCP tool or workflow,
- every test states the minimum profile,
- every test gives the output or oracle that should be treated as success,
- environment-dependent tests are marked explicitly.

## Profile Definitions

- `math`: core computation, session state, knowledge (some network-backed), debug, and symbol tools
- `notebook`: graphics, notebook, and data workflows
- `full`: repository, knowledge, frontend, ML, admin, and environment-dependent workflows

## Success Oracle Types

- `Exact`: exact structural equality.
- `Symbolic`: mathematical equivalence under `Simplify`/`FullSimplify`.
- `Numeric`: within a fixed tolerance.
- `Structural`: correct `Head`, dimensions, field presence, or non-failure status.
- `Artifact`: file/image/notebook object exists and is non-empty.
- `Semantic`: output satisfies a domain invariant even if formatting differs.
- `SkipIfUnavailable`: do not fail the suite if the environment lacks frontend, network, subkernels, or repository access.

## Environment Tags

- `offline`: should pass without network.
- `frontend`: needs the Mathematica front end.
- `network`: depends on Wolfram/cloud/web data.
- `subkernels`: depends on parallel kernels.
- `resource`: depends on the Wolfram resource system.

## Oracle Hardening Rules

- Prefer structured fields when they exist.
- If a tool returns `success=true` together with `parse_error=true`, inspect the `raw` payload before marking failure.
- Distinguish `skip_unavailable` from `fail_regression`.
- Use `skip_unavailable` only for missing frontend, missing network/resource access, or missing subkernels.
- Treat serialization errors, malformed schemas, empty message channels after known errors, and broken live-notebook access as regressions when the relevant capability is otherwise available.
- For notebook-file readers, prefer the stable fixture `tests/fixtures/complex_notebook.nb`.
- For live notebook workflows, use a disposable notebook title and temp paths under `/tmp`.

## Tool Validation Layer

These tests are intentionally tool-first. They should run before the larger Wolfram-domain corpus because they validate whether the MCP server surface itself is behaving correctly.

### T1. System, Syntax, and Lifecycle Tools

| ID | Profile | Env | MCP Tool / Workflow | Preferred Input | Success Oracle |
|---|---|---|---|---|---|
| TOOL-SYS-01 | math | offline | `get_feature_status` | no args | Structural: `success=true`; includes `features.profile`; `tool_groups` is non-empty |
| TOOL-SYS-02 | math | offline | `check_syntax` | `code="Plot[Sin[x], {x, 0, 2 Pi}]"` | Exact: `valid=true` |
| TOOL-SYS-03 | math | offline | `check_syntax` | `code="Plot[Sin[x], {x, 0, 2 Pi"` | Exact: `valid=false` |
| TOOL-SYS-04 | math | offline | `get_kernel_state` | no args | Structural/raw: returns version and memory metadata; accept either parsed fields or `raw` containing `kernel_version ->` |
| TOOL-SYS-05 | math | offline | `get_mathematica_status` | no args | Structural: returns connection status with frontend/kernel version, system id, notebook count, port, and connection mode; mark `skip_unavailable` only if the status endpoint itself times out |
| TOOL-SYS-06 | math | offline | `execute_code` then `get_messages` | first run `1/0`, then `get_messages(count=5)` | Regression probe: stronger success is a non-empty warning list containing `Power::infy`; empty or malformed message payload after an induced error should be flagged |
| TOOL-SYS-07 | math | offline | `restart_kernel` then `execute_code` | restart, then compute `2+2` | Structural: restart returns success and follow-up execution still works |
| TOOL-SYS-08 | math | offline | `restart_kernel` state probe | set sentinel variable before restart, then inspect variable list after restart | Regression probe: sentinel variable should be gone after restart |

### T2. Session, Symbol Lookup, Knowledge, and Math Alias Tools

| ID | Profile | Env | MCP Tool / Workflow | Preferred Input | Success Oracle |
|---|---|---|---|---|---|
| TOOL-SES-01 | math | offline | `set_variable` | `name="testVarCorpus", value="42"` | Exact: `success=true`, `head="Integer"` |
| TOOL-SES-02 | math | offline | `get_variable` | `name="testVarCorpus"` | Exact: `value="42"` |
| TOOL-SES-03 | math | offline | `list_variables` | `include_system=false` | Structural: variable list contains `testVarCorpus` |
| TOOL-SES-04 | math | offline | `clear_variables` | `names=["testVarCorpus"]` | Exact/structural: `count=1`; follow-up `list_variables` no longer contains `testVarCorpus` |
| TOOL-SES-05 | math | offline | `get_expression_info` | `expression="x^2 + 3 x + 1"` | Exact/structural: `head="Plus"`, `depth=3`, `leaf_count=8` |
| TOOL-SES-06 | math | offline | `get_symbol_info` | `symbol="Integrate"` | Structural: `success=true`, `symbol="Integrate"`; do not require rich usage text because current builds may return sparse metadata |
| TOOL-SES-07 | math | offline | `resolve_function` | `query="integrate", max_candidates=5` | Exact/structural: `status="resolved"` and `resolved_symbol="Integrate"` |
| TOOL-SES-08 | math | offline | `suggest_similar_functions` | `query="Integrat"` | Structural: match list includes `Integrate` |
| TOOL-SES-09 | math | offline | `get_constant` | `name="Pi"` | Raw/semantic: accept parsed or raw output containing exact `Pi` or numeric value near `3.14159265358979` |
| TOOL-SES-10 | math | offline | `convert_units` | `quantity="1 hours", target_unit="seconds"` | Raw/semantic: accept raw payload containing `3600 seconds` or numeric `3600` |
| TOOL-SES-11 | math | offline | `interpret_natural_language` | `text="solve x squared equals 4 for x"` | Structural/semantic: result contains both `x -> -2` and `x -> 2` |
| TOOL-SES-12 | full | network | `wolfram_alpha` | `query="derivative of sin(x^2)", return_type="result"` | Raw/semantic: response contains `2 x Cos[x^2]` |
| TOOL-SES-13 | full | network | `entity_lookup` | `entity_type="Country", name="Japan"` | Regression probe: should resolve a country entity and expose resolved fields; raw unevaluated control expressions should be treated as failure |
| TOOL-SES-14 | math | offline | `load_package` | `package_name="Developer\`"` | Raw/structural: success payload indicates package loaded |
| TOOL-SES-15 | math | offline | `list_loaded_packages` | after `load_package("Developer\`")` | Structural: loaded-package list contains `Developer\`` |
| TOOL-SES-16 | full | offline | `mathematica_integrate` | `expression="Sin[x]", variable="x"` | Exact: output `-Cos[x]` |
| TOOL-SES-17 | full | offline | `mathematica_solve` | `equation="x^2 - 5*x + 6 == 0", variable="x"` | Exact/contains: output contains `x -> 2` and `x -> 3` |
| TOOL-SES-18 | full | offline | `mathematica_simplify` | `expression="Sin[x]^2 + Cos[x]^2"` | Exact: output `1` |
| TOOL-SES-19 | full | offline | `mathematica_limit` | `expression="Sin[x]/x", variable="x", point="0"` | Exact: output `1` |
| TOOL-SES-20 | full | offline | `mathematica_series` | `expression="Sin[x]", variable="x", point="0", order=5` | Structural: output is a valid series representation containing coefficients for `x`, `-x^3/6`, `x^5/120` |

### T3. Notebook Lifecycle, Frontend, and Offline Notebook Readers

| ID | Profile | Env | MCP Tool / Workflow | Preferred Input | Success Oracle |
|---|---|---|---|---|---|
| TOOL-NB-01 | notebook | offline | `read_notebook` | fixture path, `output_format="outline"` | Exact: `section_count=5`; includes `Analysis Notebook`, `Setup`, `Results`, `Detailed Analysis`, `Appendix` |
| TOOL-NB-02 | notebook | offline | `read_notebook` | fixture path, `output_format="wolfram"` | Structural: `code_cells=3`; content includes `f[x_]:=x^2+3x+1` and `Integrate[f[x],x]` |
| TOOL-NB-03 | notebook | offline | `read_notebook` | fixture path, `output_format="markdown"` | Contains: markdown includes `## Setup`, `## Results`, and Wolfram code fences |
| TOOL-NB-04 | full | offline | `get_notebook_outline` | fixture path | Exact/structural: `count=5`; titles match the section structure above |
| TOOL-NB-05 | full | offline | `get_notebook_cell` | fixture path, `cell_index=0`, `full=true` | Exact: content is `Analysis Notebook` |
| TOOL-NB-06 | full | offline | `parse_notebook_python` | fixture path, `output_format="json"` | Structural: `cell_count=10`, `code_cells=3` |
| TOOL-NB-07 | full | offline | `read_notebook_content` | fixture path, `include_outputs=false` | Structural: `cell_count=9`; includes the input cell `Integrate[f[x],x]` |
| TOOL-NB-08 | full | offline | `convert_notebook` | fixture path, `output_format="markdown"` | Contains: converted markdown includes notebook title and Wolfram code |
| TOOL-NB-09 | notebook | frontend | `create_notebook` | `title="Corpus Validation Notebook"` | Structural: returns notebook `id`, matching `title`, `created=true` |
| TOOL-NB-10 | notebook | frontend | `get_notebooks` | after TOOL-NB-09 | Structural: open-notebook list contains the created notebook |
| TOOL-NB-11 | notebook | frontend | `get_notebook_info` | created notebook id | Structural: title matches and `cell_count=0` for a fresh notebook |
| TOOL-NB-12 | full | frontend | `write_cell` | write `1 + 1` as `Input` with `sync="strict"` | Structural: `written=true` and a `cell_id` is returned |
| TOOL-NB-13 | notebook | frontend | `get_cells` | created notebook, `include_content=true` | Structural: at least one cell exists and includes an `Input` cell preview for `1 + 1` |
| TOOL-NB-14 | full | frontend | `evaluate_cell` | evaluate the written `Input` cell | Structural: `status="evaluation_pending"` with `evaluated=false` (protocol 4 pending contract; the output cell lands after the call returns - confirm via `get_cells`) |
| TOOL-NB-15 | full | frontend | `select_cell` then `evaluate_selection` | select the input cell, then evaluate selection | Structural: `selected=true`; evaluation returns the pending contract without transport failure (output confirmed via `get_cells`) |
| TOOL-NB-16 | full | frontend | `scroll_to_cell` | scroll to the written cell | Exact/structural: `scrolled=true` |
| TOOL-NB-17 | notebook | frontend | `screenshot_cell` | screenshot the written cell | Artifact: non-empty image payload is returned |
| TOOL-NB-18 | notebook | frontend | `save_notebook` | save created notebook to `/tmp/corpus-validation.nb` | Artifact: `saved=true`; file exists and is non-empty |
| TOOL-NB-19 | notebook | frontend | `export_notebook` | export created notebook to `/tmp/corpus-validation.pdf` | Artifact: `exported=true`; file exists and is non-empty |
| TOOL-NB-20 | full | frontend | `delete_cell` | delete the original input cell | Structural: `deleted=true`; follow-up `get_cells` no longer shows that input cell |
| TOOL-NB-21 | notebook | frontend | `close_notebook` | close the created notebook | Structural: `closed=true`; follow-up `get_notebooks` no longer includes it |
| TOOL-NB-22 | notebook | frontend | `open_notebook_file` | open `tests/fixtures/complex_notebook.nb` | Regression probe: should return a live notebook reference; serialization failures count as real failures, not skips |
| TOOL-NB-23 | notebook | frontend | `get_cell_content` | query the live written cell | Regression probe: should return readable cell content; command-level Mathematica errors count as failure |
| TOOL-NB-24 | notebook | frontend | `screenshot_notebook` | screenshot a live notebook window | Regression probe: should return a notebook screenshot artifact; missing-key/path exceptions count as failure |

### T4. Graphics, Repository, Async, and Cache Tools

| ID | Profile | Env | MCP Tool / Workflow | Preferred Input | Success Oracle |
|---|---|---|---|---|---|
| TOOL-GFX-01 | notebook | offline | `export_graphics` | plot `Sin[x]` to `/tmp/mcp_sin_plot.png`, `format="PNG"`, `size=480` | Artifact: `success=true`, `bytes > 0`, file exists |
| TOOL-GFX-02 | notebook | offline | `inspect_graphics` | `expression="Plot[Sin[x], {x, 0, 2 Pi}]"` | Raw/structural: payload identifies `head -> Graphics` and exposes plot-range/image-size metadata |
| TOOL-GFX-03 | notebook | offline | `rasterize_expression` | `expression="MatrixForm[{{1,2},{3,4}}]", image_size=240` | Artifact: non-empty rendered image is returned |
| TOOL-GFX-04 | notebook | offline | `compare_plots` | expressions for sine and cosine, `labels=["Sine","Cosine"]` | Raw/structural: payload contains `success -> True` and `plot_count -> 2` |
| TOOL-GFX-05 | notebook | offline | `create_animation` | `expression="Plot[Sin[n*x], {x, 0, 2Pi}]", parameter="n", range_spec="1, 5", frames=4` | Raw/structural: payload contains `success -> True` and a non-trivial `frame_count` |
| TOOL-REPO-01 | full | resource | `search_function_repository` | `query="integer partition frequency", max_results=3` | Structural: returns at least one result including `IntegerPartitionFrequency` |
| TOOL-REPO-02 | full | resource | `get_function_repository_info` | `function_name="IntegerPartitionFrequency"` | Raw/structural: payload contains description text and a documentation link |
| TOOL-REPO-03 | full | resource | `load_resource_function` | `function_name="IntegerPartitionFrequency"` | Raw/structural: reports `loaded -> True` |
| TOOL-REPO-04 | full | resource | `search_data_repository` | any stable query such as `Titanic` or `iris` | Smoke/skip: require a structured payload; if repository search is unavailable or returns no datasets, mark `skip_unavailable` rather than fail the whole suite |
| TOOL-REPO-05 | full | resource | `get_dataset_info` / `load_dataset` | stable dataset name in environments where the repository is reachable | Smoke/skip: mark `skip_unavailable` on repository absence; otherwise require structured metadata or loaded sample data |
| TOOL-CACHE-01 | full | offline | `cache_expression` | cache `Integrate[Sin[x]^10, x]` as `sin10_probe` | Structural: `cached=true` and the stored result preview is non-empty |
| TOOL-CACHE-02 | full | offline | `get_cached` | `name="sin10_probe"` | Structural: returns the previously cached expression/result |
| TOOL-CACHE-03 | full | offline | `list_cache` | after TOOL-CACHE-01 | Structural: cache listing contains `sin10_probe` |
| TOOL-CACHE-04 | full | offline | `clear_expression_cache` | no args, then `list_cache` | Exact/structural: clear succeeds and follow-up cache count is `0` |
| TOOL-ASYNC-01 | full | offline | `submit_computation` | `code="Factor[x^40 - 1]", name="factor40"` | Structural: returns `success=true`, a `job_id`, and `status="submitted"` |
| TOOL-ASYNC-02 | full | offline | `poll_computation` | use the `job_id` from TOOL-ASYNC-01 | Structural: returns a status such as `running` or `completed` |
| TOOL-ASYNC-03 | full | offline | `get_computation_result` | query the same `job_id` | Structural: before completion it should return a coherent running message; after completion it should return a non-empty algebraic result |

## Semantics, Scoping, and Advanced Coverage

### A. Core Semantics, Scoping, and Expression Structure

| ID | Profile | Env | Preferred MCP Tool | Example / Workflow | Success Oracle |
|---|---|---|---|---|---|
| SCOP-01 | math | offline | `execute_code(style="compute")` | `x = 5; f[y_] := y^2 + x; x = 10; f[3]` | Exact: `19` |
| SCOP-02 | math | offline | `execute_code(style="compute")` | `x = 1; Module[{x = 2}, x^2] + x` | Exact: `5` |
| SCOP-03 | math | offline | `execute_code(style="compute")` | `x = 1; f := x^2; Block[{x = 2}, f]` | Exact: `4` |
| SCOP-04 | math | offline | `execute_code(style="compute")` | `With[{c = 3}, c*x]` | Symbolic: equivalent to `3 x` |
| SCOP-05 | math | offline | `execute_code(style="compute")` | `Head[123.45]` | Exact: `Real` |
| SCOP-06 | math | offline | `execute_code(style="compute")` | `Extract[f[g[a, b], c], {1, 2}]` | Exact: `b` |
| SCOP-07 | math | offline | `execute_code(style="compute")` | `MatchQ[x^2 + y^2, _Plus]` | Exact: `True` |
| SCOP-08 | math | offline | `execute_code(style="compute")` | `expr = HoldComplete[(1 + 1)^2]; ReleaseHold[expr]` | Exact: `4` |
| SCOP-09 | math | offline | `execute_code(style="compute")` | `Lookup[Association["a" -> 1, "b" -> 2], "c", 0]` | Exact: `0` |
| SCOP-10 | math | offline | `execute_code(style="compute")` | `Map[f, {{a, b}, {c, d}}, {2}]` | Exact: `{{f[a], f[b]}, {f[c], f[d]}}` |
| SCOP-11 | notebook | offline | `execute_code(style="compute")` | `ds = Dataset[{Association["id" -> 1, "x" -> 10], Association["id" -> 2, "x" -> 20]}]; Normal[ds[All, "x"]]` | Exact: `{10, 20}` |
| SCOP-12 | notebook | offline | `execute_code(style="compute")` | `KeySortBy[Association["b" -> 2, "a" -> 1], Identity]` | Semantic: keys are ordered `a`, then `b` |

### B. Advanced Symbolic and Transform Coverage

| ID | Profile | Env | Preferred MCP Tool | Example / Workflow | Success Oracle |
|---|---|---|---|---|---|
| ADV-01 | math | offline | `execute_code(style="compute")` | `ComplexExpand[Abs[x + I y]]` | Symbolic: equivalent to `Sqrt[x^2 + y^2]` |
| ADV-02 | math | offline | `execute_code(style="compute")` | `PolynomialReduce[x^2 y + x y^2, {x y - 1, y^2 - 1}, {x, y}]` | Structural: quotient/remainder form is returned and reconstitution matches input |
| ADV-03 | math | offline | `execute_code(style="compute")` | `D[Sin[x y], {x, 1}, {y, 2}]` | Symbolic: equivalent to `-x^2 Sin[x y] - 2 x Cos[x y]` |
| ADV-04 | notebook | offline | `execute_code(style="compute")` | `Integrate[x Log[x], x]` | Symbolic: equivalent to `-x^2/4 + (x^2 Log[x])/2` |
| ADV-05 | notebook | offline | `execute_code(style="compute")` | `Assuming[a > 0, Integrate[Exp[-a x], {x, 0, Infinity}]]` | Symbolic: `1/a` |
| ADV-06 | notebook | offline | `execute_code(style="compute")` | `LaplaceTransform[t Exp[-a t], t, s]` | Symbolic: `1/(a + s)^2` |
| ADV-07 | math | offline | `execute_code(style="compute")` | `Sum[1/n^2, {n, 1, Infinity}]` | Exact: `Pi^2/6` |
| ADV-08 | notebook | offline | `execute_code(style="compute")` | `Sort[x /. NSolve[x^2 == 2, x]]` | Numeric: max error below `10^-10` from `{-Sqrt[2], Sqrt[2]}` |

### C. Research-Grade Science and Engineering Coverage

| ID | Profile | Env | Preferred MCP Tool | Example / Workflow | Success Oracle |
|---|---|---|---|---|---|
| SCI-01 | full | offline | `execute_code(style="compute")` | `comm[a_, b_] := a ** b - b ** a; comm[x, p]` | Exact: `x ** p - p ** x` |
| SCI-02 | notebook | offline | `execute_code(style="compute")` | `L = 1; f[i_, x_] := Sqrt[2/L] Sin[i Pi (x + L/2)/L]; Integrate[f[1, x] f[2, x], {x, -L/2, L/2}]` | Exact: `0` |
| SCI-03 | notebook | offline | `execute_code(style="compute")` | `metric = DiagonalMatrix[{-1, 1, r^2, r^2 Sin[theta]^2}]; Inverse[metric]` | Symbolic: diagonal inverse with `Csc[theta]^2/r^2` in the last entry |
| SCI-04 | notebook | offline | `execute_code(style="compute")` | `z = 1 + Exp[-beta eps]; energy = FullSimplify[-D[Log[z], beta]]` | Symbolic: equivalent to `eps/(1 + E^(beta eps))` |
| SCI-05 | notebook | offline | `execute_code(style="compute")` | `Expand[(p1 + p2 - p3)^2 + m^2]` | Exact symbolic polynomial expansion in `p1`, `p2`, `p3`, `m` |
| SCI-06 | notebook | offline | `execute_code(style="compute")` | `u = {x^2, x y}; Grad[u, {x, y}]` | Exact: `{{2 x, 0}, {y, x}}` |
| SCI-07 | notebook | offline | `execute_code(style="compute")` | `UnitConvert[Quantity[1, "ElementaryCharge"], "Coulombs"]` | Numeric/semantic: magnitude approximately `1.602176634*^-19` coulombs |

### D. MCP-Native Workflow Coverage

| ID | Profile | Env | Preferred MCP Tool | Example / Workflow | Success Oracle |
|---|---|---|---|---|---|
| MCP-01 | math | offline | `execute_code(style="compute")` | `Table[n^2, {n, 5}]` | Structural: tool returns `success=true`; output `{1, 4, 9, 16, 25}` |
| MCP-02 | notebook | frontend | `create_notebook` | `create_notebook(title="Corpus Probe")` | Structural: returns notebook `id`, correct `title`, and `created=true` |
| MCP-03 | notebook | frontend | `execute_code(style="notebook")` | `Plot[Sin[x], {x, 0, 2 Pi}]` in the current live notebook | Structural: status `executed_in_notebook`, a `cell_id` is returned, `evaluated=true` |
| MCP-04 | full | frontend | `execute_code(style="interactive")` | `Manipulate[Plot[Sin[n x], {x, 0, 2 Pi}], {n, 1, 5, 1}]` | Semantic: interactive result is created without kernel crash; mark `SkipIfUnavailable` if frontend path times out |
| MCP-05 | notebook | offline | `read_notebook` | `read_notebook(path="tests/fixtures/complex_notebook.nb", output_format="outline")` | Exact: `section_count=5`; includes `Analysis Notebook`, `Setup`, `Results`, `Detailed Analysis`, `Appendix` |
| MCP-06 | notebook | offline | `read_notebook` | `read_notebook(path="tests/fixtures/complex_notebook.nb", output_format="wolfram")` | Structural: `code_cells=3`; content includes `f[x_]:=x^2+3x+1` and `Integrate[f[x],x]` |
| MCP-07 | notebook | offline | `read_notebook` | `read_notebook(path="tests/fixtures/complex_notebook.nb", output_format="json", include_outputs=false)` | Exact/structural: `title="Analysis Notebook"`, `cell_count=9`, `code_cells=3` |
| MCP-08 | math | offline | `verify_derivation` | `steps=["Integrate[x^2, x]", "x^3/3"]` | Structural: report or raw data shows `all_valid=true` |
| MCP-09 | math | offline | `trace_evaluation` | `trace_evaluation("Expand[(x+1)^2]", max_depth=5)` | Structural: `success=true` and final `result` is `1 + 2*x + x^2` |
| MCP-10 | math | offline | `time_expression` | `time_expression("Table[Prime[n], {n, 1000}]")` | Structural: `success=true`, positive `time_ms`, and prime list result is returned |
| MCP-11 | notebook | offline | `export_graphics` | `export_graphics("Plot[Sin[x], {x, 0, 2 Pi}]", "/tmp/mcp_sin_plot.png", format="PNG", size=480)` | Artifact: `success=true`, file path returned, `bytes > 0` |
| MCP-12 | notebook | offline | `inspect_graphics` | `inspect_graphics("Plot[Sin[x], {x, 0, 2 Pi}]")` | Structural: report identifies `head -> Graphics` and includes plot-range/image-size metadata |
| MCP-13 | notebook | offline | `rasterize_expression` | `rasterize_expression("MatrixForm[{{1,2},{3,4}}]", image_size=240)` | Artifact: image payload is returned and render is non-empty |
| MCP-14 | notebook | offline | `export_data` + `import_data` | Export `{{1,2,3},{4,5,6}}` to CSV, then re-import | Structural: export succeeds; import reports `head=List`, `dimensions={2,3}`, preview matches data |
| MCP-15 | full | resource | `search_function_repository` | Search for `integer partition frequency` | Exact/structural: at least one result, including `IntegerPartitionFrequency` |
| MCP-16 | full | resource | `load_resource_function` | Load `IntegerPartitionFrequency` | Structural: tool reports `loaded=true` |
| MCP-17 | full | resource | `load_dataset` | Load a small sample dataset | Semantic: dataset loads successfully or is marked `SkipIfUnavailable` if the repository is unavailable |
| MCP-18 | full | network | `entity_lookup` | Country lookup for `Japan` | Structural: tool returns `success=true` or a parsed/raw payload identifying the `Country` entity |
| MCP-19 | full | network | `wolfram_alpha` | Query `derivative of sin(x^2)` | Semantic: response contains the correct derivative `2 x Cos[x^2]` |
| MCP-20 | full | offline | `get_kernel_state` | Inspect kernel status | Structural: response includes kernel version, memory, and loaded-package metadata |
| MCP-21 | full | offline | `get_mathematica_status` | Inspect server/addon connection status | Structural: response should include server connection health; mark `SkipIfUnavailable` if the status tool times out in a given environment |
| MCP-22 | notebook | frontend | `create_notebook` + `execute_code(style="notebook")` + `get_cells`/`get_cell_content` | Create a new live notebook and execute one input cell | Semantic: created notebook contains a new evaluated input cell; use notebook APIs to confirm cell presence |
| MCP-23 | notebook | frontend | `CreateDocument` via `execute_code(style="notebook")` | `nb = CreateDocument[]; NotebookWrite[nb, Cell["test", "Text"]]; "ok"` | Exact/structural: returns `"ok"`; side effect is a live notebook with a text cell |
| MCP-24 | notebook | frontend | notebook truncation workflow | Put a very large expression in a notebook cell, then read it with truncation threshold | Semantic: truncated read flags truncation; full cell retrieval recovers the entire expression |

### E. Precision, Stability, and System-Integrity Coverage

| ID | Profile | Env | Preferred MCP Tool | Example / Workflow | Success Oracle |
|---|---|---|---|---|---|
| SYS-01 | math | offline | `execute_code(style="compute")` | `TimeConstrained[FactorInteger[2^1000 - 1], 1]` | Exact/semantic: returns `$Aborted` rather than hanging |
| SYS-02 | math | offline | `execute_code(style="compute")` | `MemoryConstrained[Range[10^9], 10^7]` | Exact/semantic: returns `$Aborted` rather than exhausting RAM |
| SYS-03 | math | offline | `execute_code(style="compute")` | `Block[{$RecursionLimit = 20}, Clear[f]; f[x_] := f[x + 1]; Quiet@Check[f[0], $Aborted]]` | Semantic: recursion is cut off and the kernel remains alive |
| SYS-04 | math | offline | `execute_code(style="compute")` across two calls | First call: `x = 42`; second call: `x` | Semantic: if the session is stateful, second call returns `42`; if not, the server must document stateless behavior |
| SYS-05 | full | offline | `execute_code(style="compute")` | `f = Compile[{{x, _Real}}, Sin[x]^2 + Cos[x]^2]; {f[1.0], Abs[f[1.0] - 1.0] < 10^-12}` | Exact/semantic: `{1., True}` up to machine precision |
| SYS-06 | full | subkernels | `execute_code(style="compute")` | `Table[Prime[n], {n, 20}] === ParallelTable[Prime[n], {n, 20}]` | Exact: `True`; mark `SkipIfUnavailable` if subkernels are unavailable |
| SYS-07 | math | offline | `execute_code(style="compute")` | `IntegerLength[2^100000]` | Exact: `30103` |
| SYS-08 | math | offline | `execute_code(style="compute")` | `N[1/3, 100]` | Semantic: 100-digit decimal precision is preserved |
| SYS-09 | notebook | offline | `execute_code(style="compute")` | `NIntegrate[Exp[-x^2], {x, 0, 1}, WorkingPrecision -> 50, AccuracyGoal -> 20, PrecisionGoal -> 20]` | Numeric: agrees with the high-precision reference `0.746824133...` within `10^-20` where supported |
| SYS-10 | full | offline | `execute_code(style="compute")` | `Developer` protected-mode file-read probe | Semantic: unauthorized host file reads should fail safely; keep this as an opt-in security test only |

## Broad-Domain Corpus

## Core mathematics: arithmetic, algebra, and number theory

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 1 | Arithmetic | math | `2 + 3` | `5` | Exact integer match |
| 2 | Arithmetic | math | `2^100` | `1267650600228229401496703205376` | Exact big-integer match |
| 3 | Arithmetic | math | `1/3 + 1/7` | `10/21` | Exact rational match |
| 4 | Arithmetic | math | `N[Pi, 50]` | `3.14159265358979323846264338327950288419716939937510` | 50-digit Pi matches known value |
| 5 | Arithmetic | math | `Sqrt[2] + Sqrt[3]` | `Sqrt[2] + Sqrt[3]` (stays symbolic) | Does NOT return decimal; remains symbolic |
| 6 | Arithmetic | math | `Sqrt[-1]` | `I` | Returns imaginary unit |
| 7 | Arithmetic | math | `1/0` | `ComplexInfinity` | Returns `ComplexInfinity` with `Power::infy` message |
| 8 | Arithmetic | math | `0/0` | `Indeterminate` | Returns `Indeterminate` with message |
| 9 | Algebra | math | `Expand[(x + y)^3]` | `x^3 + 3*x^2*y + 3*x*y^2 + y^3` | Expanded binomial matches |
| 10 | Algebra | math | `Factor[x^2 - 5*x + 6]` | `(x - 2)*(x - 3)` | Correct linear factors |
| 11 | Algebra | math | `Simplify[(x^2 - 1)/(x - 1)]` | `1 + x` | Common factor cancelled |
| 12 | Algebra | math | `Apart[1/(x^2 - 1)]` | `1/(2*(-1 + x)) - 1/(2*(1 + x))` | Verify via `Together` returning original |
| 13 | Algebra | math | `Together[1/x + 1/(x + 1)]` | `(1 + 2*x)/(x*(1 + x))` | Combined fraction correct |
| 14 | Algebra | math | `PolynomialQuotientRemainder[x^3 + 2*x + 1, x - 1, x]` | `{1 + x + x^2, 4}` | Quotient `x^2+x+1`, remainder `4` |
| 15 | Algebra | math | `Factor[x^2 + 1]` | `1 + x^2` | Unchanged - irreducible over rationals |
| 16 | Algebra | math | `Factor[x^2 + 1, GaussianIntegers -> True]` | `(x - I)(x + I)` | Factors over Gaussian integers |
| 17 | Algebra | math | `FullSimplify[Sin[x]^2 + Cos[x]^2]` | `1` | Pythagorean identity simplifies to 1 |
| 18 | Number Theory | math | `PrimeQ[104729]` | `True` | 104729 is the 10000th prime |
| 19 | Number Theory | math | `Prime[100]` | `541` | The 100th prime |
| 20 | Number Theory | math | `FactorInteger[1023]` | `{{3, 1}, {11, 1}, {31, 1}}` | 1023 = 3 × 11 × 31 |
| 21 | Number Theory | math | `EulerPhi[12]` | `4` | φ(12) = 4 |
| 22 | Number Theory | math | `GCD[48, 18]` | `6` | Greatest common divisor |
| 23 | Number Theory | math | `PowerMod[2, 10, 1000]` | `24` | 2^10 mod 1000 = 24 |
| 24 | Number Theory | math | `ModularInverse[3, 7]` | `5` | 3 × 5 ≡ 1 (mod 7) |
| 25 | Combinatorics | math | `Binomial[10, 3]` | `120` | C(10,3) exact |
| 26 | Combinatorics | math | `100!` | 158-digit exact integer | Last digits: `...000000000000000000000000` |
| 27 | Combinatorics | math | `PartitionsP[10]` | `42` | Partition count |
| 28 | Combinatorics | math | `CatalanNumber[5]` | `42` | C₅ = 42 |
| 29 | Combinatorics | math | `StirlingS2[5, 3]` | `25` | Stirling number S(5,3) |
| 30 | Combinatorics | math | `Multinomial[2, 3, 4]` | `1260` | 9!/(2!3!4!) |

---

## Calculus: derivatives, integrals, limits, and differential equations

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 31 | Derivative | math | `D[x^3 + 2*x, x]` | `2 + 3*x^2` | Exact symbolic match |
| 32 | Derivative | math | `D[Sin[x^2], x]` | `2*x*Cos[x^2]` | Chain rule applied |
| 33 | Derivative | math | `D[x^4, {x, 3}]` | `24*x` | Third derivative |
| 34 | Integral | notebook | `Integrate[x^2, x]` | `x^3/3` | Antiderivative |
| 35 | Integral | notebook | `Integrate[x^2, {x, 0, 1}]` | `1/3` | Definite integral exact |
| 36 | Integral | notebook | `Integrate[Sin[x]^2, x]` | `x/2 - Sin[2*x]/4` | Verify via `Simplify[result - (x/2 - Sin[2x]/4)] == 0` |
| 37 | Integral | notebook | `Assuming[a > 0, Integrate[1/(x + a), {x, 0, 1}]]` | `Log[1 + 1/a]` | Assumptions affect result |
| 38 | Integral | notebook | `Integrate[1/x^2, {x, 1, Infinity}]` | `1` | Improper integral |
| 39 | Integral | notebook | `Integrate[Exp[-x^2], {x, -Infinity, Infinity}]` | `Sqrt[Pi]` | Gaussian integral |
| 40 | Integral | notebook | `NIntegrate[Exp[-x^2], {x, 0, Infinity}]` | ≈ `0.886227` | Within 1e-5 of Sqrt[Pi]/2 |
| 41 | Limit | math | `Limit[Sin[x]/x, x -> 0]` | `1` | Classic limit |
| 42 | Limit | math | `Limit[(1 + 1/n)^n, n -> Infinity]` | `E` | Euler's number |
| 43 | Series | notebook | `Series[Exp[x], {x, 0, 4}]` | `1 + x + x^2/2 + x^3/6 + x^4/24 + O[x]^5` | Taylor coefficients match |
| 44 | ODE | notebook | `DSolve[y'[x] == y[x], y[x], x]` | `{{y[x] -> C[1]*E^x}}` | General solution |
| 45 | ODE | notebook | `DSolve[{y'[x] == -2*y[x], y[0] == 3}, y[x], x]` | `{{y[x] -> 3*E^(-2*x)}}` | IVP particular solution |
| 46 | ODE | notebook | `NDSolve[{y'[x] == -y[x]^2, y[0] == 1}, y, {x, 0, 5}]` | `{{y -> InterpolatingFunction[...]}}` | Head is InterpolatingFunction; `y[1]/.%[[1]]` ≈ 0.5 |

---

## Linear algebra and solvers

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 47 | Linear Algebra | notebook | `Det[{{1, 2}, {3, 4}}]` | `-2` | Exact |
| 48 | Linear Algebra | notebook | `Inverse[{{1, 2}, {3, 4}}]` | `{{-2, 1}, {3/2, -1/2}}` | Product with original = IdentityMatrix[2] |
| 49 | Linear Algebra | notebook | `Eigenvalues[{{2, 1}, {1, 2}}]` | `{3, 1}` | Eigenvalues correct |
| 50 | Linear Algebra | notebook | `Eigenvectors[{{2, 1}, {1, 2}}]` | `{{1, 1}, {-1, 1}}` | A.v == λ*v for each |
| 51 | Linear Algebra | notebook | `LinearSolve[{{1, 2}, {3, 5}}, {1, 2}]` | `{-1, 1}` | Verify M.x == b |
| 52 | Linear Algebra | notebook | `MatrixRank[{{1,2,3},{4,5,6},{7,8,9}}]` | `2` | Rank-deficient matrix |
| 53 | Linear Algebra | notebook | `NullSpace[{{1,2,3},{4,5,6},{7,8,9}}]` | `{{1, -2, 1}}` | M.v == {0,0,0} |
| 54 | Linear Algebra | notebook | `SingularValueList[{{1, 0}, {0, 2}}]` | `{2, 1}` | Descending order |
| 55 | Linear Algebra | notebook | `Det[{{1, 2}, {2, 4}}]` | `0` | Singular matrix |
| 56 | Linear Algebra | notebook | `Inverse[{{1, 2}, {2, 4}}]` | `Inverse::sing` message | Singularity error |
| 57 | Solve | notebook | `Solve[x^2 - 5*x + 6 == 0, x]` | `{{x -> 2}, {x -> 3}}` | Two solutions |
| 58 | Solve | notebook | `Solve[x^2 + 1 == 0, x]` | `{{x -> -I}, {x -> I}}` | Complex roots |
| 59 | Solve | notebook | `Solve[x^2 + 1 == 0, x, Reals]` | `{}` | No real solutions → empty |
| 60 | Solve | notebook | `Solve[{x + y == 3, 2*x - y == 0}, {x, y}]` | `{{x -> 1, y -> 2}}` | System of equations |
| 61 | Solve | notebook | `NSolve[x^5 - x - 1 == 0, x]` | 5 complex roots; one real ≈ 1.16730 | All 5 satisfy equation |
| 62 | Solve | notebook | `FindRoot[Cos[x] == x, {x, 0}]` | `{x -> 0.739085}` | Dottie number; Cos[x] ≈ x at result |
| 63 | Solve | notebook | `Reduce[x^2 - 4 < 0, x, Reals]` | `-2 < x < 2` | Interval |

---

## Optimization

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 64 | Optimization | notebook | `Minimize[x^2 - 4*x + 5, x]` | `{1, {x -> 2}}` | Min value 1 at x=2 |
| 65 | Optimization | notebook | `Maximize[{x + y, x^2 + y^2 <= 1}, {x, y}]` | `{Sqrt[2], {x -> 1/Sqrt[2], y -> 1/Sqrt[2]}}` | Constrained max on unit disk |
| 66 | Optimization | notebook | `NMinimize[{x^2 + y^2, x + y >= 1}, {x, y}]` | ≈ `{0.5, {x -> 0.5, y -> 0.5}}` | Numeric tolerance 1e-4 |
| 67 | Optimization | notebook | `FindMinimum[x^4 - 3*x^2 + 2, {x, 0.5}]` | ≈ `{-0.25, {x -> 1.22474}}` | Local minimum |
| 68 | Optimization | notebook | `Minimize[{x + 2*y, x + y >= 10, x >= 0, y >= 0}, {x, y}, Integers]` | `{10, {x -> 10, y -> 0}}` | Integer programming |

---

## Statistics, probability, and data analysis

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 69 | Statistics | notebook | `Mean[{1, 2, 3, 4, 5}]` | `3` | Exact |
| 70 | Statistics | notebook | `Median[{1, 3, 5, 7, 9, 11}]` | `6` | (5+7)/2 |
| 71 | Statistics | notebook | `StandardDeviation[{2, 4, 4, 4, 5, 5, 7, 9}]` | `2` | Sample std dev |
| 72 | Statistics | notebook | `Variance[{1, 2, 3, 4, 5}]` | `5/2` | Exact rational |
| 73 | Statistics | notebook | `PDF[NormalDistribution[0, 1], 0]` | `1/Sqrt[2*Pi]` | Symbolic PDF value |
| 74 | Statistics | notebook | `CDF[NormalDistribution[0, 1], 0]` | `1/2` | Exact |
| 75 | Statistics | notebook | `Correlation[{1,2,3,4,5}, {2,4,6,8,10}]` | `1` | Perfect correlation |
| 76 | Statistics | notebook | `Length[RandomVariate[NormalDistribution[0,1], 100]]` | `100` | Structural: list of length 100 |

---

## String manipulation and pattern matching

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 77 | String | math | `StringJoin["Hello", " ", "World"]` | `"Hello World"` | Exact string |
| 78 | String | math | `StringReplace["the cat sat on the mat", "cat" -> "dog"]` | `"the dog sat on the mat"` | Exact string |
| 79 | String | math | `StringCases["the year 2025 and 2026", DigitCharacter..]` | `{"2025", "2026"}` | Two digit-strings |
| 80 | String | math | `StringMatchQ["Hello123", LetterCharacter.. ~~ DigitCharacter..]` | `True` | Boolean True |
| 81 | String | math | `StringSplit["a-b-c-d", "-"]` | `{"a", "b", "c", "d"}` | List of 4 strings |
| 82 | String | math | `StringCases["abc123def456", RegularExpression["[0-9]+"]]` | `{"123", "456"}` | Regex extraction |
| 83 | String | math | `StringLength["Wolfram"]` | `7` | Exact integer |
| 84 | Pattern | notebook | `Cases[{1, "a", 2, "b", 3}, _Integer]` | `{1, 2, 3}` | Integer extraction |
| 85 | Pattern | notebook | `{1, 2, 3} /. x_Integer :> x^2` | `{1, 4, 9}` | RuleDelayed squares |
| 86 | Pattern | notebook | `Cases[{1, 2, "x", 3.5, 4}, _?NumericQ]` | `{1, 2, 3.5, 4}` | PatternTest filter |
| 87 | Pattern | notebook | `Cases[{1, 2, 3, 4, 5}, x_ /; x > 3]` | `{4, 5}` | Condition filter |
| 88 | Pattern | notebook | `Cases[{1, "a", 2.0, "b"}, _Integer \| _String]` | `{1, "a", "b"}` | Alternatives pattern |
| 89 | Pattern | notebook | `ReplaceAll[f[1,2] + f[3,4], f[x_,y_] :> x*y]` | `14` | f[1,2]→2, f[3,4]→12, sum=14 |

---

## List manipulation and functional programming

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 90 | List | math | `Table[i^2, {i, 1, 5}]` | `{1, 4, 9, 16, 25}` | Exact list |
| 91 | List | math | `Map[# + 1 &, {1, 2, 3}]` | `{2, 3, 4}` | Each incremented |
| 92 | List | math | `Select[{1,2,3,4,5,6}, EvenQ]` | `{2, 4, 6}` | Even filter |
| 93 | List | math | `Sort[{3,1,4,1,5,9}, Greater]` | `{9, 5, 4, 3, 1, 1}` | Descending |
| 94 | List | math | `Flatten[{{1,2},{3,{4,5}}}]` | `{1, 2, 3, 4, 5}` | All nesting removed |
| 95 | List | math | `Partition[{1,2,3,4,5,6}, 2]` | `{{1,2},{3,4},{5,6}}` | Three pairs |
| 96 | List | math | `FoldList[Plus, 0, {1,2,3,4}]` | `{0, 1, 3, 6, 10}` | Cumulative sums |
| 97 | List | notebook | `Association["a" -> 1, "b" -> 2, "c" -> 3]` | `<\|"a"->1,"b"->2,"c"->3\|>` | Valid Association |
| 98 | List | notebook | `GroupBy[{1,2,3,4,5,6}, EvenQ]` | `<\|False->{1,3,5}, True->{2,4,6}\|>` | Grouped by predicate |
| 99 | List | notebook | `Merge[{<\|"a"->1,"b"->2\|>, <\|"a"->3,"c"->4\|>}, Total]` | `<\|"a"->4,"b"->2,"c"->4\|>` | Merged with sum |
| 100 | Functional | math | `Nest[# + 1 &, 0, 5]` | `5` | Applied 5 times |
| 101 | Functional | math | `NestList[2*# &, 1, 4]` | `{1, 2, 4, 8, 16}` | Doubling list |
| 102 | Functional | math | `FixedPoint[Floor[#/2] &, 100]` | `0` | Converges to 0 |
| 103 | Functional | math | `Through[{Min, Max}[{3,1,4,1,5}]]` | `{1, 5}` | Simultaneous application |
| 104 | Functional | math | `Composition[Sqrt, Abs][-9]` | `3` | Abs then Sqrt |
| 105 | Functional | math | `Apply[Plus, {1,2,3,4,5}]` | `15` | Head replacement |
| 106 | Functional | math | `MapThread[Plus, {{1,2,3},{10,20,30}}]` | `{11, 22, 33}` | Element-wise add |

---

## Data import/export and date/time

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 107 | Import/Export | notebook | `ExportString[{{1,2,3},{4,5,6}}, "CSV"]` | `"1,2,3\n4,5,6\n"` | CSV formatted |
| 108 | Import/Export | notebook | `ImportString["1,2,3\n4,5,6", "CSV"]` | `{{1,2,3},{4,5,6}}` | Parsed to nested list |
| 109 | Import/Export | notebook | `ExportString[{"a"->1,"b"->2}, "JSON"]` | Valid JSON string | Contains `"a"` and `"b"` keys |
| 110 | Import/Export | notebook | `ImportString["{\"x\":10,\"y\":20}", "RawJSON"]` | `<\|"x"->10,"y"->20\|>` | Association with correct keys |
| 111 | DateTime | notebook | `DateObject[{2025, 1, 1}]` | DateObject for Jan 1, 2025 | Head is DateObject |
| 112 | DateTime | notebook | `DateDifference[{2025,1,1},{2025,12,31}]` | `Quantity[364, "Days"]` | 364 days |
| 113 | DateTime | notebook | `DateString[{2025,7,4}, {"Year","-","Month","-","Day"}]` | `"2025-07-04"` | ISO formatted |
| 114 | DateTime | notebook | `AbsoluteTime[{2000, 1, 1}]` | `3155673600` | Seconds from epoch |

---

## Boolean algebra

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 115 | Boolean | notebook | `BooleanMinimize[a && b \|\| a && !b]` | `a` | Simplified to `a` |
| 116 | Boolean | notebook | `SatisfiableQ[a && !a]` | `False` | Contradiction |
| 117 | Boolean | notebook | `SatisfiableQ[a \|\| b]` | `True` | Satisfiable |
| 118 | Boolean | notebook | `BooleanConvert[Implies[a, b], "CNF"]` | `!a \|\| b` | CNF form |

---

## Plotting and visualization (kernel returns symbolic Graphics)

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 119 | Plot | notebook | `Plot[Sin[x], {x, 0, 2 Pi}]` | Graphics object | `Head[%] === Graphics` |
| 120 | Plot | notebook | `Plot[x^2 - 3x + 1, {x,-2,5}, PlotStyle->{Red,Thick}, PlotLabel->"Quadratic"]` | Styled Graphics | Head is Graphics; not $Failed |
| 121 | Plot | notebook | `Plot[{Sin[x],Cos[x]}, {x,-Pi,Pi}, PlotLegends->"Expressions"]` | Legended Graphics | Head is Legended or Graphics |
| 122 | Plot | notebook | `Plot[1/x, {x, -1, 1}]` | Graphics (singularity at 0 handled) | Head is Graphics; no crash |
| 123 | Plot3D | notebook | `Plot3D[Sin[x]*Cos[y], {x,-Pi,Pi}, {y,-Pi,Pi}]` | Graphics3D object | `Head[%] === Graphics3D` |
| 124 | Plot3D | notebook | `Plot3D[Sin[x^2+y^2], {x,-2,2}, {y,-2,2}, ColorFunction->"Rainbow"]` | Graphics3D with color | `Head[%] === Graphics3D` |
| 125 | ListPlot | notebook | `ListPlot[{1.2,2.5,3.1,2.8,4.0,3.7,5.2}]` | Graphics with points | `Head[%] === Graphics` |
| 126 | ContourPlot | notebook | `ContourPlot[x^2+y^2, {x,-3,3}, {y,-3,3}, Contours->10]` | Graphics with contours | `Head[%] === Graphics` |
| 127 | ParametricPlot | notebook | `ParametricPlot[{Sin[2t],Cos[3t]}, {t,0,2Pi}]` | Graphics (Lissajous curve) | `Head[%] === Graphics` |
| 128 | PolarPlot | notebook | `PolarPlot[1 + 2 Cos[t], {t, 0, 2 Pi}]` | Graphics (limaçon) | `Head[%] === Graphics` |
| 129 | Show | notebook | `Show[Plot[Sin[x],{x,0,2Pi},PlotStyle->Red], Plot[Cos[x],{x,0,2Pi},PlotStyle->Blue]]` | Combined Graphics | Single Graphics, not $Failed |
| 130 | Charts | notebook | `BarChart[{3, 7, 2, 5, 9}]` | Graphics (bar chart) | `Head[%] === Graphics` |
| 131 | RegionPlot | notebook | `RegionPlot[x^2 + y^2 < 1, {x,-1.5,1.5}, {y,-1.5,1.5}]` | Graphics (filled disk) | `Head[%] === Graphics` |

---

## Image processing

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 132 | Image | full | `Image[Table[Mod[i+j,2], {i,100}, {j,100}]]` | Image (checkerboard) | `Head[%]===Image`; `ImageDimensions[%]==={100,100}` |
| 133 | Image | full | `ImageResize[Image[RandomReal[1,{100,100}]], {50,50}]` | Resized Image | `ImageDimensions[%]==={50,50}` |
| 134 | Image | full | `ColorConvert[Image[RandomReal[1,{64,64,3}]], "Grayscale"]` | Grayscale Image | `ImageChannels[%]===1` |
| 135 | Image | full | `Binarize[Image[RandomReal[1,{64,64}]], 0.5]` | Binary Image | Pixel values only 0 or 1 |
| 136 | Image | full | `EdgeDetect[Image[Table[If[30<i<70&&30<j<70,1.,0.],{i,100},{j,100}]]]` | Edge Image | `Head[%]===Image` |

---

## Graph theory

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 137 | Graph | notebook | `Graph[{1->2, 2->3, 3->1, 3->4}]` | Graph object | `GraphQ[%]===True` |
| 138 | Graph | notebook | `CompleteGraph[5]` | K₅ graph | `VertexCount[%]===5`; `EdgeCount[%]===10` |
| 139 | Graph | notebook | `FindShortestPath[Graph[{1<->2,2<->3,3<->4,1<->4,2<->4}], 1, 3]` | Path list, e.g. `{1,2,3}` | `First[%]===1`; `Last[%]===3` |
| 140 | Graph | notebook | `Module[{g=CycleGraph[6]}, {GraphDistance[g,1,4], ConnectedGraphQ[g], VertexDegree[g,1]}]` | `{3, True, 2}` | Exact match |
| 141 | Graph | notebook | `AdjacencyMatrix[CompleteGraph[3]]//Normal` | `{{0,1,1},{1,0,1},{1,1,0}}` | Exact matrix |

---

## Computational geometry and signal processing

| # | Category | Profile | Input Expression | Expected Output | Success Criteria |
|---|----------|---------|-----------------|-----------------|------------------|
| 142 | Geometry | notebook | `RegionMeasure[Disk[{0,0}, 1]]` | `Pi` | Area of unit disk |
| 143 | Geometry | notebook | `Volume[Ball[{0,0,0}, 1]]` | `4 Pi/3` | Volume of unit sphere |
| 144 | Geometry | full | `ConvexHullMesh[RandomReal[1, {20, 2}]]` | BoundaryMeshRegion | `Head[%]===BoundaryMeshRegion` |
| 145 | Geometry | full | `DelaunayMesh[RandomReal[1, {15, 2}]]` | MeshRegion | `Head[%]===MeshRegion` |
| 146 | Signal | full | `Fourier[Table[Sin[2 Pi k/32], {k, 0, 127}]]` | List of 128 complex numbers | `Length[%]===128`; all numeric |
| 147 | Signal | full | `InverseFourier[Fourier[{1,2,3,4,5,6,7,8}]]` | ≈ `{1,2,3,4,5,6,7,8}` | `Chop[%-{1,2,3,4,5,6,7,8}]==={0,0,0,0,0,0,0,0}` |
| 148 | Signal | full | `LowpassFilter[Table[Sin[2Pi*0.05*k]+0.5Sin[2Pi*0.4*k],{k,0,255}],0.2]` | List of 256 reals | `Length[%]===256` |

---

## Units, quantities, and entities (knowledge base)

| # | Category | Profile | Input Expression | Expected Output | Success Criteria | Notes |
|---|----------|---------|-----------------|-----------------|------------------|-------|
| 149 | Units | notebook | `Quantity[100, "Kilometers"]` | `Quantity[100, "Kilometers"]` | Head is Quantity | No internet |
| 150 | Units | notebook | `UnitConvert[Quantity[100,"Kilometers"], "Miles"]` | ≈ `Quantity[62.1371, "Miles"]` | Magnitude ≈ 62.14 | No internet |
| 151 | Units | notebook | `QuantityMagnitude[UnitConvert[Quantity[1,"Hours"],"Seconds"]]` | `3600` | Exact | No internet |
| 152 | Units | notebook | `CompatibleUnitQ[Quantity[1,"Miles"], Quantity[1,"Kilometers"]]` | `True` | Boolean True | No internet |
| 153 | Entity | full | `Entity["Country","UnitedStates"]["Population"]` | Quantity > 300 million | `QuantityMagnitude[%] > 300000000` | **Internet required** |
| 154 | Entity | full | `EntityValue[Entity["Country","France"], "Capital"]` | Entity for Paris | Contains "Paris" | **Internet required** |
| 155 | Entity | full | `Entity["Element","Oxygen"]["AtomicNumber"]` | `8` | Exact | **Internet required** |
| 156 | Entity | full | `Entity["Planet","Mars"]["Mass"]` | ≈ 6.39×10²³ kg | Quantity in kg | **Internet required** |

---

## Machine learning and NLP

| # | Category | Profile | Input Expression | Expected Output | Success Criteria | Notes |
|---|----------|---------|-----------------|-----------------|------------------|-------|
| 157 | ML | full | `FindClusters[{1,2,3,100,101,102}]` | `{{1,2,3},{100,101,102}}` | Two correctly separated clusters | No internet |
| 158 | ML | full | `ClusteringComponents[{1,2,3,100,101,102}]` | `{1,1,1,2,2,2}` or equivalent | Two distinct labels | No internet |
| 159 | ML | full | `Nearest[{1,2,5,10,15}, 6, 3]` | `{5, 10, 2}` | Three nearest elements | No internet |
| 160 | ML | full | `Predict[{1->1,2->4,3->9}, 4]` | ≈ `16` | Numeric ≈ 16 | No internet |
| 161 | NLP | full | `TextWords["The quick brown fox jumps over the lazy dog"]` | List of 9 words | `Length[%]===9` | No internet |
| 162 | NLP | full | `WordCount["The quick brown fox"]` | `4` | Exact | No internet |
| 163 | NLP | full | `TextSentences["Hello world. How are you? I am fine."]` | List of 3 sentences | `Length[%]===3` | No internet |

---

## Specialized domains: finance, control, tensors, and more

| # | Category | Profile | Input Expression | Expected Output | Success Criteria | Notes |
|---|----------|---------|-----------------|-----------------|------------------|-------|
| 164 | Financial | full | `TimeValue[1000, 0.05, 10]` | ≈ `1628.89` | FV of $1000 at 5% for 10 years | No internet |
| 165 | Financial | full | `TimeValue[Annuity[100,10], 0.05, 0]` | ≈ `772.17` | PV of annuity | No internet |
| 166 | Control | full | `TransferFunctionModel[1/(s^2+s+1), s]` | TransferFunctionModel object | Head matches | No internet |
| 167 | Control | full | `SystemsModelQ[TransferFunctionModel[1/(s+1), s]]` | `True` | Boolean True | No internet |
| 168 | Tensor | full | `TensorProduct[{a,b},{c,d}]` | `{{a*c,a*d},{b*c,b*d}}` | Outer product correct | No internet |
| 169 | Tensor | full | `TensorContract[{{a,b},{c,d}}, {{1,2}}]` | `a + d` (trace) | Symbolic sum | No internet |
| 170 | Tensor | full | `ArrayReshape[Range[12], {3,4}]` | `{{1,2,3,4},{5,6,7,8},{9,10,11,12}}` | 3×4 matrix | No internet |
| 171 | DiffGeom | full | `CoordinateTransform["Cartesian"->"Polar", {3,4}]` | `{5, ArcTan[4/3]}` | r=5, θ=ArcTan[4/3] | No internet |
| 172 | DiffGeom | full | `CoordinateTransform["Cartesian"->"Spherical", {1,1,1}]` | `{Sqrt[3], ArcCos[1/Sqrt[3]], Pi/4}` | 3 coordinates | No internet |
| 173 | Coding | full | `HammingDistance[{0,1,0,1},{1,1,0,0}]` | `2` | Exact | No internet |
| 174 | Coding | full | `HammingDistance["karolin","kathrin"]` | `3` | Exact | No internet |
| 175 | Astronomy | full | `MoonPhase[]` | Number 0–1 or icon | Numeric or graphic result | Uses system clock |
| 176 | Chemistry | full | `Entity["Element","Hydrogen"]["AtomicMass"]` | ≈ Quantity[1.008, "AtomicMassUnit"] | Head is Quantity | **Internet required** |
| 177 | Audio | full | `AudioGenerator[{"Sin",440}, 1]` | Audio object (1s, 440Hz) | `Head[%]===Audio` | No internet |
| 178 | Parallel | full | `ParallelTable[i^2, {i,1,10}]` | `{1,4,9,16,25,36,49,64,81,100}` | Same as serial Table | Launches subkernels |
| 179 | FileSystem | full | `Directory[]` | String path | Valid path string | No internet |
| 180 | FileSystem | full | `FileExistsQ[$InstallationDirectory]` | `True` | Boolean True | No internet |
| 181 | External | full | `HTTPRequest["https://httpbin.org/get"]` | HTTPRequest object | Head is HTTPRequest (no network call) | No internet for object creation |
| 182 | Dynamic | full | `Manipulate[Plot[Sin[n*x],{x,0,2Pi}],{n,1,10}]` | Symbolic Manipulate expression | Returns Manipulate[...] form | **Requires frontend** for rendering |
| 183 | Notebook | full | `CreateDocument[{TextCell["Test"]}]` | NotebookObject | Returns NotebookObject | **Requires frontend** |
| 184 | Geo | full | `GeoDistance[GeoPosition[{40.7128,-74.006}],GeoPosition[{51.5074,-0.1278}]]` | ≈ 5570 km | Quantity > 5000 km | **Internet required** |

---

## Seventeen critical edge cases that break naive implementations

These tricky expressions are specifically chosen to expose common MCP server failures around assumptions, precision, output types, and error handling.

| # | Category | Profile | Input Expression | Expected Output | What it tests |
|---|----------|---------|-----------------|-----------------|---------------|
| E1 | Edge | math | `Sqrt[2] // N` | `1.41421` (approx) | Forcing numeric evaluation of symbolic |
| E2 | Edge | notebook | `Integrate[Sin[Sin[x]], {x, 0, Pi}]` | Unevaluated (no closed form) | Handling expressions without closed forms |
| E3 | Edge | notebook | `Assuming[x > 0, Simplify[Sqrt[x^2]]]` | `x` | Assumption-dependent simplification |
| E4 | Edge | notebook | `Solve[x^2 + 1 == 0, x, Reals]` | `{}` | Empty solution set handling |
| E5 | Edge | notebook | `NMinimize[{x, x > 1 && x < 0}, x]` | Infeasible - returns `Infinity` | Contradictory constraints |
| E6 | Edge | math | `$MachinePrecision` | ≈ `15.9546` | Machine precision query |
| E7 | Edge | math | `N[1/3, 100]` | 100-digit decimal | Arbitrary precision arithmetic |
| E8 | Edge | notebook | `Plot[{}, {x, 0, 1}]` | Empty Graphics (axes only) | Empty plot handling |
| E9 | Edge | notebook | `Inverse[{{1,2},{2,4}}]` | Error: `Inverse::sing` | Singular matrix error handling |
| E10 | Edge | notebook | `NDSolve[{y'[x]==-y[x]^2,y[0]==1},y,{x,0,5}]` | InterpolatingFunction | Numeric ODE → non-standard output type |
| E11 | Edge | full | `CountryData["France","Population"]` | Quantity with "People" | Network-dependent entity lookup |
| E12 | Edge | full | `Classify["Sentiment","I love this product!"]` | `"Positive"` | Built-in classifier (may need download) |
| E13 | Edge | full | `BlockchainData["Bitcoin","BlockCount"]` | Large integer | Network + blockchain API |
| E14 | Edge | math | `Limit[1/x, x -> 0, Direction -> "FromAbove"]` | `Infinity` | Directional limit |
| E15 | Edge | math | `Limit[1/x, x -> 0, Direction -> "FromBelow"]` | `-Infinity` | Opposite direction |
| E16 | Edge | notebook | `Series[1/(1-x), {x, 0, 5}]` | `1+x+x^2+x^3+x^4+x^5+O[x]^6` | SeriesData object handling |
| E17 | Edge | full | `Manipulate[x^2, {x,0,10}]` | Symbolic form (no render) | Frontend-absent behavior |

---


## CI Buckets

For CI, split this corpus into four executable buckets:

1. `core-math`: deterministic `math` profile tests only
2. `notebook-offline`: graphics, import/export, notebook parsing, rasterization
3. `full-offline`: ML, image, signals, control, resource tools that do not require the network
4. `optional-env`: frontend, network, subkernels, security probes

## Minimal Harness Policy

For each row, store:

- `id`
- `profile`
- `env`
- `tool`
- `code_or_workflow`
- `success_oracle_type`
- `success_oracle`
- `skip_policy`

Recommended result states:

- `pass`
- `fail`
- `skip_unavailable`
- `skip_not_enabled`
- `error_harness`
