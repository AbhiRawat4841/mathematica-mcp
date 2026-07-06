# Performance Benchmarks

Historical benchmark snapshots, each dated at its source. All numbers are median unless noted.

> **Current architecture (v1.0):** the warm persistent-kernel funnel (`evaluate_wl` routes through a persistent wolframclient session with cold `wolframscript` fallback) and `state_delta` response gating landed **after** all measurements below were taken. The tables reflect the pre-1.0 architecture at their stated dates and have not been re-measured; treat them as historical context, not current numbers.

---

## Offline Operations (No Wolfram Runtime Required)

**Source:** `benchmarks/results/phase_v080.json`
**Date:** 2026-04-04 | **Python:** 3.11.5

| Operation | Median | Mean | Iterations | Notes |
|-----------|--------|------|------------|-------|
| `cold_import` | 515ms | 462ms | 3 | `import mathematica_mcp.server` |
| `symbol_index_build` (raw) | 12,270ms | 12,270ms | 1 | One-time wolframscript build, ~7,800 System symbols |
| `ensure_index` (cold) | ~11,897ms | ~11,897ms | 1 | Full cold start: disk cache miss + wolframscript build + disk cache write |
| `ensure_index` (warm) | 0.9ms | 1.0ms | 3 | Disk cache hit (in-memory invalidated each iteration) |
| `symbol_index_search` | 0.6ms | 0.7ms | 2,000 | Per-query name-match substep |
| `symbol_subprocess_old` | 12,045ms | 12,178ms | 4 | Old path: full `Names[]` scan per call |
| `lookup_end_to_end` (cold) | 13,860ms | 13,860ms | 2 | Subprocess fallback |
| `lookup_end_to_end` (hot) | 0.7ms | 0.7ms | 20 | Cached index search |
| `raster_cache` (miss) | 0.001ms | 0.001ms | 10,000 | Cache lookup only |
| `raster_cache` (hit) | 0.005ms | 0.005ms | 10,000 | Cache lookup + path return |
| `notebook_python_parse` (cold) | 5.2ms | 5.2ms | 1 | Integration.nb, 2 cells |
| `notebook_python_parse` (warm) | 0.6ms | 0.6ms | 3 | Subsequent parses |
| `cache_epoch` (hit) | 0.001ms | 0.002ms | 50,000 | Epoch validation check |
| `cache_epoch` (miss) | 0.001ms | 0.001ms | 50,000 | Epoch mismatch |
| `telemetry_overhead` (disabled) | 0.13ms | 0.14ms | 5,000 | Boolean check only |
| `telemetry_overhead` (enabled) | 0.13ms | 0.14ms | 5,000 | With timing capture |

### Symbol Index Speedup

| Path | Median | Speedup |
|------|--------|---------|
| Old (subprocess per call) | 12,045ms | baseline |
| New (disk-cached index, warm) | 0.9ms | ~13,400x (cold start eliminated) |
| New (in-memory index, hot) | 0.7ms | ~17,200x |

The symbol index is now persisted to disk at `~/.cache/mathematica-mcp/symbols/`. On subsequent process starts, the index loads from disk in <100ms instead of rebuilding via wolframscript (~14s). The cache key is derived from the resolved wolframscript binary identity (path + mtime + size), so it auto-invalidates when the Wolfram Language installation changes.

---

## Live Addon Operations (Wolfram Runtime Required)

**Source:** `benchmarks/benchmark_results_baseline.json`
**Date:** 2026-01-22 | **Pre-optimization baseline**

| Benchmark Name | Median | Mean | Iterations |
|---------------|--------|------|------------|
| `execute_code_simple` | 28.6ms | 28.9ms | 10 |
| `execute_code_notebook_kernel` | 74.0ms | 71.6ms | 5 |
| `execute_code_notebook_frontend` | 11,263ms | 11,269ms | 3 |
| `execute_code_notebook_kernel_integrate` | 60.2ms | 60.9ms | 5 |
| `execute_code_notebook_kernel_plot` | 104.8ms | 105.3ms | 3 |
| `get_cells` | 122.0ms | 125.2ms | 10 |
| `screenshot_notebook` | 529.0ms | 533.1ms | 3 |
| `get_notebook_info` | 21.5ms | 27.0ms | 10 |
| `write_cell` | 39.4ms | 37.9ms | 5 |

> **Note:** These are pre-optimization baseline timings from January 2026. The notebook benchmark now uses a dedicated session (`session_id`) for all operations, matching the production usage pattern. The addon's `timing_ms` field reflects actual evaluation time (not socket transport), so do not compare it directly to wall-clock latency above.

---

## Frontend Polling Fix (v0.9.4)

**Source:** `benchmarks/benchmark_bottleneck_diagnosis.py`
**Date:** 2026-04-09 | **Post-fix measurements**

The `executeCodeNotebookFrontend` polling loop was using `Length[Cells[nb]]` cell-count checking to detect output cell creation. From the preemptive link, this check never resolved, burning the entire `max_wait` timeout. Replaced with `CurrentValue[nb, Evaluating]` polling.

| Benchmark | Before (v0.9.3) | After (v0.9.4) | Speedup |
|-----------|-----------------|-----------------|---------|
| `exec_notebook frontend 1+1` | 10,893ms | 129ms | **85x** |
| `evaluate_cell 2+2` | 134ms | 98ms | 1.4x |
| `exec_notebook kernel 1+1` | 55ms | 47ms | 1.2x |
| `execute_code 1+1` (preemptive) | 29ms | 30ms | (baseline) |
| `ping` (socket round-trip) | 28ms | 27ms | (baseline) |

### Bottleneck Diagnosis Results

The benchmark also confirmed these architectural properties:

| Scenario | Finding |
|----------|---------|
| Main link blocked (Pause[15]) | Preemptive path unaffected (11ms); evaluate_cell returns at max_wait |
| Preemptive link blocked (Pause[8]) | ALL MCP commands stall (ping: 7,015ms queued behind blocker) |
| Socket health | Cold connect: 5ms; reconnect cycle: 6ms - negligible overhead |
| Memory pressure (100MB loaded) | No degradation (3ms before and after) |
| RestartMCPServer[] | Performance identical before/after (only fixes broken TCP state) |

---

## Addon Timing Accuracy

The addon's reported `timing_ms` now correctly measures evaluation time. Previously, `ToExpression[code]` was called before `AbsoluteTiming`, causing all reported timings to be ~0ms. The fix uses `HoldComplete` to defer evaluation into the timed block, which also corrects context isolation and timeout behavior.

| Aspect | Before | After |
|--------|--------|-------|
| `timing_ms` for `1+1` | 0 | 0 (correct: sub-ms rounds to 0) |
| `timing_ms` for `Integrate[Sin[x]^10, x]` | 0 | Actual ms (e.g., 15-50ms) |
| Context isolation | No-op (code already evaluated) | Correctly applied |
| Timeout (`TimeConstrained`) | No-op (code already evaluated) | Correctly enforced |
| Syntax error detection | Not distinguished | `error_type: "syntax_error"` |

---

## Large Response Handling

Responses that previously hard-failed with "Response too large" at 20MB are now handled gracefully:

- **Command-level truncation**: `execute_code` and notebook kernel mode skip redundant representations (FullForm, TeXForm) when the primary InputForm output exceeds 5M characters.
- **processCommand safety net**: A total-budget check (15M characters across all string fields) truncates oversized fields before JSON serialization, preventing the 20MB wire cap from being hit.
- **Explicit truncation flag**: Truncated responses include `"truncated": true` and `"truncated_fields"` metadata indicating which fields were shortened and their original sizes.

---

## Profile Surface

| Profile | Tool Count | Use Case |
|---------|-----------|----------|
| `lean` (default) | 12 | Consolidated v1.0 surface (`evaluate`, `notebooks`, `cells`, ...) |
| `math` | 28 | Pure computation, no file/notebook ops |
| `notebook` | 48 | Math + notebook reading + `create_notebook` |
| `classic` / `full` | 82 | Complete legacy feature set (all pre-1.0 tool names) |

Fewer tools = smaller schema payload during MCP negotiation.

---

## Reproducing Benchmarks

### Offline benchmarks (no Mathematica needed)

```bash
PYTHONPATH=src python benchmarks/benchmark_perf_phases.py my_run
```

Results are written to `benchmarks/results/phase_<run_name>.json`.

The offline suite includes:
- `symbol_index_build` - raw wolframscript build cost (no cache)
- `ensure_index_cold` - full cold start with disk cache cleared
- `ensure_index_warm` - disk cache load (in-memory invalidated each iteration)
- Symbol search, raster cache, notebook parse, cache epoch, telemetry overhead

### Live addon benchmarks (Mathematica + addon required)

```bash
python benchmarks/benchmark_notebook_ops.py my_run
```

The live benchmark creates a dedicated notebook session (`session_id=benchmark-session`) and threads it through all operations. The session is cleaned up at the end.

### Profile surface analysis

```bash
PYTHONPATH=src python benchmarks/profile_surface.py
```

### Caveats

- Online benchmarks require a running Mathematica instance with the addon loaded
- Some fixtures (e.g., `Integration.nb`) are optional and benchmarks handle their absence gracefully
- Timings vary with hardware, kernel state, and whether the kernel is warm or cold
- Addon `timing_ms` measures evaluation time only (excludes socket transport and frontend overhead)
- Benchmark results are validated before recording: builds that produce 0 symbols are reported as failed, not as valid timings
