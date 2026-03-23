# Performance Benchmarks

Benchmark data from two separate measurement runs. All numbers are median unless noted.

---

## Offline Operations (No Wolfram Runtime Required)

**Source:** `benchmarks/results/phase_current.json`
**Date:** 2026-03-23 | **Python:** 3.11.5

| Operation | Median | Mean | Iterations | Notes |
|-----------|--------|------|------------|-------|
| `cold_import` | 695ms | 781ms | 3 | `import mathematica_mcp.server` |
| `symbol_index_build` | 13,659ms | 13,659ms | 1 | One-time build, 7,800 System symbols |
| `symbol_index_search` | 1.2ms | 1.4ms | 2,000 | Per-query name-match substep |
| `symbol_subprocess_old` | 14,012ms | 16,131ms | 4 | Old path: full `Names[]` scan per call |
| `lookup_end_to_end` (cold) | 23,394ms | 23,394ms | 2 | Subprocess fallback |
| `lookup_end_to_end` (hot) | 1.2ms | 1.2ms | 20 | Cached index search |
| `raster_cache` (miss) | 0.001ms | 0.001ms | 10,000 | Cache lookup only |
| `raster_cache` (hit) | 0.008ms | 0.009ms | 10,000 | Cache lookup + path return |
| `notebook_python_parse` (cold) | 9.8ms | 9.8ms | 1 | Integration.nb, 2 cells |
| `notebook_python_parse` (warm) | 1.5ms | 1.5ms | 3 | Subsequent parses |
| `cache_epoch` (hit) | 0.003ms | 0.003ms | 50,000 | Epoch validation check |
| `cache_epoch` (miss) | 0.002ms | 0.003ms | 50,000 | Epoch mismatch |
| `telemetry_overhead` (disabled) | 0.41ms | 0.59ms | 5,000 | Boolean check only |
| `telemetry_overhead` (enabled) | 0.31ms | 0.54ms | 5,000 | With timing capture |

### Symbol Index Speedup

| Path | Median | Speedup |
|------|--------|---------|
| Old (subprocess per call) | 14,012ms | baseline |
| New (cached index, hot) | 1.2ms | ~11,700x |

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

> These are pre-optimization baseline timings from January 2026. Post-optimization timings should be re-measured to capture symbol index, raster cache, and cache epoch improvements.

---

## Profile Surface

| Profile | Tool Count | Use Case |
|---------|-----------|----------|
| `math` | ~25 | Pure computation, no file/notebook ops |
| `notebook` | ~44 | Math + notebook reading |
| `full` | ~79 | Complete feature set |

Fewer tools = smaller schema payload during MCP negotiation.

---

## Reproducing Benchmarks

### Offline benchmarks (no Mathematica needed)

```bash
PYTHONPATH=src python benchmarks/benchmark_perf_phases.py my_run
```

Results are written to `benchmarks/results/phase_<run_name>.json`.

### Live addon benchmarks (Mathematica + addon required)

```bash
python benchmarks/benchmark_notebook_ops.py my_run
```

### Profile surface analysis

```bash
PYTHONPATH=src python benchmarks/profile_surface.py
```

### Caveats

- Online benchmarks require a running Mathematica instance with the addon loaded
- Some fixtures (e.g., `Integration.nb`) are optional and benchmarks handle their absence gracefully
- Timings vary with hardware, kernel state, and whether the kernel is warm or cold
