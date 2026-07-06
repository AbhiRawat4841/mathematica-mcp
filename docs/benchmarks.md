# Performance Benchmarks

Benchmark snapshots, each dated at its source. All numbers are median unless noted.

> The **v1.1.1 section below** covers the optimization round (prewarm, compact responses, screenshot cache, addon rung, transport-floor attribution). The **v1.0.1 section** reflects the prior architecture snapshot (warm persistent-kernel funnel + `state_delta` response gating). Sections after it are historical pre-1.0 snapshots kept for comparison.

---

## v1.1.1 (2026-07-06)

**Sources:** `benchmarks/results/transport_floor.json` + `.md`, `benchmarks/results/channel_split.json`, feature-level measurements from the optimization round. Same environment as v1.0.1.

### Where the ~30ms transport floor actually lives

A dedicated probe (`benchmarks/probe_transport_floor.py`) decomposed the bare-`ping` floor by measuring the identical client code path against three servers:

| Layer | p50 | Share of floor |
|-------|-----|----------------|
| Python client + loopback wire (pure-Python echo) | 0.07ms | 0.2% |
| Wolfram `SocketListen` + handler, headless kernel | 1.93ms | 5.8% |
| Live addon `ping` (front-end kernel) | 31.9ms | - |
| Front-end-kernel scheduling residue (by elimination) | ~30.0ms | 94% |

The floor is not transport, JSON, or dispatch work (those measure under 2ms combined). The remaining ~30ms is attributed by elimination to the wait for the front-end-bonded kernel to service its async socket task, roughly one scheduling tick per handler invocation - an inference, not direct instrumentation, but one corroborated by two independent signatures below (pipelining collapse and faster-under-load service). Client-side micro-optimization cannot move it. A headless kernel answers the same `SocketListen` in ~2ms, so a sub-5ms floor would require a gateway-kernel architecture (out of scope).

Two corollaries, both verified live:

- **Batching pays one tick for N commands.** The addon processes all newline-delimited messages that arrive together in a single handler invocation, and `batch_commands` runs its sub-commands inline the same way. Measured: 10 pipelined pings = 5.9ms per request; a `batch` call with 10 sub-pings = 44.6ms total, ~4.5ms per sub-command (vs ~32ms each sequentially). Note this improves throughput, not time-to-first-result: all N results return together. Agents issuing several addon calls should use the lean `batch` tool.
- **The floor is idle-state latency.** Under an active front-end evaluation the async task is serviced faster (~16-20ms observed), which is the signature of scheduling residue rather than fixed work.

### Optimization round: measured effects

| Change | Before | After |
|--------|--------|-------|
| First warm-funnel call after server start | ~12.9s (kernel boot on first use) | background prewarm at startup; boot overlaps agent setup (`MATHEMATICA_PREWARM`, default on) |
| Lean `evaluate` response, trivial result | 361 B (`standard`) | 119 B (compact default; failures keep the full shape) |
| Lean `evaluate` response, graphics + `state_delta` | 10,298 B | 1,701 B |
| Repeat `screenshot` of unchanged notebook | ~403ms per call | ~0ms on cache hit (opt-in `cache=True`, epoch-invalidated) |
| `verify_derivation` / `get_constant` during the boot window | ~12.5s (cold subprocess) | ~30ms via the opt-in addon rung (`execution_method='addon'`) |
| Frontend-mode dispatch (`execute_code` mode=frontend) | up to 10s in-handler poll, then a false "success" | ~0.3-0.4s honest `evaluation_pending` (0.2s poll cap) |
| Concurrent `ping` during a frontend evaluation | 5.1s (blocked behind the poll) | 23ms |

Failure responses are exempt from compact slimming by design (an agent recovering from an error needs `messages`, `transport_status`, `error_families`, `output_preview`); large lean outputs flow to cursor pagination instead of lossy summarization.

### Read-only second socket: measured no-go

`benchmarks/measure_channel_split.py` tested whether a second socket could unblock status calls during long evaluations. Result: no. During a long `execute_code` or a kernel-mode notebook evaluation, the kernel's single command queue blocks a second socket exactly as long as the first (measured 2.9s block on both connections during a 3s kernel-mode eval; 3.7s during a preemptive-link eval). The only case a second channel could serve (frontend-mode evaluation) is one where the existing single connection already answers in ~13ms. No product code was added.

---

## v1.0.1 (2026-07-06)

**Sources:** `benchmarks/results/phase_v101.json`, `benchmark_notebook_ops.py`, warm-funnel probe
**Environment:** macOS ARM64 (same dev machine as prior snapshots) | Mathematica 15.0 | Python 3.11

### Warm funnel (new in 1.0: these tools previously spawned a cold `wolframscript` subprocess per call)

| Tool (warm persistent session) | v1.0.1 median | Pre-1.0 (cold subprocess per call) |
|--------------------------------|---------------|-------------------------------------|
| `verify_derivation` (2 steps) | **2.3ms** | ~12,500ms |
| `get_constant("Pi")` | **1.7ms** | ~12,500ms |
| `list_loaded_packages` | 29.1ms | ~12,500ms |
| `get_kernel_state` | 47.0ms | ~12,500ms |

One-time cost: the persistent kernel session starts on first use (~12.9s, includes Wolfram kernel boot). Cold-execution counter after the probe: **0** - no `wolframscript` subprocess was spawned on any warm path.

### Live addon operations (after `state_delta` gating)

The meaningful way to read these round trips is against the **transport floor**: a bare `ping` (no kernel work, no `state_delta`) costs ~30ms of socket + JSON + handler-scheduling overhead, so an operation's real cost is what it adds *above* that floor.

| Operation | v1.0.1 median | Above floor | Note |
|-----------|---------------|-------------|------|
| `ping` (transport floor) | 29.7ms | - | pure transport, zero kernel work |
| `execute_code` (trivial) | 28.4-32.7ms | ~0ms | pure-kernel: carries no `state_delta` |
| `get_notebook_info` | 33.0-34.6ms | ~3ms | carries `state_delta` by design |
| `write_cell` | 47.9ms | ~18ms | cell creation + `state_delta` |
| `get_cells` | 68.0ms | ~38ms | O(cell count) `NotebookRead` |
| `screenshot_notebook` | 403.5ms | ~374ms | front-end PNG export |

Ranges show run-to-run variance on the identical setup (`execute_code` measured 32.7ms and 28.4ms an hour apart) - about ±4ms, larger than most cross-snapshot differences.

**Why the 2026-01 baseline is not a comparison column:** that snapshot (below) was taken on Mathematica **14.x**, pre-1.0, and recorded `get_notebook_info` at 21ms - *below today's 29.7ms empty-command floor*, which is unreachable under the current front end's transport. Its deltas against v1.0.1 reflect the Mathematica version and machine state, not code changes. The `state_delta` gating change is about *what* pure-kernel responses depend on, not raw latency: they no longer touch the front end at all (previously every response computed `SelectedNotebook[]`/`Cells[]`, ~80% of in-kernel processing, and stalled whenever the front end was busy).

### Offline operations

| Operation | v1.0.1 | v0.8.0 (2026-04) |
|-----------|--------|------------------|
| `cold_import` | 318ms | 515ms |
| `lookup_end_to_end` (hot index) | 0.7ms | 0.7ms |
| `notebook_python_parse` (cold / warm) | 4.7ms / 0.5ms | 5.2ms / 0.6ms |
| `symbol_index_build` (raw, from cold process) | 12.4s | 12.3s |

The symbol-index build now runs through the warm funnel; from a cold process its time is dominated by the one-time kernel startup (~13s). Within an already-warm server process the build is a single in-kernel `Names[]` call.

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

The addon's reported `timing_ms` now correctly measures evaluation time. Previously, `ToExpression[code]` evaluated the expression *before* `AbsoluteTiming` ran, so every reported timing was ~0ms no matter how long the computation actually took. The fix uses `HoldComplete` to defer evaluation into the timed block, which also makes context isolation and timeouts actually apply.

| Aspect | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| `timing_ms` for an expensive evaluation (e.g. `Integrate[Sin[x]^10, x]`) | ~0ms regardless of real cost | actual duration (e.g. 15-50ms) |
| Context isolation | No-op (code already evaluated) | Correctly applied |
| Timeout (`TimeConstrained`) | No-op (code already evaluated) | Correctly enforced |
| Syntax error detection | Not distinguished | `error_type: "syntax_error"` |

Trivial expressions like `1+1` still report `0ms` - they genuinely complete in under a millisecond and round down. The bug was that *everything* reported ~0, making expensive and trivial evaluations indistinguishable.

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
uv run python benchmarks/benchmark_notebook_ops.py my_run
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
