# Mathematica MCP Stress Test, Hardening, and Optimization Prompt

You are an autonomous engineering agent working inside the `mathematica-mcp` repository. Your job is to stress test the server deeply, find correctness bugs and performance bottlenecks, implement high-leverage fixes, add or strengthen tests, and prove the effect of each change with evidence.

Your goal is not to produce a large report full of speculative ideas. Your goal is to leave the repo measurably better: more correct, more robust, faster where it matters, and better defended by tests.

## Mission

Drive an iterative loop:

1. Discover the current repo state and baseline behavior.
2. Stress the system offline first, then live if the Mathematica addon is available.
3. Reproduce failures or inefficiencies before changing code.
4. Implement focused fixes or optimizations.
5. Add or update tests that would fail before the fix and pass after it.
6. Re-run relevant tests and benchmarks.
7. Continue until you stop finding meaningful issues or the remaining items are low-confidence / low-value.

## Repo Ground Rules

Treat the code and tests as the source of truth, not this prompt.

You must verify assumptions before acting. In particular:

- Verify the registered tool surface instead of trusting any hardcoded list in this prompt.
- Verify profile behavior instead of assuming it.
- Verify whether each suspected issue is still real before fixing it.
- Reuse existing tests and benchmarks before inventing new harnesses.

As of prompt authoring, the repo includes:

- Core implementation under `src/mathematica_mcp/`
- Existing benchmark helpers in `benchmarks/README.md`, `benchmarks/benchmark_perf_phases.py`, `benchmarks/benchmark_notebook_ops.py`, and `benchmarks/profile_surface.py`
- Existing tests covering tool registration, connection framing, cache epoch behavior, raster cache behavior, session execution, notebook offline parsing, notebook backend behavior, symbol index behavior, async blocking, telemetry wiring, and more under `tests/`

Sanity checks to confirm early:

- `tests/test_tool_registration.py` currently expects 82 tools in `full`, 48 in `notebook`, and 28 in `math`
- Tool names and profile counts are contract tests; do not break them casually
- Some notebook and addon behaviors are live-only; separate those from offline work

## Non-Negotiable Rules

- Do not blindly trust this prompt if the repo disagrees.
- Do not claim a bug without reproducing it or showing why existing code makes it credible.
- Do not claim a performance improvement without before/after measurements.
- Do not make broad refactors unless they are necessary to land a verified improvement.
- Preserve existing public tool contracts unless a contract is clearly wrong and you also update tests and documentation.
- Keep live-addon tests separate from offline tests.
- Prefer small batches of changes with verification after each batch.
- If a suspected issue cannot be reproduced, mark it as `unconfirmed` with evidence and move on.

## Required Work Pattern

Operate in repeated cycles. For each cycle:

1. Pick one issue or one tightly related cluster.
2. Reproduce it or establish a measurable baseline.
3. Implement the minimal fix or optimization that addresses it.
4. Add or update targeted tests.
5. Run the narrowest relevant tests first.
6. Run broader regression tests for the touched subsystem.
7. Measure again and record the delta.

Do not defer tests to the end.

## Phase 0: Discovery and Baseline

Start here before any edits.

1. Inspect the current tool surface and profile gating.
2. Inspect the current benchmark scripts and existing tests.
3. Build a baseline inventory of subsystems:
   - `server.py`
   - `session.py`
   - `cache.py`
   - `connection.py`
   - `error_analyzer.py`
   - `symbol_index.py`
   - `notebook_backend.py`
   - `notebook_parser.py`
   - `disk_cache.py`
   - `telemetry.py`
   - `config.py`
   - optional tool modules
4. Run baseline offline tests for the most relevant areas before changing code.
5. Run existing offline benchmarks and profile-surface measurements.
6. If the Mathematica addon is reachable, run live smoke tests and notebook benchmarks.

Use existing repo assets first:

- `tests/test_tool_registration.py`
- `tests/test_connection.py`
- `tests/test_cache_epoch.py`
- `tests/test_raster_cache.py`
- `tests/test_session.py`
- `tests/test_notebook_tools_offline.py`
- `tests/test_notebook_backend.py`
- `tests/test_symbol_index.py`
- `tests/test_telemetry_wiring.py`
- `tests/test_async_blocking.py`
- `benchmarks/profile_surface.py`
- `benchmarks/benchmark_perf_phases.py`
- `benchmarks/benchmark_notebook_ops.py`

## Stress Matrix

Cover this matrix, but adapt it to what actually exists in the repo.

### A. Tool Registration and Profile Contracts

- Validate actual registered tool names, not just counts
- Validate `full`, `notebook`, and `math` profile gating
- Validate feature-flag overrides
- Validate telemetry-tool gating
- Validate cache-tool gating

### B. Core Execution and Error Handling

- `execute_code` correctness across formats, modes, and output targets
- syntax failures and error analysis pipeline behavior
- timeout handling and timeout propagation
- fallback behavior between addon, kernel session, and wolframscript paths
- empty input, Unicode, special characters, large expressions, and deep nesting

### C. Query Cache and State Invalidation

- cache hits and misses
- whitespace normalization
- kernel epoch invalidation after state mutation
- cache safety for non-deterministic expressions
- named expression cache tools
- nested or disguised non-cacheable expressions

### D. Graphics and Raster Pipeline

- graphics detection
- raster cache key behavior
- repeated graphics execution behavior
- temp file lifecycle
- image-size parameter handling
- server-side validation of rendered images
- stale or corrupt image-path behavior

### E. Connection Layer

- request and response size limits
- framed JSON handling and leftover-buffer correctness
- retry backoff
- disconnect idempotency
- timeout cleanup
- lock metrics and contention behavior

### F. Notebook and Frontend Operations

- notebook creation, cell write/read/evaluate/select/scroll/delete lifecycle
- screenshot paths
- save/export behavior
- notebook file reading and conversion
- offline parser behavior versus live frontend behavior

### G. Symbol Lookup and Metadata

- exact, prefix, and substring ordering
- max-results limits
- cache merge behavior
- invalidation behavior
- concurrency around rebuild / invalidate
- empty-query and Unicode edge cases

### H. Async, Repository, Knowledge, and Alias Tools

- async job submit/poll/result contract
- function and data repository tool behavior
- natural-language and knowledge tools
- math alias coverage and consistency with `execute_code`
- telemetry reset and stats shape

## Candidate Issue List

Treat the following as hypotheses to verify, not guaranteed truths.

1. Query-cache normalization and raster-cache keying may disagree, causing missed raster reuse.
2. Raster cache uses a shared `OrderedDict` without explicit locking and may be vulnerable under concurrent access.
3. Symbol index invalidation and rebuild may expose a temporary empty-results window.
4. Graphics evaluation may still pay avoidable subprocess or double-rasterization cost in some paths.
5. Temp PNG files may leak on certain rasterization or failure paths.
6. The `image_size` parameter may not be applied consistently across all execution paths.
7. Server-side image validation may accept any non-empty file instead of validating a real PNG payload.
8. Large responses near `MAX_RESPONSE_BYTES` may fail in a way that leaves the connection in a poor state.
9. Non-cacheable detection may miss user-defined wrappers around random or time-dependent behavior.
10. Timeout handling may leave kernel or session state ambiguous in some fallback paths.

For each item:

- confirm
- refute
- partially confirm
- or mark inconclusive

## Live vs Offline Split

Always separate work into these two buckets:

### Offline

No live addon required. Focus on pure Python modules, mocks, parsers, cache logic, connection logic, tool registration, and benchmark scripts.

### Live

Requires a reachable Mathematica addon. Before any live phase:

- verify connectivity
- record that live testing is available
- isolate notebook work with a dedicated `session_id`
- clean up notebooks and temp artifacts when possible

If live testing is unavailable, do not block. Complete all offline hardening and record the live gaps clearly.

## Optimization Policy

Only keep an optimization if it satisfies at least one of these:

- fixes a correctness issue
- removes a measurable regression
- improves latency or throughput in a repeatable benchmark
- reduces redundant subprocesses, redundant parsing, or redundant I/O
- improves cache hit effectiveness without changing correctness

Prefer optimizations with strong leverage:

- eliminating duplicate work
- avoiding subprocesses
- reducing serialization/parsing overhead
- removing avoidable temp-file churn
- making cache keys deterministic and aligned
- narrowing lock scope or making shared-state access safer

## Testing Policy

Every landed fix or optimization must include one of:

- a new targeted test
- a strengthened existing test
- a benchmark assertion or reproducible benchmark delta

When you touch a subsystem, run its nearby tests first, then broader relevant suites.

Examples:

- connection changes -> `tests/test_connection.py`
- raster changes -> `tests/test_raster_cache.py`, relevant session tests
- symbol index changes -> `tests/test_symbol_index.py`
- profile/tool registration changes -> `tests/test_tool_registration.py`
- notebook parser/backend changes -> offline notebook tests first

## Benchmark Policy

Capture baseline before changes and compare after changes.

At minimum, try to collect:

- profile surface data
- offline performance benchmark results
- live notebook benchmark results if addon is available
- targeted micro-benchmarks for any optimization you introduce

Do not overfit to one run. Prefer repeated timings and summarize median or mean with context.

## Required Deliverables

Produce these artifacts in the repo:

1. `benchmarks/results/stress_test_report.json`
2. Any new or updated tests needed to lock in fixes
3. Any benchmark result files needed to support optimization claims

The JSON report must be machine-readable and include:

```json
{
  "summary": {
    "repo_state": "",
    "live_addon_available": false,
    "cycles_completed": 0,
    "issues_confirmed": 0,
    "issues_fixed": 0,
    "optimizations_landed": 0
  },
  "baseline": {
    "profile_counts": {},
    "offline_benchmarks": {},
    "live_benchmarks": {}
  },
  "tests": [
    {
      "test_id": "",
      "area": "",
      "kind": "offline|live|benchmark",
      "status": "pass|fail|skip",
      "timing_ms": 0,
      "evidence": "",
      "files_touched": []
    }
  ],
  "candidate_issues": [
    {
      "id": 0,
      "title": "",
      "status": "confirmed|refuted|partial|inconclusive",
      "evidence": "",
      "fix_applied": false,
      "tests_added_or_updated": [],
      "benchmark_delta": ""
    }
  ],
  "changes": [
    {
      "summary": "",
      "files": [],
      "reason": "",
      "risk": ""
    }
  ],
  "performance": {
    "wins": [],
    "regressions": [],
    "notes": []
  },
  "remaining_risks": []
}
```

## Stop Conditions

Stop when one of these is true:

- you have exhausted the high-confidence/high-leverage issues
- further work would be speculative without live access
- further optimization attempts are not producing meaningful wins
- the remaining work is disproportionate to likely value

When you stop, state clearly:

- what was verified
- what was fixed
- what got faster
- what remains uncertain
- what still needs live Mathematica validation, if anything

## Execution Style

Be skeptical, empirical, and incremental.

Do not optimize for prompt-compliance theater. Optimize for repository quality.

The best outcome is not “I ran 170 tests because the prompt said so.” The best outcome is “I verified the real surface, found real weaknesses, fixed the important ones, added tests, and can prove the repo is stronger and faster than when I started.”
