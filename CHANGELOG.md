# Changelog

All notable changes to this project will be documented in this file.

## [0.9.1] - 2026-04-09

### Changed

- **Guidance system refactored for lean LLM context**: Shared primitive functions (`_style_keyword_table`, `_routing_lines`, `_quick_defaults`, `_recovery_defaults`, `_avoid_lines`) replace per-builder inline copies. Server instructions now carry universal defaults, recovery tools, and anti-patterns; client-specific files (AGENTS.md, CLAUDE.md hint) are additive-only. Always-on context dropped from ~880 to ~540 words (Codex) and ~760 to ~460 words (Claude Code).
- **Profile-aware guidance conditioning**: Math profile no longer mentions live notebooks, `create_notebook`, `style="notebook"`, `style="interactive"`, or `sync`. Intro text adapts per profile. `build_server_instructions()` replaces the hardcoded `instructions` string in `server.py`.
- **Expanded intent keywords**: Added "evaluate", "solve", "graph", "visualize", "slider", "dynamic", "animate", "fresh notebook", "create notebook" across all guidance layers, README, and docs.
- **Restored lost guidance**: `sync="none"` anti-pattern, `save_notebook()`/`screenshot_notebook()`/`screenshot_cell()` routing, and `execute_code` notebook-reuse explanation re-added after earlier refactor dropped them.
- **Word-budget tests**: Guidance tests now enforce per-layer and combined always-on word budgets to prevent future context bloat.

## [0.9.0] - 2026-04-09

### Added

- **Payload shaping** (`response_detail` parameter on `execute_code`): Four detail levels — `compact` (essential fields only, token-efficient), `standard` (exact backward-compatible default), `verbose` (+ detail_level marker), `diagnostic` (+ cache epoch, routing hints). Compact mode preserves notebook IDs, transport status, error families, and handles graphics placeholders by swapping to `output_inputform`. Large outputs auto-summarized with balanced-brace list element counting. Pure response filter module with no runtime singleton dependencies.

- **Session brief** (`get_session_brief()` tool): Compact ~100-token session state snapshot showing connection mode (addon/kernel_only/disconnected), recent error families (sorted by recency with 24h age cutoff), and routing advice. Uses 500ms addon timeout and never starts a fresh kernel session.

- **Computation journal** (`get_computation_journal()`, `clear_computation_journal()` tools): In-memory ring buffer (10 entries) recording code/output previews, timing, transport status, error families, timed_out, and from_cache for each `execute_code` call. Records raw canonical results before response filtering. Survives context compaction.

- **Smart cache optimization**: Pure-System expressions (no user-defined symbols, no session-sensitive introspection like `Names`, `Contexts`, `$Packages`, `Definition`) are now epoch-insensitive — they survive `set_variable`/`clear_variables` mutations without re-evaluation. Memoized single-pass code analysis with `lru_cache(1024)`, static ~200 builtin allowlist, session-sensitive symbol denylist. Malformed input (unbalanced strings/comments) falls back to epoch-sensitive (safe degradation).

- **Expression classification**: Regex-based classifier categorizes Wolfram code into routing-relevant classes (`plot`, `frontend_dynamic`, `symbolic_heavy`, `numeric_heavy`, `io`, `general`). Strips strings and nested comments before matching. Used for expr-type-specific routing cohorts and transport-path hints.

- **Per-path transport outcomes**: Attempt-level telemetry records each transport leg independently (addon_notebook, addon_cli) with typed `AttemptOutcome` (OK, INFRA_ERROR, TIMEOUT, SEMANTIC_ERROR). Persisted in `path_transport_outcomes` on aggregate cohorts. Fixes the data flow bug where addon failures were invisible after kernel fallback.

- **Routing hints**: Advisory hints built from two sources — transport-path hints (per-path infra/timeout rates from attempt telemetry) and end-to-end hints (timeout/fallback rates from final cohort data). Structured `RoutingHint` records with severity/specificity ordering, deduplication, and 5-hint cap. Available via `get_routing_memory_stats(include_hints=True)` and in `diagnostic` response detail.

- **Conservative routing action** (opt-in): Concurrency-safe half-open circuit breaker that skips persistently failing addon_cli transport legs for compute requests. Requires both `MATHEMATICA_ROUTING_MEMORY=advise` and `MATHEMATICA_ROUTING_ACTION=compute_cli_skip`. Breaker trips on 5 consecutive `INFRA_ERROR` outcomes only (ignores timeouts and semantic errors). 60s cooldown with single-probe recovery. Idempotent finish/abort lifecycle. Skip counter observability in routing stats. Uses truthful `kernel_direct_routing_skip` execution path (not `kernel_fallback`).

- **Transport lifecycle API**: `begin_transport_attempt()` / `finish_transport_attempt()` / `abort_transport_attempt()` API. The primary CLI transport path uses a unified `finish` call that handles both persisted attempt telemetry and breaker state. Idempotent with `_completed` guard. Probe-in-flight concurrency control under lock.

- **Shared Wolfram scanner** (`wl_scan.py`): Single-pass state machine for stripping string literals and nested Mathematica comments. Preserves token boundaries with space replacement. Returns `ScanResult(cleaned, ok)` for safe degradation on malformed input. Also provides balanced-brace top-level element counting (tolerates leading whitespace, rejects trailing content).

- **Shared transport classification** (`transport_classification.py`): Single source of truth for attempt-level and final-level classification. Refactored `_classify_transport()` and `_extract_error_families()` from server.py. Handles non-dict results safely.

- **Shared constants** (`constants.py`): `ExecutionPath` labels and `AttemptOutcome` enum used across server, routing memory, and transport classification.

- **Routing memory schema v2**: Adds `path_transport_outcomes` to cohort stats. Backward-compatible: v1 files load cleanly (missing field defaults to empty dict). Zero-valued counters pruned during decay/serialization to prevent JSON growth.

- **Recent error families API** (`get_recent_error_families()`): Public accessor on routing memory, sorted by `last_seen` (not frequency). Filters `"other"` unless no better candidates. Configurable `max_age_seconds` cutoff.

### Changed

- `execute_code` now accepts `response_detail` parameter (default `"standard"` — exact backward-compatible behavior)
- `get_routing_memory_stats` now accepts `include_hints` parameter (default `False`)
- Routing memory `_SCHEMA_VERSION` bumped from 1 to 2 (backward-compatible load)
- `clear_routing_memory()` now resets runtime-only breaker state and skip counters
- `FeatureFlags` now includes `routing_action` field (default `"off"`)

## [0.8.7] - 2026-04-06

### Fixed

- **Corpus manifest tuned for live Mathematica**: all `execute_code` cases now include `output_target: "cli"` so responses return `output_inputform` instead of notebook-mode status
- **Oracle expectations match actual Mathematica output**: `Factor` returns `(-3 + x)*(-2 + x)`, `StringJoin` returns quoted strings, `DSolve` uses `2*x` notation
- **Variable workflow uses valid Wolfram names**: camelCase (`corpusTestVar`) instead of underscores (reserved for patterns)
- **Runtime probe timeout**: increased `wolfram_runtime_available` from 5s to 30s to handle cold kernel startup
- **Documentation accuracy**: fixed corpus counts and added manifest authoring guidance (`output_target`, camelCase naming)

All 31 smoke corpus tests pass against live Mathematica.

## [0.8.6] - 2026-04-06

### Added

- **Corpus-driven MCP test harness**: data-driven test infrastructure that runs Wolfram Language expressions against actual MCP server tools with structured oracle verification (11 strategies: exact, symbolic, numeric, structural, artifact, warning, workflow)
- **Corpus manifest** (`tests/corpus/mathematica_mcp_corpus.json`): executable smoke manifest with 30 cases + 1 variable lifecycle workflow, validated against live Mathematica (31/31 passing)
- **Corpus meta-tests**: 83 tests validating the harness itself (normalizer, verifiers, models, adapters, polling, cleanup) — no Mathematica needed
- **Tiered CI**: smoke meta-tests run on every PR (`ci.yml`); core/extended/probe tiers run via manual dispatch or nightly schedule (`corpus.yml`)
- **Response normalization layer**: uniform handling of JSON, dict, Image, and parse_error payloads with warning/artifact extraction
- **Workflow engine**: self-contained multi-step workflows with per-step assertions, structured polling, state extraction, cleanup error recording, and `{tmp_path}` templating

### Changed

- CI now runs corpus smoke meta-tests as a required step before the full test suite
- `CONTRIBUTING.md` updated with corpus test commands and tool-addition guidance
- `tests/README.md` rewritten to document all three test layers, corpus tiers, and verification strategies
- Added `pydantic>=2.0.0` to dev dependencies

## [0.8.5] - 2026-04-06

### Added

- **Routing memory** (`MATHEMATICA_ROUTING_MEMORY`): opt-in observability layer that collects aggregate routing statistics from `execute_code` — transport success rates, latency histograms, and error family frequencies. Modes: `off` (default), `observe`. No raw code or expressions stored. (`advise` mode reserved for future routing hints.) See [Technical Reference](docs/technical-reference.md#routing-memory).
- **Response metadata normalization**: every `execute_code` response now includes `route_variant`, `execution_path`, `transport_status`, `overall_timing_ms`, and `error_families` across all branches
- **Wolframclient message capture**: the `wolframclient` execution path now captures `$MessageList` with `Block` isolation, matching the existing `wolframscript` path for symmetric error reporting
- **Routing memory admin tools**: `get_routing_memory_stats()` and `clear_routing_memory()` (full profile only, when routing memory is enabled)
- **Routing memory benchmarks**: `bench_routing_memory_overhead`, `bench_cold_import_routing`, `bench_execute_code_normalization` in the benchmark suite

## [0.8.1] - 2026-04-04

### Fixed

- **Absolute launcher paths in generated configs**: `resolve_launcher()` resolves `uv`/`uvx` to absolute paths so GUI MCP clients (Claude Desktop, Cursor) work without shell PATH inheritance
- **Math alias numeric type hints**: Widened `lower_bound`, `upper_bound` (integrate) and `point` (limit, series) to accept `int` and `float` in addition to `str`, preventing Pydantic validation errors when LLMs send numeric literals
- **CLI startup diagnostics**: Added stderr logging and crash reporting to the MCP server entry point for easier debugging

### Changed

- Documentation examples now use `/ABSOLUTE/PATH/TO/uv` placeholder and note that GUI apps may not inherit shell PATH
- Updated tests for `resolve_launcher` integration

## [0.8.0] - 2026-04-04

### Added

- **Execution `style` parameter**: New keyword-only `style` parameter on `execute_code` bundles `output_target` + `mode` into three presets: `"compute"` (cli/kernel), `"notebook"` (notebook/kernel), `"interactive"` (notebook/frontend) — reduces LLM argument assembly errors
- **Pure execution resolver**: `_resolve_execution_params()` handles style→param resolution with explicit>style>profile priority, unknown-style errors, and CLI mode normalization
- **Unconditional execution metadata**: Every `execute_code` response now includes `executed_output_target` (and `executed_mode` when applicable) across all 6 response paths — including timeout and fallback — so callers always know what actually happened
- **Profile-aware guidance**: `build_claude_hint()`, `build_claude_command()`, and `build_codex_guidance()` now conditionally omit notebook/interactive flows for the `math` profile, matching the existing behavior of `build_mathematica_expert_prompt()`

### Fixed

- **`create_notebook` profile exposure**: Moved `create_notebook` from `notebook_advanced` to `notebook_primary` group — previously all guidance told LLMs to call it for "new notebook" flows, but it was not registered in the `notebook` profile
- **Addon robustness**: Added `StringQ` guards in `editableNotebookQ`, `jsonSanitize`, and `maybeCompressResponse` to prevent crashes from unexpected non-string values; improved `dispatchCommand` error capture with `$MessageList`
- **Oversized response handling**: Connection layer now closes socket and clears buffer on oversized responses instead of leaving stale data
- **Ruff lint violations**: Replaced `try/except OSError: pass` with `contextlib.suppress(OSError)` (SIM105); fixed formatting across cli, server, and test files

### Changed

- All guidance builders, MCP prompts, server instructions, and documentation now use `style=` parameter instead of raw `output_target`/`mode` combinations
- Documentation splits "Execution Styles" into two audiences: chat users (keywords) and tool callers (`style=` parameter)
- `execute_code` docstring leads with style selection guide instead of raw parameter combinations
- TOML config rewrite logic improved in CLI setup

## [0.7.7] - 2026-04-01

### Added

- **Execution mode keywords**: Users can steer routing with natural language keywords ("calculate", "plot", "new notebook", "interactive") — documented in README, CLAUDE.md, AGENTS.md, and server instructions
- **MCP server instructions**: FastMCP `instructions` field tells all connected agents to use MCP tools directly, never wolframscript CLI or manual .nb file creation; clarifies "notebook" means a live Mathematica window, not a file on disk
- **MCP prompts**: Five new prompts (`calculate`, `notebook`, `new_notebook`, `interactive`, `quickstart`) so MCP clients can surface structured mode selection to users
- **Codex project guidance**: `build_codex_guidance()` generates AGENTS.md content; `setup codex --project-dir .` now installs it alongside config, matching Claude Code's CLAUDE.md flow
- **Print redirect to notebook**: `Print[]` output during kernel evaluation is captured via `Block[{Print = ...}]` and written as Print-style cells in the notebook instead of the Messages window

### Fixed

- **"New notebook" creates new instead of reusing**: Guidance no longer blanket-discourages `create_notebook`; when the user says "new notebook", the LLM calls `create_notebook(title=...)` first, then `execute_code(output_target="notebook")`
- **Error response for mixed success/error results**: `processCommand` no longer drops the result when the response contains both `"error"` and `"success"` keys

### Changed

- `create_notebook` tool description now says "use when user asks for a NEW notebook" instead of "prefer execute_code instead"
- `execute_code` tool description clarifies it reuses the active notebook and notes `create_notebook` should be called first for new notebooks
- Expert prompt, CLAUDE.md hint, and command guide all include MCP-first directive and keyword→mode routing table
- `--project-dir` setup flag now works for both Claude Code (CLAUDE.md) and Codex (AGENTS.md)

## [0.7.6] - 2026-04-01

### Fixed

- **Addon timing correctness**: `execute_code` and notebook kernel mode now correctly time evaluation by deferring `ToExpression` with `HoldComplete` inside `AbsoluteTiming`. Previously all `timing_ms` values reported ~0 because code was evaluated before the timed block. This also fixes context isolation and timeout enforcement, which were silently broken.
- **Notebook kernel error semantics**: `executeCodeNotebookKernel` now reports `success: false` for `$Failed` results and includes `error_type` (syntax_error/evaluation_error/timeout), consistent with `cmdExecuteCode`.
- **Large response hard-fail**: Responses exceeding 20MB previously returned an opaque "Response too large" error. Added command-level truncation (skip redundant FullForm/TeXForm for large results) and a processCommand safety net with total field budget. Truncated responses include `"truncated": true` metadata.

### Added

- **Symbol index disk persistence**: The symbol index (~7,800 symbols) is now cached to `~/.cache/mathematica-mcp/symbols/` with a cache key derived from the resolved wolframscript binary identity (realpath + mtime + size). Eliminates the ~16s cold-start subprocess call on subsequent process starts (warm load <100ms). Uses a singleflight state machine with generation counter to prevent stale publication after invalidation.
- **Benchmark session management**: Notebook benchmarks now create a dedicated session (`session_id=benchmark-session`) and thread it through all operations, matching production usage patterns.
- **Benchmark validation**: Symbol index build benchmarks validate that symbols were actually loaded before recording timings. Failed builds report `status: failed` instead of misleading zero-symbol timings.
- **Cold/warm startup benchmarks**: New `ensure_index_cold` and `ensure_index_warm` benchmarks measure the full startup path with and without disk cache.

### Changed

- Updated benchmark and technical reference documentation to reflect timing fix, disk caching, and truncation behavior

## [0.7.5] - 2026-03-24

### Changed

- Removed Codecov integration from CI while keeping terminal coverage output
- Switched PyPI publishing to run from version tag pushes or manual dispatch
- Clarified README badges to distinguish repo version from latest published version

## [0.7.0] - 2026-03-24

### Added

- GitHub Actions CI with test matrix (Linux/macOS, Python 3.10-3.12) and package verification
- Release gating via reusable CI workflow with secrets inheritance
- Ruff linter and formatter with enforced checks in CI
- SECURITY.md with threat model, trust boundaries, and known input handling gaps
- CONTRIBUTING.md with dev setup, tool addition guide, and commit conventions
- Benchmark documentation with offline and live-addon timing data
- Two polished example sessions (symbolic calculus, notebook analysis)
- GitHub issue templates for bug reports and feature requests
- CI and PyPI badges in README

### Changed

- Restructured README with audience table, 3 concrete workflow examples, and expanded doc links
- Fixed misleading "secure sandbox" claim to accurately describe timeout/size-limit controls

### Fixed

- FakeSocket test mocks missing settimeout method in telemetry wiring tests
- 26 ruff lint errors and 28 formatting issues across the codebase

## [0.6.5] - 2026-03-24

### Fixed

- End-to-end execution timeout enforcement for all code paths

## [0.6.0] - 2026-03-23

### Added

- Symbol index with cached search for fast symbol lookup
- Raster cache (50-entry LRU for graphics)
- Cache epoch-based invalidation on state mutations
- Telemetry wiring with p50/p95 timing
- Notebook routing: Python parser vs kernel backend dispatch

## [0.5.0] - 2026-03-22

### Performance

- Eliminated duplicate graphics execution and fixed socket/cache hot paths

### Documentation

- Updated README, installation guide, and technical reference for v0.4.0

## [0.4.0] - 2026-03-20

### Added

- Tool profiles: math (~25), notebook (~44), full (~79)
- Feature flags via environment variables
- LLM guidance for optimized call routing

## [0.3.0] - 2026-03-16

### Added

- Notebook backend abstraction with capability-based dispatch

## [0.2.0] - 2026-03-14

### Changed

- Extracted optional tools into modules, added lazy loading

## [0.1.0] – [0.1.5] - 2026-01-23

### Added

- Initial release: MCP server with Mathematica addon
- PyPI publish workflow
- Multi-client setup (Claude Desktop, Cursor, VS Code, Codex, Gemini, Claude Code)
- Addon bundling in wheel

*Note: versions 0.1.0, 0.1.2, 0.1.3, 0.1.4, 0.1.5 are tagged (v0.1.1 was never tagged). Versions 0.2.0–0.6.5 were commit-history milestones. v0.7.0 is the first tagged release since v0.1.5.*
