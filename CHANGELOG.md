# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-07-06

### ⚠ BREAKING

- **Addon protocol is now `4`** (response-contract change for front-end evaluations, below). Reinstall the addon (`wolframscript -file addon/install.wl`, or `setup`) and restart Mathematica or re-`Get` the package; a stale addon is detected and surfaced by `status()`.

### Added

- **Background kernel prewarm**: the persistent kernel session now boots in a daemon thread at server startup, so the ~13s boot overlaps setup instead of blocking the first warm call. Calls arriving inside the boot window take the addon rung when eligible (measured: `verify_derivation` 56ms via addon mid-boot, 1ms warm after); without a connected addon they wait for the boot instead of triggering it serially. `MATHEMATICA_PREWARM` (default on; `0`/`false`/`no`/`off` disables). An unused prewarmed kernel is still reclaimed by the idle reaper.
- **Opt-in screenshot cache**: lean `screenshot` gains `cache=True` for notebook/cell scope, invalidated by a mutation epoch bumped on every notebook-mutating MCP command (repeat captures of an unchanged notebook drop from ~400ms to ~0ms). Requires a stable target (`notebook=` or `session_id=`); the focused-notebook default is never cached. May serve stale pixels if the notebook changes outside MCP (manual edits, `Dynamic` repaints); see the Technical Reference caveats.
- **Addon fallback rung in the warm funnel**: while the warm kernel is booting (or cold-forced), kernel-independent pure-math calls (`verify_derivation`, `get_constant`, `convert_units`) run on the already-connected addon kernel (~30ms, `execution_method="addon"`) instead of a ~12.5s cold subprocess. Kernel-time on the user's front-end kernel is capped at 30s per call; kernel-identity-sensitive calls (variables, packages, kernel state) never take this rung; any parse ambiguity falls through to the always-correct cold path.
- **Transport-floor attribution**: `benchmarks/probe_transport_floor.py` decomposes the ~30ms addon round-trip floor (client/JSON/dispatch measure under 2ms combined; the ~94% remainder is attributed by elimination to front-end-kernel async-task scheduling) and validates that `batch` amortizes it to ~4.5ms per sub-command. `benchmarks/measure_channel_split.py` documents the measured no-go on a read-only second socket. See [docs/benchmarks.md](docs/benchmarks.md).

### Changed

- **Lean `evaluate` responses default to `compact`**: empty fields stripped, verbose duplicates dropped (trivial responses shrink ~67%, graphics responses ~83%). Failure responses (including notebook `timeout` / `executed_with_errors` shapes) keep the complete diagnostic shape, and large outputs flow to cursor pagination instead of lossy summarization. Override with `MATHEMATICA_RESPONSE_DETAIL=standard`. Classic-profile tools are unchanged.
- **Transient kernel-boot failures no longer latch the process cold**: a failed boot (e.g. license contention at startup) now sets a 60s retry cooldown instead of forcing cold `wolframscript` subprocesses until restart; permanent cold mode is reserved for missing `wolframclient`/kernel. A failed half-started session is terminated instead of leaking until exit.

### Fixed

- **Front-end evaluations no longer report false completion**: `execute_code(mode="frontend")`, `evaluate_cell`, and `evaluate_selection` used to return `success`/`evaluated: true` with empty output for anything running longer than ~0.5s while the notebook was still computing. All three now share one honest contract: `status: "evaluation_pending"`, `evaluation_complete: false`, `waited_seconds`, and guidance to re-check via `get_cells` (the output cell lands in the notebook when the evaluation finishes). The in-handler wait is capped at 0.2s because the front end provably cannot complete while the handler runs; previously the poll could stall every addon command for up to 10s per call (a concurrent `ping` measured 5.1s blocked before, 23ms after). Closing a notebook with a pending evaluation orphans its output into the system Messages window; the docs now warn about this.
- **`state_delta.kernel_busy` reads the command's target notebook**, not whichever notebook happens to be focused. It still cannot observe an in-flight front-end evaluation (front-end state does not resolve from the socket handler); use the `evaluation_pending`/`evaluation_complete` contract for progress instead.
- **Kernel-session creation race**: `get_kernel_session()` had no creation mutex; a background prewarm plus a first tool call could boot two kernels and leak a license seat. Creation is now serialized with double-checked locking.

### Fixed

- **Addon reinstall was a no-op for existing installs**: `install.wl` skipped any `init.m` that already contained a MathematicaMCP section, so upgrades never re-pointed the loader at the new package copy and `status()` kept reporting protocol skew. The installer now replaces the section idempotently (re-runs are byte-identical).
- Documentation accuracy: `read_notebook_file` claims corrected (kernel-first when available, Python fallback offline, `.wl` requires the kernel), `session_id` vs `isolate_context` clarified, broken anchors fixed, and the addon protocol docs now cover `protocol_version` and `state_delta`.

### Changed

- Docs: README slimmed to a product introduction (tool tables, profile/toolset details, and architecture notes moved to the [Technical Reference](docs/technical-reference.md)); em-dashes replaced with plain punctuation across all documentation; profile table alignment fixed.

## [1.0.0] - 2026-07-06

### ⚠ BREAKING

- **Default profile is now `lean` (12 tools), not `full` (82 tools).** If you do nothing, agents see the 12 consolidated lean tools. Set `MATHEMATICA_PROFILE=classic` (alias `full`) to keep the complete pre-1.0 surface. `classic` is byte-identical to the old `full` - same tools, same implementations, no shims. See [docs/MIGRATION.md](docs/MIGRATION.md).
- **Reinstall the Mathematica addon.** 1.0 adds a Python↔addon `protocol_version` handshake, now at `3`. The addon lives in `$UserBaseDirectory/Kernel/init.m` and does **not** update with `pip`/`uvx`; run `setup` (or `wolframscript -file addon/install.wl`) again. A stale addon is detected and surfaced by `status()`.
- **`state_delta` is no longer attached to every addon response** (a response-contract change; the handshake ships at protocol `3`). It now appears only on notebook-touching commands (`create_notebook`, `write_cell`, `evaluate_cell`, `get_cells`, ... plus `batch_commands`); pure-kernel responses omit it. `kernel_busy` inside it reports the focused notebook's actual `Evaluating` state.
- **`vars(action='clear')` requires `name=` or `pattern=`.** A bare `clear` no longer wipes the session - it errors and points at `vars(action='clear_all')`, which is now the explicit way to clear everything. Anyone scripting a bare `clear` must switch to `clear_all`.

### Added

- **Lean default profile**: 12 consolidated tools (`status`, `notebooks`, `cells`, `edit_cells`, `evaluate`, `screenshot`, `verify_derivation`, `kernel`, `vars`, `read_notebook_file`, `guide`, `batch`) tagged `@_tool("lean")`, thin wrappers over the same internals `classic` uses. Tool-schema context drops from 82 tools / ~61 KB (~15k tokens) to 12 tools / ~11.5 KB (~2.9k tokens).
- **`MATHEMATICA_TOOLSETS`** opt-in extras for lean (comma-separated): `data_io`, `graphics_plus`, `cloud`, `debug`, `notebook_files`, `notebook_edit`, `symbols`, `math_aliases`, `repository`, `async_jobs`, `cache`. Can only add tools to lean, never remove the 12 core tools.
- **Warm funnel**: the 12 previously-cold `wolframscript` tools (including `verify_derivation`) and the symbol-index build now run on the persistent `WolframLanguageSession` (sub-second warm) with a flagged, counted cold-subprocess fallback. Every response carries `execution_method`, and `status()` surfaces a **cold-execution counter** (0 on the lean happy path), kernel-session liveness, and the idle-shutdown timeout. Remaining cold sites migrated too: function/data repository lookups, symbol usage hydration, graphics re-rasterization on query-cache hits, and notebook→TeX conversion.
- **JSON-first kernel result parsing**: warm-funnel results are JSON-exported kernel-side, with a sanitizer that stringifies non-JSON-safe leaves and association/rule *keys* (e.g. `EntityProperty[...] -> value`); the regex Association parser remains only as a fallback. An undefined-symbol filter ensures the literal `Sym::usage` text of an unevaluated `MessageName` is never returned or cached as real usage.
- **Idle kernel shutdown**: the persistent kernel shuts down after `MATHEMATICA_KERNEL_IDLE_TIMEOUT` seconds of inactivity (default `1800`; `0` disables) and restarts on the next call; never reaps mid-evaluation or mid-startup. The kernel is also closed at process exit (threading/atexit hook).
- **Guidance v2**: failed evaluations carry `error_analysis` with `suggested_fix` and `next_step` on **all** evaluate paths; `retry_with` is a concrete corrected call when derivable from context, otherwise `null`. Oversized output is capped (`MATHEMATICA_MAX_OUTPUT_CHARS`, default `4000`) with a continuation `cursor` you pass back to the same tool; notebook-touching addon responses carry a `state_delta` (`notebook` / `cell_count` / `kernel_busy`); `guide(topic)` gives on-demand help, including a `batch` topic documenting the `batch` command vocabulary.
- **Profile-aware guidance**: server instructions and the 6 MCP prompts condition on the active profile; the lean profile gets its own instructions and prompt text (the previous text referenced classic-only tools and misdirected lean clients).
- **Lean input validation**: `evaluate` rejects ambiguous calls (`code=` + `file=`, or `dry_run=True` + `file=`); `notebooks` validates `format` per action (`save`: `Notebook|PDF|HTML|TeX`; `export`: `PDF|HTML|TeX|Markdown`); structured errors carry a `next_step` with the corrected call.
- **Mathematica 15 first-class**: agent-created notebooks set `ShowChatbar->False` on ≥15 (override with `show_chatbar=True`). 14.x supported behind `$VersionNumber >= 15.` guards, with `MMCP_FORCE_V14=1` to force the `<15` branch for testing. See [V14_VALIDATION.md](V14_VALIDATION.md).
- **Addon protocol handshake**: `protocol_version` (now `3`) added to `ping`/`get_status`; the Python client detects a stale addon and instructs a reinstall.
- **LLM trace driver**: `benchmarks/llm_driver.py` replays tool-call scenarios against the live server (with an offline `--stub` mode), `benchmarks/scenarios.json` gains lean-profile variants (`lean_preferred_tools`, `lean_forbidden_sequences`), and `benchmarks/score_trace_corpus.py` accepts `--profile lean`.
- **Docs**: repositioned README as a front-end / notebook automation layer that runs beside the official Wolfram Local MCP (with a comparison table and a `setup --with-official` flag), plus [docs/MIGRATION.md](docs/MIGRATION.md) and [V14_VALIDATION.md](V14_VALIDATION.md).

### Fixed

- **`get_cells` errored with `Command returned unexpected result head: List`** for any client that omitted the pagination params (`offset`/`limit`/`include_content`); it now always returns an Association, and non-integer `offset`/`limit` values are coerced instead of producing degenerate JSON.
- **WL string injection**: user text containing quotes or backslashes (e.g. `Quantity[100, "Centimeters"]` as a `verify_derivation` step, quoted strings in `get_constant`/`inspect_graphics`/`wolfram_alpha` inputs) broke the generated Wolfram Language and returned parse errors; all interpolation is now escaped.
- **`verify_derivation` returned "Could not parse verification results"** whenever a step produced an expression-valued result (e.g. `-4 + Pi^2`, whose 2-D OutputForm rendering defeated the old text parser); it now returns the structured step-by-step report.
- **`get_kernel_state` / `list_loaded_packages` / cloud tools returned `{"raw": ..., "parse_error": true}`** instead of structured fields; results are now JSON-exported kernel-side and parsed reliably.
- **Undefined symbols' usage was faked and cached**: looking up a nonexistent symbol echoed the literal `Sym::usage` text as real usage and stored it in the symbol index.
- **Graphics re-rasterization on query-cache hits spawned a ~12.5s cold `wolframscript` subprocess**, making a cache hit slower than a miss; it now renders the cached output expression on the warm session without re-running user code.
- **`execute_code`'s warm path ignored `timeout`**, so a runaway evaluation could block the kernel session indefinitely; evaluations are now bounded by `TimeConstrained`.
- **Idle kernel shutdown could silently never fire on freshly-booted hosts**: the reaper compared against a re-read monotonic clock instead of the sampled one, so on machines with short uptime (CI runners, fresh boots) the idle timeout was never considered reached.
- **User-expression tools leaked symbol definitions into the shared kernel's `` Global` ``** (e.g. `inspect_graphics`, `get_constant`); user text now evaluates in a throwaway scratch context.

### Changed

- `verify_derivation` now runs warm on the persistent session (moat tool, shared by `lean` and `classic`). No API change.
- `error_analysis` is now attached and preserved in compact responses across every evaluate path, not just the notebook path.
- Kernel lifecycle hardening: the health-check ping runs under the session lock, and its interval was raised from 5s to 30s.
- Performance: `mcpStateDelta[]` was ~80% of in-kernel processing time on trivial addon calls - now gated to notebook-touching commands (see BREAKING); cold `wolframscript` spawns (~12.5s each) eliminated from ~14 call paths via the warm funnel.
- Repo hygiene: draft docs moved out of the repo root to `docs/drafts/` (untracked); removed two tracked test fossils (`tests/TEST_SUMMARY.md`, `tests/demo_error_detection.py`).
- `__init__.__version__` synced with `pyproject.toml` (0.9.5 → 1.0.0); enforced by a version-sync test.

## [0.9.5] - 2026-07-03

### Fixed

- **Windows path escaping**: backslashes in Windows paths broke the generated Wolfram Language in `addon/install.wl` (the `init.m` load entry) and `session.py` temp-file paths; paths are now normalized before interpolation. (Community contribution by @lecojari.)
- **Addon robustness for open-notebook flows**: `get_status` no longer fails when `SystemInformation["FrontEnd"]` is unavailable (reports `"Unavailable"` instead); notebook listing and creation are guarded against an invalid front-end state; the `MATHEMATICA_MCP_TOKEN` auth token is read defensively from the environment.

## [0.9.4] - 2026-04-09

### Fixed

- **Frontend evaluation polling (85x speedup)**: `executeCodeNotebookFrontend` was using a broken `Length[Cells[nb]]` cell-count check to detect evaluation completion. From the preemptive link, this check never resolved, causing the polling loop to burn the entire `max_wait` timeout (10-30 seconds) even for trivial expressions like `1+1`. Replaced with `CurrentValue[nb, Evaluating]` - the same proven pattern used by `cmdEvaluateCell` and `cmdExecuteSelection`. Frontend mode latency dropped from **10,893ms to 129ms** (median).
- **Evaluation polling intervals halved**: `cmdEvaluateCell` and `cmdExecuteSelection` polling intervals reduced from 100ms to 50ms, cutting average detection latency for fast-completing computations (~134ms to ~98ms).

### Changed

- Updated technical reference, benchmarks documentation, and addon README to reflect the frontend polling fix and evaluation architecture

## [0.9.3] - 2026-04-09

### Fixed

- **`get_messages` recency**: `get_messages` now returns the most recent N messages instead of the oldest N

### Added

- **Recency ordering test**: New test to verify `get_messages` returns the newest entries

## [0.9.2] - 2026-04-09

### Fixed

- **Strict notebook transport**: Notebook requests no longer silently reroute through CLI fallback when notebook transport fails -- the server returns a structured notebook error instead
- **Message-bearing results preserved**: Valid Mathematica results with messages (e.g. `ComplexInfinity`) are no longer collapsed into evaluation failures
- **`get_messages` counter/reporting**: Fixed counter and reporting issues in the `get_messages` tool

### Added

- **`response_detail` aliases**: `short`, `medium`, and `long` accepted as aliases for compact, standard, and verbose detail levels
- **Regression tests**: New coverage for schema exposure, transport behavior, warning-bearing evaluations, and image validation

### Changed

- Improved dispatch and message logging in the Mathematica addon
- Updated README, technical reference, and generated guidance to match current behavior

## [0.9.1] - 2026-04-09

### Changed

- **Guidance system refactored for lean LLM context**: Shared primitive functions (`_style_keyword_table`, `_routing_lines`, `_quick_defaults`, `_recovery_defaults`, `_avoid_lines`) replace per-builder inline copies. Server instructions now carry universal defaults, recovery tools, and anti-patterns; client-specific files (AGENTS.md, CLAUDE.md hint) are additive-only. Always-on context dropped from ~880 to ~540 words (Codex) and ~760 to ~460 words (Claude Code).
- **Profile-aware guidance conditioning**: Math profile no longer mentions live notebooks, `create_notebook`, `style="notebook"`, `style="interactive"`, or `sync`. Intro text adapts per profile. `build_server_instructions()` replaces the hardcoded `instructions` string in `server.py`.
- **Expanded intent keywords**: Added "evaluate", "solve", "graph", "visualize", "slider", "dynamic", "animate", "fresh notebook", "create notebook" across all guidance layers, README, and docs.
- **Restored lost guidance**: `sync="none"` anti-pattern, `save_notebook()`/`screenshot_notebook()`/`screenshot_cell()` routing, and `execute_code` notebook-reuse explanation re-added after earlier refactor dropped them.
- **Word-budget tests**: Guidance tests now enforce per-layer and combined always-on word budgets to prevent future context bloat.

## [0.9.0] - 2026-04-09

### Added

- **Payload shaping** (`response_detail` parameter on `execute_code`): Four detail levels - `compact` (essential fields only, token-efficient), `standard` (exact backward-compatible default), `verbose` (+ detail_level marker), `diagnostic` (+ cache epoch, routing hints). Compact mode preserves notebook IDs, transport status, error families, and handles graphics placeholders by swapping to `output_inputform`. Large outputs auto-summarized with balanced-brace list element counting. Pure response filter module with no runtime singleton dependencies.

- **Session brief** (`get_session_brief()` tool): Compact ~100-token session state snapshot showing connection mode (addon/kernel_only/disconnected), recent error families (sorted by recency with 24h age cutoff), and routing advice. Uses 500ms addon timeout and never starts a fresh kernel session.

- **Computation journal** (`get_computation_journal()`, `clear_computation_journal()` tools): In-memory ring buffer (10 entries) recording code/output previews, timing, transport status, error families, timed_out, and from_cache for each `execute_code` call. Records raw canonical results before response filtering. Survives context compaction.

- **Smart cache optimization**: Pure-System expressions (no user-defined symbols, no session-sensitive introspection like `Names`, `Contexts`, `$Packages`, `Definition`) are now epoch-insensitive - they survive `set_variable`/`clear_variables` mutations without re-evaluation. Memoized single-pass code analysis with `lru_cache(1024)`, static ~200 builtin allowlist, session-sensitive symbol denylist. Malformed input (unbalanced strings/comments) falls back to epoch-sensitive (safe degradation).

- **Expression classification**: Regex-based classifier categorizes Wolfram code into routing-relevant classes (`plot`, `frontend_dynamic`, `symbolic_heavy`, `numeric_heavy`, `io`, `general`). Strips strings and nested comments before matching. Used for expr-type-specific routing cohorts and transport-path hints.

- **Per-path transport outcomes**: Attempt-level telemetry records each transport leg independently (addon_notebook, addon_cli) with typed `AttemptOutcome` (OK, INFRA_ERROR, TIMEOUT, SEMANTIC_ERROR). Persisted in `path_transport_outcomes` on aggregate cohorts. Fixes the data flow bug where addon failures were invisible after kernel fallback.

- **Routing hints**: Advisory hints built from two sources - transport-path hints (per-path infra/timeout rates from attempt telemetry) and end-to-end hints (timeout/fallback rates from final cohort data). Structured `RoutingHint` records with severity/specificity ordering, deduplication, and 5-hint cap. Available via `get_routing_memory_stats(include_hints=True)` and in `diagnostic` response detail.

- **Conservative routing action** (opt-in): Concurrency-safe half-open circuit breaker that skips persistently failing addon_cli transport legs for compute requests. Requires both `MATHEMATICA_ROUTING_MEMORY=advise` and `MATHEMATICA_ROUTING_ACTION=compute_cli_skip`. Breaker trips on 5 consecutive `INFRA_ERROR` outcomes only (ignores timeouts and semantic errors). 60s cooldown with single-probe recovery. Idempotent finish/abort lifecycle. Skip counter observability in routing stats. Uses truthful `kernel_direct_routing_skip` execution path (not `kernel_fallback`).

- **Transport lifecycle API**: `begin_transport_attempt()` / `finish_transport_attempt()` / `abort_transport_attempt()` API. The primary CLI transport path uses a unified `finish` call that handles both persisted attempt telemetry and breaker state. Idempotent with `_completed` guard. Probe-in-flight concurrency control under lock.

- **Shared Wolfram scanner** (`wl_scan.py`): Single-pass state machine for stripping string literals and nested Mathematica comments. Preserves token boundaries with space replacement. Returns `ScanResult(cleaned, ok)` for safe degradation on malformed input. Also provides balanced-brace top-level element counting (tolerates leading whitespace, rejects trailing content).

- **Shared transport classification** (`transport_classification.py`): Single source of truth for attempt-level and final-level classification. Refactored `_classify_transport()` and `_extract_error_families()` from server.py. Handles non-dict results safely.

- **Shared constants** (`constants.py`): `ExecutionPath` labels and `AttemptOutcome` enum used across server, routing memory, and transport classification.

- **Routing memory schema v2**: Adds `path_transport_outcomes` to cohort stats. Backward-compatible: v1 files load cleanly (missing field defaults to empty dict). Zero-valued counters pruned during decay/serialization to prevent JSON growth.

- **Recent error families API** (`get_recent_error_families()`): Public accessor on routing memory, sorted by `last_seen` (not frequency). Filters `"other"` unless no better candidates. Configurable `max_age_seconds` cutoff.

### Changed

- `execute_code` now accepts `response_detail` parameter (default `"standard"` - exact backward-compatible behavior)
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
- **Corpus meta-tests**: 83 tests validating the harness itself (normalizer, verifiers, models, adapters, polling, cleanup) - no Mathematica needed
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

- **Routing memory** (`MATHEMATICA_ROUTING_MEMORY`): opt-in observability layer that collects aggregate routing statistics from `execute_code` - transport success rates, latency histograms, and error family frequencies. Modes: `off` (default), `observe`. No raw code or expressions stored. (`advise` mode reserved for future routing hints.) See [Technical Reference](docs/technical-reference.md#routing-memory).
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

- **Execution `style` parameter**: New keyword-only `style` parameter on `execute_code` bundles `output_target` + `mode` into three presets: `"compute"` (cli/kernel), `"notebook"` (notebook/kernel), `"interactive"` (notebook/frontend) - reduces LLM argument assembly errors
- **Pure execution resolver**: `_resolve_execution_params()` handles style→param resolution with explicit>style>profile priority, unknown-style errors, and CLI mode normalization
- **Unconditional execution metadata**: Every `execute_code` response now includes `executed_output_target` (and `executed_mode` when applicable) across all 6 response paths - including timeout and fallback - so callers always know what actually happened
- **Profile-aware guidance**: `build_claude_hint()`, `build_claude_command()`, and `build_codex_guidance()` now conditionally omit notebook/interactive flows for the `math` profile, matching the existing behavior of `build_mathematica_expert_prompt()`

### Fixed

- **`create_notebook` profile exposure**: Moved `create_notebook` from `notebook_advanced` to `notebook_primary` group - previously all guidance told LLMs to call it for "new notebook" flows, but it was not registered in the `notebook` profile
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

- **Execution mode keywords**: Users can steer routing with natural language keywords ("calculate", "plot", "new notebook", "interactive") - documented in README, CLAUDE.md, AGENTS.md, and server instructions
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
