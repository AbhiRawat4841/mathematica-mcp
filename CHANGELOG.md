# Changelog

All notable changes to this project will be documented in this file.

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
