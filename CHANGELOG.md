# Changelog

All notable changes to this project will be documented in this file.

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
- CI, codecov, and PyPI badges in README

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
