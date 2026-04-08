# Contributing to mathematica-mcp

## Development Setup

```bash
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp
uv sync --extra dev
```

## Running Tests

```bash
# All tests (wolframscript-dependent tests auto-skip if runtime is absent)
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/mathematica_mcp --cov-report=term-missing

# Corpus meta-tests only (no Mathematica needed — validates test infrastructure)
uv run pytest tests/test_corpus_normalize.py tests/test_corpus_verifiers.py tests/test_corpus_infra.py -v

# Corpus smoke tier (needs wolframscript)
uv run pytest tests/test_corpus_runner.py -m "tier_smoke" -v
```

## Linting

```bash
uv run ruff check src/ tests/ benchmarks/
uv run ruff format --check src/ tests/ benchmarks/

# Auto-fix
uv run ruff check --fix src/ tests/ benchmarks/
uv run ruff format src/ tests/ benchmarks/
```

## Architecture

See [docs/technical-reference.md](docs/technical-reference.md) for the full architecture, tool tiers, and component overview.

## Adding a New Tool

1. **Define the function** in `src/mathematica_mcp/server.py` (for core tools) or the relevant `optional_*.py` module (e.g., `optional_repository_tools.py`, `optional_symbol_tools.py`)
2. **Register with a tool group** using the `@_tool("group_name")` decorator
3. **Gate via profile** — ensure the tool group is included in the appropriate profiles in `src/mathematica_mcp/config.py` (`PROFILE_TOOL_GROUPS`)
4. **Add tests** — prefer offline tests where possible (mock kernel interactions)
5. **Update registration tests** — add the tool name to the expected sets in `tests/test_tool_registration.py`
6. **Add corpus coverage** — add test cases to `tests/corpus/mathematica_mcp_corpus.json` with the appropriate tier, backend, oracle, and required capabilities (see `tests/README.md` for details)

### Key Modules

| Module | Purpose |
|--------|---------|
| `server.py` | Core MCP server, tool registration, execute_code routing |
| `config.py` | Feature flags, profiles, environment variable resolution |
| `routing_memory.py` | Routing statistics, expression classification, transport lifecycle, breaker |
| `transport_classification.py` | Shared attempt/final transport classification (single source of truth) |
| `response_filter.py` | Pure response payload shaping (`response_detail` parameter) |
| `journal.py` | In-memory computation journal (ring buffer) |
| `cache.py` | Query cache with epoch-insensitive optimization, code analysis |
| `wl_scan.py` | Wolfram Language scanner (string/comment stripping, brace counting) |
| `constants.py` | Shared ExecutionPath labels and AttemptOutcome enum |
| `session.py` | Kernel lifecycle, execution, raster cache |
| `guidance.py` | Dynamic tool docstrings, prompts, session brief |
| `connection.py` | Socket-based communication with Mathematica addon |

## Pull Request Process

1. Fork the repository and create a feature branch
2. Ensure `ruff check` and `ruff format --check` pass
3. Ensure all tests pass (`uv run pytest tests/ -v`)
4. Submit a PR against the `main` branch

## Commit Conventions

- `feat:` — new features
- `fix:` — bug fixes
- `perf:` — performance improvements
- `chore:` — maintenance (version bumps, CI, dependencies)
- `docs:` — documentation changes
- `style:` — formatting, no logic change
- `refactor:` — code restructuring without behavior change

## Branch Protection

CI workflows enforce lint, test, and packaging checks. For true merge gating, required status checks must also be configured in GitHub repo Settings > Branches > Branch protection rules. After adding new CI jobs (e.g., lint), the branch protection rules should be updated to require those jobs as well.
