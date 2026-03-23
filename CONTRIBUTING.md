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
