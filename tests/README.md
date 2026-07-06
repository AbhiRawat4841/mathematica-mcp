# Mathematica MCP Test Suite

This directory contains comprehensive tests for the Mathematica MCP server.

## Test Architecture

Tests are organized into three layers:

### 1. Unit & Integration Tests (hand-written)

Core tests that validate individual modules and live Mathematica interactions.

| File | Purpose | Requires Runtime |
|------|---------|-----------------|
| `test_session.py` | Calculus, algebra, linear algebra, special functions, JSON parsing | Yes |
| `test_error_detection.py` | Error analyzer, message analysis, LLM formatting (33 tests) | Partial (2 classes need runtime) |
| `test_notebook_backend.py` | Notebook abstraction layer, cell models, serialization | No |
| `test_notebook_tools_offline.py` | Offline notebook parsing and conversion | No |
| `test_notebook_optimizations.py` | Kernel-mode fast path (~152x perf improvement) | Yes (addon) |
| `test_tool_registration.py` | Profile-based tool sets (lean/classic/math/notebook/full; lean is the default) | No |
| `test_lean_tools.py` | Lean consolidated-tool dispatch and param guards (mocked) | No |
| `test_lean_tools_live.py` | Lean tools against a live addon | Yes (addon) |
| `test_lean_hardening.py` | Lean tool edge cases and error paths | No |
| `test_profiles_parity.py` | classic == full surface; lean == 12 tools; toolset opt-ins | No |
| `test_schema_budget.py` | Per-profile schema-size budget (CI gate: lean <= 16 KB / 18 tools) | No |
| `test_warm_funnel.py` | Warm persistent-kernel funnel vs cold wolframscript fallback | Partial (live parity tests need runtime) |
| `test_llm_driver_stub.py` | `benchmarks/llm_driver.py --stub` end-to-end + scorer | No |
| `test_guidance_lean.py` | Lean-profile guidance uses lean vocabulary | No |
| `test_connection.py` | Socket connection management, backoff/retry | No |
| `test_cache_epoch.py` | Cache invalidation via kernel epoch | No |
| `test_execution_style.py` | Execution style parameter resolution | No |
| `test_async_blocking.py` | Async/await with asyncio.to_thread | No |
| `test_derivation_verification.py` | Mathematical derivation verification | Yes |
| `test_guidance.py` | Profile-aware guidance, layering, word budgets, and cross-layer duplication guards | No |
| `test_lazy_tool_imports.py` | Lazy importing of optional tool modules | No |
| `test_raster_cache.py` | Raster graphics caching | No |
| `test_routing_memory.py` | Routing memory, expression classification, hints, transport lifecycle, breaker | No |
| `test_transport_classification.py` | Shared attempt/final transport classification | No |
| `test_response_filter.py` | Payload shaping (compact/standard/verbose/diagnostic) | No |
| `test_journal.py` | Computation journal ring buffer | No |
| `test_wl_scan.py` | Wolfram scanner (string/comment stripping, brace counting) | No |
| `test_config.py` | Feature flags (routing_action) | No |
| `test_session_backend.py` | Session backend implementation | No |
| `test_symbol_index.py` | Symbol lookup and indexing | No |
| `test_telemetry_wiring.py` | Telemetry integration | No |
| `test_cli.py` | CLI and addon installation | No |
| `test_readme_commands.py` | README example validation | No |
| `test_execute_code_metadata.py` | Response normalization, transport classification, error family extraction | No |
| `test_image_validation.py` | Server-side PNG validation (`_is_valid_png`, `_attach_image_if_valid`) | No |

### 2. Corpus Test Runner (data-driven)

A parametrized test harness that runs Wolfram Language expressions against the actual MCP server tools and verifies results using structured oracles.

| File | Purpose | Requires Runtime |
|------|---------|-----------------|
| `test_corpus_runner.py` | Two parametrized entry points: `test_corpus_case` + `test_corpus_workflow` | Yes (for live cases) |
| `corpus/mathematica_mcp_corpus.json` | Executable manifest - the source of truth for all corpus tests | N/A (data) |
| `corpus/models.py` | Pydantic schema: `CorpusCase`, `CorpusWorkflow`, `Oracle`, `PollCondition` | N/A |
| `corpus/normalize.py` | Normalizes all MCP tool responses (JSON, dict, Image, parse_error) into one shape | N/A |
| `corpus/verifiers.py` | 11 verification strategies (exact, symbolic, numeric, structural, artifact, etc.) | Partial (symbolic needs kernel) |
| `corpus/adapters.py` | 4 execution adapters: server_tool, offline_notebook_file, live_frontend, alias_codegen | N/A |
| `corpus/capabilities.py` | Session-scoped capability probes (runtime, addon, frontend, network) | N/A |

**Key design decisions:**
- MCP server tools are the primary execution path - `execute_in_kernel()` is only the oracle engine for symbolic/numeric equivalence
- Stateful tests (variable lifecycle, notebook lifecycle) are self-contained workflows, not cross-test dependencies
- Corpus manifest is JSON - the markdown corpus files are documentation only, never read at runtime

### 3. Corpus Meta-Tests (test the test infrastructure)

These verify the corpus harness itself is correct - no Mathematica needed.

| File | Purpose | Tests |
|------|---------|-------|
| `test_corpus_normalize.py` | Normalizer against all response shapes (JSON, dict, Image, warnings, artifacts) | ~24 |
| `test_corpus_verifiers.py` | All 11 verifiers with synthetic NormalizedResult objects | ~39 |
| `test_corpus_infra.py` | Models validation, adapter dispatch, polling, cleanup templating | ~19 |

## Running Tests

```bash
# All tests (wolframscript-dependent tests auto-skip if runtime is absent)
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/mathematica_mcp --cov-report=term-missing

# Corpus meta-tests only (no Mathematica needed)
uv run pytest tests/test_corpus_normalize.py tests/test_corpus_verifiers.py tests/test_corpus_infra.py -v

# Corpus smoke tier (needs wolframscript)
uv run pytest tests/test_corpus_runner.py -m "tier_smoke" -v

# Corpus by section
uv run pytest tests/test_corpus_runner.py -k "ARITH or ALG" -v

# Corpus workflows only
uv run pytest tests/test_corpus_runner.py::test_corpus_workflow -v
```

## Corpus Tiers

| Tier | Purpose | CI |
|------|---------|-----|
| `smoke` | 30 cases + 1 workflow (31 total) | Required (meta-tests in CI, live tests in manual/nightly) |
| `core` | planned (~120 stable offline math/algebra/calculus) - **not yet populated**; `-m tier_core` collects 0 tests | Nightly schedule (no-op today) |
| `extended` | planned (~80 broader coverage) - **not yet populated** | Manual dispatch (no-op today) |
| `probe` | planned (~30 fragile/regression paths) - **not yet populated** | Manual dispatch, non-blocking |

## Corpus Verification Strategies

| Strategy | Description |
|----------|-------------|
| `exact_text` | Output text matches expected value exactly (whitespace-normalized) |
| `symbolic_equiv` | Mathematically equivalent via `Simplify[a - b] === 0` (kernel as oracle) |
| `numeric_tol` | Within tolerance: `abs(actual - expected) < tolerance` |
| `boolean` | Output is exactly `True` or `False` |
| `field_equals` | Dot-path field in parsed response equals expected value |
| `field_contains` | Dot-path field contains expected substring |
| `structural_fields` | Multiple field checks (e.g., `parsed.success == True`) |
| `artifact_exists` | File/image artifact is present and non-empty |
| `warning_tag` | Expected warning tag found in normalized warnings |
| `raw_contains` | Expected substrings found in raw response |
| `workflow_assert` | All workflow steps passed, no cleanup errors |

## CI/CD Integration

Two CI workflows:

- **`ci.yml`** (required, runs on every PR):
  - Lint (`ruff check` + `ruff format`)
  - Corpus smoke meta-tests (no Mathematica needed)
  - Full unit test suite
  - Package verification

- **`corpus.yml`** (non-blocking, manual dispatch or nightly):
  - Smoke/core/extended/probe tiers
  - Requires wolframscript in the runner environment
  - All steps use `continue-on-error: true`

## Requirements

- Python 3.10+
- pytest 7.0+
- pydantic 2.0+ (for corpus models)
- Wolfram Language / Mathematica (for runtime-dependent tests)
- wolframscript in PATH (for session/corpus live tests)

## Contributing

When adding new tests:

1. **Unit/integration tests**: Follow existing patterns in the relevant `test_*.py` file
2. **Corpus test cases**: Add entries to `corpus/mathematica_mcp_corpus.json` with the correct tier, backend, oracle, and required capabilities. For `execute_code` cases, always include `"output_target": "cli"` in params - the default output target is profile-dependent (`cli` for lean/math, `notebook` for classic/notebook/full), and notebook mode returns no `output_inputform`. Use camelCase for Wolfram variable names (underscores are reserved for patterns).
3. **Corpus workflows**: Use self-contained workflow items with per-step assertions and cleanup
4. **Tool registration**: Update expected tool sets in `test_tool_registration.py`
5. **Always run meta-tests** after modifying corpus infrastructure: `uv run pytest tests/test_corpus_normalize.py tests/test_corpus_verifiers.py tests/test_corpus_infra.py -v`
