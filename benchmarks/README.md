# Benchmarking Mathematica MCP Guidance and Profiles

This directory contains benchmarks for performance measurement and routing/profile work.

## Goals

- Measure tool-surface reduction by profile
- Measure schema-size reduction by profile
- Measure symbol index cold/warm startup and search performance
- Measure live addon operation latency with dedicated notebook sessions
- Score LLM task traces before and after guidance/skill changes
- Detect notebook anti-patterns such as `create_notebook -> write_cell -> evaluate_cell`

## Files

- `benchmark_perf_phases.py`: offline performance benchmarks (symbol index build/warm/cold, search, caching, telemetry). Results validate that builds succeed (symbol_count > 0) before recording timings.
- `benchmark_notebook_ops.py`: live addon benchmarks with dedicated session (`session_id=benchmark-session`). Creates a notebook, threads session_id through all operations, cleans up at the end.
- `benchmark_bottleneck_diagnosis.py`: live latency diagnosis (socket round-trip, preemptive vs main link, frontend polling).
- `probe_transport_floor.py`: decomposes the ~30ms addon round-trip floor into client/wire, Wolfram socket stack, and front-end-kernel scheduling shares (pure-Python echo + headless WL echo + live addon; pipelining and payload sweeps). Writes `results/transport_floor.json` and `.md`.
- `measure_channel_split.py`: measures where a concurrent status call blocks during long evaluations (Python lock vs kernel queue); the data behind the read-only-second-socket no-go. Uses an explicit-id scratch notebook and drains queued frontend evals before closing it.
- `scenarios.json`: canonical task corpus (13 scenarios, including error-recovery ones). Each scenario carries both classic (`preferred_tools` / `forbidden_sequences`) and lean (`lean_preferred_tools` / `lean_forbidden_sequences`) expectations.
- `profile_surface.py`: reports tool count and schema byte size for each profile
- `llm_driver.py`: drives the scenario corpus through a tool-calling LLM (or an offline stub) and emits trace JSONL for the scorer
- `score_trace_corpus.py`: compares two trace corpora, such as baseline vs updated (`--profile lean|classic` selects which expectations to score against)

## Generating traces with llm_driver.py

`llm_driver.py` replays `scenarios.json` and writes trace JSONL in the exact shape `score_trace_corpus.py` consumes.

```bash
# Offline stub mode: scripted policy, no server, no kernel, no API key
uv run python benchmarks/llm_driver.py --stub --profile lean -o stub_lean.jsonl

# Live mode (Anthropic): needs ANTHROPIC_API_KEY; spawns the MCP server over stdio
uv run python benchmarks/llm_driver.py --provider anthropic --model claude-opus-4-8 \
    --profile lean -o live_lean.jsonl

# Live mode (OpenAI-compatible): needs OPENAI_API_KEY, honors OPENAI_BASE_URL
uv run python benchmarks/llm_driver.py --provider openai --model gpt-5 \
    --profile lean -o live_lean_openai.jsonl

# Score the traces (use --profile lean for lean-vocabulary expectations)
uv run python benchmarks/score_trace_corpus.py stub_lean.jsonl live_lean.jsonl --profile lean
```

The stub mode is exercised end-to-end in CI by `tests/test_llm_driver_stub.py`.

## Suggested workflow

1. On the baseline commit, capture profile surface data:
   `PYTHONPATH=src python benchmarks/profile_surface.py > baseline_surface.json`
2. Run your client task corpus and save traces as JSONL:
   `baseline_traces.jsonl`
3. Apply the routing/profile/skill changes.
4. Capture the new surface data:
   `PYTHONPATH=src python benchmarks/profile_surface.py > updated_surface.json`
5. Run the same client task corpus again and save:
   `updated_traces.jsonl`
6. Compare traces:
   `python benchmarks/score_trace_corpus.py baseline_traces.jsonl updated_traces.jsonl`
   (add `--profile lean` when the traces were produced against the lean tool surface)

## Trace format

Each JSON line should look like:

```json
{
  "scenario": "plot_surface",
  "tools": ["execute_code", "screenshot_notebook"],
  "success": true,
  "first_useful_ms": 2400,
  "total_ms": 3200
}
```

The benchmark scripts do not require any specific client. They only need a
repeatable scenario corpus and recorded tool sequences.
