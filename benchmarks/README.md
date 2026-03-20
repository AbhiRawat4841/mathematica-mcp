# Benchmarking Mathematica MCP Guidance and Profiles

This directory contains lightweight benchmarks for the routing/profile work.

## Goals

- Measure tool-surface reduction by profile
- Measure schema-size reduction by profile
- Score LLM task traces before and after guidance/skill changes
- Detect notebook anti-patterns such as `create_notebook -> write_cell -> evaluate_cell`

## Files

- `scenarios.json`: canonical task corpus and forbidden tool patterns
- `profile_surface.py`: reports tool count and schema byte size for each profile
- `score_trace_corpus.py`: compares two trace corpora, such as baseline vs updated

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
