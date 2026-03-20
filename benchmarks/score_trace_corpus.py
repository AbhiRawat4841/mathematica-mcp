from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = REPO_ROOT / "benchmarks" / "scenarios.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _contains_sequence(tools: list[str], sequence: list[str]) -> bool:
    if not sequence:
        return False
    for index in range(len(tools) - len(sequence) + 1):
        if tools[index : index + len(sequence)] == sequence:
            return True
    return False


def _score(rows: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tool_counts = []
    first_useful = []
    total_times = []
    success_total = 0
    anti_patterns = 0
    scenario_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "success": 0, "anti_patterns": 0}
    )

    for row in rows:
        scenario = scenarios.get(row["scenario"], {})
        tools = row.get("tools", [])
        tool_counts.append(len(tools))
        if row.get("first_useful_ms") is not None:
            first_useful.append(row["first_useful_ms"])
        if row.get("total_ms") is not None:
            total_times.append(row["total_ms"])
        if row.get("success"):
            success_total += 1

        forbidden_sequences = scenario.get("forbidden_sequences", [])
        hit_anti_pattern = any(
            _contains_sequence(tools, sequence) for sequence in forbidden_sequences
        )
        if hit_anti_pattern:
            anti_patterns += 1

        scenario_entry = scenario_stats[row["scenario"]]
        scenario_entry["runs"] += 1
        scenario_entry["success"] += int(bool(row.get("success")))
        scenario_entry["anti_patterns"] += int(hit_anti_pattern)

    def _median(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    return {
        "runs": len(rows),
        "success_rate": round(success_total / len(rows), 4) if rows else None,
        "anti_pattern_rate": round(anti_patterns / len(rows), 4) if rows else None,
        "median_tool_calls": _median(tool_counts),
        "median_first_useful_ms": _median(first_useful),
        "median_total_ms": _median(total_times),
        "scenario_stats": scenario_stats,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python benchmarks/score_trace_corpus.py BASELINE.jsonl UPDATED.jsonl")

    baseline_path = Path(sys.argv[1])
    updated_path = Path(sys.argv[2])
    scenarios = {row["id"]: row for row in json.loads(SCENARIOS_PATH.read_text())}

    baseline_rows = _load_jsonl(baseline_path)
    updated_rows = _load_jsonl(updated_path)

    baseline = _score(baseline_rows, scenarios)
    updated = _score(updated_rows, scenarios)

    report = {
        "baseline": baseline,
        "updated": updated,
        "delta": {
            "success_rate": None
            if baseline["success_rate"] is None or updated["success_rate"] is None
            else round(updated["success_rate"] - baseline["success_rate"], 4),
            "anti_pattern_rate": None
            if baseline["anti_pattern_rate"] is None or updated["anti_pattern_rate"] is None
            else round(updated["anti_pattern_rate"] - baseline["anti_pattern_rate"], 4),
            "median_tool_calls": None
            if baseline["median_tool_calls"] is None or updated["median_tool_calls"] is None
            else updated["median_tool_calls"] - baseline["median_tool_calls"],
            "median_first_useful_ms": None
            if baseline["median_first_useful_ms"] is None or updated["median_first_useful_ms"] is None
            else updated["median_first_useful_ms"] - baseline["median_first_useful_ms"],
            "median_total_ms": None
            if baseline["median_total_ms"] is None or updated["median_total_ms"] is None
            else updated["median_total_ms"] - baseline["median_total_ms"],
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
