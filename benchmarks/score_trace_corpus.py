from __future__ import annotations

import argparse
import json
import statistics
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
    return any(tools[index : index + len(sequence)] == sequence for index in range(len(tools) - len(sequence) + 1))


def _score(
    rows: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]], profile: str = "classic"
) -> dict[str, Any]:
    tool_counts = []
    first_useful = []
    total_times = []
    success_total = 0
    anti_patterns = 0
    scenario_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"runs": 0, "success": 0, "anti_patterns": 0})

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

        forbidden_key = "lean_forbidden_sequences" if profile == "lean" else "forbidden_sequences"
        forbidden_sequences = scenario.get(forbidden_key, [])
        hit_anti_pattern = any(_contains_sequence(tools, sequence) for sequence in forbidden_sequences)
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
    parser = argparse.ArgumentParser(description="Score trace corpora against benchmarks/scenarios.json")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("updated", type=Path)
    parser.add_argument("--profile", choices=["classic", "lean"], default="classic")
    opts = parser.parse_args()

    scenarios = {row["id"]: row for row in json.loads(SCENARIOS_PATH.read_text())}

    baseline_rows = _load_jsonl(opts.baseline)
    updated_rows = _load_jsonl(opts.updated)

    baseline = _score(baseline_rows, scenarios, opts.profile)
    updated = _score(updated_rows, scenarios, opts.profile)

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
