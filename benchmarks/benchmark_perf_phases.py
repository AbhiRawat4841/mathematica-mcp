#!/usr/bin/env python3
"""Benchmark suite for the Phase 0-4 performance optimizations.

Measures the offline (no-addon) paths that the optimizations target:
  - Cold import time
  - Symbol index build + search vs subprocess
  - Raster cache hit/miss/eviction
  - Notebook routing (Python vs kernel backend dispatch)
  - Cache epoch overhead on query cache hit/miss
  - Telemetry wrapper overhead (enabled vs disabled)

Run:  PYTHONPATH=src python benchmarks/benchmark_perf_phases.py [suffix]

Results are written to benchmarks/results/phase_<suffix>.json
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Ensure src is importable
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _timed(func, iterations=1):
    """Run *func* for *iterations*, return list of elapsed ms."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return times


def _stats(times):
    """Compute summary statistics from a list of ms timings."""
    if not times:
        return {}
    return {
        "iterations": len(times),
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "stdev_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
    }


# ---- Benchmarks ----


def bench_cold_import():
    """Measure cold import of mathematica_mcp.server in a subprocess."""
    script = (
        "import time; s=time.perf_counter(); "
        "import mathematica_mcp.server; "
        "print(f'{(time.perf_counter()-s)*1000:.3f}')"
    )
    times = []
    for _ in range(3):
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": SRC_ROOT},
            cwd=REPO_ROOT,
        )
        if r.returncode == 0:
            times.append(float(r.stdout.strip()))
    return {"name": "cold_import", **_stats(times)}


def bench_symbol_index_build():
    """Measure raw wolframscript build cost (no cache logic).

    This benchmarks _build_index_sync() directly — the pure subprocess
    path.  No disk cache clear needed since _build_index_sync() never
    reads the cache.
    """
    from mathematica_mcp import symbol_index

    symbol_index.invalidate()

    times = _timed(symbol_index._build_index_sync, iterations=1)
    count = len(symbol_index._system_symbols)
    version = symbol_index.get_version()

    if count == 0:
        return {
            "name": "symbol_index_build",
            "status": "failed",
            "reason": "no symbols loaded (wolframscript may not be available or returned non-zero exit)",
            **_stats(times),
        }

    return {
        "name": "symbol_index_build",
        "symbol_count": count,
        "kernel_version": version,
        **_stats(times),
    }


def bench_ensure_index_cold():
    """Measure full ensure_index() cold-start cost (disk cache cleared)."""
    from mathematica_mcp import symbol_index

    symbol_index.invalidate()
    symbol_index.clear_disk_cache()

    times = _timed(symbol_index.ensure_index, iterations=1)
    count = len(symbol_index._system_symbols)

    if count == 0:
        return {
            "name": "ensure_index_cold",
            "status": "failed",
            "reason": "no symbols loaded",
            **_stats(times),
        }

    return {
        "name": "ensure_index_cold",
        "symbol_count": count,
        **_stats(times),
    }


def bench_ensure_index_warm():
    """Measure ensure_index() warm-load cost (disk cache populated).

    Each iteration invalidates in-memory state first so every call
    actually loads from disk, not from the in-memory fast path.
    """
    from mathematica_mcp import symbol_index

    # Ensure disk cache is populated first
    symbol_index.ensure_index()
    count_before = len(symbol_index._system_symbols)
    if count_before == 0:
        return {
            "name": "ensure_index_warm",
            "status": "failed",
            "reason": "could not populate index for warm test",
        }

    # Measure disk-cache loads: invalidate in-memory before each iteration
    times = []
    for _ in range(3):
        symbol_index.invalidate()  # clear in-memory, disk cache remains
        start = time.perf_counter()
        symbol_index.ensure_index()
        times.append((time.perf_counter() - start) * 1000)

    count = len(symbol_index._system_symbols)

    return {
        "name": "ensure_index_warm",
        "symbol_count": count,
        **_stats(times),
    }


def bench_symbol_index_search():
    """Measure the pure-Python index search substep (no subprocess).

    This is the name-matching component only.  The full resolve_function
    tool call may additionally invoke a Global-namespace subprocess and
    a usage-hydration subprocess on the first call for each symbol set.
    """
    from mathematica_mcp import symbol_index

    symbol_index.ensure_index()

    queries = ["Plot", "Sin", "Integrate", "Table", "NDSolve", "Graphics", "List", "Map", "Sort", "Select"]
    all_times = []
    per_query = {}
    for q in queries:
        times = _timed(lambda q=q: symbol_index.search(q), iterations=200)
        all_times.extend(times)
        per_query[q] = round(statistics.mean(times), 4)

    return {
        "name": "symbol_index_search_substep",
        "note": "name-match substep only; full tool call adds Global search + usage hydration on first call",
        "per_query_mean_ms": per_query,
        **_stats(all_times),
    }


def bench_symbol_subprocess():
    """Measure the old subprocess-based symbol search (full Names scan).

    This is what every resolve_function call paid before the index existed.
    """
    import shutil

    ws = shutil.which("wolframscript")
    if not ws:
        return {"name": "symbol_subprocess_old", "skipped": True, "reason": "wolframscript not found"}

    queries = ["Plot", "Sin"]
    all_times = []
    per_query = {}
    for q in queries:

        def run(q=q):
            subprocess.run(
                [ws, "-code", f'Names["*{q}*"]'],
                capture_output=True,
                text=True,
                timeout=30,
            )

        times = _timed(run, iterations=2)
        all_times.extend(times)
        per_query[q] = round(statistics.mean(times), 1)

    return {
        "name": "symbol_subprocess_old",
        "note": "old path: full Names scan subprocess per call",
        "per_query_mean_ms": per_query,
        **_stats(all_times),
    }


def bench_lookup_end_to_end():
    """End-to-end _lookup_symbols_in_kernel: index-hot vs index-cold.

    Index-hot:  index populated → pure Python search, no subprocess.
    Index-cold: index invalidated → falls through to subprocess.

    This is a before/after comparison within the same run.
    """
    import shutil

    from mathematica_mcp import symbol_index
    from mathematica_mcp.server import _lookup_symbols_in_kernel

    ws = shutil.which("wolframscript")
    if not ws:
        return {"name": "lookup_end_to_end", "skipped": True, "reason": "wolframscript not found"}

    queries = ["Plot", "Sin"]

    # --- Before: index cold (subprocess fallback) ---
    symbol_index.invalidate()
    from unittest.mock import patch

    cold_times = []
    for q in queries:
        with patch.object(symbol_index, "ensure_index", return_value=False):
            times = _timed(lambda q=q: _lookup_symbols_in_kernel(q), iterations=1)
            cold_times.extend(times)

    # --- After: index hot (pure Python) ---
    symbol_index.ensure_index()
    hot_times = []
    for q in queries:
        times = _timed(lambda q=q: _lookup_symbols_in_kernel(q), iterations=10)
        hot_times.extend(times)

    return {
        "name": "lookup_end_to_end",
        "note": "end-to-end _lookup_symbols_in_kernel; cold=subprocess fallback, hot=index search",
        "before_cold": _stats(cold_times),
        "after_hot": _stats(hot_times),
    }


def bench_raster_cache():
    """Measure raster cache layer hit/miss overhead (microbenchmark).

    This measures the cache lookup itself, NOT the full execute_code path.
    The cache hit avoids a subprocess rasterization call (~2-20s), but the
    full tool path also includes query-cache lookup, JSON shaping, and
    asyncio overhead.  This benchmark proves the cache layer adds negligible
    overhead; the real savings are the avoided subprocess calls.
    """
    from mathematica_mcp.session import (
        _get_cached_raster,
        _put_cached_raster,
        clear_raster_cache,
    )

    clear_raster_cache()

    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, b"fake PNG " * 100)
    os.close(fd)

    code = "Plot[Sin[x], {x, 0, 2 Pi}]"

    miss_times = _timed(lambda: _get_cached_raster(code), iterations=10000)
    _put_cached_raster(code, path)
    hit_times = _timed(lambda: _get_cached_raster(code), iterations=10000)

    clear_raster_cache()
    # path is deleted by clear_raster_cache

    return {
        "name": "raster_cache_layer",
        "note": "cache lookup microbenchmark only; real savings are avoided subprocess rasterization calls",
        "miss": _stats(miss_times),
        "hit": _stats(hit_times),
    }


def bench_notebook_python_parse():
    """Measure offline Python notebook parsing."""
    nb_path = os.path.join(REPO_ROOT, "Integration.nb")
    if not os.path.exists(nb_path):
        return {"name": "notebook_python_parse", "skipped": True, "reason": "Integration.nb not found"}

    import asyncio

    from mathematica_mcp.notebook_backend import PythonSyntaxBackend

    backend = PythonSyntaxBackend()

    # Cold parse
    cold_times = _timed(
        lambda: asyncio.run(backend.extract(nb_path, truncation_threshold=25000)),
        iterations=1,
    )
    # Warm parse (parser cache may help)
    warm_times = _timed(
        lambda: asyncio.run(backend.extract(nb_path, truncation_threshold=25000)),
        iterations=3,
    )
    result = asyncio.run(backend.extract(nb_path, truncation_threshold=25000))

    return {
        "name": "notebook_python_parse",
        "cell_count": len(result.cells),
        "cold": _stats(cold_times),
        "warm": _stats(warm_times),
    }


def bench_cache_epoch():
    """Measure query cache hit/miss with epoch in key."""
    from types import SimpleNamespace

    import mathematica_mcp.cache as cache_mod

    original = cache_mod.FEATURES
    cache_mod.FEATURES = SimpleNamespace(expression_cache=True)
    cache_mod._kernel_epoch = 0

    cache = cache_mod.QueryCache()
    cache.put("benchcode", {"output": "result"}, output_format="text")

    hit_times = _timed(
        lambda: cache.get("benchcode", output_format="text"),
        iterations=50000,
    )
    miss_times = _timed(
        lambda: cache.get("cachemiss_xyz_999", output_format="text"),
        iterations=50000,
    )

    cache_mod.FEATURES = original

    return {
        "name": "cache_epoch",
        "hit": _stats(hit_times),
        "miss": _stats(miss_times),
    }


def bench_telemetry_overhead():
    """Measure telemetry wrapper cost (enabled vs disabled)."""
    import asyncio
    from types import SimpleNamespace

    import mathematica_mcp.telemetry as tel

    original = tel.FEATURES

    # Disabled
    tel.FEATURES = SimpleNamespace(telemetry=False)
    from mathematica_mcp.telemetry import telemetry_tool

    @telemetry_tool("bench_disabled")
    async def noop_disabled():
        return 42

    disabled_times = _timed(lambda: asyncio.run(noop_disabled()), iterations=5000)

    # Enabled
    tel.FEATURES = SimpleNamespace(telemetry=True)
    tel.reset_stats()

    @telemetry_tool("bench_enabled")
    async def noop_enabled():
        return 42

    enabled_times = _timed(lambda: asyncio.run(noop_enabled()), iterations=5000)

    tel.FEATURES = original

    return {
        "name": "telemetry_overhead",
        "disabled": _stats(disabled_times),
        "enabled": _stats(enabled_times),
    }


# ---- Runner ----


def run_all():
    benchmarks = [
        ("Cold import", bench_cold_import),
        ("Symbol index build (raw)", bench_symbol_index_build),
        ("ensure_index cold (disk cache cleared)", bench_ensure_index_cold),
        ("ensure_index warm (disk cache hit)", bench_ensure_index_warm),
        ("Symbol index search (substep)", bench_symbol_index_search),
        ("Symbol subprocess (old path)", bench_symbol_subprocess),
        ("Lookup end-to-end (before/after)", bench_lookup_end_to_end),
        ("Raster cache layer", bench_raster_cache),
        ("Notebook Python parse", bench_notebook_python_parse),
        ("Cache epoch overhead", bench_cache_epoch),
        ("Telemetry overhead", bench_telemetry_overhead),
    ]

    results = []
    for label, fn in benchmarks:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        try:
            r = fn()
            results.append(r)
            # Print summary
            if r.get("skipped"):
                print(f"  SKIPPED: {r.get('reason')}")
            elif "mean_ms" in r:
                print(
                    f"  mean={r['mean_ms']:.3f}ms  median={r['median_ms']:.3f}ms  "
                    f"min={r['min_ms']:.3f}ms  max={r['max_ms']:.3f}ms"
                )
            else:
                for k, v in r.items():
                    if k == "name":
                        continue
                    if isinstance(v, dict) and "mean_ms" in v:
                        print(f"  {k}: mean={v['mean_ms']:.3f}ms  median={v['median_ms']:.3f}ms")
                    elif k == "per_query_mean_ms":
                        for q, t in v.items():
                            print(f"    {q}: {t}ms")
                    else:
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": label, "error": str(e)})

    return results


def save_results(results, suffix):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"phase_{suffix}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {path}")
    return path


def print_comparison(results):
    """Print a human-readable comparison table."""
    print(f"\n{'=' * 72}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'=' * 72}")

    for r in results:
        name = r.get("name", "?")
        if r.get("skipped") or r.get("error"):
            print(f"  {name:<30} SKIPPED")
            continue

        if "before_cold" in r and "after_hot" in r:
            b = r["before_cold"]
            a = r["after_hot"]
            speedup = b["mean_ms"] / a["mean_ms"] if a["mean_ms"] > 0 else 0
            print(f"  {name:<30} BEFORE={b['mean_ms']:.0f}ms  AFTER={a['mean_ms']:.3f}ms  ({speedup:.0f}x)")
        elif "mean_ms" in r:
            print(f"  {name:<30} {r['mean_ms']:>10.3f}ms mean")
        elif "hit" in r and "miss" in r:
            hit = r["hit"]
            miss = r["miss"]
            print(f"  {name:<30} hit={hit['mean_ms']:.4f}ms  miss={miss['mean_ms']:.4f}ms")
        elif "cold" in r and "warm" in r:
            cold = r["cold"]
            warm = r["warm"]
            print(f"  {name:<30} cold={cold['mean_ms']:.1f}ms  warm={warm['mean_ms']:.1f}ms")
        elif "disabled" in r and "enabled" in r:
            d = r["disabled"]
            e = r["enabled"]
            overhead = e["mean_ms"] - d["mean_ms"]
            print(
                f"  {name:<30} disabled={d['mean_ms']:.4f}ms  enabled={e['mean_ms']:.4f}ms  overhead={overhead:.4f}ms"
            )
        elif "per_query_mean_ms" in r:
            pq = r["per_query_mean_ms"]
            avg = statistics.mean(pq.values())
            print(f"  {name:<30} {avg:>10.4f}ms mean across {len(pq)} queries")


if __name__ == "__main__":
    suffix = sys.argv[1] if len(sys.argv) > 1 else "current"
    results = run_all()
    print_comparison(results)
    save_results(results, suffix)
