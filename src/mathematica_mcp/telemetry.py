import functools
import logging
import math
import threading
import time
from typing import Callable, Any, Dict, List, Set

from .config import FEATURES

logger = logging.getLogger("mathematica_mcp.telemetry")

# Maximum number of individual timings to keep per tool for percentile calculations.
_MAX_TIMING_HISTORY = 200

_usage_stats: Dict[str, Dict[str, Any]] = {}
_stats_lock = threading.Lock()

# Track which tools have been wrapped with telemetry.
_instrumented_tools: Set[str] = set()


def _empty_stats() -> Dict[str, Any]:
    return {"calls": 0, "total_time_ms": 0, "errors": 0, "timings": []}


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def telemetry_tool(name: str):
    """Decorator that instruments an async tool function with timing and error tracking.

    When FEATURES.telemetry is disabled, the wrapper is still applied but
    short-circuits to the original function with zero overhead beyond one
    boolean check.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not FEATURES.telemetry:
                return await func(*args, **kwargs)

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                with _stats_lock:
                    stats = _usage_stats.setdefault(name, _empty_stats())
                    stats["calls"] += 1
                    stats["total_time_ms"] += elapsed_ms
                    timings = stats["timings"]
                    timings.append(elapsed_ms)
                    if len(timings) > _MAX_TIMING_HISTORY:
                        del timings[: len(timings) - _MAX_TIMING_HISTORY]
                logger.debug(f"Telemetry: {name} completed in {elapsed_ms}ms")
                return result
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                with _stats_lock:
                    stats = _usage_stats.setdefault(name, _empty_stats())
                    stats["calls"] += 1
                    stats["errors"] += 1
                    timings = stats["timings"]
                    timings.append(elapsed_ms)
                    if len(timings) > _MAX_TIMING_HISTORY:
                        del timings[: len(timings) - _MAX_TIMING_HISTORY]
                logger.debug(f"Telemetry: {name} failed with {type(e).__name__}")
                raise

        wrapper._telemetry_instrumented = True  # type: ignore[attr-defined]
        wrapper._telemetry_name = name  # type: ignore[attr-defined]
        _instrumented_tools.add(name)
        return wrapper

    return decorator


def get_usage_stats() -> Dict[str, Dict[str, Any]]:
    """Return per-tool usage statistics including percentiles.

    Each tool entry contains: calls, total_time_ms, errors, p50_ms, p95_ms,
    min_ms, max_ms.  The raw timings list is omitted to keep the response
    compact.
    """
    with _stats_lock:
        result: Dict[str, Dict[str, Any]] = {}
        for name, stats in _usage_stats.items():
            timings = stats["timings"]
            entry: Dict[str, Any] = {
                "calls": stats["calls"],
                "total_time_ms": stats["total_time_ms"],
                "errors": stats["errors"],
            }
            if timings:
                entry["p50_ms"] = round(_percentile(timings, 50), 1)
                entry["p95_ms"] = round(_percentile(timings, 95), 1)
                entry["min_ms"] = min(timings)
                entry["max_ms"] = max(timings)
            result[name] = entry
        return result


def get_instrumented_tools() -> Set[str]:
    """Return the set of tool names that have been wrapped with telemetry."""
    return set(_instrumented_tools)


def reset_stats():
    global _usage_stats
    with _stats_lock:
        _usage_stats = {}
