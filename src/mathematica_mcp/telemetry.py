import functools
import logging
import time
from typing import Callable, Any, Dict
from collections import defaultdict

from .config import FEATURES

logger = logging.getLogger("mathematica_mcp.telemetry")

_usage_stats: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"calls": 0, "total_time_ms": 0, "errors": 0}
)


def telemetry_tool(name: str):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not FEATURES.telemetry:
                return await func(*args, **kwargs)

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                _usage_stats[name]["calls"] += 1
                _usage_stats[name]["total_time_ms"] += elapsed_ms
                logger.debug(f"Telemetry: {name} completed in {elapsed_ms}ms")
                return result
            except Exception as e:
                _usage_stats[name]["calls"] += 1
                _usage_stats[name]["errors"] += 1
                logger.debug(f"Telemetry: {name} failed with {type(e).__name__}")
                raise

        return wrapper

    return decorator


def get_usage_stats() -> Dict[str, Dict[str, Any]]:
    return dict(_usage_stats)


def reset_stats():
    global _usage_stats
    _usage_stats = defaultdict(lambda: {"calls": 0, "total_time_ms": 0, "errors": 0})
