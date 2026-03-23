import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .config import FEATURES


@dataclass
class CachedExpression:
    name: str
    expression: str
    result: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


_expression_cache: dict[str, CachedExpression] = {}

# Kernel state epoch – monotonically increasing counter included in query
# cache keys.  Bumped by any operation that mutates kernel state (e.g.
# set_variable, clear_variables, restart_kernel, load_package, run_script).
# After a bump, existing cache entries become unreachable by key, ensuring
# no stale results are returned.
_kernel_epoch: int = 0


def bump_kernel_epoch() -> int:
    """Increment the kernel epoch, logically invalidating all query cache entries."""
    global _kernel_epoch
    _kernel_epoch += 1
    return _kernel_epoch


def get_kernel_epoch() -> int:
    """Return the current kernel state epoch."""
    return _kernel_epoch


NON_CACHEABLE_PATTERNS = [
    "Random",
    "Now",
    "Date",
    "AbsoluteTime",
    "SessionTime",
    "$Line",
    "Dynamic",
    "Button",
    "Manipulate",
    "CurrentValue",
]


class QueryCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash_code(
        self,
        code: str,
        output_format: str,
        render_graphics: bool,
        deterministic_seed: int | None,
        context_key: str | None,
    ) -> str:
        normalized = " ".join(code.split())
        options = (
            f"|fmt={output_format}|gfx={int(render_graphics)}|seed={deterministic_seed}"
            f"|ctx={context_key}|epoch={_kernel_epoch}"
        )
        return hashlib.sha256((normalized + options).encode()).hexdigest()[:16]

    def _is_cacheable(self, code: str) -> bool:
        return not any(pattern in code for pattern in NON_CACHEABLE_PATTERNS)

    def get(
        self,
        code: str,
        *,
        output_format: str = "text",
        render_graphics: bool = True,
        deterministic_seed: int | None = None,
        context_key: str | None = None,
    ) -> dict[str, Any] | None:
        if not FEATURES.expression_cache or not self._is_cacheable(code):
            return None

        key = self._hash_code(code, output_format, render_graphics, deterministic_seed, context_key)
        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]
        if time.time() - entry["created_at"] > self.ttl:
            del self._cache[key]
            self.misses += 1
            return None

        self._cache.move_to_end(key)
        entry["access_count"] = entry.get("access_count", 0) + 1
        self.hits += 1
        return entry["result"]

    def put(
        self,
        code: str,
        result: dict[str, Any],
        *,
        output_format: str = "text",
        render_graphics: bool = True,
        deterministic_seed: int | None = None,
        context_key: str | None = None,
    ) -> None:
        if not FEATURES.expression_cache or not self._is_cacheable(code):
            return

        key = self._hash_code(code, output_format, render_graphics, deterministic_seed, context_key)
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "code": code,
            "result": result,
            "created_at": time.time(),
            "access_count": 0,
        }

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0


_query_cache = QueryCache()


def cache_expression(name: str, expression: str, result: str) -> bool:
    if not FEATURES.expression_cache:
        return False

    _expression_cache[name] = CachedExpression(
        name=name,
        expression=expression,
        result=result,
        created_at=time.time(),
    )
    return True


def get_cached_expression(name: str) -> CachedExpression | None:
    if not FEATURES.expression_cache:
        return None

    cached = _expression_cache.get(name)
    if cached:
        cached.access_count += 1
        cached.last_accessed = time.time()
    return cached


def list_cached_expressions() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "expression": item.expression,
            "result_preview": item.result[:100] + "..." if len(item.result) > 100 else item.result,
            "access_count": item.access_count,
            "age_seconds": int(time.time() - item.created_at),
        }
        for name, item in _expression_cache.items()
    }


def clear_cache():
    global _expression_cache
    _expression_cache = {}


def remove_cached_expression(name: str) -> bool:
    if name in _expression_cache:
        del _expression_cache[name]
        return True
    return False
