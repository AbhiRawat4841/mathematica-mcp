import hashlib
import time
from collections import OrderedDict
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .config import FEATURES


@dataclass
class CachedExpression:
    name: str
    expression: str
    result: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


_expression_cache: Dict[str, CachedExpression] = {}


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
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash_code(self, code: str) -> str:
        normalized = " ".join(code.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _is_cacheable(self, code: str) -> bool:
        return not any(pattern in code for pattern in NON_CACHEABLE_PATTERNS)

    def get(self, code: str) -> Optional[Dict[str, Any]]:
        if not FEATURES.expression_cache or not self._is_cacheable(code):
            return None

        key = self._hash_code(code)
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

    def put(self, code: str, result: Dict[str, Any]) -> None:
        if not FEATURES.expression_cache or not self._is_cacheable(code):
            return

        key = self._hash_code(code)
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "code": code,
            "result": result,
            "created_at": time.time(),
            "access_count": 0,
        }

    def stats(self) -> Dict[str, Any]:
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


def get_cached_expression(name: str) -> Optional[CachedExpression]:
    if not FEATURES.expression_cache:
        return None

    cached = _expression_cache.get(name)
    if cached:
        cached.access_count += 1
        cached.last_accessed = time.time()
    return cached


def list_cached_expressions() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "expression": item.expression,
            "result_preview": item.result[:100] + "..."
            if len(item.result) > 100
            else item.result,
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
