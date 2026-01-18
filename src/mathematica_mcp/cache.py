import time
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
