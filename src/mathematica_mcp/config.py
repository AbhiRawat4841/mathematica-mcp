"""Feature flags and configuration for mathematica-mcp."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


@dataclass
class FeatureFlags:
    function_repository: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_FUNCTION_REPO", True)
    )
    data_repository: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_DATA_REPO", True)
    )
    async_computation: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_ASYNC", True)
    )
    symbol_lookup: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_LOOKUP", True)
    )
    math_aliases: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_MATH_ALIASES", True)
    )
    expression_cache: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_CACHE", True)
    )
    telemetry: bool = field(
        default_factory=lambda: _env_bool("MATHEMATICA_ENABLE_TELEMETRY", False)
    )

    def is_enabled(self, feature: str) -> bool:
        return getattr(self, feature, False)

    def to_dict(self) -> Dict[str, bool]:
        return {
            "function_repository": self.function_repository,
            "data_repository": self.data_repository,
            "async_computation": self.async_computation,
            "symbol_lookup": self.symbol_lookup,
            "math_aliases": self.math_aliases,
            "expression_cache": self.expression_cache,
            "telemetry": self.telemetry,
        }


FEATURES = FeatureFlags()
