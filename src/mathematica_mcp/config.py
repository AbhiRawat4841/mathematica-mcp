"""Feature flags and configuration for mathematica-mcp."""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_PROFILES = ("math", "notebook", "full")

PROFILE_TOOL_GROUPS: dict[str, frozenset[str]] = {
    "math": frozenset(
        {
            "core",
            "session",
            "knowledge",
            "debug",
            "kernel_tools",
        }
    ),
    "notebook": frozenset(
        {
            "core",
            "session",
            "knowledge",
            "debug",
            "kernel_tools",
            "notebook_primary",
            "data",
            "graphics",
        }
    ),
    "full": frozenset(
        {
            "core",
            "session",
            "knowledge",
            "debug",
            "kernel_tools",
            "notebook_primary",
            "notebook_advanced",
            "file_legacy",
            "data",
            "graphics",
            "admin",
        }
    ),
}

PROFILE_FEATURE_DEFAULTS: dict[str, dict[str, bool]] = {
    "math": {
        "function_repository": False,
        "data_repository": False,
        "async_computation": False,
        "symbol_lookup": True,
        "math_aliases": False,
        "expression_cache": True,
        "telemetry": False,
    },
    "notebook": {
        "function_repository": False,
        "data_repository": False,
        "async_computation": False,
        "symbol_lookup": True,
        "math_aliases": False,
        "expression_cache": True,
        "telemetry": False,
    },
    "full": {
        "function_repository": True,
        "data_repository": True,
        "async_computation": True,
        "symbol_lookup": True,
        "math_aliases": True,
        "expression_cache": True,
        "telemetry": False,
    },
}

FEATURE_ENV_KEYS = {
    "function_repository": "MATHEMATICA_ENABLE_FUNCTION_REPO",
    "data_repository": "MATHEMATICA_ENABLE_DATA_REPO",
    "async_computation": "MATHEMATICA_ENABLE_ASYNC",
    "symbol_lookup": "MATHEMATICA_ENABLE_LOOKUP",
    "math_aliases": "MATHEMATICA_ENABLE_MATH_ALIASES",
    "expression_cache": "MATHEMATICA_ENABLE_CACHE",
    "telemetry": "MATHEMATICA_ENABLE_TELEMETRY",
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _env_explicit_bool(key: str) -> bool | None:
    value = os.getenv(key)
    if value is None:
        return None
    return _parse_bool(value)


def _env_profile(default: str = "full") -> str:
    profile = os.getenv("MATHEMATICA_PROFILE", default).strip().lower()
    if profile in VALID_PROFILES:
        return profile
    return default


def _resolve_feature(name: str, profile: str) -> bool:
    explicit = _env_explicit_bool(FEATURE_ENV_KEYS[name])
    if explicit is not None:
        return explicit
    return PROFILE_FEATURE_DEFAULTS[profile][name]


@dataclass(frozen=True)
class FeatureFlags:
    profile: str
    tool_groups: frozenset[str]
    function_repository: bool
    data_repository: bool
    async_computation: bool
    symbol_lookup: bool
    math_aliases: bool
    expression_cache: bool
    telemetry: bool
    cache_tools: bool
    default_output_target: str

    @classmethod
    def from_env(cls, profile_override: str | None = None) -> FeatureFlags:
        profile = profile_override or _env_profile()
        if profile not in VALID_PROFILES:
            profile = "full"
        expression_cache = _resolve_feature("expression_cache", profile)
        telemetry = _resolve_feature("telemetry", profile)
        return cls(
            profile=profile,
            tool_groups=PROFILE_TOOL_GROUPS[profile],
            function_repository=_resolve_feature("function_repository", profile),
            data_repository=_resolve_feature("data_repository", profile),
            async_computation=_resolve_feature("async_computation", profile),
            symbol_lookup=_resolve_feature("symbol_lookup", profile),
            math_aliases=_resolve_feature("math_aliases", profile),
            expression_cache=expression_cache,
            telemetry=telemetry,
            cache_tools=expression_cache and profile == "full",
            default_output_target="cli" if profile == "math" else "notebook",
        )

    def is_enabled(self, feature: str) -> bool:
        return getattr(self, feature, False)

    def tool_group_enabled(self, group: str) -> bool:
        return group in self.tool_groups

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "default_output_target": self.default_output_target,
            "tool_groups": sorted(self.tool_groups),
            "function_repository": self.function_repository,
            "data_repository": self.data_repository,
            "async_computation": self.async_computation,
            "symbol_lookup": self.symbol_lookup,
            "math_aliases": self.math_aliases,
            "expression_cache": self.expression_cache,
            "cache_tools": self.cache_tools,
            "telemetry": self.telemetry,
        }


FEATURES = FeatureFlags.from_env()
