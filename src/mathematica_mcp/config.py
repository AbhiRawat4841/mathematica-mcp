"""Feature flags and configuration for mathematica-mcp."""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_PROFILES = ("lean", "classic", "math", "notebook", "full")

# Groups reused by both classic and full (classic is the byte-identical current
# surface; full aliases to it at 1.0).
_FULL_GROUPS = frozenset(
    {
        "core",
        "session",
        "knowledge",
        "debug",
        "moat",  # verify_derivation — shared by lean and classic
        "kernel_tools",
        "notebook_primary",
        "notebook_advanced",
        "file_legacy",
        "data",
        "graphics",
        "admin",
    }
)

PROFILE_TOOL_GROUPS: dict[str, frozenset[str]] = {
    # New default at 1.0: the 12 consolidated lean tools tagged @_tool("lean").
    # Extra tool groups are opt-in via MATHEMATICA_TOOLSETS (see _resolve_toolsets).
    "lean": frozenset({"lean", "moat"}),
    "classic": _FULL_GROUPS,
    "math": frozenset(
        {
            "core",
            "session",
            "knowledge",
            "debug",
            "moat",
            "kernel_tools",
        }
    ),
    "notebook": frozenset(
        {
            "core",
            "session",
            "knowledge",
            "debug",
            "moat",
            "kernel_tools",
            "notebook_primary",
            "data",
            "graphics",
        }
    ),
    "full": _FULL_GROUPS,
}

# All extras off for lean; opt in via MATHEMATICA_TOOLSETS. expression_cache stays
# on (internal query cache, registers no tools).
_LEAN_FEATURES = {
    "function_repository": False,
    "data_repository": False,
    "async_computation": False,
    "symbol_lookup": False,
    "math_aliases": False,
    "expression_cache": True,
    "telemetry": False,
}
_FULL_FEATURES = {
    "function_repository": True,
    "data_repository": True,
    "async_computation": True,
    "symbol_lookup": True,
    "math_aliases": True,
    "expression_cache": True,
    "telemetry": False,
}

PROFILE_FEATURE_DEFAULTS: dict[str, dict[str, bool]] = {
    "lean": _LEAN_FEATURES,
    "classic": _FULL_FEATURES,
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
    "full": _FULL_FEATURES,
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


# Opt-in extras for the lean profile (MATHEMATICA_TOOLSETS=data_io,cloud,...).
# Toolset names map to either extra tool groups or optional-module feature flags.
# Applied to the lean profile only; can only enable, never disable.
_TOOLSET_GROUPS: dict[str, frozenset[str]] = {
    "data_io": frozenset({"data"}),
    "graphics_plus": frozenset({"graphics"}),
    "cloud": frozenset({"knowledge"}),
    "debug": frozenset({"debug"}),  # trace/time/journal — verify_derivation is lean-core (moat group)
    "notebook_files": frozenset({"file_legacy"}),
    "notebook_edit": frozenset({"notebook_advanced"}),
}
_TOOLSET_FEATURES: dict[str, dict[str, bool]] = {
    "symbols": {"symbol_lookup": True},
    "math_aliases": {"math_aliases": True},
    "repository": {"function_repository": True, "data_repository": True},
    "async_jobs": {"async_computation": True},
    "cache": {"cache_tools": True},
}


def _resolve_toolsets() -> tuple[frozenset[str], dict[str, bool]]:
    """Parse MATHEMATICA_TOOLSETS into (extra tool groups, feature overrides)."""
    raw = os.getenv("MATHEMATICA_TOOLSETS", "")
    names = {n.strip().lower() for n in raw.split(",") if n.strip()}
    groups: set[str] = set()
    features: dict[str, bool] = {}
    for name in names:
        groups |= _TOOLSET_GROUPS.get(name, frozenset())
        features.update(_TOOLSET_FEATURES.get(name, {}))
    return frozenset(groups), features


_VALID_ROUTING_MEMORY_MODES = frozenset({"off", "observe", "advise"})


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _env_explicit_bool(key: str) -> bool | None:
    value = os.getenv(key)
    if value is None:
        return None
    return _parse_bool(value)


def _env_profile(default: str = "lean") -> str:
    # v1.0 breaking change: lean is the default profile. Set MATHEMATICA_PROFILE=classic
    # (or full) to keep the legacy 82-tool surface. See docs/MIGRATION.md.
    profile = os.getenv("MATHEMATICA_PROFILE", default).strip().lower()
    if profile in VALID_PROFILES:
        return profile
    return default


def _resolve_feature(name: str, profile: str) -> bool:
    explicit = _env_explicit_bool(FEATURE_ENV_KEYS[name])
    if explicit is not None:
        return explicit
    return PROFILE_FEATURE_DEFAULTS[profile][name]


def _resolve_routing_memory() -> str:
    """Resolve routing memory mode from env. Defaults to 'off' in all profiles."""
    explicit = os.getenv("MATHEMATICA_ROUTING_MEMORY", "").strip().lower()
    return explicit if explicit in _VALID_ROUTING_MEMORY_MODES else "off"


_VALID_ROUTING_ACTIONS = frozenset({"off", "compute_cli_skip"})


def _resolve_routing_action(routing_memory: str) -> str:
    """Resolve routing action from env. Forced to 'off' unless routing_memory='advise'."""
    if routing_memory != "advise":
        return "off"
    explicit = os.getenv("MATHEMATICA_ROUTING_ACTION", "").strip().lower()
    return explicit if explicit in _VALID_ROUTING_ACTIONS else "off"


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
    routing_memory: str  # "off" | "observe" | "advise"
    routing_action: str  # "off" | "compute_cli_skip" (requires advise)
    default_output_target: str

    @classmethod
    def from_env(cls, profile_override: str | None = None) -> FeatureFlags:
        profile = profile_override or _env_profile()
        if profile not in VALID_PROFILES:
            profile = "lean"
        expression_cache = _resolve_feature("expression_cache", profile)
        telemetry = _resolve_feature("telemetry", profile)
        routing_memory = _resolve_routing_memory()
        # Toolset opt-ins apply to lean only; they can enable, never disable.
        extra_groups, ts_features = _resolve_toolsets() if profile == "lean" else (frozenset(), {})

        def feat(name: str) -> bool:
            return _resolve_feature(name, profile) or ts_features.get(name, False)

        return cls(
            profile=profile,
            tool_groups=PROFILE_TOOL_GROUPS[profile] | extra_groups,
            function_repository=feat("function_repository"),
            data_repository=feat("data_repository"),
            async_computation=feat("async_computation"),
            symbol_lookup=feat("symbol_lookup"),
            math_aliases=feat("math_aliases"),
            expression_cache=expression_cache,
            telemetry=telemetry,
            cache_tools=(expression_cache and profile in ("full", "classic")) or ts_features.get("cache_tools", False),
            routing_memory=routing_memory,
            routing_action=_resolve_routing_action(routing_memory),
            # lean's evaluate defaults to target="kernel" (cli); it always passes
            # output_target explicitly, so this only affects guidance text.
            default_output_target="cli" if profile in ("math", "lean") else "notebook",
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
            "routing_memory": self.routing_memory,
            "routing_action": self.routing_action,
            "telemetry": self.telemetry,
        }


FEATURES = FeatureFlags.from_env()
