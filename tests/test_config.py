"""Tests for config.py — routing_action feature flag."""

from __future__ import annotations

import os


class TestRoutingAction:
    def test_default_off(self):
        env = {k: v for k, v in os.environ.items()}
        env.pop("MATHEMATICA_ROUTING_ACTION", None)
        env.pop("MATHEMATICA_ROUTING_MEMORY", None)
        from mathematica_mcp.config import _resolve_routing_action

        assert _resolve_routing_action("off") == "off"
        assert _resolve_routing_action("observe") == "off"

    def test_forced_off_when_not_advise(self, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        from mathematica_mcp.config import _resolve_routing_action

        # Even with env set, forced off when routing_memory != advise
        assert _resolve_routing_action("observe") == "off"
        assert _resolve_routing_action("off") == "off"

    def test_env_override_with_advise(self, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        from mathematica_mcp.config import _resolve_routing_action

        assert _resolve_routing_action("advise") == "compute_cli_skip"

    def test_invalid_env_defaults_to_off(self, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "invalid_value")
        from mathematica_mcp.config import _resolve_routing_action

        assert _resolve_routing_action("advise") == "off"

    def test_to_dict_includes_routing_action(self, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")

        import importlib

        import mathematica_mcp.config as config_mod

        importlib.reload(config_mod)
        features = config_mod.FeatureFlags.from_env()
        d = features.to_dict()
        assert "routing_action" in d
        assert d["routing_action"] == "compute_cli_skip"

    def test_to_dict_shows_effective_value(self, monkeypatch):
        """to_dict reflects the effective value after forcing."""
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "observe")
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")

        import importlib

        import mathematica_mcp.config as config_mod

        importlib.reload(config_mod)
        features = config_mod.FeatureFlags.from_env()
        # Forced to off because routing_memory is observe, not advise
        assert features.routing_action == "off"
        assert features.to_dict()["routing_action"] == "off"
