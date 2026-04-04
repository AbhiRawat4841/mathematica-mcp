"""Phase 1 – Cache correctness via kernel epoch.

Verifies that state-mutating operations bump the epoch and invalidate
cached query results, while pure expressions continue to hit the cache
within the same epoch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _enable_cache_and_reset():
    """Enable expression cache and reset cache + epoch for every test."""
    import mathematica_mcp.cache as cache_mod

    original = cache_mod.FEATURES
    cache_mod.FEATURES = SimpleNamespace(expression_cache=True)
    # Reset epoch and caches.
    cache_mod._kernel_epoch = 0
    cache_mod._query_cache.clear()
    cache_mod.clear_cache()
    yield
    cache_mod.FEATURES = original


# ---------------------------------------------------------------------------
# QueryCache + epoch integration
# ---------------------------------------------------------------------------


class TestEpochInCacheKey:
    def test_cache_hit_within_same_epoch(self):
        from mathematica_mcp.cache import _query_cache

        _query_cache.put("1+1", {"output": "2"}, output_format="text")
        result = _query_cache.get("1+1", output_format="text")
        assert result is not None
        assert result["output"] == "2"

    def test_cache_miss_after_epoch_bump(self):
        from mathematica_mcp.cache import _query_cache, bump_kernel_epoch

        _query_cache.put("1+1", {"output": "2"}, output_format="text")

        bump_kernel_epoch()

        result = _query_cache.get("1+1", output_format="text")
        assert result is None

    def test_new_entry_visible_in_new_epoch(self):
        from mathematica_mcp.cache import _query_cache, bump_kernel_epoch

        _query_cache.put("1+1", {"output": "2"}, output_format="text")

        bump_kernel_epoch()

        # Miss on old entry.
        assert _query_cache.get("1+1", output_format="text") is None

        # New entry at new epoch.
        _query_cache.put("1+1", {"output": "2"}, output_format="text")
        result = _query_cache.get("1+1", output_format="text")
        assert result is not None
        assert result["output"] == "2"

    def test_multiple_bumps_invalidate_all_prior(self):
        from mathematica_mcp.cache import _query_cache, bump_kernel_epoch

        _query_cache.put("a", {"output": "1"}, output_format="text")
        bump_kernel_epoch()
        _query_cache.put("b", {"output": "2"}, output_format="text")
        bump_kernel_epoch()

        # Both are gone at epoch 2.
        assert _query_cache.get("a", output_format="text") is None
        assert _query_cache.get("b", output_format="text") is None

    def test_epoch_does_not_affect_different_code(self):
        from mathematica_mcp.cache import _query_cache, bump_kernel_epoch

        _query_cache.put("Sin[x]", {"output": "Sin[x]"}, output_format="text")
        _query_cache.put("Cos[x]", {"output": "Cos[x]"}, output_format="text")

        # Both visible before bump.
        assert _query_cache.get("Sin[x]", output_format="text") is not None
        assert _query_cache.get("Cos[x]", output_format="text") is not None

        bump_kernel_epoch()

        # Both gone after bump.
        assert _query_cache.get("Sin[x]", output_format="text") is None
        assert _query_cache.get("Cos[x]", output_format="text") is None

    def test_hit_rate_tracks_epoch_misses(self):
        from mathematica_mcp.cache import _query_cache, bump_kernel_epoch

        _query_cache.put("x", {"output": "x"}, output_format="text")

        # Hit before bump.
        _query_cache.get("x", output_format="text")
        assert _query_cache.hits == 1

        bump_kernel_epoch()

        # Miss after bump.
        _query_cache.get("x", output_format="text")
        assert _query_cache.misses >= 1


class TestEpochCounter:
    def test_initial_epoch_is_zero(self):
        from mathematica_mcp.cache import get_kernel_epoch

        assert get_kernel_epoch() == 0

    def test_bump_increments(self):
        from mathematica_mcp.cache import bump_kernel_epoch, get_kernel_epoch

        assert bump_kernel_epoch() == 1
        assert get_kernel_epoch() == 1
        assert bump_kernel_epoch() == 2
        assert get_kernel_epoch() == 2

    def test_bump_returns_new_epoch(self):
        from mathematica_mcp.cache import bump_kernel_epoch

        new = bump_kernel_epoch()
        assert new == 1
        new = bump_kernel_epoch()
        assert new == 2


# ---------------------------------------------------------------------------
# State-mutating tool hooks bump epoch
# ---------------------------------------------------------------------------


class TestMutatorsBumpEpoch:
    """Verify that state-mutating tools in server.py bump the epoch.

    These tests mock the addon connection so no Mathematica is needed.
    """

    def _make_monkeypatch_server(self, monkeypatch):
        """Set up monkeypatched server with mocked addon and kernel session."""
        import mathematica_mcp.server as srv

        # Mock _addon_result to return success.
        async def fake_addon_result(command, params=None):
            return {"success": True, "pong": True}

        monkeypatch.setattr(srv, "_addon_result", fake_addon_result)

        # Mock _run_blocking to just call the function.
        async def fake_run_blocking(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr(srv, "_run_blocking", fake_run_blocking)

        # Mock close_kernel_session to be a no-op.
        monkeypatch.setattr(srv, "close_kernel_session", lambda: None)

        return srv

    def test_restart_kernel_bumps_epoch(self, monkeypatch):
        import asyncio

        import mathematica_mcp.cache as cache_mod

        srv = self._make_monkeypatch_server(monkeypatch)

        # Populate cache.
        cache_mod._query_cache.put("x", {"output": "x"}, output_format="text")
        assert cache_mod._query_cache.get("x", output_format="text") is not None

        asyncio.run(srv.restart_kernel())

        assert cache_mod.get_kernel_epoch() >= 1
        # Cache should be fully cleared.
        assert cache_mod._query_cache.stats()["size"] == 0

    def test_set_variable_bumps_epoch(self, monkeypatch):
        import asyncio

        import mathematica_mcp.cache as cache_mod

        srv = self._make_monkeypatch_server(monkeypatch)
        epoch_before = cache_mod.get_kernel_epoch()

        asyncio.run(srv.set_variable("x", "5"))

        assert cache_mod.get_kernel_epoch() > epoch_before

    def test_clear_variables_bumps_epoch(self, monkeypatch):
        import asyncio

        import mathematica_mcp.cache as cache_mod

        srv = self._make_monkeypatch_server(monkeypatch)
        epoch_before = cache_mod.get_kernel_epoch()

        asyncio.run(srv.clear_variables(names=["x"]))

        assert cache_mod.get_kernel_epoch() > epoch_before

    def test_run_script_bumps_epoch(self, monkeypatch):
        import asyncio

        import mathematica_mcp.cache as cache_mod

        srv = self._make_monkeypatch_server(monkeypatch)
        epoch_before = cache_mod.get_kernel_epoch()

        asyncio.run(srv.run_script("/tmp/test.wl"))

        assert cache_mod.get_kernel_epoch() > epoch_before

    def test_load_package_bumps_epoch(self, monkeypatch):
        import asyncio

        import mathematica_mcp.cache as cache_mod

        srv = self._make_monkeypatch_server(monkeypatch)

        # Mock the lazy_wolfram_tools.load_package.
        import mathematica_mcp.lazy_wolfram_tools as lwt

        async def fake_load_package(pkg, **kwargs):
            return '{"success": true}'

        monkeypatch.setattr(lwt, "load_package", fake_load_package)
        epoch_before = cache_mod.get_kernel_epoch()

        asyncio.run(srv.load_package("Developer`"))

        assert cache_mod.get_kernel_epoch() > epoch_before


class TestCacheCorrectnessLifecycle:
    """End-to-end lifecycle: cache → mutate → cache miss → re-execute → cache hit."""

    def test_cache_survives_across_pure_calls(self):
        from mathematica_mcp.cache import _query_cache

        _query_cache.put("2+2", {"output": "4"}, output_format="text")

        for _ in range(5):
            result = _query_cache.get("2+2", output_format="text")
            assert result is not None
            assert result["output"] == "4"

    def test_cache_invalidated_by_mutation_then_repopulated(self):
        from mathematica_mcp.cache import _query_cache, bump_kernel_epoch

        # Initial cache.
        _query_cache.put("x + 1", {"output": "6"}, output_format="text")
        assert _query_cache.get("x + 1", output_format="text") is not None

        # Mutation.
        bump_kernel_epoch()
        assert _query_cache.get("x + 1", output_format="text") is None

        # Re-execute and re-cache.
        _query_cache.put("x + 1", {"output": "11"}, output_format="text")
        result = _query_cache.get("x + 1", output_format="text")
        assert result is not None
        assert result["output"] == "11"

    def test_restart_clears_both_caches(self, monkeypatch):
        import asyncio

        import mathematica_mcp.cache as cache_mod
        import mathematica_mcp.server as srv

        # Mock addon.
        async def fake_addon_result(command, params=None):
            return {"success": True, "pong": True}

        monkeypatch.setattr(srv, "_addon_result", fake_addon_result)

        async def fake_run_blocking(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr(srv, "_run_blocking", fake_run_blocking)
        monkeypatch.setattr(srv, "close_kernel_session", lambda: None)

        # Populate both caches.
        cache_mod._query_cache.put("x", {"output": "5"}, output_format="text")
        cache_mod.cache_expression("myexpr", "x^2", "25")

        assert cache_mod._query_cache.stats()["size"] > 0
        assert cache_mod.get_cached_expression("myexpr") is not None

        asyncio.run(srv.restart_kernel())

        assert cache_mod._query_cache.stats()["size"] == 0
        assert cache_mod.get_cached_expression("myexpr") is None


class TestNonCacheableDetection:
    """Verify non-cacheable expression detection covers edge cases."""

    def test_direct_random_not_cached(self):
        from mathematica_mcp.cache import _query_cache

        _query_cache.put("RandomReal[]", {"output": "0.5"}, output_format="text")
        assert _query_cache.get("RandomReal[]", output_format="text") is None

    def test_nested_random_not_cached(self):
        from mathematica_mcp.cache import _query_cache

        code = "Module[{x = RandomReal[]}, x^2]"
        _query_cache.put(code, {"output": "0.25"}, output_format="text")
        assert _query_cache.get(code, output_format="text") is None

    def test_now_not_cached(self):
        from mathematica_mcp.cache import _query_cache

        _query_cache.put("Now", {"output": "DateObject[...]"}, output_format="text")
        assert _query_cache.get("Now", output_format="text") is None

    def test_dynamic_not_cached(self):
        from mathematica_mcp.cache import _query_cache

        code = "Dynamic[Clock[]]"
        _query_cache.put(code, {"output": "Dynamic[...]"}, output_format="text")
        assert _query_cache.get(code, output_format="text") is None

    def test_pure_expression_is_cached(self):
        from mathematica_mcp.cache import _query_cache

        _query_cache.put("1 + 1", {"output": "2"}, output_format="text")
        result = _query_cache.get("1 + 1", output_format="text")
        assert result is not None
        assert result["output"] == "2"

    def test_whitespace_normalization_in_cache(self):
        from mathematica_mcp.cache import _query_cache

        _query_cache.put("Sin[x]  +  Cos[x]", {"output": "result"}, output_format="text")
        # Same expression with different whitespace should hit cache
        result = _query_cache.get("Sin[x] + Cos[x]", output_format="text")
        assert result is not None
        assert result["output"] == "result"
