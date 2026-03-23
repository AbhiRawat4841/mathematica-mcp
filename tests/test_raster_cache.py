"""Phase 4 – Raster cache for repeated graphics outputs.

Verifies that cached raster files are reused on query-cache hits,
properly invalidated, and cleaned up.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _reset_raster_cache():
    """Clear raster cache before each test."""
    from mathematica_mcp.session import clear_raster_cache

    clear_raster_cache()
    yield
    clear_raster_cache()


class TestRasterCacheBasics:
    def test_cache_miss_returns_none(self):
        from mathematica_mcp.session import _get_cached_raster

        assert _get_cached_raster("Plot[Sin[x], {x, 0, Pi}]") is None

    def test_cache_roundtrip(self):
        from mathematica_mcp.session import _get_cached_raster, _put_cached_raster

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"PNG fake content")
        os.close(fd)

        _put_cached_raster("Plot[Sin[x], {x, 0, Pi}]", path)

        cached = _get_cached_raster("Plot[Sin[x], {x, 0, Pi}]")
        assert cached == path

        os.remove(path)

    def test_stale_file_evicted(self):
        from mathematica_mcp.session import _get_cached_raster, _put_cached_raster

        _put_cached_raster("code", "/tmp/nonexistent_raster.png")

        assert _get_cached_raster("code") is None

    def test_empty_file_evicted(self):
        from mathematica_mcp.session import _get_cached_raster, _put_cached_raster

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)  # Empty file, 0 bytes.

        _put_cached_raster("code", path)
        assert _get_cached_raster("code") is None

        os.remove(path)

    def test_different_code_different_entries(self):
        from mathematica_mcp.session import _get_cached_raster, _put_cached_raster

        fd1, path1 = tempfile.mkstemp(suffix=".png")
        os.write(fd1, b"PNG1")
        os.close(fd1)

        fd2, path2 = tempfile.mkstemp(suffix=".png")
        os.write(fd2, b"PNG2")
        os.close(fd2)

        _put_cached_raster("Plot[Sin[x]]", path1)
        _put_cached_raster("Plot[Cos[x]]", path2)

        assert _get_cached_raster("Plot[Sin[x]]") == path1
        assert _get_cached_raster("Plot[Cos[x]]") == path2

        os.remove(path1)
        os.remove(path2)

    def test_same_code_different_size_different_entries(self):
        from mathematica_mcp.session import _get_cached_raster, _put_cached_raster

        fd1, path1 = tempfile.mkstemp(suffix=".png")
        os.write(fd1, b"PNG1")
        os.close(fd1)

        fd2, path2 = tempfile.mkstemp(suffix=".png")
        os.write(fd2, b"PNG2")
        os.close(fd2)

        _put_cached_raster("Plot[Sin[x]]", path1, image_size=500)
        _put_cached_raster("Plot[Sin[x]]", path2, image_size=1000)

        assert _get_cached_raster("Plot[Sin[x]]", image_size=500) == path1
        assert _get_cached_raster("Plot[Sin[x]]", image_size=1000) == path2

        os.remove(path1)
        os.remove(path2)


class TestRasterCacheClear:
    def test_clear_removes_files(self):
        from mathematica_mcp.session import (
            _put_cached_raster,
            _get_cached_raster,
            clear_raster_cache,
        )

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"PNG content")
        os.close(fd)

        _put_cached_raster("code", path)
        assert os.path.exists(path)

        clear_raster_cache()
        assert not os.path.exists(path)
        assert _get_cached_raster("code") is None

    def test_overwrite_deletes_old_file(self):
        """Overwriting a key must delete the previous file, not leak it."""
        from mathematica_mcp.session import _put_cached_raster, _get_cached_raster

        fd1, path1 = tempfile.mkstemp(suffix=".png")
        os.write(fd1, b"OLD PNG")
        os.close(fd1)

        fd2, path2 = tempfile.mkstemp(suffix=".png")
        os.write(fd2, b"NEW PNG")
        os.close(fd2)

        _put_cached_raster("same_code", path1)
        assert _get_cached_raster("same_code") == path1

        # Overwrite with new path.
        _put_cached_raster("same_code", path2)
        assert _get_cached_raster("same_code") == path2

        # Old file should have been deleted.
        assert not os.path.exists(path1)
        # New file should still exist.
        assert os.path.exists(path2)

        os.remove(path2)

    def test_clear_handles_already_deleted_files(self):
        from mathematica_mcp.session import _put_cached_raster, clear_raster_cache

        _put_cached_raster("code", "/tmp/already_deleted.png")

        # Should not raise.
        clear_raster_cache()

    def test_restart_kernel_clears_raster_cache(self, monkeypatch):
        import asyncio
        import mathematica_mcp.server as srv
        from mathematica_mcp.session import _put_cached_raster, _get_cached_raster

        async def fake_addon_result(command, params=None):
            return {"success": True, "pong": True}

        monkeypatch.setattr(srv, "_addon_result", fake_addon_result)

        async def fake_run_blocking(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr(srv, "_run_blocking", fake_run_blocking)
        monkeypatch.setattr(srv, "close_kernel_session", lambda: None)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"PNG content")
        os.close(fd)

        _put_cached_raster("Plot[x]", path)
        assert _get_cached_raster("Plot[x]") == path

        asyncio.run(srv.restart_kernel())

        assert _get_cached_raster("Plot[x]") is None
        assert not os.path.exists(path)


class TestRasterCacheKey:
    def test_key_deterministic(self):
        from mathematica_mcp.session import _raster_cache_key

        k1 = _raster_cache_key("Plot[Sin[x]]", 500)
        k2 = _raster_cache_key("Plot[Sin[x]]", 500)
        assert k1 == k2

    def test_different_code_different_key(self):
        from mathematica_mcp.session import _raster_cache_key

        k1 = _raster_cache_key("Plot[Sin[x]]")
        k2 = _raster_cache_key("Plot[Cos[x]]")
        assert k1 != k2

    def test_different_size_different_key(self):
        from mathematica_mcp.session import _raster_cache_key

        k1 = _raster_cache_key("Plot[Sin[x]]", 500)
        k2 = _raster_cache_key("Plot[Sin[x]]", 1000)
        assert k1 != k2

    def test_wrapped_code_matches_lookup(self):
        """Store and lookup must use the same key form (wrapped_code)."""
        from mathematica_mcp.session import (
            _get_cached_raster,
            _put_cached_raster,
            _wrap_code_for_context,
            _wrap_code_for_determinism,
            _session_context,
        )

        raw_code = "Plot[Sin[x], {x, 0, Pi}]"
        ctx = _session_context("test_session")
        wrapped = _wrap_code_for_context(raw_code, ctx)
        wrapped = _wrap_code_for_determinism(wrapped, 42)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"PNG")
        os.close(fd)

        # Store under wrapped code.
        _put_cached_raster(wrapped, path)

        # Lookup under wrapped code should hit.
        assert _get_cached_raster(wrapped) == path
        # Lookup under raw code should miss (different key).
        assert _get_cached_raster(raw_code) is None

        os.remove(path)


class TestRasterCacheBounded:
    def test_evicts_oldest_when_full(self):
        from mathematica_mcp.session import (
            _put_cached_raster,
            _get_cached_raster,
            _raster_cache,
            _MAX_RASTER_ENTRIES,
        )

        paths = []
        for i in range(_MAX_RASTER_ENTRIES + 5):
            fd, path = tempfile.mkstemp(suffix=".png")
            os.write(fd, f"PNG{i}".encode())
            os.close(fd)
            paths.append(path)
            _put_cached_raster(f"code_{i}", path)

        # Cache should not exceed max size.
        assert len(_raster_cache) <= _MAX_RASTER_ENTRIES

        # Oldest entries should have been evicted (and files deleted).
        for i in range(5):
            assert _get_cached_raster(f"code_{i}") is None
            assert not os.path.exists(paths[i])

        # Newest entries should still be present.
        for i in range(_MAX_RASTER_ENTRIES + 5 - 3, _MAX_RASTER_ENTRIES + 5):
            assert _get_cached_raster(f"code_{i}") == paths[i]

        # Cleanup remaining files.
        for p in paths:
            if os.path.exists(p):
                os.remove(p)

    def test_lru_move_to_end_on_hit(self):
        from mathematica_mcp.session import (
            _put_cached_raster,
            _get_cached_raster,
            _raster_cache,
            _MAX_RASTER_ENTRIES,
        )

        # Fill cache to near capacity.
        paths = []
        for i in range(_MAX_RASTER_ENTRIES - 1):
            fd, path = tempfile.mkstemp(suffix=".png")
            os.write(fd, f"PNG{i}".encode())
            os.close(fd)
            paths.append(path)
            _put_cached_raster(f"fill_{i}", path)

        # Access the first entry (moves it to end / most recently used).
        _get_cached_raster("fill_0")

        # Add more entries to trigger eviction.
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix=".png")
            os.write(fd, f"EXTRA{i}".encode())
            os.close(fd)
            paths.append(path)
            _put_cached_raster(f"extra_{i}", path)

        # fill_0 should still be present (LRU moved it to end).
        assert _get_cached_raster("fill_0") is not None
        # fill_1 should have been evicted (it was oldest after fill_0 moved).
        assert _get_cached_raster("fill_1") is None

        for p in paths:
            if os.path.exists(p):
                os.remove(p)
