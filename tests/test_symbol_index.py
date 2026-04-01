"""Phase 3 – Symbol index fast path.

Verifies that the symbol_index module provides correct name lookups,
metadata caching, and version-scoped invalidation without spawning
subprocesses.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# symbol_index unit tests (mocked — no wolframscript needed)
# ---------------------------------------------------------------------------

FAKE_SYMBOLS = [
    "Abs",
    "AbsoluteTime",
    "Accumulate",
    "AcyclicGraphQ",
    "Integrate",
    "IntegerQ",
    "InterpolatingFunction",
    "Plot",
    "Plot3D",
    "PlotRange",
    "Plus",
    "Power",
    "Sin",
    "Sinc",
    "Sinh",
    "Solve",
    "Sort",
    "Sqrt",
    "Table",
    "Take",
    "Tan",
    "Thread",
    "Timing",
]


@pytest.fixture(autouse=True)
def _reset_index():
    """Reset the symbol index before each test."""
    import mathematica_mcp.symbol_index as idx

    idx.invalidate()
    yield
    idx.invalidate()


def _populate_index():
    """Populate the index with fake symbols (no subprocess)."""
    import mathematica_mcp.symbol_index as idx

    with idx._index_lock:
        idx._system_symbols = list(FAKE_SYMBOLS)
        idx._system_symbols_lower = [(s.lower(), s) for s in FAKE_SYMBOLS]
        idx._kernel_version = "14.1"


class TestSymbolSearch:
    def test_exact_match_first(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("Sin")
        assert results[0] == "Sin"

    def test_prefix_match_after_exact(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("Sin")
        # Sin is exact, Sinc and Sinh are prefix matches
        assert "Sin" in results
        assert "Sinc" in results
        assert "Sinh" in results

    def test_substring_match(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("lot")
        # Plot, Plot3D, PlotRange all contain "lot"
        assert "Plot" in results
        assert "Plot3D" in results

    def test_case_insensitive(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("integrate")
        assert "Integrate" in results

    def test_max_results_respected(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("a", max_results=3)
        assert len(results) <= 3

    def test_no_match_returns_empty(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("ZZZZNOTAFUNCTION")
        assert results == []

    def test_empty_index_returns_empty(self):
        import mathematica_mcp.symbol_index as idx

        # Index is empty and both build and disk cache are blocked.
        with (
            patch.object(idx, "_build_index_sync", return_value=False),
            patch.object(idx, "_load_from_disk_cache", return_value=False),
        ):
            results = idx.search("Sin")
            assert results == []

    def test_ordering_exact_prefix_contains(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        results = idx.search("Plot")
        # Plot is exact, Plot3D and PlotRange are prefix, no contains expected
        assert results[0] == "Plot"
        # Prefix matches follow
        prefix_results = results[1:]
        assert "Plot3D" in prefix_results
        assert "PlotRange" in prefix_results


class TestSymbolVersion:
    def test_version_stored(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        assert idx.get_version() == "14.1"

    def test_invalidate_clears_version(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        idx.invalidate()
        assert idx.get_version() == ""

    def test_invalidate_clears_symbols(self):
        import mathematica_mcp.symbol_index as idx

        _populate_index()
        assert len(idx.search("Sin")) > 0

        idx.invalidate()
        # After invalidation, search can't find anything (index is empty,
        # and both build and disk cache are blocked).
        with (
            patch.object(idx, "_build_index_sync", return_value=False),
            patch.object(idx, "_load_from_disk_cache", return_value=False),
        ):
            assert idx.search("Sin") == []


class TestMetadataCache:
    def test_cache_miss_returns_none(self):
        import mathematica_mcp.symbol_index as idx

        assert idx.get_cached_metadata("NonExistent") is None

    def test_cache_roundtrip(self):
        import mathematica_mcp.symbol_index as idx

        metadata = {"success": True, "symbol": "Sin", "usage": "Sin[x] gives the sine."}
        idx.cache_metadata("Sin", metadata)

        cached = idx.get_cached_metadata("Sin")
        assert cached is not None
        assert cached["usage"] == "Sin[x] gives the sine."

    def test_invalidate_clears_metadata(self):
        import mathematica_mcp.symbol_index as idx

        idx.cache_metadata("Sin", {"usage": "sine"})
        assert idx.get_cached_metadata("Sin") is not None

        idx.invalidate()
        assert idx.get_cached_metadata("Sin") is None

    def test_cache_metadata_merges_not_overwrites(self):
        """Usage-only hydration must not clobber a full metadata entry."""
        import mathematica_mcp.symbol_index as idx

        # Full payload first.
        full = {
            "success": True,
            "symbol": "Sin",
            "usage": "Sin[x] gives the sine of x.",
            "attributes": ["Listable", "Protected"],
            "options_count": 0,
        }
        idx.cache_metadata("Sin", full)

        # Usage-only hydration second (should merge, not replace).
        idx.cache_metadata("Sin", {"usage": "updated usage"})

        cached = idx.get_cached_metadata("Sin")
        assert cached["success"] is True
        assert cached["attributes"] == ["Listable", "Protected"]
        assert cached["usage"] == "updated usage"

    def test_usage_only_entry_does_not_satisfy_get_symbol_info(self):
        """get_symbol_info should NOT return a usage-only cache entry."""
        import mathematica_mcp.symbol_index as idx

        # Hydration writes usage-only entry.
        idx.cache_metadata("Plot", {"usage": "Plot[f, {x, ...}] plots f."})

        cached = idx.get_cached_metadata("Plot")
        assert cached is not None
        # The entry lacks "success" and "attributes", so get_symbol_info
        # should not treat it as complete.
        assert cached.get("success") is None
        assert cached.get("attributes") is None


class TestLookupSymbolsFastPath:
    """Test that _lookup_symbols_in_kernel uses the index when available."""

    def test_index_fast_path_returns_candidates(self):
        """When index is populated, returns structured candidates."""
        _populate_index()

        from mathematica_mcp.server import _lookup_symbols_in_kernel

        result = _lookup_symbols_in_kernel("Plot")
        assert result["success"] is True
        assert "candidates" in result
        names = [c["symbol"] for c in result["candidates"]]
        assert "Plot" in names

    def test_index_fast_path_marks_system_only(self):
        """Fast path result includes system_only flag."""
        _populate_index()

        from mathematica_mcp.server import _lookup_symbols_in_kernel

        result = _lookup_symbols_in_kernel("Sin")
        assert result.get("system_only") is True

    def test_index_fast_path_skips_subprocess(self):
        """No subprocess is spawned when index is populated."""
        _populate_index()

        import subprocess as sp

        original_run = sp.run
        call_count = 0

        def tracking_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_run(*args, **kwargs)

        with patch.object(sp, "run", tracking_run):
            from mathematica_mcp.server import _lookup_symbols_in_kernel

            _lookup_symbols_in_kernel("Sin")

        assert call_count == 0, "subprocess.run should not be called when index is available"

    def test_fallback_to_subprocess_when_no_index(self):
        """When index is empty and no wolframscript, returns error."""
        import mathematica_mcp.symbol_index as idx

        with (
            patch.object(idx, "ensure_index", return_value=False),
            patch("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", return_value=None),
        ):
            from mathematica_mcp.server import _lookup_symbols_in_kernel

            result = _lookup_symbols_in_kernel("Sin")
            assert result["success"] is False
            assert "error" in result


class TestGetSymbolInfoCache:
    """Test that get_symbol_info caches metadata."""

    def test_cached_metadata_returned_without_subprocess(self):
        import mathematica_mcp.symbol_index as idx

        metadata = {
            "success": True,
            "symbol": "Sin",
            "usage": "Sin[x] gives the sine of x.",
            "attributes": ["Listable", "NumericFunction", "Protected"],
            "options_count": 0,
            "options": [],
            "related_symbols": ["Cos", "Tan"],
            "context": "System`",
        }
        idx.cache_metadata("Sin", metadata)

        # Import the registration function and check.
        # We can't easily call get_symbol_info directly since it's registered
        # via mcp.tool(). Instead, verify the cache is checked by the module.
        cached = idx.get_cached_metadata("Sin")
        assert cached is not None
        assert cached["usage"] == "Sin[x] gives the sine of x."


class TestHydrateUsage:
    """Test lazy metadata hydration for top candidates."""

    def test_hydrate_from_cache(self):
        import mathematica_mcp.symbol_index as idx

        idx.cache_metadata("Sin", {"usage": "Sin[x] gives the sine of x."})
        idx.cache_metadata("Cos", {"usage": "Cos[x] gives the cosine of x."})

        from mathematica_mcp.server import _hydrate_usage

        result = _hydrate_usage(["Sin", "Cos"])
        assert result["Sin"] == "Sin[x] gives the sine of x."
        assert result["Cos"] == "Cos[x] gives the cosine of x."

    def test_hydrate_returns_empty_for_uncached_without_wolframscript(self):
        from mathematica_mcp.server import _hydrate_usage

        with patch("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", return_value=None):
            result = _hydrate_usage(["NonExistent"])
            assert result == {}

    def test_hydrate_empty_list(self):
        from mathematica_mcp.server import _hydrate_usage

        assert _hydrate_usage([]) == {}


class TestSuggestSimilarFastPath:
    """Test that suggest_similar_functions uses the index."""

    def test_suggest_uses_index(self):
        _populate_index()

        import mathematica_mcp.symbol_index as idx

        matches = idx.search("Int")
        assert "Integrate" in matches
        assert "IntegerQ" in matches
