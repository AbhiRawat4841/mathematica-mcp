"""Tests for routing_memory.py — Phase 1C."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mathematica_mcp.routing_memory import (
    RoutingMemory,
    _latency_bucket,
    _reset_for_tests,
    init,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path):
    """Reset singleton and use temp storage for every test."""
    _reset_for_tests()
    yield tmp_path
    _reset_for_tests()


def _make_mem(tmp_path: Path, mode: str = "observe") -> RoutingMemory:
    path = tmp_path / "routing_memory.json"
    mem = RoutingMemory(mode, storage_path=path)
    return mem


# ---------------------------------------------------------------------------
# Mode semantics
# ---------------------------------------------------------------------------


class TestModes:
    def test_off_returns_none(self, _isolate):
        assert init("off") is None

    def test_observe_returns_instance(self, _isolate):
        mem = init("observe", storage_path=_isolate / "rm.json")
        assert mem is not None
        assert mem.mode == "observe"

    def test_observe_hints_empty(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        assert mem.get_routing_hints("full") == []

    def test_reentry_same_mode_returns_same_instance(self, _isolate):
        path = _isolate / "rm.json"
        a = init("observe", storage_path=path)
        b = init("observe", storage_path=path)
        assert a is b

    def test_reentry_different_mode_replaces(self, _isolate):
        path = _isolate / "rm.json"
        a = init("observe", storage_path=path)
        b = init("advise", storage_path=path)
        assert a is not b
        assert b.mode == "advise"


# ---------------------------------------------------------------------------
# Counter increment semantics
# ---------------------------------------------------------------------------


class TestRecordSemantics:
    def test_ok_increments_ok_count(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        stats = mem._cohorts["full|compute"]
        assert stats.ok_count == 1
        assert stats.degraded_fallback_count == 0

    def test_degraded_fallback_increments_correctly(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "kernel_fallback", "degraded_fallback", 500, [])
        stats = mem._cohorts["full|compute"]
        assert stats.degraded_fallback_count == 1
        assert stats.ok_count == 0

    def test_timeout_increments_correctly(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "notebook_kernel", "addon_notebook", "timeout", 30000, [])
        stats = mem._cohorts["full|notebook_kernel"]
        assert stats.timeout_count == 1

    def test_infra_error_increments_correctly(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "infra_error", 0, [])
        stats = mem._cohorts["full|compute"]
        assert stats.infra_error_count == 1

    def test_semantic_error_independent_of_transport(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, ["Syntax"])
        stats = mem._cohorts["full|compute"]
        assert stats.ok_count == 1  # transport ok
        assert stats.semantic_error_count == 1  # semantic error also counted

    def test_unclassified_failure_when_infra_no_families(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "infra_error", 0, [])
        stats = mem._cohorts["full|compute"]
        assert stats.infra_error_count == 1
        assert stats.unclassified_failure_count == 1

    def test_no_unclassified_when_families_present(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "infra_error", 0, ["Syntax"])
        stats = mem._cohorts["full|compute"]
        assert stats.infra_error_count == 1
        assert stats.unclassified_failure_count == 0

    def test_path_counts_nested(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        mem.record("full", "compute", "kernel_fallback", "degraded_fallback", 500, [])
        mem.record("full", "compute", "addon_cli", "ok", 80, [])
        stats = mem._cohorts["full|compute"]
        assert stats.path_counts == {"addon_cli": 2, "kernel_fallback": 1}

    def test_error_family_stats_per_unique_family(self, _isolate):
        mem = _make_mem(_isolate)
        # Two families in one call
        mem.record("full", "compute", "addon_cli", "ok", 100, ["Syntax", "Part"])
        assert mem._error_families["Syntax"].count == 1
        assert mem._error_families["Part"].count == 1

    def test_error_family_deduped_within_call(self, _isolate):
        mem = _make_mem(_isolate)
        # Same family twice in one call → only counted once
        mem.record("full", "compute", "addon_cli", "ok", 100, ["Syntax", "Syntax"])
        assert mem._error_families["Syntax"].count == 1


# ---------------------------------------------------------------------------
# Latency histogram
# ---------------------------------------------------------------------------


class TestLatencyHistogram:
    def test_bucket_boundaries(self):
        assert _latency_bucket(0) == 0
        assert _latency_bucket(49) == 0
        assert _latency_bucket(50) == 1
        assert _latency_bucket(99) == 1
        assert _latency_bucket(100) == 2
        assert _latency_bucket(249) == 2
        assert _latency_bucket(250) == 3
        assert _latency_bucket(499) == 3
        assert _latency_bucket(500) == 4
        assert _latency_bucket(999) == 4
        assert _latency_bucket(1000) == 5
        assert _latency_bucket(4999) == 5
        assert _latency_bucket(5000) == 6
        assert _latency_bucket(50000) == 6

    def test_histogram_recording(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 49, [])
        mem.record("full", "compute", "addon_cli", "ok", 150, [])
        mem.record("full", "compute", "addon_cli", "ok", 6000, [])
        stats = mem._cohorts["full|compute"]
        assert stats.latency_hist[0] == 1  # <50
        assert stats.latency_hist[2] == 1  # 100-250
        assert stats.latency_hist[6] == 1  # >5000


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_flush_creates_file(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        mem.flush()
        assert path.exists()

    def test_roundtrip(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem1 = RoutingMemory("observe", storage_path=path)
        for _ in range(5):
            mem1.record("full", "compute", "addon_cli", "ok", 120, [])
        mem1.record("full", "compute", "addon_cli", "ok", 100, ["Syntax"])
        mem1.flush()

        mem2 = RoutingMemory("observe", storage_path=path)
        stats2 = mem2._cohorts["full|compute"]
        assert stats2.ok_count == 6
        assert stats2.semantic_error_count == 1
        assert mem2._error_families["Syntax"].count == 1

    def test_invalid_json_starts_fresh(self, _isolate):
        path = _isolate / "routing_memory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json!!!")
        mem = RoutingMemory("observe", storage_path=path)
        assert len(mem._cohorts) == 0

    def test_unknown_schema_starts_fresh(self, _isolate):
        path = _isolate / "routing_memory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 999}))
        mem = RoutingMemory("observe", storage_path=path)
        assert len(mem._cohorts) == 0

    def test_persistence_downgrade_after_failures(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])

        # Simulate 3 consecutive write failures
        with patch.object(mem, "_write_atomic", side_effect=OSError("disk full")):
            for _ in range(3):
                mem.flush()

        assert mem._persistence_enabled is False
        assert mem._flush_failure_count == 3

    def test_flush_failure_count_resets_on_success(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])

        # Fail twice, then succeed
        call_count = 0
        real_write = mem._write_atomic

        def flaky_write(data):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OSError("transient failure")
            real_write(data)

        with patch.object(mem, "_write_atomic", side_effect=flaky_write):
            mem.flush()
            assert mem._flush_failure_count == 1
            mem.flush()
            assert mem._flush_failure_count == 2
            mem.flush()
            assert mem._flush_failure_count == 0  # reset on success


# ---------------------------------------------------------------------------
# Exponential decay
# ---------------------------------------------------------------------------


class TestDecay:
    def test_no_decay_within_one_day(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        for _ in range(100):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        mem.flush()

        # Reload within the same day — no decay
        mem2 = RoutingMemory("observe", storage_path=path)
        assert mem2._cohorts["full|compute"].ok_count == 100

    def test_decay_after_14_days_on_load(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        for _ in range(100):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        mem.flush()

        # Manually set last_decay_ts to 14 days ago in the persisted file
        data = json.loads(path.read_text())
        data["last_decay_ts"] = time.time() - (14 * 86400)
        path.write_text(json.dumps(data))

        mem2 = RoutingMemory("observe", storage_path=path)
        # 0.5^(14/7) = 0.25 → 100 * 0.25 ≈ 25 (int truncation may give 24-25)
        decayed = mem2._cohorts["full|compute"].ok_count
        assert 20 <= decayed <= 26, f"Expected ~25 after 14-day decay, got {decayed}"

    def test_fresh_event_not_decayed_after_idle(self, _isolate):
        """A new event recorded after a long idle period should NOT be decayed."""
        mem = _make_mem(_isolate)
        # Simulate 14 days idle
        mem._last_decay_ts = time.time() - (14 * 86400)
        # Record one fresh event
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        # The fresh event should survive intact (old history decayed, but new event = 1)
        assert mem._cohorts["full|compute"].ok_count == 1

    def test_old_history_decayed_before_fresh_event(self, _isolate):
        """Old counters should be decayed before the new event is added."""
        mem = _make_mem(_isolate)
        for _ in range(100):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        # Simulate 14 days idle
        mem._last_decay_ts = time.time() - (14 * 86400)
        # Record one more event
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        # Old 100 decayed to ~25, plus 1 fresh = ~26
        count = mem._cohorts["full|compute"].ok_count
        assert 21 <= count <= 27, f"Expected ~26 (decayed 100 + 1 fresh), got {count}"

    def test_decay_during_flush_for_long_running_process(self, _isolate):
        """Verify that flush() applies decay, not just load()."""
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        for _ in range(100):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])

        # Simulate process running for 14 days without restart
        mem._last_decay_ts = time.time() - (14 * 86400)
        mem.flush()

        # Flush should have applied decay
        decayed = mem._cohorts["full|compute"].ok_count
        assert 20 <= decayed <= 26, f"Expected ~25 after flush decay, got {decayed}"


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_record_exception_swallowed(self, _isolate):
        mem = _make_mem(_isolate)
        # Force an error in _record_inner
        with patch.object(mem, "_record_inner", side_effect=RuntimeError("boom")):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
            # Should not raise

    def test_off_mode_no_file_created(self, _isolate):
        path = _isolate / "routing_memory.json"
        assert init("off") is None
        assert not path.exists()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_structure(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        stats = mem.get_stats()
        assert stats["mode"] == "observe"
        assert stats["persistence_enabled"] is True
        assert stats["flush_failure_count"] == 0
        assert stats["total_calls"] == 1
        assert stats["cohort_count"] == 1

    def test_clear_resets_everything(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        for _ in range(10):
            mem.record("full", "compute", "addon_cli", "ok", 100, ["Syntax"])
        mem.flush()
        assert path.exists()

        mem.clear()
        assert len(mem._cohorts) == 0
        assert len(mem._error_families) == 0
        assert not path.exists()

    def test_clear_resets_persistence_state(self, _isolate):
        """clear() should re-enable persistence after it was disabled."""
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])

        # Simulate persistence being disabled
        mem._persistence_enabled = False
        mem._flush_failure_count = 3

        mem.clear()
        assert mem._persistence_enabled is True
        assert mem._flush_failure_count == 0

    def test_clear_suppresses_stale_flush(self, _isolate):
        """A flush snapshot taken before clear() should not overwrite the cleared state."""
        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        for _ in range(10):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])

        # Simulate: take a snapshot inside lock (what flush does before writing)
        with mem._lock:
            stale_snapshot = mem._serialize()
            stale_gen = mem._flush_generation
            mem._flush_generation += 1

        # Now clear — this should advance _last_written_gen past stale_gen
        mem.clear()

        # Attempt to write the stale snapshot — should be suppressed by generation check
        with mem._flush_lock:
            if stale_gen >= mem._last_written_gen:
                # This should NOT execute
                mem._write_atomic(stale_snapshot)
                raise AssertionError("Stale snapshot should have been suppressed")

        # Verify: file should not exist (clear deleted it, stale write was suppressed)
        assert not path.exists()
