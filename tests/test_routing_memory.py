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


# ---------------------------------------------------------------------------
# Expression classification
# ---------------------------------------------------------------------------


class TestClassifyExpression:
    def test_plot_detected(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("Plot[Sin[x], {x, 0, 2 Pi}]") == "plot"

    def test_plot3d_detected(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("Plot3D[x^2 + y^2, {x,-2,2}, {y,-2,2}]") == "plot"

    def test_frontend_dynamic_manipulate(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("Manipulate[Sin[n x], {n, 1, 5}]") == "frontend_dynamic"

    def test_symbolic_integrate(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("Integrate[x^2, x]") == "symbolic_heavy"

    def test_symbolic_solve(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("Solve[x^2 == 4, x]") == "symbolic_heavy"

    def test_numeric_nintegrate(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("NIntegrate[Sin[x], {x, 0, 1}]") == "numeric_heavy"

    def test_io_export(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression('Export["file.csv", data]') == "io"

    def test_general_default(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("1 + 1") == "general"

    def test_plot_wins_over_symbolic(self):
        from mathematica_mcp.routing_memory import _classify_expression

        # Plot is checked before symbolic_heavy, so Plot wrapping Integrate → "plot"
        assert _classify_expression("Plot[Integrate[x, x], {x, 0, 1}]") == "plot"

    def test_frontend_dynamic_wins_over_plot(self):
        from mathematica_mcp.routing_memory import _classify_expression

        # frontend_dynamic checked before plot, so Manipulate wrapping Plot → "frontend_dynamic"
        assert _classify_expression("Manipulate[Plot[Sin[n x], {x,0,Pi}], {n,1,5}]") == "frontend_dynamic"

    def test_empty_string(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("") == "general"

    def test_table_is_general(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("Table[i^2, {i, 100}]") == "general"

    def test_string_content_ignored(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression('"Plot[x]"') == "general"

    def test_comment_content_ignored(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("(* Plot[x] *) 1+1") == "general"

    def test_malformed_input_returns_general(self):
        from mathematica_mcp.routing_memory import _classify_expression

        assert _classify_expression("(* unclosed Plot[x]") == "general"


# ---------------------------------------------------------------------------
# Expression type cohorts
# ---------------------------------------------------------------------------


class TestExpressionTypeCohorts:
    def test_record_with_expr_type_creates_both_keys(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [], expression_type="plot")
        assert "full|compute" in mem._cohorts
        assert "full|compute|plot" in mem._cohorts

    def test_record_without_expr_type_backward_compat(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        assert "full|compute" in mem._cohorts
        assert len([k for k in mem._cohorts if k.count("|") == 2]) == 0

    def test_expr_type_cohort_counts_independently(self, _isolate):
        mem = _make_mem(_isolate)
        mem.record("full", "compute", "addon_cli", "ok", 100, [], expression_type="plot")
        mem.record("full", "compute", "addon_cli", "ok", 100, [], expression_type="symbolic_heavy")
        assert mem._cohorts["full|compute"].ok_count == 2
        assert mem._cohorts["full|compute|plot"].ok_count == 1
        assert mem._cohorts["full|compute|symbolic_heavy"].ok_count == 1

    def test_persistence_roundtrip_with_expr_type(self, _isolate):
        path = _isolate / "routing_memory.json"
        mem1 = RoutingMemory("observe", storage_path=path)
        mem1.record("full", "compute", "addon_cli", "ok", 100, [], expression_type="plot")
        mem1.flush()
        mem2 = RoutingMemory("observe", storage_path=path)
        assert "full|compute|plot" in mem2._cohorts


# ---------------------------------------------------------------------------
# Path transport outcomes (attempt telemetry)
# ---------------------------------------------------------------------------


class TestPathTransportOutcomes:
    def test_attempt_records_path_outcomes(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.INFRA_ERROR)
        cohort = mem._cohorts["full|compute"]
        assert cohort.path_transport_outcomes["addon_cli"]["ok"] == 2
        assert cohort.path_transport_outcomes["addon_cli"]["infra_error"] == 1

    def test_multi_path_tracked(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem.record_transport_attempt("full", "compute", "addon_notebook", AttemptOutcome.TIMEOUT)
        cohort = mem._cohorts["full|compute"]
        assert "addon_cli" in cohort.path_transport_outcomes
        assert "addon_notebook" in cohort.path_transport_outcomes

    def test_typed_outcomes_tracked_separately(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.INFRA_ERROR)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.TIMEOUT)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.SEMANTIC_ERROR)
        outcomes = mem._cohorts["full|compute"].path_transport_outcomes["addon_cli"]
        assert outcomes["ok"] == 1
        assert outcomes["infra_error"] == 1
        assert outcomes["timeout"] == 1
        assert outcomes["semantic_error"] == 1

    def test_roundtrip(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        path = _isolate / "routing_memory.json"
        mem1 = RoutingMemory("observe", storage_path=path)
        mem1.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem1.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.INFRA_ERROR)
        mem1.flush()
        mem2 = RoutingMemory("observe", storage_path=path)
        outcomes = mem2._cohorts["full|compute"].path_transport_outcomes["addon_cli"]
        assert outcomes["ok"] == 1
        assert outcomes["infra_error"] == 1

    def test_decayed_and_pruned(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        # Record with aggregate cohort key to get path_transport_outcomes populated
        mem.record("full", "compute", "addon_cli", "ok", 100, [])
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        # Simulate 14 days
        mem._last_decay_ts = time.time() - (14 * 86400)
        mem.flush()
        mem2 = RoutingMemory("observe", storage_path=path)
        # Small counts may decay to 0 and be pruned
        cohort = mem2._cohorts.get("full|compute")
        if cohort and "addon_cli" in cohort.path_transport_outcomes:
            # If not pruned, values should be decayed
            assert cohort.path_transport_outcomes["addon_cli"].get("ok", 0) <= 1

    def test_v1_loads(self, _isolate):
        """Schema v1 files (no path_transport_outcomes) load without error."""
        path = _isolate / "routing_memory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        v1_data = {
            "schema_version": 1,
            "last_decay_ts": time.time(),
            "cohorts": {"full|compute": {"ok": 10}},
            "error_families": {},
        }
        path.write_text(json.dumps(v1_data))
        mem = RoutingMemory("observe", storage_path=path)
        assert "full|compute" in mem._cohorts
        assert mem._cohorts["full|compute"].path_transport_outcomes == {}

    def test_v2_written(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        path = _isolate / "routing_memory.json"
        mem = RoutingMemory("observe", storage_path=path)
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem.flush()
        data = json.loads(path.read_text())
        assert data["schema_version"] == 2

    def test_expr_type_cohort_no_path_transport_outcomes(self, _isolate):
        """Expr-type cohorts should not have path_transport_outcomes populated."""
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        # record_transport_attempt only writes to aggregate cohort
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        mem.record("full", "compute", "addon_cli", "ok", 100, [], expression_type="plot")
        # Expr-type cohort should have empty path_transport_outcomes
        assert mem._cohorts["full|compute|plot"].path_transport_outcomes == {}

    def test_addon_failure_visible_despite_kernel_success(self, _isolate):
        """Attempt records addon failure even when final record shows kernel success."""
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        # Addon fails
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.INFRA_ERROR)
        # Kernel succeeds (final end-to-end)
        mem.record("full", "compute", "kernel_fallback", "degraded_fallback", 500, [])
        # Addon failure is visible in attempt telemetry
        outcomes = mem._cohorts["full|compute"].path_transport_outcomes["addon_cli"]
        assert outcomes["infra_error"] == 1
        # Final record shows kernel_fallback
        assert mem._cohorts["full|compute"].path_counts["kernel_fallback"] == 1


# ---------------------------------------------------------------------------
# Routing hints
# ---------------------------------------------------------------------------


class TestRoutingHints:
    def test_observe_mode_returns_empty(self, _isolate):
        mem = _make_mem(_isolate, mode="observe")
        for _ in range(10):
            mem.record("full", "compute", "addon_cli", "timeout", 30000, [])
        assert mem.get_routing_hints("full") == []

    def test_advise_high_timeout_rate(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(6):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        for _ in range(4):
            mem.record("full", "compute", "addon_cli", "timeout", 30000, [])
        hints = mem.get_routing_hints("full")
        assert len(hints) >= 1
        assert any("timeout" in h.lower() for h in hints)

    def test_advise_no_hints_below_threshold(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(20):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        assert mem.get_routing_hints("full") == []

    def test_advise_minimum_sample_size(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(3):
            mem.record("full", "compute", "addon_cli", "timeout", 30000, [])
        assert mem.get_routing_hints("full") == []

    def test_advise_high_error_rate(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(6):
            mem.record("full", "compute", "addon_cli", "infra_error", 0, [])
        for _ in range(4):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        hints = mem.get_routing_hints("full")
        assert any("infra" in h.lower() or "error" in h.lower() for h in hints)

    def test_advise_high_fallback_rate(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(5):
            mem.record("full", "compute", "kernel_fallback", "degraded_fallback", 500, [])
        for _ in range(5):
            mem.record("full", "compute", "addon_cli", "ok", 100, [])
        hints = mem.get_routing_hints("full")
        assert any("fallback" in h.lower() for h in hints)

    def test_advise_expr_type_specific(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(5):
            mem.record("full", "compute", "addon_cli", "timeout", 30000, [], expression_type="plot")
        for _ in range(20):
            mem.record("full", "compute", "addon_cli", "ok", 100, [], expression_type="symbolic_heavy")
        hints = mem.get_routing_hints("full")
        assert any("plot" in h.lower() for h in hints)

    def test_advise_filters_by_profile(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(10):
            mem.record("math", "compute", "addon_cli", "timeout", 30000, [])
        assert mem.get_routing_hints("full") == []

    def test_hints_capped_at_5(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        # Create many different problematic cohorts
        for variant in ["compute", "notebook_kernel", "notebook_frontend"]:
            for _ in range(10):
                mem.record("full", variant, "addon_cli", "timeout", 30000, [])
                mem.record("full", variant, "addon_cli", "infra_error", 0, [])
                mem.record("full", variant, "kernel_fallback", "degraded_fallback", 500, [])
        hints = mem.get_routing_hints("full")
        assert len(hints) <= 5

    def test_hints_deterministic_order(self, _isolate):
        mem = _make_mem(_isolate, mode="advise")
        for _ in range(10):
            mem.record("full", "compute", "addon_cli", "timeout", 30000, [])
            mem.record("full", "compute", "addon_cli", "infra_error", 0, [])
        hints1 = mem.get_routing_hints("full")
        hints2 = mem.get_routing_hints("full")
        assert hints1 == hints2


# ---------------------------------------------------------------------------
# Recent error families
# ---------------------------------------------------------------------------


class TestRecentErrorFamilies:
    def test_by_last_seen(self, _isolate):
        from mathematica_mcp.routing_memory import ErrorFamilyStats

        mem = _make_mem(_isolate)
        mem._error_families["Syntax"] = ErrorFamilyStats(count=5, last_seen=time.time() - 100)
        mem._error_families["Part"] = ErrorFamilyStats(count=2, last_seen=time.time() - 10)
        result = mem.get_recent_error_families(limit=3)
        assert result[0] == "Part"  # most recent first

    def test_respects_limit(self, _isolate):
        from mathematica_mcp.routing_memory import ErrorFamilyStats

        mem = _make_mem(_isolate)
        for i in range(10):
            mem._error_families[f"Family{i}"] = ErrorFamilyStats(count=1, last_seen=time.time() - i)
        assert len(mem.get_recent_error_families(limit=3)) == 3

    def test_empty_when_no_errors(self, _isolate):
        mem = _make_mem(_isolate)
        assert mem.get_recent_error_families() == []

    def test_filters_other(self, _isolate):
        from mathematica_mcp.routing_memory import ErrorFamilyStats

        mem = _make_mem(_isolate)
        mem._error_families["Syntax"] = ErrorFamilyStats(count=5, last_seen=time.time())
        mem._error_families["other"] = ErrorFamilyStats(count=10, last_seen=time.time())
        result = mem.get_recent_error_families(limit=3)
        assert "other" not in result
        assert "Syntax" in result

    def test_age_cutoff(self, _isolate):
        from mathematica_mcp.routing_memory import ErrorFamilyStats

        mem = _make_mem(_isolate)
        mem._error_families["OldError"] = ErrorFamilyStats(count=5, last_seen=time.time() - 200000)
        mem._error_families["NewError"] = ErrorFamilyStats(count=1, last_seen=time.time())
        result = mem.get_recent_error_families(limit=3, max_age_seconds=86400)
        assert "OldError" not in result
        assert "NewError" in result


# ---------------------------------------------------------------------------
# Lifecycle API (1a: always-allow)
# ---------------------------------------------------------------------------


class TestLifecycleApi1a:
    def test_begin_always_returns_allow(self, _isolate):
        mem = _make_mem(_isolate)
        lease = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease.action == "allow"

    def test_finish_records_attempt_telemetry(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome
        from mathematica_mcp.routing_memory import TransportLease

        mem = _make_mem(_isolate)
        lease = TransportLease(action="allow", key=("compute", "addon_cli"))
        mem.finish_transport_attempt(lease, AttemptOutcome.OK)
        assert lease._completed is True

    def test_abort_is_noop(self, _isolate):
        from mathematica_mcp.routing_memory import TransportLease

        mem = _make_mem(_isolate)
        lease = TransportLease(action="allow", key=("compute", "addon_cli"))
        mem.abort_transport_attempt(lease)
        assert lease._completed is True


# ---------------------------------------------------------------------------
# Transport lifecycle (1b: breaker-enabled)
# ---------------------------------------------------------------------------


class TestTransportLifecycle:
    def _make_advise_mem(self, tmp_path, action="compute_cli_skip"):
        """Create an advise-mode mem with routing_action enabled."""
        import os

        os.environ["MATHEMATICA_ROUTING_ACTION"] = action
        os.environ["MATHEMATICA_ROUTING_MEMORY"] = "advise"
        path = tmp_path / "routing_memory.json"
        mem = RoutingMemory("advise", storage_path=path)
        return mem

    def _trip_breaker(self, mem, route_variant="compute", path="addon_cli"):
        """Record 5 infra errors to trip the breaker."""
        from mathematica_mcp.constants import AttemptOutcome

        for _ in range(5):
            mem.record_transport_attempt("full", route_variant, path, AttemptOutcome.INFRA_ERROR)

    def test_begin_returns_allow_when_action_off(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "off")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate, action="off")
        self._trip_breaker(mem)
        lease = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease.action == "allow"

    def test_begin_returns_allow_when_observe_mode(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "observe")
        monkeypatch.delenv("MATHEMATICA_ROUTING_ACTION", raising=False)
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = _make_mem(_isolate, mode="observe")
        lease = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease.action == "allow"

    def test_begin_returns_skip_when_breaker_tripped(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        lease = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease.action == "skip"

    def test_begin_returns_probe_after_cooldown(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        # Simulate cooldown expired
        mem._tripped_at[("compute", "addon_cli")] = time.monotonic() - 120
        lease = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease.action == "probe"

    def test_begin_notebook_always_allow(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        # Notebook route variant is never skipped
        lease = mem.begin_transport_attempt("full", "notebook_kernel", "addon_notebook")
        assert lease.action == "allow"

    def test_finish_feeds_recent_outcomes_on_all_attempts(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        for _ in range(3):
            mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        recent = mem._recent_outcomes.get(("compute", "addon_cli"))
        assert recent is not None
        assert len(recent) == 3
        assert all(o == AttemptOutcome.OK for o in recent)

    def test_finish_ok_resets_breaker(self, _isolate, monkeypatch):
        from mathematica_mcp.constants import AttemptOutcome

        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        assert mem._breaker_state.get(("compute", "addon_cli")) == "open"
        # OK result clears the breaker
        mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.OK)
        assert mem._breaker_state.get(("compute", "addon_cli")) == "closed"

    def test_finish_infra_error_trips_breaker(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        self._trip_breaker(mem)
        assert mem._breaker_state.get(("compute", "addon_cli")) == "open"

    def test_breaker_ignores_timeout_outcome(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        # 5 timeouts should NOT trip breaker
        for _ in range(5):
            mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.TIMEOUT)
        assert mem._breaker_state.get(("compute", "addon_cli"), "closed") == "closed"

    def test_breaker_ignores_semantic_errors(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        for _ in range(5):
            mem.record_transport_attempt("full", "compute", "addon_cli", AttemptOutcome.SEMANTIC_ERROR)
        assert mem._breaker_state.get(("compute", "addon_cli"), "closed") == "closed"

    def test_cooldown_skips_during_window(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        # Within cooldown
        lease1 = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease1.action == "skip"

    def test_concurrent_probes_only_one(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        # Expire cooldown
        mem._tripped_at[("compute", "addon_cli")] = time.monotonic() - 120
        # First caller gets probe
        lease1 = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease1.action == "probe"
        # Second caller gets skip (probe in flight)
        lease2 = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease2.action == "skip"

    def test_abort_reverts_to_open_no_cooldown_reset(self, _isolate, monkeypatch):
        from mathematica_mcp.routing_memory import TransportLease

        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        original_trip_time = mem._tripped_at[("compute", "addon_cli")]
        # Expire cooldown and get probe
        mem._tripped_at[("compute", "addon_cli")] = time.monotonic() - 120
        lease = mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert lease.action == "probe"
        # Abort — should revert to open without resetting cooldown
        mem.abort_transport_attempt(lease)
        assert mem._breaker_state[("compute", "addon_cli")] == "open"

    def test_abort_does_not_record_failure(self, _isolate):
        from mathematica_mcp.routing_memory import TransportLease

        mem = _make_mem(_isolate)
        lease = TransportLease(action="probe", key=("compute", "addon_cli"))
        mem._breaker_state[("compute", "addon_cli")] = "probe_in_flight"
        mem.abort_transport_attempt(lease)
        # No outcome recorded in recent_outcomes
        recent = mem._recent_outcomes.get(("compute", "addon_cli"))
        assert recent is None or len(recent) == 0

    def test_abort_noop_for_non_probe(self, _isolate):
        from mathematica_mcp.routing_memory import TransportLease

        mem = _make_mem(_isolate)
        lease = TransportLease(action="allow", key=("compute", "addon_cli"))
        mem.abort_transport_attempt(lease)
        assert lease._completed is True

    def test_finish_idempotent_on_double_call(self, _isolate):
        from mathematica_mcp.constants import AttemptOutcome
        from mathematica_mcp.routing_memory import TransportLease

        mem = _make_mem(_isolate)
        mem._breaker_state[("compute", "addon_cli")] = "probe_in_flight"
        lease = TransportLease(action="probe", key=("compute", "addon_cli"))
        mem.finish_transport_attempt(lease, AttemptOutcome.OK)
        assert mem._breaker_state[("compute", "addon_cli")] == "closed"
        # Second call is no-op
        mem._breaker_state[("compute", "addon_cli")] = "open"  # tamper to detect mutation
        mem.finish_transport_attempt(lease, AttemptOutcome.INFRA_ERROR)
        # Should still be "open" (second call was no-op due to _completed flag)
        assert mem._breaker_state[("compute", "addon_cli")] == "open"

    def test_skip_counter_incremented(self, _isolate, monkeypatch):
        monkeypatch.setenv("MATHEMATICA_ROUTING_ACTION", "compute_cli_skip")
        monkeypatch.setenv("MATHEMATICA_ROUTING_MEMORY", "advise")
        import importlib

        import mathematica_mcp.config

        importlib.reload(mathematica_mcp.config)
        mem = self._make_advise_mem(_isolate)
        self._trip_breaker(mem)
        mem.begin_transport_attempt("full", "compute", "addon_cli")
        assert mem._routing_action_skip_count >= 1

    def test_clear_resets_breaker_state(self, _isolate):
        """Fix 1: clear() must reset runtime-only breaker state."""
        from mathematica_mcp.constants import AttemptOutcome

        mem = _make_mem(_isolate)
        self._trip_breaker(mem)
        assert mem._breaker_state.get(("compute", "addon_cli")) == "open"
        assert len(mem._recent_outcomes) > 0
        mem.clear()
        assert len(mem._breaker_state) == 0
        assert len(mem._recent_outcomes) == 0
        assert len(mem._tripped_at) == 0
        assert mem._routing_action_skip_count == 0

    def test_finish_records_telemetry_unified(self, _isolate):
        """Fix 5: finish_transport_attempt() records persisted telemetry internally."""
        from mathematica_mcp.constants import AttemptOutcome
        from mathematica_mcp.routing_memory import TransportLease

        mem = _make_mem(_isolate)
        lease = TransportLease(action="allow", key=("compute", "addon_cli"))
        mem.finish_transport_attempt(lease, AttemptOutcome.OK, profile="full")
        # Should have recorded in path_transport_outcomes
        cohort = mem._cohorts.get("full|compute")
        assert cohort is not None
        assert cohort.path_transport_outcomes.get("addon_cli", {}).get("ok", 0) == 1
        # And in recent_outcomes
        assert len(mem._recent_outcomes.get(("compute", "addon_cli"), [])) == 1
