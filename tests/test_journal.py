"""Tests for journal.py — computation journal."""

from __future__ import annotations

from mathematica_mcp.journal import ComputationJournal


class TestComputationJournal:
    def test_record_adds_entry(self):
        j = ComputationJournal()
        j.record("1+1", "2", success=True, timing_ms=5)
        assert len(j.get_entries()) == 1

    def test_entry_has_required_fields(self):
        j = ComputationJournal()
        j.record(
            "Plot[Sin[x], {x,0,Pi}]",
            "-Graphics-",
            success=True,
            timing_ms=100,
            route_variant="compute",
            execution_path="addon_cli",
            transport_status="ok",
            error_families=["Syntax"],
            timed_out=False,
            from_cache=True,
        )
        entry = j.get_entries()[0]
        assert "timestamp" in entry
        assert "code_preview" in entry
        assert "output_preview" in entry
        assert entry["success"] is True
        assert entry["timing_ms"] == 100
        assert entry["route_variant"] == "compute"
        assert entry["execution_path"] == "addon_cli"
        assert entry["transport_status"] == "ok"
        assert entry["error_families"] == ["Syntax"]
        assert entry["timed_out"] is False
        assert entry["from_cache"] is True

    def test_code_preview_truncated(self):
        j = ComputationJournal()
        j.record("x" * 200, "result", success=True, timing_ms=5)
        assert len(j.get_entries()[0]["code_preview"]) == 100

    def test_output_preview_truncated(self):
        j = ComputationJournal()
        j.record("code", "y" * 300, success=True, timing_ms=5)
        assert len(j.get_entries()[0]["output_preview"]) == 100

    def test_max_entries_eviction(self):
        j = ComputationJournal(max_entries=3)
        for i in range(5):
            j.record(f"code_{i}", f"out_{i}", success=True, timing_ms=i)
        entries = j.get_entries()
        assert len(entries) == 3

    def test_eviction_preserves_newest(self):
        j = ComputationJournal(max_entries=3)
        for i in range(5):
            j.record(f"code_{i}", f"out_{i}", success=True, timing_ms=i)
        entries = j.get_entries()
        assert entries[0]["code_preview"] == "code_2"
        assert entries[2]["code_preview"] == "code_4"

    def test_clear_empties(self):
        j = ComputationJournal()
        j.record("1+1", "2", success=True, timing_ms=5)
        j.clear()
        assert len(j.get_entries()) == 0

    def test_to_dict_structure(self):
        j = ComputationJournal()
        j.record("1+1", "2", success=True, timing_ms=5)
        d = j.to_dict()
        assert d["entry_count"] == 1
        assert len(d["entries"]) == 1

    def test_session_id_optional(self):
        j = ComputationJournal()
        j.record("1+1", "2", success=True, timing_ms=5)
        assert "session_id" not in j.get_entries()[0]
        j.record("1+1", "2", success=True, timing_ms=5, session_id="sess123")
        assert j.get_entries()[1]["session_id"] == "sess123"

    def test_error_families_default_empty_list(self):
        j = ComputationJournal()
        j.record("1+1", "2", success=True, timing_ms=5)
        assert j.get_entries()[0]["error_families"] == []
