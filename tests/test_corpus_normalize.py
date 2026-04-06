"""Meta-tests for the corpus normalization layer.

These run WITHOUT wolframscript — all synthetic data.
Ensures the normalizer correctly handles every response shape
this repo produces before any live Mathematica tests run.
"""

from __future__ import annotations

import json

from corpus.normalize import Artifact, NormalizedResult, normalize


class TestNormalizeJSONString:
    def test_success_response(self):
        raw = json.dumps(
            {"success": True, "output_inputform": "42", "warnings": []}
        )
        r = normalize(raw)
        assert r.ok is True
        assert r.output_text == "42"
        assert r.warnings == []
        assert r.parse_error is False

    def test_failed_response(self):
        raw = json.dumps(
            {"success": False, "error": "Syntax error", "warnings": []}
        )
        r = normalize(raw)
        assert r.ok is False
        assert r.error_text == "Syntax error"

    def test_parse_error_with_raw(self):
        raw = json.dumps(
            {"success": True, "parse_error": True, "raw": "some raw content"}
        )
        r = normalize(raw)
        assert r.parse_error is True
        assert r.parsed is not None
        # The inner "raw" field must be preserved, not the JSON wrapper
        assert r.raw == "some raw content"

    def test_parse_error_raw_used_for_contains(self):
        """raw_contains should inspect the inner Wolfram raw output,
        not the JSON envelope."""
        inner_raw = "kernel_version -> 14.0, memory -> 12345"
        raw = json.dumps(
            {"success": True, "parse_error": True, "raw": inner_raw}
        )
        r = normalize(raw)
        assert r.raw == inner_raw
        assert "kernel_version" in r.raw

    def test_malformed_json(self):
        r = normalize("this is not json at all")
        assert r.parse_error is True
        assert r.raw == "this is not json at all"
        assert r.parsed is None

    def test_output_text_precedence(self):
        """output_inputform takes precedence over output and result."""
        raw = json.dumps(
            {
                "success": True,
                "output_inputform": "preferred",
                "output": "fallback1",
                "result": "fallback2",
            }
        )
        r = normalize(raw)
        assert r.output_text == "preferred"

    def test_output_text_fallback_to_output(self):
        raw = json.dumps({"success": True, "output": "fallback1"})
        r = normalize(raw)
        assert r.output_text == "fallback1"

    def test_meta_extraction(self):
        raw = json.dumps(
            {
                "success": True,
                "output_inputform": "x",
                "timing_ms": 123,
                "transport_status": "ok",
            }
        )
        r = normalize(raw)
        assert r.meta["timing_ms"] == 123
        assert r.meta["transport_status"] == "ok"


class TestNormalizeDict:
    def test_dict_passthrough(self):
        r = normalize({"success": True, "output": "hello"})
        assert r.ok is True
        assert r.output_text == "hello"
        assert isinstance(r.parsed, dict)

    def test_list_passthrough(self):
        r = normalize([1, 2, 3])
        assert r.ok is True
        assert r.parsed == [1, 2, 3]
        assert r.output_text == "[1, 2, 3]"

    def test_plain_string_non_json(self):
        r = normalize("just a plain string")
        assert r.parse_error is True


class TestNormalizeImage:
    def test_image_object(self):
        try:
            from mcp.server.fastmcp import Image
        except ImportError:
            import pytest

            pytest.skip("mcp package not available")

        img = Image(data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 10, format="png")
        r = normalize(img)
        assert r.ok is True
        assert len(r.artifacts) == 1
        assert r.artifacts[0].kind == "image"
        assert r.artifacts[0].format == "png"
        assert r.artifacts[0].data is not None


class TestWarningNormalization:
    def test_string_warnings(self):
        raw = json.dumps(
            {
                "success": True,
                "warnings": ["Power::infy: infinite expression"],
            }
        )
        r = normalize(raw)
        assert len(r.warnings) == 1
        assert "Power::infy" in r.warnings[0]

    def test_dict_message_records(self):
        raw = json.dumps(
            {
                "success": True,
                "messages": [
                    {
                        "type": "error",
                        "tag": "Syntax::sntxf",
                        "text": "missing bracket",
                    }
                ],
            }
        )
        r = normalize(raw)
        assert any("Syntax::sntxf" in w for w in r.warnings)

    def test_mixed_warnings_and_messages(self):
        raw = json.dumps(
            {
                "success": True,
                "warnings": ["Power::infy: inf"],
                "messages": [
                    {"type": "error", "tag": "Set::setraw", "text": "assign"}
                ],
            }
        )
        r = normalize(raw)
        assert len(r.warnings) == 2

    def test_empty_tag_skipped(self):
        raw = json.dumps(
            {
                "success": True,
                "messages": [{"type": "info", "tag": "", "text": "ok"}],
            }
        )
        r = normalize(raw)
        assert len(r.warnings) == 0


class TestArtifactExtraction:
    def test_image_path_extracted(self):
        raw = json.dumps(
            {
                "success": True,
                "image_path": "/tmp/plot.png",
                "is_graphics": True,
            }
        )
        r = normalize(raw)
        assert len(r.artifacts) == 1
        assert r.artifacts[0].kind == "file"
        assert r.artifacts[0].path == "/tmp/plot.png"

    def test_rendered_image_extracted(self):
        raw = json.dumps(
            {"success": True, "rendered_image": "/tmp/rendered.png"}
        )
        r = normalize(raw)
        assert len(r.artifacts) == 1

    def test_path_field_extracted(self):
        """Notebook/export tools return artifacts under 'path'."""
        raw = json.dumps(
            {"success": True, "path": "/tmp/saved_notebook.nb"}
        )
        r = normalize(raw)
        assert len(r.artifacts) == 1
        assert r.artifacts[0].path == "/tmp/saved_notebook.nb"

    def test_no_artifacts_from_clean_response(self):
        raw = json.dumps({"success": True, "output_inputform": "42"})
        r = normalize(raw)
        assert len(r.artifacts) == 0

    def test_file_path_extracted(self):
        raw = json.dumps(
            {"success": True, "file_path": "/tmp/export.csv"}
        )
        r = normalize(raw)
        assert len(r.artifacts) == 1
        assert r.artifacts[0].path == "/tmp/export.csv"


class TestNormalizedResultDefaults:
    def test_default_construction(self):
        r = NormalizedResult()
        assert r.ok is False
        assert r.parsed is None
        assert r.warnings == []
        assert r.artifacts == []
        assert r.meta == {}

    def test_artifact_construction(self):
        a = Artifact(kind="file", path="/tmp/test.png")
        assert a.kind == "file"
        assert a.data is None
