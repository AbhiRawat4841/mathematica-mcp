"""Tests for server-side PNG validation (_is_valid_png, _attach_image_if_valid)."""

from __future__ import annotations

import os
import tempfile

import pytest


class TestIsValidPng:
    def test_valid_png_file(self):
        from mathematica_mcp.server import _is_valid_png

        fd, path = tempfile.mkstemp(suffix=".png")
        # Write valid PNG header + minimal data
        os.write(fd, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        os.close(fd)

        assert _is_valid_png(path) is True
        os.remove(path)

    def test_empty_file_rejected(self):
        from mathematica_mcp.server import _is_valid_png

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        assert _is_valid_png(path) is False
        os.remove(path)

    def test_nonexistent_file_rejected(self):
        from mathematica_mcp.server import _is_valid_png

        assert _is_valid_png("/tmp/does_not_exist_12345.png") is False

    def test_wrong_magic_bytes_rejected(self):
        from mathematica_mcp.server import _is_valid_png

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"NOT A PNG FILE CONTENT HERE")
        os.close(fd)

        assert _is_valid_png(path) is False
        os.remove(path)

    def test_too_short_file_rejected(self):
        from mathematica_mcp.server import _is_valid_png

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"\x89PNG")  # Only 4 bytes, need 8
        os.close(fd)

        assert _is_valid_png(path) is False
        os.remove(path)

    def test_jpeg_file_rejected(self):
        from mathematica_mcp.server import _is_valid_png

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # JPEG magic
        os.close(fd)

        assert _is_valid_png(path) is False
        os.remove(path)


class TestAttachImageIfValid:
    def test_valid_png_attached(self):
        from mathematica_mcp.server import _attach_image_if_valid

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        os.close(fd)

        result = {"is_graphics": True, "image_path": path, "output": "Graphics[...]"}
        _attach_image_if_valid(result)

        assert result["rendered_image"] == path
        assert "tip" in result
        os.remove(path)

    def test_invalid_png_stripped_and_file_deleted(self):
        from mathematica_mcp.server import _attach_image_if_valid

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"NOT A PNG")
        os.close(fd)
        assert os.path.exists(path)

        result = {
            "is_graphics": True,
            "image_path": path,
            "output": "[Graphics rendered to image: ...]",
            "output_inputform": "Graphics[Circle[]]",
        }
        _attach_image_if_valid(result)

        assert "rendered_image" not in result
        assert "image_path" not in result
        assert result["output"] == "Graphics[Circle[]]"
        # The invalid file must be deleted from disk, not just hidden.
        assert not os.path.exists(path)

    def test_missing_file_stripped(self):
        from mathematica_mcp.server import _attach_image_if_valid

        result = {
            "is_graphics": True,
            "image_path": "/tmp/nonexistent_png_test.png",
            "output": "[Graphics rendered]",
            "output_inputform": "Graphics[...]",
        }
        _attach_image_if_valid(result)

        assert "rendered_image" not in result
        assert result["output"] == "Graphics[...]"


class TestImageFromResult:
    """Verify _image_from_result validates PNG before returning bytes."""

    def test_valid_png_returns_image(self):
        import asyncio

        from mathematica_mcp.server import _image_from_result

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        os.close(fd)

        result = {"path": path}
        img = asyncio.run(_image_from_result(result))
        assert img.data.startswith(b"\x89PNG")
        # File should be cleaned up after read
        assert not os.path.exists(path)

    def test_invalid_png_raises_and_cleans_up(self):
        import asyncio

        from mathematica_mcp.server import _image_from_result

        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"NOT A PNG FILE")
        os.close(fd)
        assert os.path.exists(path)

        result = {"path": path}
        with pytest.raises(ValueError, match="Invalid or corrupt PNG"):
            asyncio.run(_image_from_result(result))

        # Invalid file must be deleted
        assert not os.path.exists(path)

    def test_missing_file_raises(self):
        import asyncio

        from mathematica_mcp.server import _image_from_result

        result = {"path": "/tmp/nonexistent_image_test_12345.png"}
        with pytest.raises(ValueError, match="Invalid or corrupt PNG"):
            asyncio.run(_image_from_result(result))
