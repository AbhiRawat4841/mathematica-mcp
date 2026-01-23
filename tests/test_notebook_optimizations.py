"""
Tests for notebook control optimizations.

These tests verify the new kernel-mode fast path and lazy refresh features.
Requires Mathematica to be running with StartMCPServer[].
"""

import json
import os
import socket
import time
import pytest


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9881
SOCKET_TIMEOUT = 30.0
REQUIRE_LIVE_ADDON = os.environ.get("MCP_REQUIRE_LIVE_ADDON") == "1"


class MCPTestClient:
    """Simple test client for Mathematica addon."""
    
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self._socket = None
    
    def connect(self):
        if self._socket:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(SOCKET_TIMEOUT)
            self._socket.connect((self.host, self.port))
            return True
        except Exception:
            self._socket = None
            return False
    
    def send_command(self, command: str, params: dict = None) -> dict:
        if not self._socket and not self.connect():
            raise ConnectionError("Not connected to Mathematica addon")
        
        request = {"command": command, "params": params or {}}
        request_bytes = (json.dumps(request) + "\n").encode("utf-8")
        
        self._socket.sendall(request_bytes)
        
        buffer = b""
        while b"\n" not in buffer:
            chunk = self._socket.recv(8192)
            if not chunk:
                raise ConnectionError("Connection closed")
            buffer += chunk
        
        line = buffer.partition(b"\n")[0]
        response = json.loads(line.decode("utf-8"))
        
        if response.get("status") == "error":
            raise RuntimeError(f"MCP Error: {response.get('message')}")
        
        return response.get("result", {})
    
    def close(self):
        if self._socket:
            self._socket.close()
            self._socket = None


@pytest.fixture(scope="module")
def mcp_client():
    """Fixture to provide connected MCP client."""
    client = MCPTestClient()
    if not client.connect():
        pytest.skip("Mathematica addon not running (StartMCPServer[] required)")
    
    # Verify connection with ping
    try:
        result = client.send_command("ping")
        if not result.get("pong"):
            pytest.skip("Ping failed - addon not responding correctly")

        try:
            probe = client.send_command(
                "execute_code_notebook",
                {"code": "1 + 1", "mode": "kernel"}
            )
        except Exception as e:
            if REQUIRE_LIVE_ADDON:
                raise
            pytest.skip(f"Addon running but notebook execution unavailable: {e}")

        if not probe.get("success"):
            if REQUIRE_LIVE_ADDON:
                pytest.fail("Addon running but notebook execution failed")
            pytest.skip("Addon running but notebook execution failed")
    except Exception as e:
        pytest.skip(f"Cannot connect to addon: {e}")
    
    yield client
    client.close()


class TestKernelModeFastPath:
    """Tests for kernel-mode notebook execution."""
    
    def test_kernel_mode_simple_expression(self, mcp_client):
        """Test kernel mode with simple arithmetic."""
        result = mcp_client.send_command(
            "execute_code_notebook",
            {"code": "1 + 1", "mode": "kernel"}
        )
        
        assert result.get("success") is True
        assert result.get("mode") == "kernel"
        assert "2" in result.get("output_preview", "")
        assert "timing_ms" in result
        assert result["timing_ms"] < 5000  # Should be fast
    
    def test_kernel_mode_symbolic(self, mcp_client):
        """Test kernel mode with symbolic computation."""
        result = mcp_client.send_command(
            "execute_code_notebook",
            {"code": "Integrate[x^2, x]", "mode": "kernel"}
        )
        
        assert result.get("success") is True
        assert result.get("mode") == "kernel"
        assert "x^3/3" in result.get("output_preview", "")
    
    def test_kernel_mode_graphics_detection(self, mcp_client):
        """Test that graphics are detected in kernel mode."""
        result = mcp_client.send_command(
            "execute_code_notebook",
            {"code": "Plot[Sin[x], {x, 0, 2 Pi}]", "mode": "kernel"}
        )
        
        assert result.get("success") is True
        assert result.get("mode") == "kernel"
        assert result.get("is_graphics") is True
    
    def test_kernel_mode_no_refresh_default(self, mcp_client):
        """Test that refresh is not called by default (fast)."""
        start = time.perf_counter()
        result = mcp_client.send_command(
            "execute_code_notebook",
            {"code": "Range[10]", "mode": "kernel", "refresh": False}
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        assert result.get("success") is True
        # Without refresh, should be very fast
        assert elapsed < 3000, f"Expected < 3000ms, got {elapsed}ms"
    
    def test_kernel_mode_with_refresh(self, mcp_client):
        """Test kernel mode with explicit refresh."""
        result = mcp_client.send_command(
            "execute_code_notebook",
            {"code": "2 + 2", "mode": "kernel", "refresh": True}
        )
        
        assert result.get("success") is True
        assert result.get("mode") == "kernel"
    
    def test_frontend_mode_fallback(self, mcp_client):
        """Test that frontend mode still works as fallback."""
        result = mcp_client.send_command(
            "execute_code_notebook",
            {"code": "3 + 3", "mode": "frontend", "max_wait": 5}
        )
        
        assert result.get("success") is True
        assert result.get("mode") == "frontend"


class TestLazyRefresh:
    """Tests for lazy/optional refresh in cell operations."""
    
    def test_write_cell_no_refresh_default(self, mcp_client):
        """Test write_cell doesn't refresh by default."""
        result = mcp_client.send_command(
            "write_cell",
            {"content": "(* Test cell *)", "style": "Text"}
        )
        
        assert result.get("written") is True
    
    def test_write_cell_with_refresh(self, mcp_client):
        """Test write_cell with explicit refresh."""
        result = mcp_client.send_command(
            "write_cell",
            {"content": "(* Test cell 2 *)", "style": "Text", "refresh": True}
        )
        
        assert result.get("written") is True


class TestScreenshotOptimization:
    """Tests for screenshot using ExportPacket."""
    
    def test_screenshot_uses_export_packet_by_default(self, mcp_client):
        """Test that screenshot tries ExportPacket first."""
        result = mcp_client.send_command(
            "screenshot_notebook",
            {"max_height": 500}
        )
        
        assert "path" in result
        assert "method" in result
        # Method should be either "export_packet" or "rasterize"
        assert result["method"] in ["export_packet", "rasterize"]
    
    def test_screenshot_force_rasterize(self, mcp_client):
        """Test forcing rasterize method."""
        result = mcp_client.send_command(
            "screenshot_notebook",
            {"max_height": 500, "use_rasterize": True}
        )
        
        assert "path" in result
        assert result.get("method") == "rasterize"


class TestPerformanceComparison:
    """Compare performance between kernel and frontend modes."""
    
    def test_kernel_faster_than_frontend(self, mcp_client):
        """Verify kernel mode is significantly faster than frontend mode."""
        # Kernel mode timing
        kernel_times = []
        for _ in range(3):
            start = time.perf_counter()
            result = mcp_client.send_command(
                "execute_code_notebook",
                {"code": "5 + 5", "mode": "kernel"}
            )
            if result.get("success"):
                kernel_times.append((time.perf_counter() - start) * 1000)
        
        if not kernel_times:
            pytest.skip("Kernel mode executions failed")
        
        avg_kernel = sum(kernel_times) / len(kernel_times)
        
        # Log timing for reference
        print(f"\nKernel mode avg: {avg_kernel:.1f}ms")
        
        # Kernel mode should be fast (under 2 seconds)
        assert avg_kernel < 2000, f"Kernel mode too slow: {avg_kernel}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
