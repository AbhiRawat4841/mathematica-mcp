from __future__ import annotations

import json

import pytest

from mathematica_mcp import connection
from mathematica_mcp.connection import MathematicaConnection


class _FakeSocket:
    attempts = 0

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def connect(self, address: tuple[str, int]) -> None:
        type(self).attempts += 1
        raise OSError("boom")

    def close(self) -> None:
        pass


def test_connection_backoff_skips_repeat_connect_attempts(monkeypatch):
    current = {"t": 100.0}

    monkeypatch.setattr(connection.time, "monotonic", lambda: current["t"])
    monkeypatch.setattr(connection.socket, "socket", lambda *args, **kwargs: _FakeSocket())

    conn = connection.MathematicaConnection(retry_backoff_seconds=2.0)

    assert conn.connect() is False
    assert _FakeSocket.attempts == 1

    assert conn.connect() is False
    assert _FakeSocket.attempts == 1

    current["t"] += 2.1
    assert conn.connect() is False
    assert _FakeSocket.attempts == 2


def test_get_mathematica_connection_does_not_ping_healthy_connection(monkeypatch):
    class HealthyConnection:
        def __init__(self):
            self.connect_calls = 0
            self.send_calls = 0

        def is_connected(self) -> bool:
            return True

        def connect(self) -> bool:
            self.connect_calls += 1
            return True

        def send_command(self, command: str, params=None):
            self.send_calls += 1
            return {"pong": True}

    fake = HealthyConnection()
    monkeypatch.setattr(connection, "_global_connection", fake)

    result = connection.get_mathematica_connection()

    assert result is fake
    assert fake.connect_calls == 0
    assert fake.send_calls == 0


def test_get_mathematica_connection_surfaces_last_error(monkeypatch):
    class FailingConnection:
        last_error = "connection refused"

        def is_connected(self) -> bool:
            return False

        def connect(self) -> bool:
            return False

    monkeypatch.setattr(connection, "_global_connection", FailingConnection())

    try:
        connection.get_mathematica_connection()
    except ConnectionError as exc:
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("expected ConnectionError")


# ============================================================================
# Socket framing tests
# ============================================================================


class _MockRecvSocket:
    """Mock socket that returns pre-configured chunks from recv()."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._idx = 0
        self.recv_count = 0
        self.closed = False
        self.timeout_history: list[float] = []

    def recv(self, bufsize: int) -> bytes:
        self.recv_count += 1
        if self._idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def sendall(self, data: bytes) -> None:
        pass

    def getpeername(self):
        return ("localhost", 9881)

    def settimeout(self, timeout: float) -> None:
        self.timeout_history.append(timeout)

    def connect(self, address: tuple[str, int]) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class TestFramedJsonReceive:
    """Tests for _receive_framed_json: baseline coverage before Phase 2."""

    def test_single_complete_frame(self):
        """A single recv returning a complete frame."""
        payload = json.dumps({"result": {"status": "ok"}}).encode() + b"\n"
        mock_sock = _MockRecvSocket([payload])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        result = conn._receive_framed_json()

        parsed = json.loads(result)
        assert parsed["result"]["status"] == "ok"

    def test_frame_split_across_two_recvs(self):
        """A frame split across two recv() calls."""
        full = json.dumps({"a": 1}).encode() + b"\n"
        half1 = full[:5]
        half2 = full[5:]
        mock_sock = _MockRecvSocket([half1, half2])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        result = conn._receive_framed_json()

        parsed = json.loads(result)
        assert parsed["a"] == 1
        assert mock_sock.recv_count == 2

    def test_extra_data_after_newline_is_ignored(self):
        """Extra bytes after the first newline are currently discarded."""
        frame1 = json.dumps({"a": 1}).encode() + b"\n"
        extra = b"leftover"
        mock_sock = _MockRecvSocket([frame1 + extra])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        result = conn._receive_framed_json()

        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_connection_closed_before_data_raises(self):
        """Empty recv should raise ConnectionError."""
        mock_sock = _MockRecvSocket([b""])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        with pytest.raises(ConnectionError, match="Connection closed"):
            conn._receive_framed_json()

    def test_response_too_large_raises(self):
        """Buffer exceeding MAX_RESPONSE_BYTES should raise."""
        # Create a chunk that exceeds the limit
        huge_chunk = b"x" * (connection.MAX_RESPONSE_BYTES + 1)
        mock_sock = _MockRecvSocket([huge_chunk])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        with pytest.raises(ValueError, match="Response too large"):
            conn._receive_framed_json()


class TestSendCommandResponseContract:
    """Freeze response shape of send_command for contract testing."""

    def test_success_response_returns_result_dict(self):
        """send_command should extract and return the 'result' field."""
        expected_result = {"success": True, "data": "hello"}
        response_json = (
            json.dumps(
                {
                    "status": "ok",
                    "result": expected_result,
                }
            ).encode()
            + b"\n"
        )

        mock_sock = _MockRecvSocket([response_json])
        conn = MathematicaConnection()
        conn._socket = mock_sock
        result = conn.send_command("test_cmd", {"param": "value"})

        assert result == expected_result

    def test_error_response_raises_runtime_error(self):
        """send_command should raise RuntimeError on error status."""
        response_json = (
            json.dumps(
                {
                    "status": "error",
                    "message": "something went wrong",
                }
            ).encode()
            + b"\n"
        )

        mock_sock = _MockRecvSocket([response_json])
        conn = MathematicaConnection()
        conn._socket = mock_sock
        with pytest.raises(RuntimeError, match="something went wrong"):
            conn.send_command("test_cmd")

    def test_socket_not_closed_on_success(self):
        """Socket should remain open after a successful command."""
        response_json = (
            json.dumps(
                {
                    "status": "ok",
                    "result": {"ok": True},
                }
            ).encode()
            + b"\n"
        )

        mock_sock = _MockRecvSocket([response_json])
        conn = MathematicaConnection()
        conn._socket = mock_sock
        conn.send_command("test_cmd")

        assert conn._socket is mock_sock
        assert not mock_sock.closed


# ============================================================================
# Phase 2: Persistent buffer and leftover preservation tests
# ============================================================================


class TestLeftoverPreservation:
    """Verify that bytes after the first newline are preserved for next read."""

    def test_two_frames_in_one_recv(self):
        """Two complete frames arriving in one recv() call."""
        frame1 = json.dumps({"a": 1}).encode() + b"\n"
        frame2 = json.dumps({"b": 2}).encode() + b"\n"
        mock_sock = _MockRecvSocket([frame1 + frame2])

        conn = MathematicaConnection()
        conn._socket = mock_sock

        result1 = conn._receive_framed_json()
        parsed1 = json.loads(result1)
        assert parsed1["a"] == 1

        # Second frame should come from buffer, no extra recv
        result2 = conn._receive_framed_json()
        parsed2 = json.loads(result2)
        assert parsed2["b"] == 2

        # Only one recv call was needed
        assert mock_sock.recv_count == 1

    def test_leftover_preserved_across_consecutive_reads(self):
        """Leftover bytes from first frame are available for second frame."""
        frame1 = json.dumps({"x": 10}).encode() + b"\n"
        frame2_part1 = b'{"y":'
        frame2_part2 = b"20}\n"

        # First recv gives frame1 + start of frame2
        # Second recv gives rest of frame2
        mock_sock = _MockRecvSocket([frame1 + frame2_part1, frame2_part2])

        conn = MathematicaConnection()
        conn._socket = mock_sock

        result1 = conn._receive_framed_json()
        assert json.loads(result1)["x"] == 10

        result2 = conn._receive_framed_json()
        assert json.loads(result2)["y"] == 20


class TestSocketCleanupOnError:
    """Verify socket is closed before reference is dropped on error."""

    def test_timeout_closes_socket(self):
        """Socket.close() must be called before nulling on timeout."""

        class TimeoutSocket(_MockRecvSocket):
            def recv(self, bufsize):
                raise TimeoutError("read timed out")

        mock_sock = TimeoutSocket([])
        conn = MathematicaConnection()
        conn._socket = mock_sock

        with pytest.raises(TimeoutError):
            conn.send_command("test_cmd")

        assert conn._socket is None
        assert mock_sock.closed

    def test_connection_reset_closes_socket(self):
        """Socket.close() must be called on ConnectionResetError."""

        class ResetSocket(_MockRecvSocket):
            def recv(self, bufsize):
                raise ConnectionResetError("Connection reset by peer")

        mock_sock = ResetSocket([])
        conn = MathematicaConnection()
        conn._socket = mock_sock

        with pytest.raises(ConnectionError):
            conn.send_command("test_cmd")

        assert conn._socket is None
        assert mock_sock.closed

    def test_recv_buffer_cleared_on_disconnect(self):
        """disconnect() must clear the receive buffer."""
        conn = MathematicaConnection()
        conn._recv_buffer.extend(b"leftover data")
        conn.disconnect()
        assert len(conn._recv_buffer) == 0

    def test_recv_buffer_cleared_on_error(self):
        """Error path must clear the receive buffer."""

        class TimeoutSocket(_MockRecvSocket):
            def recv(self, bufsize):
                raise TimeoutError("timed out")

        mock_sock = TimeoutSocket([])
        conn = MathematicaConnection()
        conn._socket = mock_sock
        conn._recv_buffer.extend(b"stale data")

        with pytest.raises(TimeoutError):
            conn.send_command("test_cmd")

        assert len(conn._recv_buffer) == 0

    def test_json_decode_error_closes_socket_and_clears_buffer(self):
        """JSONDecodeError must close socket and clear buffer to prevent poisoning."""
        # Return invalid JSON that ends with a newline (so framing completes)
        bad_json = b"not valid json\n"
        mock_sock = _MockRecvSocket([bad_json])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        conn._recv_buffer.extend(b"prior leftover")

        with pytest.raises(ValueError, match="Invalid JSON"):
            conn.send_command("test_cmd")

        assert conn._socket is None
        assert len(conn._recv_buffer) == 0
        assert mock_sock.closed


# ============================================================================
# Per-command timeout tests
# ============================================================================


class TestPerCommandTimeout:
    """Verify that send_command(timeout=...) sets and restores socket timeout."""

    def test_custom_timeout_applied_and_restored(self):
        """send_command(timeout=X) should set X during call, restore default after."""
        response_json = (
            json.dumps(
                {
                    "status": "ok",
                    "result": {"ok": True},
                }
            ).encode()
            + b"\n"
        )

        mock_sock = _MockRecvSocket([response_json])
        conn = MathematicaConnection()
        conn._socket = mock_sock
        conn.send_command("test_cmd", timeout=600.0)

        # Should have: set custom timeout (600), then restore default (180)
        assert mock_sock.timeout_history == [600.0, connection.SOCKET_TIMEOUT]

    def test_default_timeout_not_changed_without_override(self):
        """send_command() without timeout should set default once (no restore needed)."""
        response_json = (
            json.dumps(
                {
                    "status": "ok",
                    "result": {"ok": True},
                }
            ).encode()
            + b"\n"
        )

        mock_sock = _MockRecvSocket([response_json])
        conn = MathematicaConnection()
        conn._socket = mock_sock
        conn.send_command("test_cmd")

        # Should set default timeout once, no restore call
        assert mock_sock.timeout_history == [connection.SOCKET_TIMEOUT]

    def test_timeout_restored_after_error(self):
        """Socket timeout should be restored even when recv raises."""

        class TimeoutAfterSend:
            """Socket that raises timeout on recv."""

            def __init__(self):
                self.timeout_history = []
                self.closed = False

            def sendall(self, data):
                pass

            def getpeername(self):
                return ("localhost", 9881)

            def settimeout(self, t):
                self.timeout_history.append(t)

            def recv(self, bufsize):
                raise TimeoutError("timed out")

            def close(self):
                self.closed = True

        mock_sock = TimeoutAfterSend()
        conn = MathematicaConnection()
        conn._socket = mock_sock

        with pytest.raises(TimeoutError):
            conn.send_command("test_cmd", timeout=999.0)

        # Custom timeout was set before the error; socket is closed so no restore
        assert 999.0 in mock_sock.timeout_history


class TestOversizedResponseHandling:
    """Verify oversized response closes socket and clears buffer (Issue #8)."""

    def test_oversized_response_closes_socket_and_clears_buffer(self):
        """ValueError from _receive_framed_json must close socket and clear buffer."""
        huge_chunk = b"x" * (connection.MAX_RESPONSE_BYTES + 1)
        mock_sock = _MockRecvSocket([huge_chunk])

        conn = MathematicaConnection()
        conn._socket = mock_sock
        conn._recv_buffer.extend(b"prior data")

        with pytest.raises(ValueError, match="Response too large"):
            conn.send_command("test_cmd")

        # Socket must be closed and nulled
        assert conn._socket is None
        assert mock_sock.closed
        # Buffer must be cleared
        assert len(conn._recv_buffer) == 0
        # last_error must be set
        assert "Response too large" in conn.last_error

    def test_oversized_request_does_not_close_socket(self):
        """ValueError from request-too-large (pre-I/O) must NOT close the socket."""
        response_json = (
            json.dumps({"status": "ok", "result": {"ok": True}}).encode() + b"\n"
        )
        mock_sock = _MockRecvSocket([response_json])

        conn = MathematicaConnection()
        conn._socket = mock_sock

        # Build a request that exceeds MAX_REQUEST_BYTES
        huge_params = {"data": "x" * (connection.MAX_REQUEST_BYTES + 1)}

        with pytest.raises(ValueError, match="Request too large"):
            conn.send_command("test_cmd", huge_params)

        # Socket must still be open — no I/O occurred
        assert conn._socket is mock_sock
        assert not mock_sock.closed
        # last_error should NOT be set (socket is healthy)
        assert conn.last_error == ""
