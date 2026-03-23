import json
import socket
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("mathematica_mcp.connection")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9881
SOCKET_TIMEOUT = 180.0
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 5 * 1024 * 1024


@dataclass
class MathematicaConnection:
    host: str = field(
        default_factory=lambda: os.getenv("MATHEMATICA_HOST", DEFAULT_HOST)
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("MATHEMATICA_PORT", DEFAULT_PORT))
    )
    _socket: Optional[socket.socket] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _auth_token: str = field(
        default_factory=lambda: os.getenv("MATHEMATICA_MCP_TOKEN", "")
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: float(os.getenv("MATHEMATICA_RETRY_BACKOFF", "2.0"))
    )
    _next_retry_at: float = field(default=0.0, repr=False)
    _last_error: str = field(default="", repr=False)
    _recv_buffer: bytearray = field(default_factory=bytearray, repr=False)
    _lock_acquisitions: int = field(default=0, repr=False)
    _lock_wait_times: list = field(default_factory=list, repr=False)
    _lock_hold_times: list = field(default_factory=list, repr=False)

    _MAX_METRIC_SAMPLES: int = field(default=200, repr=False)

    def connect(self) -> bool:
        if self._socket:
            return True
        if not self.can_retry():
            return False

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(SOCKET_TIMEOUT)
            self._socket.connect((self.host, self.port))
            self._last_error = ""
            self._next_retry_at = 0.0
            self._recv_buffer.clear()
            logger.info(f"Connected to Mathematica at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Mathematica: {e}")
            self._last_error = str(e)
            self._next_retry_at = time.monotonic() + self.retry_backoff_seconds
            self._socket = None
            return False

    def disconnect(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._recv_buffer.clear()
        logger.info("Disconnected from Mathematica")

    def _close_socket_on_error(self) -> None:
        """Close the socket and clear buffer on error, before nulling."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._recv_buffer.clear()

    def can_retry(self) -> bool:
        return time.monotonic() >= self._next_retry_at

    @property
    def last_error(self) -> str:
        return self._last_error

    def is_connected(self) -> bool:
        if not self._socket:
            return False
        try:
            self._socket.getpeername()
            return True
        except Exception:
            self._socket = None
            return False

    def send_command(
        self, command: str, params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        lock_wait_start = time.monotonic()
        with self._lock:
            lock_acquired_at = time.monotonic()
            wait_ms = (lock_acquired_at - lock_wait_start) * 1000
            self._lock_acquisitions += 1
            self._lock_wait_times.append(wait_ms)
            if len(self._lock_wait_times) > self._MAX_METRIC_SAMPLES:
                del self._lock_wait_times[
                    : len(self._lock_wait_times) - self._MAX_METRIC_SAMPLES
                ]
            try:
                if not self._socket and not self.connect():
                    raise ConnectionError(
                        "Not connected to Mathematica addon. "
                        "Make sure Mathematica is running with the addon loaded. "
                        "Run StartMCPServer[] in Mathematica."
                    )

                request = {"command": command, "params": params or {}}
                if self._auth_token:
                    request["token"] = self._auth_token

                request_bytes = (json.dumps(request) + "\n").encode("utf-8")
                if len(request_bytes) > MAX_REQUEST_BYTES:
                    raise ValueError("Request too large")

                assert self._socket is not None

                self._socket.sendall(request_bytes)

                response_bytes = self._receive_framed_json()
                response = json.loads(response_bytes.decode("utf-8"))

                if response.get("status") == "error":
                    error_msg = response.get(
                        "message", "Unknown error from Mathematica"
                    )
                    raise RuntimeError(f"Mathematica error: {error_msg}")

                return response.get("result", {})

            except socket.timeout:
                self._last_error = "Timeout waiting for Mathematica response"
                self._next_retry_at = time.monotonic() + self.retry_backoff_seconds
                self._close_socket_on_error()
                raise TimeoutError(
                    "Timeout waiting for Mathematica response. The computation may be too slow."
                )
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                self._last_error = str(e)
                self._next_retry_at = time.monotonic() + self.retry_backoff_seconds
                self._close_socket_on_error()
                raise ConnectionError(f"Connection to Mathematica lost: {e}")
            except json.JSONDecodeError as e:
                self._close_socket_on_error()
                raise ValueError(f"Invalid JSON response from Mathematica: {e}")
            finally:
                hold_ms = (time.monotonic() - lock_acquired_at) * 1000
                self._lock_hold_times.append(hold_ms)
                if len(self._lock_hold_times) > self._MAX_METRIC_SAMPLES:
                    del self._lock_hold_times[
                        : len(self._lock_hold_times) - self._MAX_METRIC_SAMPLES
                    ]

    def get_lock_metrics(self) -> dict[str, Any]:
        """Return lock contention metrics for diagnostic reporting."""
        from .telemetry import _percentile

        return {
            "acquisitions": self._lock_acquisitions,
            "wait_p50_ms": round(_percentile(self._lock_wait_times, 50), 2),
            "wait_p95_ms": round(_percentile(self._lock_wait_times, 95), 2),
            "hold_p50_ms": round(_percentile(self._lock_hold_times, 50), 2),
            "hold_p95_ms": round(_percentile(self._lock_hold_times, 95), 2),
        }

    def _receive_framed_json(self, buffer_size: int = 65536) -> bytes:
        """Read one newline-delimited JSON frame.

        Uses a persistent _recv_buffer so leftover bytes from a previous
        recv are available for the next frame.
        """
        assert self._socket is not None

        # Check if a complete frame is already in the buffer
        idx = self._recv_buffer.find(b"\n")
        if idx != -1:
            line = bytes(self._recv_buffer[:idx])
            del self._recv_buffer[:idx + 1]
            return line

        while True:
            chunk = self._socket.recv(buffer_size)
            if not chunk:
                raise ConnectionError("Connection closed before receiving data")

            self._recv_buffer.extend(chunk)
            if len(self._recv_buffer) > MAX_RESPONSE_BYTES:
                raise ValueError("Response too large")

            idx = self._recv_buffer.find(b"\n")
            if idx != -1:
                line = bytes(self._recv_buffer[:idx])
                del self._recv_buffer[:idx + 1]  # preserve remainder
                return line


_global_connection: Optional[MathematicaConnection] = None


def get_mathematica_connection() -> MathematicaConnection:
    global _global_connection

    if _global_connection is None:
        _global_connection = MathematicaConnection()

    if not _global_connection.is_connected():
        if not _global_connection.connect():
            error_detail = f" Last error: {_global_connection.last_error}." if _global_connection.last_error else ""
            raise ConnectionError(
                "Could not connect to Mathematica addon. "
                "Ensure Mathematica is running and execute: StartMCPServer[]."
                f"{error_detail}"
            )

    return _global_connection


def close_connection():
    global _global_connection
    if _global_connection:
        _global_connection.disconnect()
        _global_connection = None
