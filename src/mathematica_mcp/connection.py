import json
import socket
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("mathematica_mcp.connection")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9881
SOCKET_TIMEOUT = 180.0


@dataclass
class MathematicaConnection:
    host: str = field(
        default_factory=lambda: os.getenv("MATHEMATICA_HOST", DEFAULT_HOST)
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("MATHEMATICA_PORT", DEFAULT_PORT))
    )
    _socket: Optional[socket.socket] = field(default=None, repr=False)

    def connect(self) -> bool:
        if self._socket:
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(SOCKET_TIMEOUT)
            self._socket.connect((self.host, self.port))
            logger.info(f"Connected to Mathematica at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Mathematica: {e}")
            self._socket = None
            return False

    def disconnect(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            logger.info("Disconnected from Mathematica")

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
        if not self._socket and not self.connect():
            raise ConnectionError(
                "Not connected to Mathematica addon. "
                "Make sure Mathematica is running with the addon loaded. "
                "Run StartMCPServer[] in Mathematica."
            )

        request = {"command": command, "params": params or {}}
        assert self._socket is not None

        try:
            request_bytes = json.dumps(request).encode("utf-8")
            self._socket.sendall(request_bytes)

            response_bytes = self._receive_complete_json()
            response = json.loads(response_bytes.decode("utf-8"))

            if response.get("status") == "error":
                error_msg = response.get("message", "Unknown error from Mathematica")
                raise RuntimeError(f"Mathematica error: {error_msg}")

            return response.get("result", {})

        except socket.timeout:
            self._socket = None
            raise TimeoutError(
                "Timeout waiting for Mathematica response. The computation may be too slow."
            )
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            self._socket = None
            raise ConnectionError(f"Connection to Mathematica lost: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Mathematica: {e}")

    def _receive_complete_json(self, buffer_size: int = 8192) -> bytes:
        chunks: list[bytes] = []
        assert self._socket is not None

        while True:
            chunk = self._socket.recv(buffer_size)
            if not chunk:
                if not chunks:
                    raise ConnectionError("Connection closed before receiving data")
                break

            chunks.append(chunk)

            try:
                data = b"".join(chunks)
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                continue

        data = b"".join(chunks)
        json.loads(data.decode("utf-8"))
        return data


_global_connection: Optional[MathematicaConnection] = None


def get_mathematica_connection() -> MathematicaConnection:
    global _global_connection

    if _global_connection is None:
        _global_connection = MathematicaConnection()

    if not _global_connection.is_connected():
        try:
            result = _global_connection.send_command("ping")
            if not result.get("pong"):
                raise ConnectionError("Ping failed")
        except Exception:
            _global_connection = MathematicaConnection()
            if not _global_connection.connect():
                raise ConnectionError(
                    "Could not connect to Mathematica addon."
                    "Ensure Mathematica is running and execute: StartMCPServer[]"
                )

    return _global_connection


def close_connection():
    global _global_connection
    if _global_connection:
        _global_connection.disconnect()
        _global_connection = None
