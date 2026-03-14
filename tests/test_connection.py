from __future__ import annotations

import socket

from mathematica_mcp import connection


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
