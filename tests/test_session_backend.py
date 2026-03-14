from __future__ import annotations

from mathematica_mcp import session


class _FakeKernelSession:
    def __init__(self):
        self.evaluate_calls = 0

    def evaluate(self, expr):
        self.evaluate_calls += 1
        return 1

    def terminate(self):
        pass


def test_get_kernel_session_skips_health_check_within_interval(monkeypatch):
    fake = _FakeKernelSession()

    monkeypatch.setattr(session, "_kernel_session", fake)
    monkeypatch.setattr(session, "_use_wolframscript", False)
    monkeypatch.setattr(session, "_last_kernel_health_check", 100.0)
    monkeypatch.setattr(session.time, "monotonic", lambda: 101.0)

    result = session.get_kernel_session()

    assert result is fake
    assert fake.evaluate_calls == 0


def test_get_kernel_session_rechecks_after_interval(monkeypatch):
    fake = _FakeKernelSession()

    monkeypatch.setattr(session, "_kernel_session", fake)
    monkeypatch.setattr(session, "_use_wolframscript", False)
    monkeypatch.setattr(session, "_last_kernel_health_check", 100.0)
    monkeypatch.setattr(
        session.time,
        "monotonic",
        lambda: 100.0 + session.KERNEL_HEALTH_CHECK_INTERVAL + 1.0,
    )

    result = session.get_kernel_session()

    assert result is fake
    assert fake.evaluate_calls == 1
