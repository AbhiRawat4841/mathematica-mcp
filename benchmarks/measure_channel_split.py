#!/usr/bin/env python3
"""Measure where a concurrent status call blocks during a long evaluation.

Splits the wait into "Python lock" vs "kernel preemptive link" to decide
whether a read-only second socket can help.

Two experiments, 3 iterations each, medians reported:
  (a) long FRONTEND-mode notebook eval. The addon fires the cell to the
      frontend (main link) then polls; the poll gives up ~0.5s in because the
      frontend's Evaluating flag is not observable on the preemptive link, so
      send_command RETURNS while the computation keeps running in the
      background. get_notebook_info measured:
        (a-i)   SAME connection, issued while send_command still holds the lock
                -> waits out that ~0.5-0.7s handshake only
        (a-ii)  SECOND raw socket, during the ongoing computation
                -> preemptive link free (~30-90ms)
        (a-iii) SAME connection, during the ongoing computation (lock already
                released) -> also fast; this is the linchpin: no long lock hold
  (b) long execute_code (Pause[4], runs entirely on the preemptive link and
      holds the lock the whole time). get_notebook_info via a SECOND raw socket
        -> preemptive queue busy, second socket cannot help (stall ~4s)

Run with: uv run python -m benchmarks.measure_channel_split
"""

import json
import os
import socket
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mathematica_mcp.connection import MathematicaConnection  # noqa: E402

DEFAULT_HOST = os.getenv("MATHEMATICA_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("MATHEMATICA_PORT", "9881"))
TOKEN = os.getenv("MATHEMATICA_MCP_TOKEN", "")
ITERATIONS = 3
PAUSE_SECONDS = 4


def raw_rpc(command, params=None, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=30.0):
    """One request/response on a fresh raw socket; returns (result, rtt_ms)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    try:
        request = {"command": command, "params": params or {}}
        if TOKEN:
            request["token"] = TOKEN
        t0 = time.monotonic()
        s.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                raise ConnectionError("closed before response")
            buf += chunk
        rtt_ms = (time.monotonic() - t0) * 1000
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8")), rtt_ms
    finally:
        s.close()


def _spawn_long_eval(conn, command, params):
    """Run a blocking send_command in a daemon thread; return the thread."""

    def _run():
        try:
            conn.send_command(command, params, timeout=PAUSE_SECONDS + 15)
        except Exception as e:  # noqa: BLE001 - measurement, just note it
            print(f"    [long-eval thread] {command} raised: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def experiment_a(conn, target):
    """Frontend-mode eval; probe same connection and second socket during it.

    Uses a Pause-based loop so the computation outlives send_command (which
    returns ~0.6s in once the poll gives up).
    """
    code = f"Do[Pause[0.1], {{{PAUSE_SECONDS * 10}}}]; 1 + 1"  # ~PAUSE_SECONDS on the frontend
    fired_at = time.monotonic()
    t = _spawn_long_eval(
        conn,
        "execute_code_notebook",
        {"code": code, "mode": "frontend", "max_wait": PAUSE_SECONDS + 10, **target},
    )
    time.sleep(0.1)  # eval now holds the primary lock during its poll handshake

    # (a-i) same connection while send_command still holds the lock: waits out
    # the ~0.5-0.7s handshake, then returns (NOT the full computation).
    t0 = time.monotonic()
    conn.send_command("get_notebook_info", dict(target))
    wait_same_handshake = (time.monotonic() - t0) * 1000

    # The computation is still running in the background now. Probe both paths.
    # (a-ii) second raw socket during the ongoing computation.
    _, rtt_second = raw_rpc("get_notebook_info", dict(target))

    # (a-iii) same connection during the ongoing computation (lock released).
    t0 = time.monotonic()
    conn.send_command("get_notebook_info", dict(target))
    wait_same_during = (time.monotonic() - t0) * 1000

    t.join(timeout=PAUSE_SECONDS + 15)
    # send_command returned long before the frontend finished; wait the eval
    # out HERE. Leaving it queued and closing the notebook later orphans it,
    # and the front end dumps the orphaned Out[..] into the Messages window.
    _wait_frontend_idle(conn, target, fired_at)
    return wait_same_handshake, rtt_second, wait_same_during


def _wait_frontend_idle(conn, target, fired_at):
    """Block until the scratch notebook's queued frontend eval is done.

    Belt one: never return before the eval's nominal duration has elapsed
    (kernel_busy can read False BEFORE the frontend starts evaluating - that
    race is why the addon's own poll gives up early). Belt two: poll
    kernel_busy until it reads False twice in a row or a deadline expires.
    """
    if fired_at is not None:
        remaining = PAUSE_SECONDS + 2 - (time.monotonic() - fired_at)
        if remaining > 0:
            time.sleep(remaining)
    idle_polls, deadline = 0, time.monotonic() + PAUSE_SECONDS + 15
    while idle_polls < 2 and time.monotonic() < deadline:
        info = conn.send_command("get_notebook_info", dict(target))
        busy = bool(((info or {}).get("state_delta") or {}).get("kernel_busy"))
        idle_polls = 0 if busy else idle_polls + 1
        time.sleep(0.5)


def experiment_b(conn, target):
    """execute_code Pause; probe via raw socket (preemptive link is busy)."""
    t = _spawn_long_eval(conn, "execute_code", {"code": f"Pause[{PAUSE_SECONDS}]"})
    time.sleep(0.3)

    _, rtt_second = raw_rpc("get_notebook_info", dict(target), timeout=PAUSE_SECONDS + 15)

    t.join(timeout=PAUSE_SECONDS + 15)
    return rtt_second


def run():
    conn = MathematicaConnection()
    if not conn.connect():
        print(f"ERROR: could not connect to addon: {conn.last_error}")
        return None

    # ping both paths to confirm the addon and a second socket are live.
    _, ping_second = raw_rpc("ping")
    print(f"Second-socket ping: {ping_second:.1f}ms")

    session_id = "channel-split-measure"
    created = conn.send_command("create_notebook", {"title": "Channel Split", "session_id": session_id})
    nb_id = created.get("id") if isinstance(created, dict) else None
    if not nb_id:
        sys.exit(f"ABORT: create_notebook returned no notebook id: {created!r}")

    # Belt: verify the EXPLICIT id (no session_id) resolves to OUR scratch
    # notebook before evaluating anything. If a kernel restart dropped the
    # session map or the notebook vanished, resolveNotebook would fall back to
    # the user's front-most window; catch that here instead of writing cells
    # into it.
    check = conn.send_command("get_notebook_info", {"notebook": nb_id})
    if not (isinstance(check, dict) and check.get("title") == "Channel Split"):
        conn.send_command("close_notebook", {"notebook": nb_id, "session_id": session_id, "save": False})
        sys.exit(f"ABORT: explicit notebook id did not resolve to 'Channel Split' (got {check!r})")

    # Every subsequent command carries BOTH the explicit id and the session id.
    target = {"notebook": nb_id, "session_id": session_id}

    a_i, a_ii, a_iii, b = [], [], [], []
    try:
        for i in range(ITERATIONS):
            print(f"\n[iter {i + 1}/{ITERATIONS}]")
            wait_handshake, rtt_second, wait_during = experiment_a(conn, target)
            print(f"  (a-i)   same-connection during send_command handshake: {wait_handshake:.1f}ms")
            print(f"  (a-ii)  second-socket during ongoing computation:     {rtt_second:.1f}ms")
            print(f"  (a-iii) same-connection during ongoing computation:   {wait_during:.1f}ms")
            a_i.append(wait_handshake)
            a_ii.append(rtt_second)
            a_iii.append(wait_during)

            rtt_b = experiment_b(conn, target)
            print(f"  (b)     second-socket during execute_code:            {rtt_b:.1f}ms")
            b.append(rtt_b)
    finally:
        # Always close the scratch notebook, even if an experiment raised -
        # but only after any still-queued eval has drained (see
        # _wait_frontend_idle: closing early orphans Out cells into Messages).
        try:
            _wait_frontend_idle(conn, target, fired_at=None)
        except Exception as e:  # noqa: BLE001 - cleanup must reach close
            print(f"    [cleanup] idle wait failed: {e}")
        conn.send_command("close_notebook", {"notebook": nb_id, "session_id": session_id, "save": False})
        conn.disconnect()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pause_seconds": PAUSE_SECONDS,
        "iterations": ITERATIONS,
        "experiments": {
            "a_i_same_connection_handshake_ms": {"samples": a_i, "median": statistics.median(a_i)},
            "a_ii_second_socket_during_compute_ms": {"samples": a_ii, "median": statistics.median(a_ii)},
            "a_iii_same_connection_during_compute_ms": {"samples": a_iii, "median": statistics.median(a_iii)},
            "b_second_socket_execute_code_ms": {"samples": b, "median": statistics.median(b)},
        },
    }

    print("\n" + "=" * 60)
    print("MEDIANS")
    print("=" * 60)
    print(f"(a-i)   same conn, frontend handshake window : {statistics.median(a_i):.1f}ms")
    print(f"(a-ii)  second socket, during computation    : {statistics.median(a_ii):.1f}ms")
    print(f"(a-iii) same conn, during computation        : {statistics.median(a_iii):.1f}ms")
    print(f"(b)     second socket, execute_code          : {statistics.median(b):.1f}ms")
    return results


def save(results, filename):
    if results:
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    results = run()
    if results:
        out = os.path.join(os.path.dirname(__file__), "results", "channel_split.json")
        save(results, out)
