#!/usr/bin/env python3
"""
Bottleneck diagnosis benchmark for Mathematica MCP evaluation latency.

Tests the hypothesis: when does restarting the MCP server vs restarting the
kernel fix slow notebook evaluations?

Architecture under test:
  Claude Code → Python MCP → TCP :9881 → MathematicaMCP.wl (SocketListen)
                                              │
                              ┌────────────────┴────────────────┐
                              │                                 │
                        PREEMPTIVE LINK                    MAIN LINK
                        • execute_code                    • evaluate_cell
                        • kernel-mode notebook            • frontend-mode notebook
                        • Dynamic/Manipulate              • Shift+Enter in notebook

Scenarios tested:
  A. Clean baseline — all paths
  B. Main link artificially blocked — preemptive should be fine
  C. Preemptive link artificially blocked — everything stalls
  D. Socket/connection degradation
  E. Kernel memory pressure
  F. RestartMCPServer[] recovery

Prerequisites:
  - Mathematica running with StartMCPServer[] active
  - A notebook open (for evaluate_cell tests)

Run:
  cd /path/to/mathematica-mcp
  python benchmarks/benchmark_bottleneck_diagnosis.py
"""

import json
import socket
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# ── Connection ──────────────────────────────────────────────────────────────

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9881


class DiagConnection:
    """Minimal socket connection to the Mathematica MCP addon."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=60.0, label=""):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.label = label
        self._socket = None

    def connect(self) -> bool:
        if self._socket:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            return True
        except Exception as e:
            print(f"  [{self.label}] Connect failed: {e}")
            self._socket = None
            return False

    def send_command(self, command: str, params: dict | None = None, timeout: float | None = None) -> dict:
        if not self._socket and not self.connect():
            raise ConnectionError("Not connected")

        request = {"command": command, "params": params or {}}
        request_bytes = (json.dumps(request) + "\n").encode("utf-8")

        if timeout is not None:
            self._socket.settimeout(timeout)
        else:
            self._socket.settimeout(self.timeout)

        self._socket.sendall(request_bytes)

        buffer = b""
        while b"\n" not in buffer:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed by server")
            buffer += chunk

        line = buffer.partition(b"\n")[0]
        response = json.loads(line.decode("utf-8"))

        if response.get("status") == "error":
            raise RuntimeError(f"Addon error: {response.get('message')}")

        return response.get("result", {})

    def close(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def reconnect(self) -> bool:
        self.close()
        return self.connect()


# ── Measurement helpers ─────────────────────────────────────────────────────


@dataclass
class Measurement:
    name: str
    times_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.times_ms) > 0

    @property
    def mean(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else float("inf")

    @property
    def median(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else float("inf")

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    def summary(self) -> str:
        if not self.times_ms:
            errs = "; ".join(self.errors[:3])
            return f"FAILED ({errs})"
        parts = [f"mean={self.mean:.1f}ms", f"median={self.median:.1f}ms"]
        if len(self.times_ms) > 1:
            parts.append(f"stdev={self.stdev:.1f}ms")
        parts.append(f"min={min(self.times_ms):.1f}ms")
        parts.append(f"max={max(self.times_ms):.1f}ms")
        parts.append(f"n={len(self.times_ms)}")
        return ", ".join(parts)


def measure(conn: DiagConnection, name: str, command: str, params: dict | None = None,
            iterations: int = 5, warmup: int = 1, timeout: float | None = None) -> Measurement:
    """Run a command repeatedly and collect timing."""
    m = Measurement(name=name)

    for _ in range(warmup):
        try:
            conn.send_command(command, params, timeout=timeout)
        except Exception as e:
            m.errors.append(f"warmup: {e}")
            return m

    for i in range(iterations):
        t0 = time.perf_counter()
        try:
            conn.send_command(command, params, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            m.times_ms.append(elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            m.errors.append(f"iter {i}: {e} ({elapsed_ms:.0f}ms)")

    return m


def print_measurement(m: Measurement, indent: str = "    "):
    """Print a measurement result with visual indicator."""
    if m.ok:
        avg = m.mean
        if avg < 50:
            indicator = "●"  # fast
        elif avg < 200:
            indicator = "◐"  # moderate
        elif avg < 1000:
            indicator = "◑"  # slow
        else:
            indicator = "○"  # very slow
        print(f"{indent}{indicator} {m.name}: {m.summary()}")
    else:
        print(f"{indent}✗ {m.name}: {m.summary()}")


# ── Scenario implementations ───────────────────────────────────────────────

def scenario_a_baseline(conn: DiagConnection) -> list[Measurement]:
    """Clean-state baseline for all evaluation paths."""
    print("\n" + "=" * 70)
    print("SCENARIO A: Clean-state baseline")
    print("=" * 70)
    print("  Tests each evaluation path with no contention.\n")

    results = []

    # 1. Ping (pure socket round-trip, no computation)
    m = measure(conn, "ping (socket round-trip)", "ping", iterations=10, warmup=2)
    print_measurement(m)
    results.append(m)

    # 2. execute_code trivial (preemptive link, no notebook)
    m = measure(conn, "execute_code 1+1 (preemptive)", "execute_code",
                {"code": "1+1"}, iterations=10, warmup=2)
    print_measurement(m)
    results.append(m)

    # 3. execute_code moderate
    m = measure(conn, "execute_code Integrate (preemptive)", "execute_code",
                {"code": "Integrate[Sin[x]^2, x]"}, iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    # 4. execute_code with graphics (heavier preemptive)
    m = measure(conn, "execute_code Plot (preemptive)", "execute_code",
                {"code": "Plot[Sin[x], {x, 0, 2 Pi}]"}, iterations=3, warmup=1)
    print_measurement(m)
    results.append(m)

    # 5. execute_code_notebook kernel mode (preemptive + cell write)
    m = measure(conn, "exec_notebook kernel 1+1", "execute_code_notebook",
                {"code": "1+1", "mode": "kernel", "session_id": "diag-bench"}, iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    # 6. evaluate_cell (main link — need a cell to evaluate)
    #    Write a cell first, then evaluate it
    print("\n    [Setting up evaluate_cell test...]")
    try:
        write_result = conn.send_command("write_cell", {
            "content": "2+2", "style": "Input", "session_id": "diag-bench"
        })
        cell_id = write_result.get("cell_id")
        if cell_id:
            m = measure(conn, "evaluate_cell 2+2 (main link)", "evaluate_cell",
                        {"cell_id": cell_id, "max_wait": 10, "session_id": "diag-bench"},
                        iterations=5, warmup=1)
            print_measurement(m)
            results.append(m)
        else:
            print("    ✗ Could not get cell_id for evaluate_cell test")
    except Exception as e:
        print(f"    ✗ evaluate_cell setup failed: {e}")

    # 7. execute_code_notebook frontend mode (main link)
    m = measure(conn, "exec_notebook frontend 1+1 (main link)", "execute_code_notebook",
                {"code": "1+1", "mode": "frontend", "max_wait": 10, "session_id": "diag-bench"},
                iterations=3, warmup=1)
    print_measurement(m)
    results.append(m)

    return results


def scenario_b_main_link_blocked(conn: DiagConnection) -> list[Measurement]:
    """Block the main link and test whether preemptive path is unaffected."""
    print("\n" + "=" * 70)
    print("SCENARIO B: Main link blocked")
    print("=" * 70)
    print("  A long computation occupies the main evaluation queue.")
    print("  Hypothesis: execute_code (preemptive) stays fast,")
    print("              evaluate_cell (main link) becomes slow.\n")

    results = []

    # Start a long-running computation on the main link.
    # Use evaluate_cell with max_wait=0.5 so the handler returns quickly
    # but the cell keeps running on the main link for ~15 seconds.
    print("    [Injecting Pause[15] on main link via evaluate_cell...]")
    try:
        # Write a cell with long computation
        write_result = conn.send_command("write_cell", {
            "content": "Pause[15]; \"main_link_unblocked\"",
            "style": "Input", "session_id": "diag-bench"
        })
        blocker_cell_id = write_result.get("cell_id")
        if not blocker_cell_id:
            print("    ✗ Could not create blocker cell")
            return results

        # Start evaluation but return after 0.5s (cell keeps running on main link)
        conn.send_command("evaluate_cell", {
            "cell_id": blocker_cell_id, "max_wait": 0.5, "session_id": "diag-bench"
        })
        print("    [Main link now blocked for ~15 seconds]")
        time.sleep(0.5)  # small buffer for eval to start

    except Exception as e:
        print(f"    ✗ Failed to block main link: {e}")
        return results

    # Now test while main link is blocked
    block_start = time.perf_counter()

    # Preemptive path should be fast
    m = measure(conn, "execute_code 1+1 WHILE main blocked (preemptive)", "execute_code",
                {"code": "1+1"}, iterations=5, warmup=0)
    print_measurement(m)
    results.append(m)

    m = measure(conn, "ping WHILE main blocked", "ping", iterations=5, warmup=0)
    print_measurement(m)
    results.append(m)

    # Main link path should be slow/blocked
    # Write another cell to evaluate
    try:
        write_result = conn.send_command("write_cell", {
            "content": "3+3", "style": "Input", "session_id": "diag-bench"
        })
        test_cell_id = write_result.get("cell_id")
        if test_cell_id:
            m = measure(conn, "evaluate_cell 3+3 WHILE main blocked", "evaluate_cell",
                        {"cell_id": test_cell_id, "max_wait": 3, "session_id": "diag-bench"},
                        iterations=1, warmup=0)
            print_measurement(m)
            results.append(m)
            # Check: did the evaluate_cell return quickly (because max_wait expired)
            # but the cell is still queued? That's the expected behavior.
            if m.ok and m.mean > 2500:
                print("      ^ Timed out waiting — cell queued behind Pause[15]. EXPECTED.")
            elif m.ok and m.mean < 500:
                print("      ^ Returned fast — either main link wasn't blocked or max_wait triggered return.")
    except Exception as e:
        print(f"    ✗ evaluate_cell test during block failed: {e}")

    # Check if Evaluating is true on the notebook
    try:
        status = conn.send_command("get_notebook_info", {"session_id": "diag-bench"})
        evaluating = status.get("evaluating", "unknown")
        print(f"\n    Notebook evaluating status: {evaluating}")
    except Exception as e:
        print(f"    Could not check notebook status: {e}")

    elapsed = time.perf_counter() - block_start
    remaining = max(0, 15 - elapsed - 1)
    if remaining > 0:
        print(f"\n    [Waiting {remaining:.0f}s for main link to clear...]")
        time.sleep(remaining)

    # Verify main link is free again
    m = measure(conn, "execute_code 1+1 AFTER main unblocked", "execute_code",
                {"code": "1+1"}, iterations=3, warmup=0)
    print_measurement(m)
    results.append(m)

    return results


def scenario_c_preemptive_blocked(conn: DiagConnection) -> list[Measurement]:
    """Block the preemptive link and test that EVERYTHING stalls.

    This is the key scenario: if execute_code runs a long computation,
    the SocketListen handler is occupied, and ALL other MCP commands
    (including ping) queue behind it.
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: Preemptive link blocked")
    print("=" * 70)
    print("  A long execute_code occupies the preemptive evaluation link.")
    print("  Hypothesis: ALL MCP commands stall until it completes.\n")

    results = []
    blocker_done = threading.Event()
    blocker_result = {"elapsed_ms": None, "error": None}

    def blocker_thread():
        """Send a long execute_code on a SEPARATE socket (conn1)."""
        conn1 = DiagConnection(label="blocker", timeout=30.0)
        if not conn1.connect():
            blocker_result["error"] = "Could not connect blocker socket"
            blocker_done.set()
            return
        try:
            t0 = time.perf_counter()
            # This runs Pause[8] on the preemptive link, blocking the handler
            conn1.send_command("execute_code", {"code": "Pause[8]; \"preemptive_done\""}, timeout=30.0)
            blocker_result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            blocker_result["error"] = str(e)
        finally:
            conn1.close()
            blocker_done.set()

    # Start the blocker on a separate thread/socket
    print("    [Starting Pause[8] on preemptive link via separate socket...]")
    t = threading.Thread(target=blocker_thread, daemon=True)
    t.start()
    time.sleep(1.0)  # give it time to reach the kernel

    # Now try to use the MAIN connection — should be blocked
    print("    [Testing commands on primary socket while preemptive is blocked...]")

    # Ping should be very slow (queued behind the Pause[8])
    m = measure(conn, "ping WHILE preemptive blocked", "ping",
                iterations=1, warmup=0, timeout=15.0)
    print_measurement(m)
    results.append(m)
    if m.ok and m.mean > 3000:
        print("      ^ Slow ping confirms preemptive link was blocked. EXPECTED.")

    # execute_code should also be slow
    m = measure(conn, "execute_code 1+1 WHILE preemptive blocked", "execute_code",
                {"code": "1+1"}, iterations=1, warmup=0, timeout=15.0)
    print_measurement(m)
    results.append(m)

    # Wait for blocker to finish
    blocker_done.wait(timeout=15.0)
    if blocker_result["error"]:
        print(f"    Blocker thread error: {blocker_result['error']}")
    else:
        print(f"    Blocker thread completed in {blocker_result['elapsed_ms']:.0f}ms")

    # Post-block: should be fast again
    m = measure(conn, "ping AFTER preemptive unblocked", "ping",
                iterations=3, warmup=0)
    print_measurement(m)
    results.append(m)

    return results


def scenario_d_socket_degradation(conn: DiagConnection) -> list[Measurement]:
    """Test socket/connection health scenarios."""
    print("\n" + "=" * 70)
    print("SCENARIO D: Socket/connection health")
    print("=" * 70)
    print("  Tests reconnection overhead and stale-socket detection.\n")

    results = []

    # 1. Fresh connection latency (cold connect + first command)
    print("    [Testing cold connection + first command...]")
    cold_conn = DiagConnection(label="cold", timeout=10.0)
    t0 = time.perf_counter()
    cold_conn.connect()
    cold_conn.send_command("ping")
    cold_ms = (time.perf_counter() - t0) * 1000
    cold_conn.close()
    m = Measurement(name="cold connect + ping", times_ms=[cold_ms])
    print_measurement(m)
    results.append(m)

    # 2. Warm connection (already connected)
    m = measure(conn, "warm ping (reused socket)", "ping", iterations=10, warmup=2)
    print_measurement(m)
    results.append(m)

    # 3. Rapid reconnection cycle
    print("    [Testing rapid disconnect/reconnect cycle...]")
    reconnect_times = []
    for i in range(5):
        rc = DiagConnection(label=f"reconnect-{i}", timeout=10.0)
        t0 = time.perf_counter()
        rc.connect()
        rc.send_command("execute_code", {"code": "1+1"})
        elapsed = (time.perf_counter() - t0) * 1000
        reconnect_times.append(elapsed)
        rc.close()
    m = Measurement(name="reconnect + execute_code cycle", times_ms=reconnect_times)
    print_measurement(m)
    results.append(m)

    # 4. Many concurrent sockets (simulates MCP server under load)
    print("    [Testing 5 concurrent connections...]")
    concurrent_results = []

    def concurrent_ping(idx):
        c = DiagConnection(label=f"concurrent-{idx}", timeout=10.0)
        if not c.connect():
            return None
        t0 = time.perf_counter()
        try:
            c.send_command("execute_code", {"code": f"{idx}+1"})
            return (time.perf_counter() - t0) * 1000
        except Exception:
            return None
        finally:
            c.close()

    threads = []
    thread_results = [None] * 5
    for i in range(5):
        def worker(idx=i):
            thread_results[idx] = concurrent_ping(idx)
        t = threading.Thread(target=worker)
        threads.append(t)

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    wall_ms = (time.perf_counter() - t0) * 1000

    valid = [r for r in thread_results if r is not None]
    m = Measurement(name="5 concurrent execute_code (individual)", times_ms=valid)
    print_measurement(m)
    results.append(m)
    print(f"      Wall-clock for all 5: {wall_ms:.0f}ms")
    if valid:
        serial_estimate = sum(valid)
        print(f"      Sum of individual times: {serial_estimate:.0f}ms")
        if wall_ms > serial_estimate * 0.8:
            print("      ^ Near-serial execution — confirms preemptive link is single-threaded")
        else:
            print("      ^ Some parallelism detected")

    return results


def scenario_e_memory_pressure(conn: DiagConnection) -> list[Measurement]:
    """Test latency under kernel memory pressure."""
    print("\n" + "=" * 70)
    print("SCENARIO E: Kernel memory pressure")
    print("=" * 70)
    print("  Creates large data in kernel, measures latency impact.\n")

    results = []

    # Baseline before loading data
    m = measure(conn, "execute_code 1+1 (before load)", "execute_code",
                {"code": "1+1"}, iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    # Create ~100MB of data in kernel
    print("\n    [Loading ~100MB table into kernel...]")
    try:
        t0 = time.perf_counter()
        conn.send_command("execute_code", {
            "code": "$mcpBenchData = RandomReal[1, {1000, 1000}]; ByteCount[$mcpBenchData]"
        }, timeout=30.0)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"    Loaded in {load_ms:.0f}ms")
    except Exception as e:
        print(f"    ✗ Failed to load data: {e}")
        return results

    # Measure after loading
    m = measure(conn, "execute_code 1+1 (after 100MB load)", "execute_code",
                {"code": "1+1"}, iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    # Trigger computation that touches the large data
    m = measure(conn, "execute_code Total[$mcpBenchData]", "execute_code",
                {"code": "Total[$mcpBenchData, 2]"}, iterations=3, warmup=0, timeout=30.0)
    print_measurement(m)
    results.append(m)

    # Cleanup
    print("\n    [Cleaning up benchmark data...]")
    try:
        conn.send_command("execute_code", {"code": "Remove[$mcpBenchData]"})
    except Exception:
        pass

    return results


def scenario_f_restart_mcp(conn: DiagConnection) -> list[Measurement]:
    """Test RestartMCPServer[] recovery and measure reconnection cost."""
    print("\n" + "=" * 70)
    print("SCENARIO F: RestartMCPServer[] recovery")
    print("=" * 70)
    print("  Restarts the socket listener, measures reconnection overhead.")
    print("  NOTE: This does NOT restart the kernel — only the TCP listener.\n")

    results = []

    # Pre-restart baseline
    m = measure(conn, "ping (before restart)", "ping", iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    # Trigger RestartMCPServer[] via execute_code
    # This will: StopMCPServer[] → Pause[0.5] → StartMCPServer[]
    # The current socket will be closed as a side effect
    print("\n    [Calling RestartMCPServer[]...]")
    try:
        conn.send_command("execute_code", {"code": "RestartMCPServer[]"}, timeout=10.0)
    except (ConnectionError, BrokenPipeError, ConnectionResetError, OSError):
        # Expected: the socket gets closed during restart
        pass
    except Exception as e:
        print(f"    Warning during restart: {type(e).__name__}: {e}")

    # Wait for server to come back up
    time.sleep(2.0)

    # Reconnect
    print("    [Reconnecting after restart...]")
    t0 = time.perf_counter()
    if not conn.reconnect():
        # Try a couple more times
        for attempt in range(3):
            time.sleep(1.0)
            if conn.reconnect():
                break
        else:
            print("    ✗ Could not reconnect after RestartMCPServer[]")
            return results
    reconnect_ms = (time.perf_counter() - t0) * 1000
    m = Measurement(name="reconnect after RestartMCPServer[]", times_ms=[reconnect_ms])
    print_measurement(m)
    results.append(m)

    # Post-restart baseline
    m = measure(conn, "ping (after restart)", "ping", iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    m = measure(conn, "execute_code 1+1 (after restart)", "execute_code",
                {"code": "1+1"}, iterations=5, warmup=1)
    print_measurement(m)
    results.append(m)

    return results


# ── Diagnosis summary ──────────────────────────────────────────────────────

def print_diagnosis(all_results: dict[str, list[Measurement]]):
    """Print actionable diagnosis based on collected measurements."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    print("""
┌──────────────────────┬──────────────────────────┬────────────────────┬──────────────┐
│ Symptom              │ Root cause               │ Fix                │ Scenario     │
├──────────────────────┼──────────────────────────┼────────────────────┼──────────────┤
│ 1+1 "Running..." in  │ Main link queue blocked  │ Restart KERNEL     │ B confirmed  │
│ notebook (Shift+Ent) │ by previous computation  │                    │ if preempt   │
│                      │                          │                    │ was fast     │
├──────────────────────┼──────────────────────────┼────────────────────┼──────────────┤
│ ALL MCP commands     │ Preemptive link blocked  │ Restart KERNEL     │ C confirmed  │
│ slow (even ping)     │ by long execute_code     │ (kills addon too)  │ if all cmds  │
│                      │ or evaluate_cell polling │                    │ were slow    │
├──────────────────────┼──────────────────────────┼────────────────────┼──────────────┤
│ MCP timeout/error    │ TCP socket stale or      │ RestartMCPServer[] │ D confirmed  │
│ but notebook works   │ addon connection dropped │ (or reconnect)     │ if reconnect │
│ when typed manually  │                          │                    │ fixes it     │
├──────────────────────┼──────────────────────────┼────────────────────┼──────────────┤
│ Everything slow      │ Kernel memory pressure   │ Restart KERNEL     │ E shows if   │
│ (gradual onset)      │ or accumulated state     │                    │ load causes  │
│                      │                          │                    │ degradation  │
├──────────────────────┼──────────────────────────┼────────────────────┼──────────────┤
│ Manipulate/Dynamic   │ Preemptive link free,    │ Restart KERNEL     │ This IS the  │
│ works but cells      │ main link blocked        │ to clear main      │ Chebyshev    │
│ show "Running..."    │                          │ link queue         │ paradox      │
└──────────────────────┴──────────────────────────┴────────────────────┴──────────────┘

Key architectural insight:
  • SocketListen handlers run on the PREEMPTIVE link (can interrupt main link)
  • execute_code runs ENTIRELY on the preemptive link (fast if main is blocked)
  • evaluate_cell dispatches to MAIN link, then polls with Pause[] loop
    → The Pause[] loop ALSO blocks the preemptive link for up to max_wait seconds!
    → This is a DOUBLE penalty: blocks main link AND preemptive link
  • Dynamic/Manipulate content uses the preemptive link → works when main is blocked

When to restart what:
  • Restart KERNEL when: cells show "Running...", gradual slowdown, accumulated state
    ⚠ This also kills the MCP server — must re-run StartMCPServer[] after
  • RestartMCPServer[] when: MCP timeouts but manual evaluation works fine
    ✓ Preserves kernel state, variables, loaded packages
  • Restart NEITHER (just wait): after a long execute_code finishes
""")

    # Show scenario-specific results
    for scenario_name, measurements in all_results.items():
        if measurements:
            print(f"\n  {scenario_name}:")
            for m in measurements:
                print_measurement(m, indent="    ")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Mathematica MCP Bottleneck Diagnosis Benchmark                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("Prerequisites: Mathematica running with StartMCPServer[] active")
    print()

    conn = DiagConnection(label="primary", timeout=30.0)
    if not conn.connect():
        print("ERROR: Could not connect to Mathematica addon on port 9881.")
        print("Make sure Mathematica is running and StartMCPServer[] has been called.")
        sys.exit(1)

    # Verify connection
    try:
        result = conn.send_command("ping")
        print(f"Connected to addon: {result}")
    except Exception as e:
        print(f"Ping failed: {e}")
        sys.exit(1)

    # Create a notebook session for tests that need one
    print("\n[Setup] Creating diagnostic notebook session...")
    try:
        conn.send_command("create_notebook", {"title": "Bottleneck Diagnosis", "session_id": "diag-bench"})
        conn.send_command("execute_code_notebook", {
            "code": "(* Bottleneck diagnosis session *)", "mode": "kernel", "session_id": "diag-bench"
        })
        print("  Notebook session ready")
    except Exception as e:
        print(f"  Warning: Could not create notebook session: {e}")
        print("  evaluate_cell tests may fail")

    all_results = {}

    # Select which scenarios to run
    scenarios = sys.argv[1:] if len(sys.argv) > 1 else ["a", "b", "c", "d", "e", "f"]

    if "a" in scenarios:
        all_results["A: Baseline"] = scenario_a_baseline(conn)

    if "b" in scenarios:
        all_results["B: Main link blocked"] = scenario_b_main_link_blocked(conn)

    if "c" in scenarios:
        all_results["C: Preemptive link blocked"] = scenario_c_preemptive_blocked(conn)

    if "d" in scenarios:
        all_results["D: Socket health"] = scenario_d_socket_degradation(conn)

    if "e" in scenarios:
        all_results["E: Memory pressure"] = scenario_e_memory_pressure(conn)

    if "f" in scenarios:
        all_results["F: MCP restart"] = scenario_f_restart_mcp(conn)

    # Print diagnosis
    print_diagnosis(all_results)

    # Cleanup
    print("\n[Cleanup]")
    try:
        conn.send_command("close_notebook", {"session_id": "diag-bench", "save": False})
        print("  Closed diagnostic notebook")
    except Exception as e:
        print(f"  Cleanup warning: {e}")
    conn.close()
    print("  Done.")


if __name__ == "__main__":
    main()
