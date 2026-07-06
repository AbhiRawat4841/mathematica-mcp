#!/usr/bin/env python3
"""Attribute the ~30ms addon transport floor. MEASUREMENT ONLY - no product code.

A bare `ping` to the live addon costs ~30ms while in-kernel work is ~0.1-3ms.
This probe splits that floor across the layers it could live in:

  1. Python client + loopback wire     -> pure-Python echo server (in-process thread)
  2. Wolfram SocketListen+handler stack -> headless wolframscript echo (separate kernel,
                                           NO front end, NO addon dispatch), same handler
                                           shape as the addon (SourceSocket + DataBytes)
  3. Live addon ping (front-end kernel) -> persistent vs fresh socket
  4. Pipelining K=10 on one socket      -> is the ~30ms per-message or per-handler-invocation?
  5. Payload sweep (execute_code)       -> does RTT scale with response size beyond wire time?

Safe to re-run. Traffic to the live addon is `ping` and trivial pure `execute_code`
(StringRepeat) only - never notebook-touching. The headless echo runs on a DIFFERENT
port (9899) and is killed in a finally block (process group).

Run:  cd /Users/48073636/mcp/mathematica-mcp && uv run python benchmarks/probe_transport_floor.py
Out:  benchmarks/results/transport_floor.json  (+ stdout summary)
"""

import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import time

HOST = "127.0.0.1"
ADDON_PORT = int(os.getenv("MATHEMATICA_PORT", "9881"))
PY_ECHO_PORT = 9898
WL_ECHO_PORT = 9899
TOKEN = os.getenv("MATHEMATICA_MCP_TOKEN", "")

N = 100  # samples per experiment after warmup
WARMUP = 5
PAYLOAD_N = 50  # fewer iters for the payload sweep (bigger responses)
PIPE_K = 10  # requests per pipelined batch
PIPE_BATCHES = 100

# Working dir for the throwaway WL echo script + log (portable across machines).
SCRATCH = tempfile.mkdtemp(prefix="mmcp_transport_probe_")

PING_REQ = {"command": "ping", "params": {}}


# --------------------------------------------------------------------------- stats
def summarize(samples):
    """p50/p90/p99 + mean/min/max, in ms, from a list of floats."""
    if not samples:
        return None
    s = sorted(samples)

    def pct(q):
        k = (len(s) - 1) * q
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    return {
        "n": len(s),
        "p50": round(pct(0.50), 3),
        "p90": round(pct(0.90), 3),
        "p99": round(pct(0.99), 3),
        "mean": round(sum(s) / len(s), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


# --------------------------------------------------------------------- client path
def rpc_once(sock, req):
    """Full client code path: json encode -> sendall -> recv-until-newline -> json decode.
    Returns (rtt_ms, decoded_response). Matches connection.py's NDJSON framing."""
    payload = (json.dumps(req) + "\n").encode("utf-8")
    t0 = time.perf_counter()
    sock.sendall(payload)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("peer closed before response")
        buf += chunk
    rtt = (time.perf_counter() - t0) * 1000
    line = buf.split(b"\n", 1)[0]
    obj = json.loads(line.decode("utf-8"))  # exercise decode on the client path
    return rtt, obj


def measure_persistent(port, req, n=N, warmup=WARMUP, timeout=30.0):
    """send+recv on ONE reused socket - excludes connection setup."""
    sock = socket.create_connection((HOST, port), timeout=timeout)
    sock.settimeout(timeout)
    try:
        for _ in range(warmup):
            rpc_once(sock, req)
        return [rpc_once(sock, req)[0] for _ in range(n)]
    finally:
        sock.close()


def measure_fresh(port, req, n=N, warmup=WARMUP, timeout=30.0):
    """connect+send+recv, fresh socket each call - INCLUDES connection setup."""

    def one():
        t0 = time.perf_counter()
        sock = socket.create_connection((HOST, port), timeout=timeout)
        try:
            sock.settimeout(timeout)
            sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("peer closed before response")
                buf += chunk
            return (time.perf_counter() - t0) * 1000
        finally:
            sock.close()

    for _ in range(warmup):
        one()
    return [one() for _ in range(n)]


def measure_pipeline(port, req, k=PIPE_K, batches=PIPE_BATCHES, warmup=WARMUP, timeout=30.0):
    """Send k requests back-to-back on one socket BEFORE reading any response, then
    read all k. Returns list of per-request latencies (batch_total / k)."""
    sock = socket.create_connection((HOST, port), timeout=timeout)
    sock.settimeout(timeout)
    blob = (json.dumps(req) + "\n").encode("utf-8") * k
    try:

        def batch():
            t0 = time.perf_counter()
            sock.sendall(blob)
            buf = b""
            while buf.count(b"\n") < k:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("peer closed mid-batch")
                buf += chunk
            total = (time.perf_counter() - t0) * 1000
            return total, total / k

        for _ in range(warmup):
            batch()
        per_req, totals = [], []
        for _ in range(batches):
            t, pr = batch()
            totals.append(t)
            per_req.append(pr)
        return per_req, totals
    finally:
        sock.close()


def measure_payload(port, size, n=PAYLOAD_N, warmup=3, timeout=30.0):
    """execute_code returning StringRepeat["x", size]; measure RTT + response bytes."""
    req = {"command": "execute_code", "params": {"code": f'StringRepeat["x", {size}]'}}
    if TOKEN:
        req["token"] = TOKEN
    sock = socket.create_connection((HOST, port), timeout=timeout)
    sock.settimeout(timeout)
    try:
        for _ in range(warmup):
            rpc_once(sock, req)
        rtts, resp_bytes = [], 0
        for _ in range(n):
            payload = (json.dumps(req) + "\n").encode("utf-8")
            t0 = time.perf_counter()
            sock.sendall(payload)
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("peer closed before response")
                buf += chunk
            rtts.append((time.perf_counter() - t0) * 1000)
            resp_bytes = buf.split(b"\n", 1)[0].__len__()
        return rtts, resp_bytes
    finally:
        sock.close()


# ------------------------------------------------------------------ python echo srv
def start_py_echo(port):
    """Raw-byte TCP echo server in a daemon thread. Returns (server_socket, stop_fn)."""
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(8)
    srv.settimeout(0.5)
    stop = threading.Event()

    def handle(conn):
        conn.settimeout(2.0)
        try:
            while not stop.is_set():
                data = conn.recv(65536)
                if not data:
                    break
                conn.sendall(data)  # echo raw bytes, newline included
        except OSError:
            pass
        finally:
            conn.close()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()

    def stop_fn():
        stop.set()
        with contextlib.suppress(OSError):
            srv.close()

    return srv, stop_fn


# -------------------------------------------------------------------- wl echo srv
WL_ECHO_SCRIPT = """
(* Headless echo: same handler shape as the addon (SourceSocket + DataBytes),
   but the handler just writes the received bytes straight back. No front end,
   no dispatch, no processCommand. Isolates Wolfram's SocketListen+handler stack. *)
listener = SocketListen[{{"127.0.0.1", {port}}},
  Function[a, BinaryWrite[a["SourceSocket"], a["DataBytes"]]],
  HandlerFunctionsKeys -> {{"SourceSocket", "DataBytes"}}
];
Print["WL_ECHO_READY"];
While[True, Pause[1]];
"""


def start_wl_echo(port):
    """Launch headless wolframscript echo on `port`. Returns Popen, or None on failure.
    Uses its own process group so we can kill the wrapper AND the WolframKernel child."""
    script_path = os.path.join(SCRATCH, "wl_echo_server.wl")
    with open(script_path, "w") as f:
        f.write(WL_ECHO_SCRIPT.format(port=port))
    log_path = os.path.join(SCRATCH, "wl_echo_server.log")
    # The child inherits its own copy of the fd, so the parent's handle can
    # close as soon as Popen returns.
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["wolframscript", "-file", script_path],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group
        )
    # Wait for the port to accept (kernel launch can take many seconds).
    deadline = time.time() + 40
    while time.time() < deadline:
        if proc.poll() is not None:
            return None  # died during startup
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                time.sleep(0.3)  # let the listener settle
                return proc
        except OSError:
            time.sleep(0.3)
    return proc if proc.poll() is None else None


def kill_wl_echo(proc):
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)


# --------------------------------------------------------------------------- run
def run():
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "N": N,
            "WARMUP": WARMUP,
            "PAYLOAD_N": PAYLOAD_N,
            "PIPE_K": PIPE_K,
            "PIPE_BATCHES": PIPE_BATCHES,
            "addon_port": ADDON_PORT,
        },
        "experiments": {},
    }
    exp = results["experiments"]

    # --- confirm the live addon is up and answering ping
    _, pong = rpc_once(socket.create_connection((HOST, ADDON_PORT), timeout=10), PING_REQ)
    print(f"addon reachable: pong={pong.get('status')} proto={((pong.get('result') or {}).get('protocol_version'))}")

    # --- 1. Python client + wire floor (in-process echo)
    print("\n[1] python echo (client + loopback wire floor)...")
    srv, stop = start_py_echo(PY_ECHO_PORT)
    try:
        exp["py_echo_persistent"] = summarize(measure_persistent(PY_ECHO_PORT, PING_REQ))
    finally:
        stop()
    print("   ", exp["py_echo_persistent"])

    # --- 2. WL SocketListen + handler floor (headless, no front end, no dispatch)
    print("\n[2] wl echo (SocketListen+handler stack, headless kernel)...")
    wl = None
    try:
        wl = start_wl_echo(WL_ECHO_PORT)
        if wl is None:
            exp["wl_echo_persistent"] = {"error": "headless wolframscript echo failed to start"}
            print("    FAILED to start (see wl_echo_server.log)")
        else:
            exp["wl_echo_persistent"] = summarize(measure_persistent(WL_ECHO_PORT, PING_REQ))
            print("   ", exp["wl_echo_persistent"])
    finally:
        kill_wl_echo(wl)

    # --- 3. Live addon ping: persistent vs fresh socket
    print("\n[3] live addon ping...")
    exp["addon_ping_persistent"] = summarize(measure_persistent(ADDON_PORT, PING_REQ))
    print("    persistent:", exp["addon_ping_persistent"])
    exp["addon_ping_fresh"] = summarize(measure_fresh(ADDON_PORT, PING_REQ))
    print("    fresh sock:", exp["addon_ping_fresh"])

    # --- 4. Pipelining K=10 on the live addon
    print("\n[4] pipelining K=10 on the live addon...")
    per_req, totals = measure_pipeline(ADDON_PORT, PING_REQ)
    exp["addon_pipeline_per_request"] = summarize(per_req)
    exp["addon_pipeline_batch_total"] = summarize(totals)
    print("    per-request:", exp["addon_pipeline_per_request"])
    print("    batch total:", exp["addon_pipeline_batch_total"])

    # --- 5. Payload sweep on the live addon
    print("\n[5] payload sweep (execute_code StringRepeat)...")
    exp["payload_sweep"] = {}
    for size in (1000, 10000, 100000):
        rtts, resp_bytes = measure_payload(ADDON_PORT, size)
        exp["payload_sweep"][str(size)] = {
            "response_bytes": resp_bytes,
            **summarize(rtts),
        }
        print(f"    size={size:>6}  resp={resp_bytes:>7}B  {exp['payload_sweep'][str(size)]['p50']}ms p50")

    return results


def main():
    results = run()
    out = os.path.join(os.path.dirname(__file__), "results", "transport_floor.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out}")

    # Compact attribution to stdout.
    e = results["experiments"]
    py = (e.get("py_echo_persistent") or {}).get("p50")
    wl = (e.get("wl_echo_persistent") or {}).get("p50")
    addon = (e.get("addon_ping_persistent") or {}).get("p50")
    print("\n=== attribution (p50, ms) ===")
    print(f"  python client + wire      : {py}")
    print(f"  WL SocketListen stack     : {wl}")
    print(f"  live addon ping           : {addon}")
    if isinstance(wl, (int, float)) and isinstance(py, (int, float)):
        print(f"  -> WL socket stack over py: {round(wl - py, 3)}")
    if isinstance(addon, (int, float)) and isinstance(wl, (int, float)):
        print(f"  -> front-end/addon residue: {round(addon - wl, 3)}")


if __name__ == "__main__":
    main()
