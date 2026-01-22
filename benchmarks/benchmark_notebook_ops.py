#!/usr/bin/env python3
"""
Benchmark script for Mathematica MCP notebook operations.

Measures performance of:
1. Notebook execution (cmdExecuteCodeNotebook)
2. Cell evaluation (cmdEvaluateCell)
3. Screenshot operations (cmdScreenshotNotebook)
4. Cell enumeration and refresh operations

Run with: python -m benchmarks.benchmark_notebook_ops
"""

import json
import socket
import time
import statistics
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9881
SOCKET_TIMEOUT = 60.0


class BenchmarkConnection:
    """Simple connection for benchmarking addon operations."""
    
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
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def send_command(self, command: str, params: dict = None) -> dict:
        if not self._socket and not self.connect():
            raise ConnectionError("Not connected")
        
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
            raise RuntimeError(f"Error: {response.get('message')}")
        
        return response.get("result", {})
    
    def close(self):
        if self._socket:
            self._socket.close()
            self._socket = None


def benchmark_operation(conn, name, command, params, iterations=5, warmup=1):
    """Run a benchmark for a single operation."""
    # Warmup
    for _ in range(warmup):
        try:
            conn.send_command(command, params)
        except Exception as e:
            print(f"  Warmup failed: {e}")
            return None
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        try:
            result = conn.send_command(command, params)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            
            # Extract internal timing if available
            if isinstance(result, dict) and "waited_seconds" in result:
                internal_wait = result["waited_seconds"] * 1000
                print(f"    Iter {i+1}: {elapsed:.1f}ms (internal wait: {internal_wait:.1f}ms)")
            else:
                print(f"    Iter {i+1}: {elapsed:.1f}ms")
        except Exception as e:
            print(f"    Iter {i+1}: FAILED - {e}")
    
    if not times:
        return None
    
    return {
        "name": name,
        "iterations": len(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("Mathematica MCP Notebook Operations Benchmark")
    print("=" * 60)
    
    conn = BenchmarkConnection()
    
    if not conn.connect():
        print("ERROR: Could not connect to Mathematica addon.")
        print("Make sure Mathematica is running with StartMCPServer[]")
        return None
    
    # Verify connection
    try:
        result = conn.send_command("ping")
        print(f"Connected: {result}")
    except Exception as e:
        print(f"Ping failed: {e}")
        return None
    
    results = []
    
    # 1. Simple execution (no notebook)
    print("\n[1] execute_code (kernel only, no notebook)")
    r = benchmark_operation(
        conn, "execute_code_simple",
        "execute_code",
        {"code": "1 + 1"},
        iterations=10
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 2. Execute code in notebook - KERNEL MODE (new fast path)
    print("\n[2] execute_code_notebook KERNEL MODE (fast path)")
    r = benchmark_operation(
        conn, "execute_code_notebook_kernel",
        "execute_code_notebook",
        {"code": "2 + 2", "mode": "kernel"},
        iterations=5
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 3. Execute code in notebook - FRONTEND MODE (legacy)
    print("\n[3] execute_code_notebook FRONTEND MODE (legacy polling)")
    r = benchmark_operation(
        conn, "execute_code_notebook_frontend",
        "execute_code_notebook",
        {"code": "2 + 2", "mode": "frontend", "max_wait": 10},
        iterations=3
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 4. Execute code in notebook - kernel mode complex expression
    print("\n[4] execute_code_notebook KERNEL MODE (Integrate)")
    r = benchmark_operation(
        conn, "execute_code_notebook_kernel_integrate",
        "execute_code_notebook",
        {"code": "Integrate[Sin[x]^2, x]", "mode": "kernel"},
        iterations=5
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 5. Execute code in notebook - kernel mode graphics
    print("\n[5] execute_code_notebook KERNEL MODE (Plot)")
    r = benchmark_operation(
        conn, "execute_code_notebook_kernel_plot",
        "execute_code_notebook",
        {"code": "Plot[Sin[x], {x, 0, 2 Pi}]", "mode": "kernel"},
        iterations=3
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 6. Get cells (enumeration cost)
    print("\n[6] get_cells (notebook cell enumeration)")
    r = benchmark_operation(
        conn, "get_cells",
        "get_cells",
        {},
        iterations=10
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 7. Screenshot notebook (with ExportPacket optimization)
    print("\n[7] screenshot_notebook (ExportPacket or Rasterize)")
    r = benchmark_operation(
        conn, "screenshot_notebook",
        "screenshot_notebook",
        {"max_height": 800},
        iterations=3
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 7. Get notebook info
    print("\n[8] get_notebook_info")
    r = benchmark_operation(
        conn, "get_notebook_info",
        "get_notebook_info",
        {},
        iterations=10
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    # 9. Write cell (without evaluation, no refresh)
    print("\n[9] write_cell (no evaluation, no refresh)")
    r = benchmark_operation(
        conn, "write_cell",
        "write_cell",
        {"content": "(* benchmark cell *)", "style": "Text"},
        iterations=5
    )
    if r:
        results.append(r)
        print(f"  => Mean: {r['mean_ms']:.1f}ms, Median: {r['median_ms']:.1f}ms")
    
    conn.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Operation':<40} {'Mean (ms)':<12} {'Median (ms)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<40} {r['mean_ms']:<12.1f} {r['median_ms']:<12.1f}")
    
    return results


def save_results(results, filename):
    """Save benchmark results to JSON file."""
    if results:
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    results = run_benchmarks()
    if results:
        # Determine filename based on args
        suffix = "baseline" if len(sys.argv) < 2 else sys.argv[1]
        filename = os.path.join(
            os.path.dirname(__file__),
            f"benchmark_results_{suffix}.json"
        )
        save_results(results, filename)
