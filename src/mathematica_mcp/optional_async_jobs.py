from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

_computation_jobs: dict[str, dict[str, Any]] = {}
_computation_jobs_lock = threading.Lock()
_MAX_JOBS = 100
_MAX_JOB_AGE_SECONDS = 3600


def register_async_computation_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    async def submit_computation(
        code: str,
        name: str | None = None,
        timeout: int = 300,
    ) -> str:
        """Submit a long-running computation for background execution."""

        def _prune_jobs(now: float) -> None:
            expired = [
                job_id
                for job_id, job in _computation_jobs.items()
                if now - job.get("submitted_at", now) > _MAX_JOB_AGE_SECONDS
            ]
            for job_id in expired:
                _computation_jobs.pop(job_id, None)

        job_id = str(uuid.uuid4())[:8]
        from .lazy_wolfram_tools import _find_wolframscript

        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps({"success": False, "error": "wolframscript not found in PATH"}, indent=2)

        now = time.time()
        with _computation_jobs_lock:
            _prune_jobs(now)
            if len(_computation_jobs) >= _MAX_JOBS:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Too many active jobs. Try again later.",
                    },
                    indent=2,
                )
            _computation_jobs[job_id] = {
                "id": job_id,
                "name": name or f"Job {job_id}",
                "code": code,
                "status": "running",
                "submitted_at": now,
                "timeout": timeout,
                "result": None,
                "error": None,
            }

        def run_computation():
            try:
                result = subprocess.run(
                    [wolframscript, "-code", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                with _computation_jobs_lock:
                    if result.returncode == 0:
                        _computation_jobs[job_id]["status"] = "completed"
                        _computation_jobs[job_id]["result"] = result.stdout.strip()
                    else:
                        _computation_jobs[job_id]["status"] = "failed"
                        _computation_jobs[job_id]["error"] = result.stderr or "Execution failed"
            except subprocess.TimeoutExpired:
                with _computation_jobs_lock:
                    _computation_jobs[job_id]["status"] = "timeout"
                    _computation_jobs[job_id]["error"] = f"Computation timed out after {timeout}s"
            except Exception as e:
                with _computation_jobs_lock:
                    _computation_jobs[job_id]["status"] = "failed"
                    _computation_jobs[job_id]["error"] = str(e)

            with _computation_jobs_lock:
                _computation_jobs[job_id]["completed_at"] = time.time()

        thread = threading.Thread(target=run_computation, daemon=True)
        thread.start()

        return json.dumps(
            {
                "success": True,
                "job_id": job_id,
                "name": name or f"Job {job_id}",
                "status": "submitted",
                "message": f"Computation submitted. Use poll_computation('{job_id}') to check status.",
            },
            indent=2,
        )

    @mcp.tool()
    async def poll_computation(job_id: str) -> str:
        """Check the status of a submitted computation."""
        with _computation_jobs_lock:
            if job_id not in _computation_jobs:
                return json.dumps({"success": False, "error": f"Job '{job_id}' not found"}, indent=2)
            job = _computation_jobs[job_id]

        elapsed = time.time() - job["submitted_at"]
        return json.dumps(
            {
                "success": True,
                "job_id": job_id,
                "name": job["name"],
                "status": job["status"],
                "elapsed_seconds": round(elapsed, 1),
                "has_result": job["result"] is not None,
            },
            indent=2,
        )

    @mcp.tool()
    async def get_computation_result(job_id: str) -> str:
        """Retrieve the result of a completed computation."""
        with _computation_jobs_lock:
            if job_id not in _computation_jobs:
                return json.dumps({"success": False, "error": f"Job '{job_id}' not found"}, indent=2)
            job = _computation_jobs[job_id]

        if job["status"] == "running":
            return json.dumps(
                {
                    "success": False,
                    "status": "running",
                    "message": "Computation still in progress. Use poll_computation to check status.",
                },
                indent=2,
            )

        return json.dumps(
            {
                "success": job["status"] == "completed",
                "job_id": job_id,
                "name": job["name"],
                "status": job["status"],
                "result": job["result"],
                "error": job["error"],
            },
            indent=2,
        )
