import atexit
import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import time
import zlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("mathematica_mcp.session")

_kernel_session: Any = None
_use_wolframscript: bool = False
# Set while get_kernel_session() is building a new session (~12s startup). The
# idle reaper must not terminate a half-started session, so it skips reaping
# whenever this is True.
_session_starting: bool = False
_last_kernel_health_check: float = 0.0
# 30s (was 5s): every eval path already self-heals on exception, so the ping is
# belt-and-suspenders and needn't burn a round-trip every 5s of wall clock.
KERNEL_HEALTH_CHECK_INTERVAL = 30.0

# Serializes access to the single persistent kernel session. wolframclient's
# WolframLanguageSession is not safe for concurrent evaluate() calls, so every
# heavy evaluation (evaluate_wl, execute_in_kernel) takes this lock. Never held
# across get_kernel_session() to avoid a re-entrant deadlock.
import threading as _eval_threading  # noqa: E402

_session_eval_lock = _eval_threading.Lock()

# Count of cold wolframscript subprocesses spawned this process. The lean happy
# path must be zero (plan §3.5); surfaced in status(). Incremented at every cold
# spawn site via _note_cold_execution().
_cold_call_count: int = 0


def cold_execution_count() -> int:
    """Number of cold wolframscript subprocesses spawned this process."""
    return _cold_call_count


def reset_cold_execution_count() -> None:
    """Reset the cold-execution counter (tests / integration assertions)."""
    global _cold_call_count
    _cold_call_count = 0


def _note_cold_execution() -> None:
    global _cold_call_count
    _cold_call_count += 1


# Idle kernel shutdown (plan §3.4): free the Wolfram license after the persistent
# kernel sits unused for MATHEMATICA_KERNEL_IDLE_TIMEOUT seconds (default 1800; 0
# disables). A daemon thread reaps; the next use pays a cold restart.
_last_activity: float = 0.0
_reaper_thread: Any = None
_DEFAULT_IDLE_TIMEOUT = 1800.0


def kernel_idle_timeout() -> float:
    """Idle-shutdown timeout in seconds (env MATHEMATICA_KERNEL_IDLE_TIMEOUT; 0 disables)."""
    try:
        return float(os.environ.get("MATHEMATICA_KERNEL_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT))
    except (TypeError, ValueError):
        return _DEFAULT_IDLE_TIMEOUT


# Back-compat private alias used internally.
_kernel_idle_timeout = kernel_idle_timeout


def _note_activity() -> None:
    global _last_activity
    _last_activity = time.monotonic()


def _maybe_reap_idle_kernel(now: float | None = None) -> bool:
    """Terminate the persistent kernel if idle beyond the timeout. Returns True if
    reaped. Never reaps mid-evaluation: it only proceeds when it can take the eval
    lock without blocking (a held lock means an evaluation is in flight = active)."""
    timeout = _kernel_idle_timeout()
    if timeout <= 0 or _kernel_session is None or _session_starting:
        return False
    now = time.monotonic() if now is None else now
    if (now - _last_activity) < timeout:
        return False
    if not _session_eval_lock.acquire(blocking=False):
        return False
    try:
        if _kernel_session is not None and (time.monotonic() - _last_activity) >= timeout:
            close_kernel_session()
            logger.info("Idle kernel shut down after %.0fs of inactivity", timeout)
            return True
    finally:
        _session_eval_lock.release()
    return False


def _start_idle_reaper() -> None:
    """Start the daemon that reaps an idle kernel. Idempotent."""
    global _reaper_thread
    if _reaper_thread is not None or _kernel_idle_timeout() <= 0:
        return

    def _loop() -> None:
        while True:
            time.sleep(60)
            with contextlib.suppress(Exception):
                _maybe_reap_idle_kernel()

    _reaper_thread = _eval_threading.Thread(target=_loop, daemon=True, name="kernel-idle-reaper")
    _reaper_thread.start()


# ---------------------------------------------------------------------------
# Raster cache – avoids re-rasterising graphics on query-cache hits.
# Key: SHA-256 prefix of (wrapped_code, image_size).
# Value: path to the temp PNG file on disk.
# Bounded: oldest entries evicted when ``_MAX_RASTER_ENTRIES`` is reached,
#          and their temp files are deleted from disk.
# ---------------------------------------------------------------------------
import threading as _raster_threading  # noqa: E402
from collections import OrderedDict  # noqa: E402

_raster_cache: OrderedDict[str, str] = OrderedDict()
_raster_lock = _raster_threading.Lock()
_MAX_RASTER_ENTRIES = 50


def _raster_cache_key(code: str, image_size: int = 500) -> str:
    return hashlib.sha256(f"{code}|sz={image_size}".encode()).hexdigest()[:16]


def _get_cached_raster(code: str, image_size: int = 500) -> str | None:
    """Return a cached raster file path if it exists and is valid."""
    key = _raster_cache_key(code, image_size)
    with _raster_lock:
        path = _raster_cache.get(key)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            _raster_cache.move_to_end(key)
            return path
        _raster_cache.pop(key, None)
    return None


def _put_cached_raster(code: str, path: str, image_size: int = 500) -> None:
    """Store a raster file path in the cache, evicting oldest if at capacity."""
    key = _raster_cache_key(code, image_size)
    with _raster_lock:
        # If already present, delete the old file before replacing.
        if key in _raster_cache:
            old_path = _raster_cache[key]
            if old_path != path:
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except OSError:
                    pass
            _raster_cache.move_to_end(key)
            _raster_cache[key] = path
            return
        # Evict oldest entries if at capacity.
        while len(_raster_cache) >= _MAX_RASTER_ENTRIES:
            _, evicted_path = _raster_cache.popitem(last=False)
            try:
                if os.path.exists(evicted_path):
                    os.remove(evicted_path)
            except OSError:
                pass
        _raster_cache[key] = path


def clear_raster_cache() -> None:
    """Remove all cached raster files from disk and clear the cache."""
    with _raster_lock:
        for path in list(_raster_cache.values()):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        _raster_cache.clear()


def _session_context(session_id: str | None) -> str | None:
    if not session_id:
        return None
    crc = zlib.crc32(session_id.encode("utf-8")) & 0xFFFFFFFF
    return f"MCP`{crc:08x}`"


def _wrap_code_for_context(code: str, context: str | None) -> str:
    if not context:
        return code
    return f'Block[{{$Context = "{context}", $ContextPath = {{"{context}", "System`"}}}}, {code}]'


def _wrap_code_for_determinism(code: str, deterministic_seed: int | None) -> str:
    if deterministic_seed is None:
        return code
    return f"BlockRandom[SeedRandom[{deterministic_seed}]; {code}]"


def find_wolfram_kernel() -> str | None:
    system = platform.system()
    potential_paths: list[str] = []

    if system == "Darwin":
        potential_paths = [
            "/Applications/Mathematica.app/Contents/MacOS/WolframKernel",
            "/Applications/Wolfram.app/Contents/MacOS/WolframKernel",
            "/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel",
        ]
    elif system == "Windows":
        program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        for version in ["15.1", "15.0", "15", "14.2", "14.1", "14.0", "13.3", "13.2", "13.1", "13.0"]:
            potential_paths.append(
                os.path.join(
                    program_files,
                    f"Wolfram Research\\Mathematica\\{version}\\WolframKernel.exe",
                )
            )
    elif system == "Linux":
        for version in ["15.1", "15.0", "15", "14.2", "14.1", "14.0", "13.3", "13.2", "13.1", "13.0"]:
            potential_paths.append(f"/usr/local/Wolfram/Mathematica/{version}/Executables/WolframKernel")

    for path in potential_paths:
        if os.path.exists(path):
            logger.info(f"Found Wolfram Kernel at: {path}")
            return path

    logger.warning("Could not find Wolfram Kernel in standard locations")
    return None


def _parse_association_output(output: str) -> dict[str, Any]:
    """Robust Association parser handling escaped quotes.

    Fallback for older Mathematica versions that don't support RawJSON export.
    Uses a regex that properly handles escaped sequences.
    """
    result: dict[str, Any] = {
        "output": output,
        "output_inputform": output,
        "output_fullform": "",
        "messages": "",
        "timing_ms": 0,
        "failed": False,
    }

    # Pattern: (?:[^"\\]|\\.)* handles escaped sequences properly
    out_match = re.search(r'"output"\s*->\s*"((?:[^"\\]|\\.)*)"', output)
    if out_match:
        result["output"] = out_match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        result["output_inputform"] = result["output"]

    input_match = re.search(r'"output_inputform"\s*->\s*"((?:[^"\\]|\\.)*)"', output)
    if input_match:
        result["output_inputform"] = input_match.group(1).replace('\\"', '"').replace("\\\\", "\\")

    full_match = re.search(r'"output_fullform"\s*->\s*"((?:[^"\\]|\\.)*)"', output)
    if full_match:
        result["output_fullform"] = full_match.group(1).replace('\\"', '"').replace("\\\\", "\\")

    tex_match = re.search(r'"output_tex"\s*->\s*"((?:[^"\\]|\\.)*)"', output)
    if tex_match:
        result["output_tex"] = tex_match.group(1).replace('\\"', '"').replace("\\\\", "\\")

    msg_match = re.search(r'"messages"\s*->\s*"((?:[^"\\]|\\.)*)"', output)

    if msg_match:
        result["messages"] = msg_match.group(1)

    timing_match = re.search(r'"timing_ms"\s*->\s*(\d+)', output)
    if timing_match:
        result["timing_ms"] = int(timing_match.group(1))

    failed_match = re.search(r'"failed"\s*->\s*(True|False)', output)
    if failed_match:
        result["failed"] = failed_match.group(1) == "True"

    return result


def _execute_via_wolframscript(
    code: str,
    output_format: str = "text",
    *,
    timeout: int = 60,
    deterministic_seed: int | None = None,
    session_id: str | None = None,
    isolate_context: bool = False,
    render_graphics: bool = False,
    image_size: int = 500,
) -> dict[str, Any]:
    """Execute code via wolframscript CLI with timing and warning capture.

    When render_graphics=True, graphics results are rasterized within the
    same subprocess invocation (no second process needed).
    """
    from .lazy_wolfram_tools import _find_wolframscript

    wolframscript = _find_wolframscript()
    if not wolframscript:
        return {
            "success": False,
            "output": "wolframscript not found in PATH",
            "error": "wolframscript not available",
            "warnings": [],
            "timing_ms": 0,
            "execution_method": "wolframscript",
        }

    # Only compute what's requested to save time and tokens
    # By default we always get InputForm (text)
    include_tex = "True" if output_format == "latex" else "False"
    include_full = "True" if output_format == "mathematica" else "False"

    context = _session_context(session_id) if isolate_context else None
    wrapped_expr = _wrap_code_for_context(code, context)
    wrapped_expr = _wrap_code_for_determinism(wrapped_expr, deterministic_seed)

    # Prepare temp file for optional graphics rasterization
    import tempfile

    raster_temp_path = ""
    if render_graphics:
        fd, raster_temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    # Escape backslashes for Mathematica string literal
    wl_raster_path = raster_temp_path.replace("\\", "\\\\")
    render_flag = "True" if render_graphics else "False"

    wrapped_code = f"""
Module[{{startTime, result, messages, timing, response, outInput, outFull="", outTex="",
         imgPath="{wl_raster_path}", didRaster=False}},
  startTime = AbsoluteTime[];
  Block[{{$MessageList = {{}}}},
    result = Quiet[Check[{wrapped_expr}, $Failed]];
    messages = $MessageList;
  ];

  (* Always compute InputForm as it is the standard text output *)
  outInput = ToString[result, InputForm];

  (* Conditionally compute expensive forms *)
  If[{include_full}, outFull = ToString[result, FullForm]];
  If[{include_tex}, outTex = ToString[TeXForm[result]]];

  (* Conditionally rasterize graphics in the SAME process *)
  If[{render_flag} && (result =!= $Failed) &&
     (MatchQ[Head[result], Graphics|Graphics3D|Legended|Image] || MatchQ[result, _Show]),
    Quiet[Export[imgPath, Rasterize[result, ImageResolution -> 144, ImageSize -> {image_size}], "PNG"]];
    didRaster = True;
  ];

  timing = Round[(AbsoluteTime[] - startTime) * 1000];

  response = <|
    "output" -> outInput,
    "output_inputform" -> outInput,
    "output_fullform" -> outFull,
    "output_tex" -> outTex,
    "messages" -> ToString[messages],
    "timing_ms" -> timing,
    "failed" -> (result === $Failed),
    "image_path" -> If[didRaster, imgPath, ""],
    "is_graphics" -> didRaster
  |>;
  ExportString[response, "RawJSON"]
]
"""

    start_time = time.time()
    _note_cold_execution()
    try:
        result = subprocess.run(
            [wolframscript, "-code", wrapped_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        python_timing = int((time.time() - start_time) * 1000)
        output = result.stdout.strip()

        if output.endswith("\nNull"):
            output = output[:-5].strip()

        if result.returncode != 0:
            # Clean up unused temp file on failure
            if raster_temp_path and os.path.exists(raster_temp_path):
                os.remove(raster_temp_path)
            return {
                "success": False,
                "output": result.stderr or output,
                "error": result.stderr or "Non-zero exit code",
                "warnings": [],
                "timing_ms": python_timing,
                "execution_method": "wolframscript",
            }

        # Parse JSON output from ExportString[..., "RawJSON"]
        try:
            parsed = json.loads(output)
            clean_output = parsed.get("output", output)
            output_inputform = parsed.get("output_inputform", clean_output)
            output_fullform = parsed.get("output_fullform", "")
            output_tex = parsed.get("output_tex", "")
            messages_str = parsed.get("messages", "")
            warnings_list = [messages_str] if messages_str and messages_str != "{}" else []
        except json.JSONDecodeError:
            # Fallback to robust Association parser for older Mathematica
            parsed = _parse_association_output(output)
            clean_output = parsed["output"]
            output_inputform = parsed.get("output_inputform", clean_output)
            output_fullform = parsed.get("output_fullform", "")
            output_tex = parsed.get("output_tex", "")
            messages_str = parsed.get("messages", "")
            warnings_list = [messages_str] if messages_str and messages_str != "{}" else []

        response: dict[str, Any] = {
            "success": True,
            "output": output_inputform,
            "output_tex": output_tex,
            "output_inputform": output_inputform,
            "output_fullform": output_fullform,
            "warnings": warnings_list,
            "timing_ms": python_timing,
            "execution_method": "wolframscript",
        }

        # Check if in-process rasterization produced an image
        image_path = parsed.get("image_path", "")
        did_raster = parsed.get("is_graphics", False)
        if did_raster and image_path and os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            response["image_path"] = image_path
            response["output"] = f"[Graphics rendered to image: {image_path}]"
            response["is_graphics"] = True
            _put_cached_raster(wrapped_expr, image_path)
        elif raster_temp_path and os.path.exists(raster_temp_path):
            # Clean up unused temp file
            os.remove(raster_temp_path)

        return response

    except subprocess.TimeoutExpired:
        if raster_temp_path and os.path.exists(raster_temp_path):
            os.remove(raster_temp_path)
        return {
            "success": False,
            "output": f"Execution timed out after {timeout} seconds",
            "error": "timeout",
            "warnings": [],
            "timing_ms": int(timeout * 1000),
            "execution_method": "wolframscript",
        }
    except Exception as e:
        if raster_temp_path and os.path.exists(raster_temp_path):
            os.remove(raster_temp_path)
        return {
            "success": False,
            "output": f"wolframscript execution failed: {e}",
            "error": str(e),
            "warnings": [],
            "timing_ms": int((time.time() - start_time) * 1000),
            "execution_method": "wolframscript",
        }


def get_kernel_session():
    global _kernel_session, _use_wolframscript, _last_kernel_health_check, _session_starting

    if _use_wolframscript:
        return None

    if _kernel_session is not None:
        if time.monotonic() - _last_kernel_health_check < KERNEL_HEALTH_CHECK_INTERVAL:
            _note_activity()
            return _kernel_session
        if not _session_eval_lock.acquire(blocking=False):
            # An evaluation is in flight, so the session is demonstrably alive;
            # pinging concurrently would violate the eval-lock contract above
            # and could corrupt the WSTP link.
            _note_activity()
            return _kernel_session
        try:
            from wolframclient.language import wlexpr

            _kernel_session.evaluate(wlexpr("1"))
            _last_kernel_health_check = time.monotonic()
            _note_activity()
            return _kernel_session
        except Exception:
            logger.warning("Kernel session unresponsive, recreating...")
            with contextlib.suppress(Exception):
                _kernel_session.terminate()
            _kernel_session = None
        finally:
            _session_eval_lock.release()

    try:
        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wlexpr
    except ImportError:
        logger.info("wolframclient not available, falling back to wolframscript")
        _use_wolframscript = True
        return None

    kernel_path = os.environ.get("MATHEMATICA_KERNEL_PATH") or find_wolfram_kernel()

    if not kernel_path:
        logger.error("No Wolfram kernel found. Set MATHEMATICA_KERNEL_PATH env var.")
        _use_wolframscript = True
        return None

    try:
        _session_starting = True
        try:
            _kernel_session = WolframLanguageSession(kernel_path)
            # Stamp activity before start()/evaluate() so the reaper never sees a
            # fresh session as stale during its ~12s startup (belt to _session_starting).
            _note_activity()
            _kernel_session.start()
            _kernel_session.evaluate(wlexpr("1+1"))
        finally:
            _session_starting = False
        _last_kernel_health_check = time.monotonic()
        _note_activity()
        # A transient cold fallback recovers here: a successful (re)creation clears
        # the permanent-cold flag.
        _use_wolframscript = False
        _start_idle_reaper()
        logger.info(f"Kernel session ready: {kernel_path}")
        return _kernel_session
    except Exception as e:
        logger.warning(f"WolframLanguageSession failed ({e}), falling back to wolframscript")
        _use_wolframscript = True
        return None


def has_existing_kernel_session() -> bool:
    """Best-effort check if a cached kernel session exists. Does not verify liveness."""
    return _kernel_session is not None and not _use_wolframscript


def close_kernel_session():
    global _kernel_session, _last_kernel_health_check, _use_wolframscript
    if _kernel_session is not None:
        with contextlib.suppress(Exception):
            _kernel_session.terminate()
        _kernel_session = None
        _last_kernel_health_check = 0.0
        logger.info("Closed kernel session")
    # Reset the permanent-cold flag unconditionally (the failure path leaves
    # _kernel_session already None) so a restart retries the warm session.
    _use_wolframscript = False


def _shutdown_at_exit() -> None:
    """Close the persistent kernel at process exit. Idempotent, bounded.

    Without this a disconnected server holds a WolframKernel license until the
    idle reaper fires (default 1800s). Runs the close in a daemon thread joined
    with a timeout so a wedged terminate() can never hang interpreter shutdown.
    """
    if _kernel_session is None:
        return
    with contextlib.suppress(Exception):
        t = _eval_threading.Thread(target=close_kernel_session, daemon=True, name="kernel-atexit-close")
        t.start()
        t.join(5.0)


# wolframclient's WolframKernelController is a NON-daemon thread, and CPython
# joins non-daemon threads BEFORE running plain atexit callbacks — a bare atexit
# handler would never fire while a kernel is still up. threading's private
# shutdown hook (used by concurrent.futures) runs before that join; keep the
# atexit registration as a fallback for interpreters without the private API.
# _shutdown_at_exit is idempotent, so double invocation is safe.
if hasattr(_eval_threading, "_register_atexit"):
    _eval_threading._register_atexit(_shutdown_at_exit)
atexit.register(_shutdown_at_exit)


@dataclass
class WLResult:
    """Transport-agnostic result of evaluating a WL expression.

    ``text`` is the OutputForm rendering of the result — byte-compatible with a
    cold ``wolframscript -code`` stdout — so callers parse it identically whether
    it came from the warm session or a cold subprocess.
    """

    text: str
    success: bool
    execution_method: str  # "wolframclient" (warm) | "wolframscript" (cold) | "none"
    error: str = ""
    timed_out: bool = False


def evaluate_wl(code: str, timeout: int = 60) -> WLResult:
    """Evaluate a WL expression (typically one returning an Association), warm first.

    Prefers the persistent kernel session; falls back to a cold ``wolframscript``
    subprocess (flagged ``execution_method='wolframscript'`` and counted). The warm
    path renders the result via ``ToString[..., OutputForm, PageWidth -> Infinity]``
    so its text matches cold wolframscript stdout, letting callers reuse the same
    Association parser. Runaway warm evaluations are bounded kernel-side with
    ``TimeConstrained``.
    """
    session = get_kernel_session()
    if session is not None:
        try:
            from wolframclient.language import wlexpr

            wrapped = (
                f"ToString[TimeConstrained[(\n{code}\n), {int(timeout)}, $Aborted], OutputForm, PageWidth -> Infinity]"
            )
            with _session_eval_lock:
                text = session.evaluate(wlexpr(wrapped))
            if isinstance(text, str):
                if text.strip() == "$Aborted":
                    return WLResult(
                        text="",
                        success=False,
                        execution_method="wolframclient",
                        error=f"Evaluation timed out after {timeout}s",
                        timed_out=True,
                    )
                _note_activity()  # idle measured from completion, not eval start
                return WLResult(text=text.strip(), success=True, execution_method="wolframclient")
            logger.warning("warm evaluate_wl returned non-string (%s); cold fallback", type(text).__name__)
        except Exception as e:  # noqa: BLE001 — degrade to cold on any warm failure
            logger.warning("warm evaluate_wl failed (%s); cold fallback", e)

    from .lazy_wolfram_tools import _find_wolframscript

    wolframscript = _find_wolframscript()
    if not wolframscript:
        return WLResult(
            text="",
            success=False,
            execution_method="none",
            error="wolframscript not found in PATH",
        )
    _note_cold_execution()
    try:
        proc = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return WLResult(
            text="",
            success=False,
            execution_method="wolframscript",
            error=f"Evaluation timed out after {timeout}s",
            timed_out=True,
        )
    except Exception as e:  # noqa: BLE001
        return WLResult(text="", success=False, execution_method="wolframscript", error=str(e))
    if proc.returncode != 0:
        return WLResult(
            text=proc.stdout.strip(),
            success=False,
            execution_method="wolframscript",
            error=proc.stderr or "Non-zero exit code",
        )
    return WLResult(text=proc.stdout.strip(), success=True, execution_method="wolframscript")


# Front-end placeholder strings (addon output) — graphics with no expression
# text to rasterize from.
_GRAPHICS_PLACEHOLDERS = frozenset({"-Graphics-", "-Graphics3D-", "-Image-"})


def _is_graphics_output(output: str) -> bool:
    """Check if output looks like a Graphics object."""
    if not output:
        return False
    output_stripped = output.strip()
    # Check for placeholder patterns (used by addon)
    if output_stripped in _GRAPHICS_PLACEHOLDERS:
        return True
    # Check for actual Graphics patterns
    graphics_patterns = [
        "Graphics[",
        "Graphics3D[",
        "Image[",
        "Legended[Graphics",
        "Show[Graphics",
    ]
    return any(output_stripped.startswith(p) for p in graphics_patterns)


def _rasterize_via_wolframscript(code: str, image_size: int = 500) -> str | None:
    """Rasterize a graphics expression via wolframscript, return temp file path."""
    import tempfile

    from .lazy_wolfram_tools import _find_wolframscript

    wolframscript = _find_wolframscript()
    if not wolframscript:
        return None

    # Create temp file for the image
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    temp_path = temp_path.replace("\\", "/")
    os.close(fd)

    rasterize_code = f'''
Module[{{result, img}},
  result = {code};
  If[Head[result] === Graphics || Head[result] === Graphics3D ||
     Head[result] === Legended || Head[result] === Image ||
     MatchQ[result, _Show],
    img = Rasterize[result, ImageResolution -> 144, ImageSize -> {image_size}];
    Export["{temp_path}", img, "PNG"];
    "success",
    "not_graphics"
  ]
]
'''

    _note_cold_execution()
    try:
        result = subprocess.run(
            [wolframscript, "-code", rasterize_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()

        if "success" in output and os.path.exists(temp_path):
            return temp_path
        else:
            # Clean up temp file if rasterization failed
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    except Exception as e:
        logger.warning(f"Rasterization via wolframscript failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


def _rasterize_cached_graphics(graphics_text: str, image_size: int = 500) -> str | None:
    """Rasterize a kernel-generated Graphics InputForm string, warm-first.

    Used on query-cache hits: re-rasterizes the CACHED output expression via
    ToExpression instead of re-running the user's original code (which could
    duplicate side effects). Isolation: the text is the kernel's own InputForm
    rendering (not raw user input), it is rebuilt inside the throwaway
    MCPScratch` context so nothing leaks into Global`, and Rasterize/Export
    only write to our temp file. evaluate_wl itself falls back to a cold
    wolframscript run of the same read-only snippet if the warm session is
    unavailable.
    """
    import tempfile

    from .lazy_wolfram_tools import _wl_string

    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    wl_path = temp_path.replace("\\", "/")

    code = f"""
Block[{{$Context = "MCPScratch`", $ContextPath = {{"System`"}}}},
  Module[{{expr}},
    expr = Quiet[Check[ToExpression[{_wl_string(graphics_text)}], $Failed]];
    If[MatchQ[Head[expr], Graphics|Graphics3D|Legended|Image] || MatchQ[expr, _Show],
      Quiet[Export["{wl_path}", Rasterize[expr, ImageResolution -> 144, ImageSize -> {image_size}], "PNG"]];
      "success",
      "not_graphics"
    ]
  ]
]
"""
    result = evaluate_wl(code, timeout=60)
    if result.success and "success" in result.text and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        return temp_path
    with contextlib.suppress(OSError):
        os.remove(temp_path)
    return None


def _cache_textual_result(cache, code: str, response: dict[str, Any], **cache_kwargs) -> None:
    """Cache only the textual pre-rasterization result.

    Strips image_path, is_graphics, and rendered_image so that cache hits
    contain the original textual output (e.g. "Graphics[...]") rather than
    placeholder strings.  This lets the caller re-rasterize on cache hits.
    """
    if not response.get("success"):
        return
    cache_result = dict(response)
    # Restore original textual output if it was replaced by a placeholder
    if cache_result.get("is_graphics") and "output_inputform" in cache_result:
        cache_result["output"] = cache_result["output_inputform"]
    cache_result.pop("image_path", None)
    cache_result.pop("is_graphics", None)
    cache_result.pop("rendered_image", None)
    cache.put(code, cache_result, **cache_kwargs)


def execute_in_kernel(
    code: str,
    output_format: str = "text",
    *,
    render_graphics: bool = True,
    deterministic_seed: int | None = None,
    session_id: str | None = None,
    isolate_context: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute code in kernel with structured metadata (timing, warnings).

    Automatically rasterizes Graphics results to images for better display.
    """
    global _use_wolframscript

    from .cache import _query_cache

    context_key = session_id if isolate_context else None
    context = _session_context(session_id) if isolate_context else None
    wrapped_code = _wrap_code_for_context(code, context)
    wrapped_code = _wrap_code_for_determinism(wrapped_code, deterministic_seed)
    cached = _query_cache.get(
        code,
        output_format=output_format,
        render_graphics=render_graphics,
        deterministic_seed=deterministic_seed,
        context_key=context_key,
    )
    if cached:
        cached_result = cached.copy()
        cached_result["from_cache"] = True
        cached_result["timing_ms"] = 0
        # Re-rasterize on cache hit if the output is Graphics.
        # Check the raster cache first to avoid a subprocess call.
        if render_graphics and _is_graphics_output(cached_result.get("output", "")):
            image_path = _get_cached_raster(wrapped_code)
            if not image_path:
                graphics_text = (cached_result.get("output_inputform") or cached_result.get("output", "")).strip()
                if graphics_text in _GRAPHICS_PLACEHOLDERS:
                    # Placeholder text carries no expression to rebuild from; the
                    # only option is a cold re-run of the code (flagged + counted
                    # inside _rasterize_via_wolframscript).
                    image_path = _rasterize_via_wolframscript(wrapped_code)
                else:
                    # Warm path: rasterize the cached output expression, never
                    # re-running the user's original (side-effecting) code.
                    image_path = _rasterize_cached_graphics(graphics_text)
                if image_path:
                    _put_cached_raster(wrapped_code, image_path)
            if image_path:
                cached_result["image_path"] = image_path
                cached_result["output"] = f"[Graphics rendered to image: {image_path}]"
                cached_result["is_graphics"] = True
        return cached_result

    session = get_kernel_session()

    if session is None or _use_wolframscript:
        result = _execute_via_wolframscript(
            code,
            output_format,
            timeout=timeout,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            render_graphics=render_graphics,
        )
        if result.get("success"):
            # Cache only the textual pre-rasterization result
            _cache_textual_result(
                _query_cache,
                code,
                result,
                output_format=output_format,
                render_graphics=render_graphics,
                deterministic_seed=deterministic_seed,
                context_key=context_key,
            )
        return result

    try:
        from wolframclient.language import wlexpr
    except ImportError:
        return _execute_via_wolframscript(
            code,
            output_format,
            timeout=timeout,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            render_graphics=render_graphics,
        )

    import tempfile

    # Prepare temp file for optional graphics rasterization (before try block
    # so cleanup in except handler can always access raster_temp_path)
    raster_temp_path = ""
    if render_graphics:
        fd, raster_temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    start_time = time.time()
    try:
        # Single evaluation: compute text forms + optional rasterization
        include_fullform = "True" if output_format == "mathematica" else "False"
        include_tex = "True" if output_format == "latex" else "False"
        wl_raster_path = raster_temp_path.replace("\\", "\\\\")
        render_flag = "True" if render_graphics else "False"

        eval_code = f"""
Module[{{res, msgs, imgPath = "{wl_raster_path}", didRaster = False}},
  Block[{{$MessageList = {{}}}},
    res = {wrapped_code};
    msgs = $MessageList;
  ];
  <|
    "inputform" -> ToString[res, InputForm],
    "fullform" -> If[{include_fullform}, ToString[res, FullForm], ""],
    "tex" -> If[{include_tex}, ToString[TeXForm[res]], ""],
    "messages" -> ToString[msgs],
    "is_graphics" -> If[{render_flag} && (res =!= $Failed) &&
        (MatchQ[Head[res], Graphics|Graphics3D|Legended|Image] || MatchQ[res, _Show]),
      Quiet[Export[imgPath, Rasterize[res, ImageResolution -> 144, ImageSize -> 500], "PNG"]];
      didRaster = True;
      True,
      False
    ],
    "image_path" -> If[didRaster, imgPath, ""]
  |>
]
"""
        # Bound the kernel-side eval so a runaway evaluation can't wedge the
        # global eval lock (and thus every kernel-routed tool) forever. Mirrors
        # evaluate_wl's TimeConstrained guard; sentinel is a String so it never
        # collides with the Association the Module returns on success.
        guarded_code = f'TimeConstrained[(\n{eval_code}\n), {int(timeout)}, "$Aborted"]'
        with _session_eval_lock:
            combined_result = session.evaluate(wlexpr(guarded_code))
        timing_ms = int((time.time() - start_time) * 1000)

        if isinstance(combined_result, str) and combined_result.strip() == "$Aborted":
            if raster_temp_path and os.path.exists(raster_temp_path):
                os.remove(raster_temp_path)
            return {
                "success": False,
                "output": f"Execution timed out after {timeout} seconds",
                "error": "timeout",
                "timed_out": True,
                "warnings": [],
                "timing_ms": timing_ms,
                "execution_method": "wolframclient",
            }

        if isinstance(combined_result, dict):
            output_inputform = str(combined_result.get("inputform", str(combined_result.get("result", ""))))
            output_fullform = str(combined_result.get("fullform", ""))
            output_tex = str(combined_result.get("tex", ""))
        else:
            output_inputform = str(combined_result)
            output_fullform = ""
            output_tex = ""

        messages_raw = str(combined_result.get("messages", "")) if isinstance(combined_result, dict) else ""
        warnings_list = [messages_raw] if messages_raw and messages_raw != "{}" else []

        response: dict[str, Any] = {
            "success": True,
            "output": output_inputform,
            "output_tex": output_tex,
            "output_inputform": output_inputform,
            "output_fullform": output_fullform,
            "warnings": warnings_list,
            "timing_ms": timing_ms,
            "execution_method": "wolframclient",
        }

        # Check if in-process rasterization produced an image
        did_raster = False
        if isinstance(combined_result, dict):
            image_path = str(combined_result.get("image_path", ""))
            did_raster = bool(combined_result.get("is_graphics", False))
        else:
            image_path = ""

        if did_raster and image_path and os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            response["image_path"] = image_path
            response["output"] = f"[Graphics rendered to image: {image_path}]"
            response["is_graphics"] = True
            _put_cached_raster(wrapped_code, image_path)
        elif raster_temp_path and os.path.exists(raster_temp_path):
            # Clean up unused temp file
            os.remove(raster_temp_path)

        # Cache only the textual pre-rasterization result
        _cache_textual_result(
            _query_cache,
            code,
            response,
            output_format=output_format,
            render_graphics=render_graphics,
            deterministic_seed=deterministic_seed,
            context_key=context_key,
        )
        _note_activity()  # idle measured from completion, not eval start
        return response
    except Exception as e:
        logger.warning(f"wolframclient evaluation failed ({e}), trying wolframscript")
        if raster_temp_path and os.path.exists(raster_temp_path):
            os.remove(raster_temp_path)
        return _execute_via_wolframscript(
            code,
            output_format,
            timeout=timeout,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            render_graphics=render_graphics,
        )
