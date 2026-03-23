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
from typing import Any

logger = logging.getLogger("mathematica_mcp.session")

_kernel_session: Any = None
_use_wolframscript: bool = False
_last_kernel_health_check: float = 0.0
KERNEL_HEALTH_CHECK_INTERVAL = 5.0

# ---------------------------------------------------------------------------
# Raster cache – avoids re-rasterising graphics on query-cache hits.
# Key: SHA-256 prefix of (wrapped_code, image_size).
# Value: path to the temp PNG file on disk.
# Bounded: oldest entries evicted when ``_MAX_RASTER_ENTRIES`` is reached,
#          and their temp files are deleted from disk.
# ---------------------------------------------------------------------------
from collections import OrderedDict  # noqa: E402

_raster_cache: OrderedDict[str, str] = OrderedDict()
_MAX_RASTER_ENTRIES = 50


def _raster_cache_key(code: str, image_size: int = 500) -> str:
    return hashlib.sha256(f"{code}|sz={image_size}".encode()).hexdigest()[:16]


def _get_cached_raster(code: str, image_size: int = 500) -> str | None:
    """Return a cached raster file path if it exists and is valid."""
    key = _raster_cache_key(code, image_size)
    path = _raster_cache.get(key)
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        _raster_cache.move_to_end(key)
        return path
    _raster_cache.pop(key, None)
    return None


def _put_cached_raster(code: str, path: str, image_size: int = 500) -> None:
    """Store a raster file path in the cache, evicting oldest if at capacity."""
    key = _raster_cache_key(code, image_size)
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
        for version in ["14.2", "14.1", "14.0", "13.3", "13.2", "13.1", "13.0"]:
            potential_paths.append(
                os.path.join(
                    program_files,
                    f"Wolfram Research\\Mathematica\\{version}\\WolframKernel.exe",
                )
            )
    elif system == "Linux":
        for version in ["14.2", "14.1", "14.0", "13.3", "13.2", "13.1", "13.0"]:
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
    global _kernel_session, _use_wolframscript, _last_kernel_health_check

    if _use_wolframscript:
        return None

    if _kernel_session is not None:
        if time.monotonic() - _last_kernel_health_check < KERNEL_HEALTH_CHECK_INTERVAL:
            return _kernel_session
        try:
            from wolframclient.language import wlexpr

            _kernel_session.evaluate(wlexpr("1"))
            _last_kernel_health_check = time.monotonic()
            return _kernel_session
        except Exception:
            logger.warning("Kernel session unresponsive, recreating...")
            with contextlib.suppress(Exception):
                _kernel_session.terminate()
            _kernel_session = None

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
        _kernel_session = WolframLanguageSession(kernel_path)
        _kernel_session.start()
        _kernel_session.evaluate(wlexpr("1+1"))
        _last_kernel_health_check = time.monotonic()
        logger.info(f"Kernel session ready: {kernel_path}")
        return _kernel_session
    except Exception as e:
        logger.warning(f"WolframLanguageSession failed ({e}), falling back to wolframscript")
        _use_wolframscript = True
        return None


def close_kernel_session():
    global _kernel_session, _last_kernel_health_check
    if _kernel_session is not None:
        with contextlib.suppress(Exception):
            _kernel_session.terminate()
        _kernel_session = None
        _last_kernel_health_check = 0.0
        logger.info("Closed kernel session")


def _is_graphics_output(output: str) -> bool:
    """Check if output looks like a Graphics object."""
    if not output:
        return False
    output_stripped = output.strip()
    # Check for placeholder patterns (used by addon)
    if output_stripped in ["-Graphics-", "-Graphics3D-", "-Image-"]:
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
                image_path = _rasterize_via_wolframscript(wrapped_code)
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
Module[{{res = {wrapped_code}, imgPath = "{wl_raster_path}", didRaster = False}},
  <|
    "inputform" -> ToString[res, InputForm],
    "fullform" -> If[{include_fullform}, ToString[res, FullForm], ""],
    "tex" -> If[{include_tex}, ToString[TeXForm[res]], ""],
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
        combined_result = session.evaluate(wlexpr(eval_code))
        timing_ms = int((time.time() - start_time) * 1000)

        if isinstance(combined_result, dict):
            output_inputform = str(combined_result.get("inputform", str(combined_result.get("result", ""))))
            output_fullform = str(combined_result.get("fullform", ""))
            output_tex = str(combined_result.get("tex", ""))
        else:
            output_inputform = str(combined_result)
            output_fullform = ""
            output_tex = ""

        response: dict[str, Any] = {
            "success": True,
            "output": output_inputform,
            "output_tex": output_tex,
            "output_inputform": output_inputform,
            "output_fullform": output_fullform,
            "warnings": [],
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
