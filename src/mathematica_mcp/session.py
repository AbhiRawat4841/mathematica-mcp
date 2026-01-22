import json
import logging
import platform
import os
import re
import subprocess
import shutil
import time
import zlib
from typing import Any, Optional

logger = logging.getLogger("mathematica_mcp.session")

_kernel_session: Any = None
_use_wolframscript: bool = False


def _session_context(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    crc = zlib.crc32(session_id.encode("utf-8")) & 0xFFFFFFFF
    return f"MCP`{crc:08x}`"


def _wrap_code_for_context(code: str, context: Optional[str]) -> str:
    if not context:
        return code
    return (
        f'Block[{{$Context = "{context}", $ContextPath = {{"{context}", "System`"}}}}, '
        f"{code}]"
    )


def _wrap_code_for_determinism(code: str, deterministic_seed: Optional[int]) -> str:
    if deterministic_seed is None:
        return code
    return f"BlockRandom[SeedRandom[{deterministic_seed}]; {code}]"


def find_wolfram_kernel() -> Optional[str]:
    system = platform.system()
    potential_paths: list[str] = []

    if system == "Darwin":
        potential_paths = [
            "/Applications/Mathematica.app/Contents/MacOS/WolframKernel",
            "/Applications/Wolfram.app/Contents/MacOS/WolframKernel",
            "/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel",
        ]
    elif system == "Windows":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        for version in ["14.2", "14.1", "14.0", "13.3", "13.2", "13.1", "13.0"]:
            potential_paths.append(
                os.path.join(
                    program_files,
                    f"Wolfram Research\\Mathematica\\{version}\\WolframKernel.exe",
                )
            )
    elif system == "Linux":
        for version in ["14.2", "14.1", "14.0", "13.3", "13.2", "13.1", "13.0"]:
            potential_paths.append(
                f"/usr/local/Wolfram/Mathematica/{version}/Executables/WolframKernel"
            )

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
        result["output_inputform"] = (
            input_match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        )

    full_match = re.search(r'"output_fullform"\s*->\s*"((?:[^"\\]|\\.)*)"', output)
    if full_match:
        result["output_fullform"] = (
            full_match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        )

    tex_match = re.search(r'"output_tex"\s*->\s*"((?:[^"\\]|\\.)*)"', output)
    if tex_match:
        result["output_tex"] = (
            tex_match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        )

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
    deterministic_seed: Optional[int] = None,
    session_id: Optional[str] = None,
    isolate_context: bool = False,
) -> dict[str, Any]:
    """Execute code via wolframscript CLI with timing and warning capture."""
    wolframscript = shutil.which("wolframscript")
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

    wrapped_code = f"""
Module[{{startTime, result, messages, timing, response, outInput, outFull="", outTex=""}},
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

  timing = Round[(AbsoluteTime[] - startTime) * 1000];
  
  response = <|
    "output" -> outInput,
    "output_inputform" -> outInput,
    "output_fullform" -> outFull,
    "output_tex" -> outTex,
    "messages" -> ToString[messages],
    "timing_ms" -> timing,
    "failed" -> (result === $Failed)
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
            warnings_list = (
                [messages_str] if messages_str and messages_str != "{}" else []
            )
        except json.JSONDecodeError:
            # Fallback to robust Association parser for older Mathematica
            parsed = _parse_association_output(output)
            clean_output = parsed["output"]
            output_inputform = parsed.get("output_inputform", clean_output)
            output_fullform = parsed.get("output_fullform", "")
            output_tex = parsed.get("output_tex", "")
            messages_str = parsed.get("messages", "")
            warnings_list = (
                [messages_str] if messages_str and messages_str != "{}" else []
            )

        return {
            "success": True,
            "output": output_inputform,
            "output_tex": output_tex,
            "output_inputform": output_inputform,
            "output_fullform": output_fullform,
            "warnings": warnings_list,
            "timing_ms": python_timing,
            "execution_method": "wolframscript",
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": f"Execution timed out after {timeout} seconds",
            "error": "timeout",
            "warnings": [],
            "timing_ms": int(timeout * 1000),
            "execution_method": "wolframscript",
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"wolframscript execution failed: {e}",
            "error": str(e),
            "warnings": [],
            "timing_ms": int((time.time() - start_time) * 1000),
            "execution_method": "wolframscript",
        }


def get_kernel_session():
    global _kernel_session, _use_wolframscript

    if _use_wolframscript:
        return None

    if _kernel_session is not None:
        try:
            from wolframclient.language import wlexpr

            _kernel_session.evaluate(wlexpr("1"))
            return _kernel_session
        except Exception:
            logger.warning("Kernel session unresponsive, recreating...")
            try:
                _kernel_session.terminate()
            except Exception:
                pass
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
        logger.info(f"Kernel session ready: {kernel_path}")
        return _kernel_session
    except Exception as e:
        logger.warning(
            f"WolframLanguageSession failed ({e}), falling back to wolframscript"
        )
        _use_wolframscript = True
        return None


def close_kernel_session():
    global _kernel_session
    if _kernel_session is not None:
        try:
            _kernel_session.terminate()
        except Exception:
            pass
        _kernel_session = None
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


def _rasterize_via_wolframscript(code: str, image_size: int = 500) -> Optional[str]:
    """Rasterize a graphics expression via wolframscript, return temp file path."""
    import tempfile

    wolframscript = shutil.which("wolframscript")
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


def execute_in_kernel(
    code: str,
    output_format: str = "text",
    *,
    render_graphics: bool = True,
    deterministic_seed: Optional[int] = None,
    session_id: Optional[str] = None,
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
        )
        output = result.get("output", "")
        if render_graphics and result.get("success") and _is_graphics_output(output):
            image_path = _rasterize_via_wolframscript(wrapped_code)
            if image_path:
                result["image_path"] = image_path
                result["output"] = f"[Graphics rendered to image: {image_path}]"
                result["is_graphics"] = True
        if result.get("success"):
            _query_cache.put(
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
        )

    start_time = time.time()
    try:
        # OPTIMIZED: Single evaluation - avoids re-evaluating code for each format
        include_fullform = "True" if output_format == "mathematica" else "False"
        include_tex = "True" if output_format == "latex" else "False"

        eval_code = f"""
Module[{{res = {wrapped_code}}},
  <|
    "result" -> res,
    "inputform" -> ToString[res, InputForm],
    "fullform" -> If[{include_fullform}, ToString[res, FullForm], ""],
    "tex" -> If[{include_tex}, ToString[TeXForm[res]], ""]
  |>
]
"""
        combined_result = session.evaluate(wlexpr(eval_code))
        timing_ms = int((time.time() - start_time) * 1000)

        if isinstance(combined_result, dict):
            output_inputform = str(
                combined_result.get("inputform", str(combined_result.get("result", "")))
            )
            output_fullform = str(combined_result.get("fullform", ""))
            output_tex = str(combined_result.get("tex", ""))
        else:
            output_inputform = str(combined_result)
            output_fullform = ""
            output_tex = ""

        response = {
            "success": True,
            "output": output_inputform,
            "output_tex": output_tex,
            "output_inputform": output_inputform,
            "output_fullform": output_fullform,
            "warnings": [],
            "timing_ms": timing_ms,
            "execution_method": "wolframclient",
        }

        if render_graphics and _is_graphics_output(output_inputform):
            image_path = _rasterize_via_wolframscript(wrapped_code)
            if image_path:
                response["image_path"] = image_path
                response["output"] = f"[Graphics rendered to image: {image_path}]"
                response["is_graphics"] = True

        _query_cache.put(
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
        return _execute_via_wolframscript(
            code,
            output_format,
            timeout=timeout,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
        )
