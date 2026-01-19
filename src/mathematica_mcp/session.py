import json
import logging
import platform
import os
import re
import subprocess
import shutil
import time
from typing import Any, Optional

logger = logging.getLogger("mathematica_mcp.session")

_kernel_session: Any = None
_use_wolframscript: bool = False


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
    code: str, output_format: str = "text"
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

    wrapped_code = f"""
Module[{{startTime, result, messages, timing, response, outInput, outFull="", outTex=""}},
  startTime = AbsoluteTime[];
  Block[{{$MessageList = {{}}}},
    result = Quiet[Check[{code}, $Failed]];
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
            timeout=60,
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
            "output": "Execution timed out after 60 seconds",
            "error": "timeout",
            "warnings": [],
            "timing_ms": 60000,
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
        return None  # Signal to use wolframscript instead

    if _kernel_session is not None:
        return _kernel_session

    try:
        from wolframclient.evaluation import WolframLanguageSession
    except ImportError:
        logger.info("wolframclient not available, falling back to wolframscript")
        _use_wolframscript = True
        return None

    kernel_path = find_wolfram_kernel()

    try:
        _kernel_session = WolframLanguageSession(kernel_path)
        # Test the session
        from wolframclient.language import wlexpr

        _kernel_session.evaluate(wlexpr("1+1"))
        logger.info("Created WolframLanguageSession (fallback kernel mode)")
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


def execute_in_kernel(code: str, output_format: str = "text") -> dict[str, Any]:
    """Execute code in kernel with structured metadata (timing, warnings).

    Automatically rasterizes Graphics results to images for better display.
    """
    global _use_wolframscript

    session = get_kernel_session()

    if session is None or _use_wolframscript:
        result = _execute_via_wolframscript(code, output_format)
        # Check if result is a Graphics object and rasterize it
        output = result.get("output", "")
        if result.get("success") and _is_graphics_output(output):
            image_path = _rasterize_via_wolframscript(code)
            if image_path:
                result["image_path"] = image_path
                result["output"] = f"[Graphics rendered to image: {image_path}]"
                result["is_graphics"] = True
        return result

    try:
        from wolframclient.language import wlexpr
    except ImportError:
        return _execute_via_wolframscript(code, output_format)

    start_time = time.time()
    try:
        result = session.evaluate(wlexpr(code))
        timing_ms = int((time.time() - start_time) * 1000)

        # Always get InputForm
        try:
            input_result = session.evaluate(wlexpr(f"ToString[InputForm[{code}]]"))
            output_inputform = str(input_result)
        except Exception:
            output_inputform = str(result)

        output_fullform = ""
        output_tex = ""

        # Only compute expensive formats if specifically requested
        if output_format == "mathematica":
            output_fullform = str(
                result
            )  # Result object string representation is usually FullForm-like or close enough

        if output_format == "latex":
            try:
                tex_result = session.evaluate(wlexpr(f"ToString[TeXForm[{code}]]"))
                output_tex = str(tex_result)
            except Exception:
                pass

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

        # Check if result is a Graphics object and rasterize it
        if _is_graphics_output(output_inputform):
            image_path = _rasterize_via_wolframscript(code)
            if image_path:
                response["image_path"] = image_path
                response["output"] = f"[Graphics rendered to image: {image_path}]"
                response["is_graphics"] = True

        return response
    except Exception as e:
        logger.warning(f"wolframclient evaluation failed ({e}), trying wolframscript")
        return _execute_via_wolframscript(code, output_format)
