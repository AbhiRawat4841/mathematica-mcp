import logging
import platform
import os
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

    wrapped_code = f"""
Module[{{startTime, result, messages, timing}},
  startTime = AbsoluteTime[];
  Block[{{$MessageList = {{}}}},
    result = Quiet[Check[{code}, $Failed]];
    messages = $MessageList;
  ];
  timing = Round[(AbsoluteTime[] - startTime) * 1000];
  <|
    "output" -> ToString[result, InputForm],
    "messages" -> ToString[messages],
    "timing_ms" -> timing,
    "failed" -> (result === $Failed)
  |>
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

        warnings_list = []
        if "messages" in output and "{" in output:
            import re

            msg_match = re.search(r'"messages"\s*->\s*"([^"]*)"', output)
            if msg_match and msg_match.group(1) and msg_match.group(1) != "{}":
                warnings_list = [msg_match.group(1)]

        tex_output = ""
        if output_format == "latex":
            try:
                tex_result = subprocess.run(
                    [wolframscript, "-code", f"ToString[TeXForm[{code}]]"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                tex_output = tex_result.stdout.strip()
            except Exception:
                pass

        clean_output = output
        import re

        out_match = re.search(r'"output"\s*->\s*"([^"]*)"', output)
        if out_match:
            clean_output = out_match.group(1)

        return {
            "success": True,
            "output": clean_output,
            "output_tex": tex_output,
            "output_inputform": clean_output,
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


def execute_in_kernel(code: str, output_format: str = "text") -> dict[str, Any]:
    """Execute code in kernel with structured metadata (timing, warnings)."""
    global _use_wolframscript

    session = get_kernel_session()

    if session is None or _use_wolframscript:
        return _execute_via_wolframscript(code, output_format)

    try:
        from wolframclient.language import wlexpr
    except ImportError:
        return _execute_via_wolframscript(code, output_format)

    start_time = time.time()
    try:
        result = session.evaluate(wlexpr(code))
        timing_ms = int((time.time() - start_time) * 1000)

        output = str(result)
        tex_output = ""

        try:
            tex_result = session.evaluate(wlexpr(f"ToString[TeXForm[{code}]]"))
            tex_output = str(tex_result)
        except Exception:
            pass

        return {
            "success": True,
            "output": output,
            "output_tex": tex_output,
            "output_inputform": str(result),
            "warnings": [],
            "timing_ms": timing_ms,
            "execution_method": "wolframclient",
        }
    except Exception as e:
        logger.warning(f"wolframclient evaluation failed ({e}), trying wolframscript")
        return _execute_via_wolframscript(code, output_format)
