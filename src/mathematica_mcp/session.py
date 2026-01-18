import logging
import platform
import os
import subprocess
import shutil
from typing import Any, Optional

logger = logging.getLogger("mathematica_mcp.session")

_kernel_session: Any = None
_use_wolframscript: bool = False  # Fallback to CLI if wolframclient fails


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
    """Execute code via wolframscript CLI (slower but more reliable)."""
    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return {
            "success": False,
            "output": "wolframscript not found in PATH",
            "error": "wolframscript not available",
        }

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()

        # Remove trailing Null from Print statements
        if output.endswith("\nNull"):
            output = output[:-5].strip()

        if result.returncode != 0:
            return {
                "success": False,
                "output": result.stderr or output,
                "error": result.stderr or "Non-zero exit code",
            }

        # Get TeXForm if requested
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

        return {
            "success": True,
            "output": output,
            "output_tex": tex_output,
            "output_inputform": output,
            "execution_method": "wolframscript",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "Execution timed out after 60 seconds",
            "error": "timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"wolframscript execution failed: {e}",
            "error": str(e),
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
    global _use_wolframscript

    session = get_kernel_session()

    # Use wolframscript if session failed to initialize
    if session is None or _use_wolframscript:
        return _execute_via_wolframscript(code, output_format)

    try:
        from wolframclient.language import wlexpr
    except ImportError:
        return _execute_via_wolframscript(code, output_format)

    try:
        result = session.evaluate(wlexpr(code))

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
            "execution_method": "wolframclient",
        }
    except Exception as e:
        # Try wolframscript as final fallback
        logger.warning(f"wolframclient evaluation failed ({e}), trying wolframscript")
        return _execute_via_wolframscript(code, output_format)
