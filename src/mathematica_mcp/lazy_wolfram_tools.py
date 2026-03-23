from __future__ import annotations

import asyncio
import functools
import json
import shutil
import subprocess
from collections.abc import Callable
from typing import Literal


@functools.lru_cache(maxsize=1)
def _find_wolframscript() -> str | None:
    """Locate wolframscript on PATH. Result is cached across calls."""
    return shutil.which("wolframscript")


def _clear_wolframscript_cache() -> None:
    """Force fresh PATH lookup. For tests or explicit environment refresh only."""
    _find_wolframscript.cache_clear()


async def _run_subprocess(*args, **kwargs) -> subprocess.CompletedProcess:
    """Run subprocess.run in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(subprocess.run, *args, **kwargs)


async def verify_derivation(
    steps: list[str],
    format: Literal["text", "latex", "mathematica"] = "text",
    timeout: int = 120,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    if len(steps) < 2:
        return json.dumps(
            {
                "success": False,
                "error": "At least two steps are required for verification",
                "steps_provided": len(steps),
            },
            indent=2,
        )

    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found in PATH"}, indent=2)

    steps_list = ", ".join([f'"{step}"' for step in steps])
    format_fn = "TeXForm" if format == "latex" else "InputForm" if format == "mathematica" else "ToString"

    verification_code = f"""
Module[{{steps, results, i, prev, current, isEqual, simplified, formatExpr}},
  steps = {{{steps_list}}};
  formatExpr = {format_fn};
  results = <|"success" -> True, "steps" -> {{}}, "summary" -> ""|>;
  Do[
    prev = ToExpression[steps[[i]]];
    current = ToExpression[steps[[i + 1]]];
    isEqual = Quiet[Check[
      TrueQ[Simplify[prev == current]] ||
      TrueQ[FullSimplify[prev == current]] ||
      TrueQ[Simplify[prev - current] == 0],
      False
    ]];
    simplified = Quiet[Check[Simplify[current], current]];
    AppendTo[results["steps"], <|
      "from" -> i,
      "to" -> i + 1,
      "expr_from" -> steps[[i]],
      "expr_to" -> steps[[i + 1]],
      "valid" -> isEqual,
      "simplified" -> ToString[formatExpr[simplified]]
    |>];
  , {{i, 1, Length[steps] - 1}}];
  results["all_valid"] = AllTrue[results["steps"], #["valid"] &];
  results["valid_count"] = Count[results["steps"], _?(#["valid"] &)];
  results["total_steps"] = Length[steps] - 1;
  results
]
"""

    try:
        result = await _run_subprocess(
            [wolframscript, "-code", verification_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or "Verification failed",
                    "stdout": result.stdout,
                },
                indent=2,
            )

        raw_output = result.stdout.strip()
        parsed = parse_wolfram_association(raw_output)
        report_lines = ["## Derivation Verification Report\n"]
        if isinstance(parsed, dict) and "steps" in parsed:
            steps_data = parsed.get("steps", [])
            all_valid = parsed.get("all_valid", False)
            for step_info in steps_data:
                if isinstance(step_info, dict):
                    from_idx = step_info.get("from", "?")
                    to_idx = step_info.get("to", "?")
                    valid = step_info.get("valid", False)
                    expr_from = step_info.get("expr_from", "")
                    expr_to = step_info.get("expr_to", "")
                    status = "✓ VALID" if valid else "✗ INVALID"
                    report_lines.append(f"Step {from_idx} → {to_idx}: {status}")
                    report_lines.append(f"  From: {expr_from}")
                    report_lines.append(f"  To:   {expr_to}")
                    report_lines.append("")
            summary = "All steps are valid!" if all_valid else "Some steps failed verification."
            report_lines.append(f"**Summary**: {summary}")
            report_lines.append(f"Valid: {parsed.get('valid_count', 0)}/{parsed.get('total_steps', 0)} steps")
        else:
            report_lines.append("Could not parse verification results.")
            report_lines.append(f"Raw output: {raw_output}")

        return json.dumps(
            {
                "success": True,
                "report": "\n".join(report_lines),
                "raw_data": parsed,
                "format": format,
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": f"Verification timed out after {timeout} seconds"},
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Verification failed: {str(e)}"}, indent=2)


async def get_kernel_state(*, parse_wolfram_association: Callable[[str], dict]) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found in PATH"}, indent=2)

    state_code = """
<|
  "success" -> True,
  "kernel_version" -> $VersionNumber,
  "version_string" -> $Version,
  "system_id" -> $SystemID,
  "machine_name" -> $MachineName,
  "memory_in_use" -> MemoryInUse[],
  "memory_in_use_mb" -> Round[MemoryInUse[] / 1024.0 / 1024.0, 0.1],
  "max_memory_used" -> MaxMemoryUsed[],
  "loaded_packages" -> Quiet[Check[
    Select[Contexts[], StringMatchQ[#, __ ~~ "`"] && !StringStartsQ[#, "System`"] && !StringStartsQ[#, "Global`"] &],
    {}
  ]],
  "global_symbols" -> Quiet[Check[
    Take[Names["Global`*"], UpTo[50]],
    {}
  ]],
  "global_symbol_count" -> Length[Names["Global`*"]],
  "session_time" -> SessionTime[],
  "process_id" -> $ProcessID
|>
"""
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", state_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return json.dumps(
                {"success": False, "error": result.stderr or "State query failed"},
                indent=2,
            )
        raw_output = result.stdout.strip()
        parsed = parse_wolfram_association(raw_output)
        return json.dumps(parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def load_package(package_name: str, *, parse_wolfram_association: Callable[[str], dict]) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found in PATH"}, indent=2)

    if not package_name.endswith("`"):
        package_name = package_name + "`"

    load_code = f"""
Module[{{beforeContexts, afterContexts, newSymbols, result}},
  beforeContexts = Contexts[];
  result = Quiet[Check[Needs["{package_name}"]; "loaded", "failed"]];
  afterContexts = Contexts[];
  newSymbols = Complement[afterContexts, beforeContexts];
  <|
    "success" -> (result === "loaded"),
    "package" -> "{package_name}",
    "new_contexts" -> newSymbols,
    "message" -> If[result === "loaded", "Package loaded successfully", "Failed to load package"]
  |>
]
"""
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", load_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw_output = result.stdout.strip()
        parsed = parse_wolfram_association(raw_output)
        return json.dumps(parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "package": package_name}, indent=2)


async def list_loaded_packages(*, parse_wolfram_association: Callable[[str], dict]) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found in PATH"}, indent=2)

    list_code = """
Module[{pkgs},
  pkgs = Select[
    Contexts[],
    StringMatchQ[#, __ ~~ "`"] &&
    !StringStartsQ[#, "System`"] &&
    !StringStartsQ[#, "Global`"] &&
    !StringStartsQ[#, "Internal`"] &
  ];
  <|"success" -> True, "packages" -> Sort[pkgs], "count" -> Length[pkgs]|>
]
"""
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", list_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        raw_output = result.stdout.strip()
        parsed = parse_wolfram_association(raw_output)
        return json.dumps(parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def wolfram_alpha(
    query: str,
    return_type: Literal["result", "data", "full"] = "result",
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)

    safe_query = query.replace('"', '\\"')
    if return_type == "full":
        code = f'''
Module[{{result}},
  result = Quiet[Check[WolframAlpha["{safe_query}", "FullOutput"], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> "{safe_query}", "result" -> ToString[result, InputForm]|>
  ]
]
'''
    elif return_type == "data":
        code = f'''
Module[{{result}},
  result = Quiet[Check[WolframAlpha["{safe_query}", {{"Result", "Input"}}], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> "{safe_query}", "result" -> ToString[result, InputForm]|>
  ]
]
'''
    else:
        code = f'''
Module[{{result}},
  result = Quiet[Check[WolframAlpha["{safe_query}", "Result"], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> "{safe_query}", "result" -> ToString[result]|>
  ]
]
'''
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Query timed out"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def interpret_natural_language(text: str) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)

    safe_text = text.replace('"', '\\"')
    code = """
Module[{query, props, inputSpec, inputExpr, inputForm, resultExpr},
  query = "__MCP_QUERY__";
  props = Quiet[Check[WolframAlpha[query, "Properties"], {}]];
  inputSpec = SelectFirst[
    props,
    (Length[#] == 2 && #[[2]] === "Input") &,
    Missing["NotAvailable"]
  ];
  inputExpr = If[inputSpec === Missing["NotAvailable"],
    Missing["NotAvailable"],
    Quiet[Check[WolframAlpha[query, {inputSpec[[1]], "Input"}], Missing["NotAvailable"]]]
  ];
  inputForm = Which[
    inputExpr === Missing["NotAvailable"] || inputExpr === {}, "",
    Head[inputExpr] === HoldComplete,
      ToString[Unevaluated[inputExpr /. HoldComplete[x_] :> x], InputForm],
    True,
      ToString[Unevaluated[inputExpr], InputForm]
  ];
  resultExpr = Quiet[Check[WolframAlpha[query, "Result"], $Failed]];
  If[resultExpr === $Failed || resultExpr === Missing["NotAvailable"] || resultExpr === {},
    ExportString[<|"success" -> False, "error" -> "Could not interpret text"|>, "JSON"],
    ExportString[
      <|
        "success" -> True,
        "input" -> query,
        "wolfram_code" -> inputForm,
        "result" -> ToString[resultExpr, InputForm],
        "tex" -> Quiet[Check[ToString[TeXForm[resultExpr]], ""]]
      |>,
      "JSON"
    ]
  ]
]
""".replace("__MCP_QUERY__", safe_text)
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return json.dumps({"success": False, "error": result.stderr or "Query failed"}, indent=2)
        output = result.stdout.strip()
        if not output:
            return json.dumps({"success": False, "error": "Empty WolframAlpha response"}, indent=2)
        try:
            return json.dumps(json.loads(output), indent=2)
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to parse WolframAlpha response: {e}",
                    "raw": output,
                },
                indent=2,
            )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def entity_lookup(
    entity_type: str,
    name: str,
    properties: list[str] | None = None,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)

    safe_name = name.replace('"', '\\"')
    if properties:
        props_str = "{" + ", ".join(f'"{p}"' for p in properties) + "}"
        code = f'''
Module[{{entity, data}},
  entity = Quiet[Check[Entity["{entity_type}", "{safe_name}"], $Failed]];
  If[entity === $Failed || !EntityQ[entity],
    <|"success" -> False, "error" -> "Entity not found"|>,
    data = EntityValue[entity, {props_str}];
    <|"success" -> True, "entity_type" -> "{entity_type}", "name" -> "{safe_name}", "properties" -> Map[ToString, data]|>
  ]
]
'''
    else:
        code = f'''
Module[{{entity, props, data}},
  entity = Quiet[Check[Entity["{entity_type}", "{safe_name}"], $Failed]];
  If[entity === $Failed || !EntityQ[entity],
    <|"success" -> False, "error" -> "Entity not found"|>,
    props = Take[EntityProperties[entity], UpTo[10]];
    data = EntityValue[entity, props];
    <|"success" -> True, "entity_type" -> "{entity_type}", "name" -> EntityValue[entity, "Name"], "properties" -> MapThread[Rule, {{props, Map[ToString, data]}}]|>
  ]
]
'''
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def convert_units(
    quantity: str,
    target_unit: str,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)

    safe_qty = quantity.replace('"', '\\"')
    safe_unit = target_unit.replace('"', '\\"')
    code = f'''
Module[{{input, result}},
  input = Quiet[Check[Interpreter["Quantity"]["{safe_qty}"], $Failed]];
  If[input === $Failed,
    <|"success" -> False, "error" -> "Could not parse quantity"|>,
    result = Quiet[Check[UnitConvert[input, "{safe_unit}"], $Failed]];
    If[result === $Failed,
      <|"success" -> False, "error" -> "Conversion failed"|>,
      <|"success" -> True, "input" -> "{safe_qty}", "target_unit" -> "{safe_unit}", "result" -> ToString[result], "numeric" -> ToString[QuantityMagnitude[result]]|>
    ]
  ]
]
'''
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def get_constant(name: str, *, parse_wolfram_association: Callable[[str], dict]) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)
    code = f'''
Module[{{constant, val, numeric}},
  constant = Quiet[Check[ToExpression["{name}"], $Failed]];
  If[constant === $Failed,
    constant = Quiet[Check[Quantity["{name}"], $Failed]]
  ];
  If[constant === $Failed,
    <|"success" -> False, "error" -> "Constant not found"|>,
    <|"success" -> True, "name" -> "{name}", "exact" -> ToString[constant, InputForm], "numeric" -> ToString[N[constant, 15]], "tex" -> Quiet[Check[ToString[TeXForm[constant]], ""]]|>
  ]
]
'''
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def trace_evaluation(
    expression: str,
    max_depth: int = 5,
    *,
    addon_result: Callable[[str, dict | None], dict],
) -> str:
    result = await addon_result("trace_evaluation", {"expression": expression, "max_depth": max_depth})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def time_expression(expression: str, *, addon_result: Callable[[str, dict | None], dict]) -> str:
    result = await addon_result("time_expression", {"expression": expression})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def check_syntax(code: str, *, addon_result: Callable[[str, dict | None], dict]) -> str:
    result = await addon_result("check_syntax", {"code": code})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def import_data(
    path: str,
    format: str | None = None,
    *,
    addon_result: Callable[[str, dict | None], dict],
    expand_path: Callable[[str], str],
) -> str:
    expanded = expand_path(path) if not path.startswith("http") else path
    result = await addon_result("import_data", {"path": expanded, "format": format or "Automatic"})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def export_data(
    expression: str,
    path: str,
    format: str | None = None,
    *,
    addon_result: Callable[[str, dict | None], dict],
    expand_path: Callable[[str], str],
) -> str:
    expanded = expand_path(path)
    result = await addon_result(
        "export_data",
        {"expression": expression, "path": expanded, "format": format or "Automatic"},
    )
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def list_supported_formats(
    *,
    addon_result: Callable[[str, dict | None], dict],
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    result = await addon_result("list_import_formats", {})
    if result.get("error"):
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)
        code = """
<|
  "success" -> True,
  "import_formats" -> $ImportFormats,
  "export_formats" -> $ExportFormats,
  "import_count" -> Length[$ImportFormats],
  "export_count" -> Length[$ExportFormats]
|>
"""
        try:
            result = await _run_subprocess(
                [wolframscript, "-code", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
    return json.dumps(result, indent=2)


async def inspect_graphics(expression: str, *, parse_wolfram_association: Callable[[str], dict]) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)
    code = f'''
Module[{{g, result}},
  g = Quiet[Check[ToExpression["{expression}"], $Failed]];
  If[g === $Failed || !MatchQ[Head[g], Graphics|Graphics3D|Graph|GeoGraphics],
    <|"success" -> False, "error" -> "Not a graphics object"|>,
    <|
      "success" -> True,
      "head" -> ToString[Head[g]],
      "primitives" -> Union[Cases[g, h_Symbol /; Context[h] === "System`" :> ToString[h], Infinity]],
      "options" -> ToString[Options[g], InputForm],
      "plot_range" -> ToString[Quiet[PlotRange /. AbsoluteOptions[g]], InputForm],
      "image_size" -> ToString[Quiet[ImageSize /. AbsoluteOptions[g]], InputForm]
    |>
  ]
]
'''
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def export_graphics(
    expression: str,
    path: str,
    format: Literal["PNG", "PDF", "SVG", "EPS", "JPEG"] = "PNG",
    size: int = 600,
    *,
    addon_result: Callable[[str, dict | None], dict],
    expand_path: Callable[[str], str],
) -> str:
    expanded = expand_path(path)
    result = await addon_result(
        "export_graphics",
        {"expression": expression, "path": expanded, "format": format, "size": size},
    )
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def compare_plots(
    expressions: list[str],
    labels: list[str] | None = None,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)
    plots_list = "{" + ", ".join(expressions) + "}"
    labels_code = "None" if not labels else "{" + ", ".join(f'"{label}"' for label in labels) + "}"
    code = f"""
Module[{{plots, labels, grid}},
  plots = {plots_list};
  labels = {labels_code};
  grid = If[labels === None,
    GraphicsRow[plots],
    GraphicsRow[MapThread[Labeled[#1, #2, Top] &, {{plots, labels}}]]
  ];
  <|"success" -> True, "combined_expression" -> ToString[grid, InputForm], "plot_count" -> Length[plots]|>
]
"""
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def create_animation(
    expression: str,
    parameter: str,
    range_spec: str,
    frames: int = 20,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wolframscript = _find_wolframscript()
    if not wolframscript:
        return json.dumps({"success": False, "error": "wolframscript not found"}, indent=2)
    code = f'''
Module[{{expr, param, range, anim}},
  expr = Hold[{expression}];
  range = {{{parameter}, {range_spec}}};
  anim = Table[
    ReleaseHold[expr /. {parameter} -> val],
    {{val, range[[2]], range[[3]], (range[[3]] - range[[2]])/{frames}}}
  ];
  <|"success" -> True, "frame_count" -> Length[anim], "parameter" -> "{parameter}", "animation_expression" -> ToString[ListAnimate[anim], InputForm]|>
]
'''
    try:
        result = await _run_subprocess(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)
