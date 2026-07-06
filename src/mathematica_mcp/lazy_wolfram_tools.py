from __future__ import annotations

import asyncio
import functools
import json
import shutil
from collections.abc import Awaitable, Callable
from typing import Literal


@functools.lru_cache(maxsize=1)
def _find_wolframscript() -> str | None:
    """Locate wolframscript on PATH. Result is cached across calls."""
    return shutil.which("wolframscript")


def _clear_wolframscript_cache() -> None:
    """Force fresh PATH lookup. For tests or explicit environment refresh only."""
    _find_wolframscript.cache_clear()


def _wl_string(s: str) -> str:
    """Return ``s`` as a safely-escaped, double-quoted WL string literal.

    Escapes backslash first, then double-quote, so arbitrary Python text can be
    interpolated where a WL string literal is expected without breaking out of the
    quotes (WL injection). e.g. ``Quantity[100, "Centimeters"]`` survives intact
    instead of mangling the surrounding expression.
    """
    escaped = s.replace(chr(92), chr(92) * 2).replace('"', '\\"')
    return f'"{escaped}"'


def _scratch_block(code: str) -> str:
    """Wrap user-expression WL ``code`` so it evaluates in a throwaway context.

    Any symbol defined while parsing/evaluating user input lands in ``MCPScratch```
    instead of the shared persistent kernel's ``Global```, so ``ToExpression[userInput]``
    can't leak Global` state across calls. Mirrors ``session._wrap_code_for_context``.
    """
    return f'Block[{{$Context = "MCPScratch`", $ContextPath = {{"System`"}}}}, {code}]'


def _json_wl(code: str) -> str:
    """Wrap WL ``code`` (which returns an Association) so its final result is a
    JSON string via ``ExportString[..., "JSON"]``.

    A String result renders verbatim through ``evaluate_wl``'s OutputForm path
    (and cold wolframscript stdout), so the JSON round-trips to ``json.loads``.
    This replaces parsing the Association's OutputForm rendering with the fragile
    regex parser, which returned ``parse_error`` fallbacks for expression-valued
    fields (e.g. get_kernel_state). If the Association holds non-JSON-safe leaves
    (Quantity, Entity, held expressions, Rationals), a second attempt recursively
    stringifies them via the Module-local ``mcpSan`` (an explicit recursion, NOT
    ReplaceAll: ReplaceAll visits heads, so the ``Association`` head symbol itself
    would get stringified and the export would always fail). Rule keys are
    stringified too (entity_lookup returns ``EntityProperty[...] -> value`` pairs).
    Only if that also fails does the raw Association pass through for the caller's
    regex fallback.
    """
    return f"""Module[{{mcpRes, mcpJson, mcpSan}},
  mcpRes = ({code});
  mcpJson = Quiet[Check[ExportString[mcpRes, "JSON", "Compact" -> True], $Failed]];
  If[!StringQ[mcpJson],
    mcpSan[x_Association] := Association[KeyValueMap[
      If[StringQ[#1], #1, ToString[#1, InputForm]] -> mcpSan[#2] &, x]];
    mcpSan[x_List] := Map[mcpSan, x];
    mcpSan[x_Rule] := Rule[
      If[StringQ[First[x]], First[x], ToString[First[x], InputForm]],
      mcpSan[Last[x]]];
    mcpSan[x : (_String | _Integer | _Real | True | False | Null)] := x;
    mcpSan[x_] := ToString[x, InputForm];
    mcpJson = Quiet[Check[ExportString[mcpSan[mcpRes], "JSON", "Compact" -> True], $Failed]]
  ];
  If[StringQ[mcpJson], mcpJson, mcpRes]
]"""


async def _run_wl_parsed(
    code: str,
    parse_wolfram_association: Callable[[str], dict],
    timeout: int = 30,
    *,
    allow_addon_fallback: bool = False,
) -> str:
    """Evaluate WL ``code`` warm-first (persistent session, cold wolframscript
    fallback), parse the Association output, and return it as a JSON string with
    ``execution_method`` attached. Shared by the migrated cold tools so warmth and
    the cold-execution counter live in one place (session.evaluate_wl).

    The Association is JSON-exported kernel-side (see ``_json_wl``) and parsed
    with ``json.loads``; the regex Association parser remains only as a fallback
    for kernels/outputs where the JSON export failed.

    ``allow_addon_fallback`` is forwarded to ``evaluate_wl``: pass it ONLY from
    scratch-wrapped pure-math callers (unqualified names are redirected to a
    throwaway context; the addon is the user's front-end kernel, and
    context-qualified input still reaches it). See ``evaluate_wl`` for the full
    safety contract.
    """
    from .session import evaluate_wl

    # Forward the opt-in only when enabled so the default funnel call keeps
    # evaluate_wl's plain (code, timeout) shape.
    addon_kw = {"allow_addon_fallback": True} if allow_addon_fallback else {}
    wl = await asyncio.to_thread(evaluate_wl, _json_wl(code), timeout, **addon_kw)
    if not wl.success:
        return json.dumps(
            {
                "success": False,
                "error": wl.error or "Execution failed",
                "execution_method": wl.execution_method,
            },
            indent=2,
        )
    try:
        parsed = json.loads(wl.text)
    except (json.JSONDecodeError, ValueError):
        parsed = parse_wolfram_association(wl.text)
    if not isinstance(parsed, dict):
        # A whole-result $Failed sanitizes to the string "$Failed" — that is a
        # kernel-side failure, not a success carrying data.
        if parsed == "$Failed":
            parsed = {"success": False, "error": "kernel returned $Failed", "result": parsed}
        else:
            parsed = {"success": True, "result": parsed}
    parsed.setdefault("execution_method", wl.execution_method)
    return json.dumps(parsed, indent=2)


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

    steps_list = ", ".join(_wl_string(step) for step in steps)
    format_fn = "TeXForm" if format == "latex" else "InputForm" if format == "mathematica" else "ToString"

    verification_code = _scratch_block(f"""
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
  ExportString[results, "JSON"]
]
""")

    from .session import evaluate_wl

    try:
        # Scratch-wrapped: the derivation is checked inside MCPScratch` (see
        # _scratch_block), so unqualified names stay off shared state and it may
        # take the addon rung when the warm session is not ready. Context-qualified
        # names in a step still reach the kernel; see evaluate_wl's opt-in contract.
        wl = await asyncio.to_thread(evaluate_wl, verification_code, timeout, allow_addon_fallback=True)
        if not wl.success:
            return json.dumps(
                {
                    "success": False,
                    "error": wl.error or "Verification failed",
                    "stdout": wl.text,
                    "execution_method": wl.execution_method,
                },
                indent=2,
            )

        raw_output = wl.text
        # ponytail: WL now emits ExportString[results, "JSON"], which evaluate_wl
        # renders verbatim via OutputForm (same proven path as interpret_natural_language),
        # so json.loads round-trips even for expression-valued fields. The fragile
        # OutputForm-Association regex stays only as a fallback for non-JSON kernel output.
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
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
                "execution_method": wl.execution_method,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Verification failed: {str(e)}"}, indent=2)


async def get_kernel_state(*, parse_wolfram_association: Callable[[str], dict]) -> str:
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
    return await _run_wl_parsed(state_code, parse_wolfram_association, timeout=30)


async def load_package(package_name: str, *, parse_wolfram_association: Callable[[str], dict]) -> str:
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
    return await _run_wl_parsed(load_code, parse_wolfram_association, timeout=60)


async def list_loaded_packages(*, parse_wolfram_association: Callable[[str], dict]) -> str:
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
    return await _run_wl_parsed(list_code, parse_wolfram_association, timeout=15)


async def wolfram_alpha(
    query: str,
    return_type: Literal["result", "data", "full"] = "result",
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wl_query = _wl_string(query)
    if return_type == "full":
        code = f"""
Module[{{result}},
  result = Quiet[Check[WolframAlpha[{wl_query}, "FullOutput"], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> {wl_query}, "result" -> ToString[result, InputForm]|>
  ]
]
"""
    elif return_type == "data":
        code = f"""
Module[{{result}},
  result = Quiet[Check[WolframAlpha[{wl_query}, {{"Result", "Input"}}], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> {wl_query}, "result" -> ToString[result, InputForm]|>
  ]
]
"""
    else:
        code = f"""
Module[{{result}},
  result = Quiet[Check[WolframAlpha[{wl_query}, "Result"], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> {wl_query}, "result" -> ToString[result]|>
  ]
]
"""
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=60)


async def interpret_natural_language(text: str) -> str:
    code = """
Module[{query, props, inputSpec, inputExpr, inputForm, resultExpr},
  query = __MCP_QUERY__;
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
""".replace("__MCP_QUERY__", _wl_string(text))
    from .session import evaluate_wl

    wl = await asyncio.to_thread(evaluate_wl, code, 30)
    if not wl.success:
        return json.dumps(
            {"success": False, "error": wl.error or "Query failed", "execution_method": wl.execution_method},
            indent=2,
        )
    output = wl.text
    if not output:
        return json.dumps(
            {
                "success": False,
                "error": "Empty WolframAlpha response",
                "execution_method": wl.execution_method,
            },
            indent=2,
        )
    # ponytail: the WL returns ExportString[..., "JSON"], and evaluate_wl renders it
    # via ToString[jsonString, OutputForm] which emits the raw JSON verbatim (escapes
    # intact), so json.loads round-trips. Falls back to reporting raw on any mismatch.
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to parse WolframAlpha response: {e}",
                "raw": output,
            },
            indent=2,
        )
    if isinstance(parsed, dict):
        parsed.setdefault("execution_method", wl.execution_method)
    return json.dumps(parsed, indent=2)


async def entity_lookup(
    entity_type: str,
    name: str,
    properties: list[str] | None = None,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wl_name = _wl_string(name)
    wl_type = _wl_string(entity_type)
    if properties:
        props_str = "{" + ", ".join(_wl_string(p) for p in properties) + "}"
        code = f"""
Module[{{entity, data}},
  entity = Quiet[Check[Entity[{wl_type}, {wl_name}], $Failed]];
  If[entity === $Failed || !EntityQ[entity],
    <|"success" -> False, "error" -> "Entity not found"|>,
    data = EntityValue[entity, {props_str}];
    <|"success" -> True, "entity_type" -> {wl_type}, "name" -> {wl_name}, "properties" -> Map[ToString, data]|>
  ]
]
"""
    else:
        code = f"""
Module[{{entity, props, data}},
  entity = Quiet[Check[Entity[{wl_type}, {wl_name}], $Failed]];
  If[entity === $Failed || !EntityQ[entity],
    <|"success" -> False, "error" -> "Entity not found"|>,
    props = Take[EntityProperties[entity], UpTo[10]];
    data = EntityValue[entity, props];
    <|"success" -> True, "entity_type" -> {wl_type}, "name" -> EntityValue[entity, "Name"], "properties" -> MapThread[Rule, {{props, Map[ToString, data]}}]|>
  ]
]
"""
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=30)


async def convert_units(
    quantity: str,
    target_unit: str,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    wl_qty = _wl_string(quantity)
    wl_unit = _wl_string(target_unit)
    code = f"""
Module[{{input, result}},
  input = Quiet[Check[Interpreter["Quantity"][{wl_qty}], $Failed]];
  If[input === $Failed,
    <|"success" -> False, "error" -> "Could not parse quantity"|>,
    result = Quiet[Check[UnitConvert[input, {wl_unit}], $Failed]];
    If[result === $Failed,
      <|"success" -> False, "error" -> "Conversion failed"|>,
      <|"success" -> True, "input" -> {wl_qty}, "target_unit" -> {wl_unit}, "result" -> ToString[result], "numeric" -> ToString[QuantityMagnitude[result]]|>
    ]
  ]
]
"""
    # Kernel-independent (Module-local Quantity/UnitConvert, no Global` touch) —
    # opts into the addon rung. See evaluate_wl's opt-in contract.
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=30, allow_addon_fallback=True)


async def get_constant(name: str, *, parse_wolfram_association: Callable[[str], dict]) -> str:
    wl_name = _wl_string(name)
    code = _scratch_block(f"""
Module[{{constant, val, numeric}},
  constant = Quiet[Check[ToExpression[{wl_name}], $Failed]];
  If[constant === $Failed,
    constant = Quiet[Check[Quantity[{wl_name}], $Failed]]
  ];
  If[constant === $Failed,
    <|"success" -> False, "error" -> "Constant not found"|>,
    <|"success" -> True, "name" -> {wl_name}, "exact" -> ToString[constant, InputForm], "numeric" -> ToString[N[constant, 15]], "tex" -> Quiet[Check[ToString[TeXForm[constant]], ""]]|>
  ]
]
""")
    # Scratch-wrapped constant lookup (unqualified names redirected; a
    # context-qualified name in `name` still reaches the kernel): opts into the
    # addon rung. See evaluate_wl's opt-in contract.
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=30, allow_addon_fallback=True)


async def trace_evaluation(
    expression: str,
    max_depth: int = 5,
    *,
    addon_result: Callable[[str, dict | None], Awaitable[dict]],
) -> str:
    result = await addon_result("trace_evaluation", {"expression": expression, "max_depth": max_depth})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def time_expression(expression: str, *, addon_result: Callable[[str, dict | None], Awaitable[dict]]) -> str:
    result = await addon_result("time_expression", {"expression": expression})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def check_syntax(code: str, *, addon_result: Callable[[str, dict | None], Awaitable[dict]]) -> str:
    result = await addon_result("check_syntax", {"code": code})
    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)
    return json.dumps(result, indent=2)


async def import_data(
    path: str,
    format: str | None = None,
    *,
    addon_result: Callable[[str, dict | None], Awaitable[dict]],
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
    addon_result: Callable[[str, dict | None], Awaitable[dict]],
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
    addon_result: Callable[[str, dict | None], Awaitable[dict]],
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    result = await addon_result("list_import_formats", {})
    if result.get("error"):
        code = """
<|
  "success" -> True,
  "import_formats" -> $ImportFormats,
  "export_formats" -> $ExportFormats,
  "import_count" -> Length[$ImportFormats],
  "export_count" -> Length[$ExportFormats]
|>
"""
        return await _run_wl_parsed(code, parse_wolfram_association, timeout=30)
    return json.dumps(result, indent=2)


async def inspect_graphics(expression: str, *, parse_wolfram_association: Callable[[str], dict]) -> str:
    code = _scratch_block(f"""
Module[{{g, result}},
  g = Quiet[Check[ToExpression[{_wl_string(expression)}], $Failed]];
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
""")
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=30)


async def export_graphics(
    expression: str,
    path: str,
    format: Literal["PNG", "PDF", "SVG", "EPS", "JPEG"] = "PNG",
    size: int = 600,
    *,
    addon_result: Callable[[str, dict | None], Awaitable[dict]],
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
    plots_list = "{" + ", ".join(expressions) + "}"
    labels_code = "None" if not labels else "{" + ", ".join(_wl_string(label) for label in labels) + "}"
    code = _scratch_block(f"""
Module[{{plots, labels, grid}},
  plots = {plots_list};
  labels = {labels_code};
  grid = If[labels === None,
    GraphicsRow[plots],
    GraphicsRow[MapThread[Labeled[#1, #2, Top] &, {{plots, labels}}]]
  ];
  <|"success" -> True, "combined_expression" -> ToString[grid, InputForm], "plot_count" -> Length[plots]|>
]
""")
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=60)


async def create_animation(
    expression: str,
    parameter: str,
    range_spec: str,
    frames: int = 20,
    *,
    parse_wolfram_association: Callable[[str], dict],
) -> str:
    code = _scratch_block(f"""
Module[{{expr, param, range, anim}},
  expr = Hold[{expression}];
  range = {{{parameter}, {range_spec}}};
  anim = Table[
    ReleaseHold[expr /. {parameter} -> val],
    {{val, range[[2]], range[[3]], (range[[3]] - range[[2]])/{frames}}}
  ];
  <|"success" -> True, "frame_count" -> Length[anim], "parameter" -> {_wl_string(parameter)}, "animation_expression" -> ToString[ListAnimate[anim], InputForm]|>
]
""")
    return await _run_wl_parsed(code, parse_wolfram_association, timeout=60)
