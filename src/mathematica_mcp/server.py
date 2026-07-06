import asyncio
import contextlib
import contextvars
import difflib
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP, Image

from .cache import (
    _query_cache,
    bump_kernel_epoch,
    clear_cache,
    get_cached_screenshot,
    get_notebook_epoch,
    list_cached_expressions,
    put_cached_screenshot,
)
from .cache import (
    cache_expression as _cache_expr,
)
from .cache import (
    get_cached_expression as _get_cached,
)
from .config import FEATURES
from .connection import get_mathematica_connection
from .constants import ExecutionPath as _EP
from .error_analyzer import analyze_messages, format_error_for_llm
from .guidance import (
    build_mathematica_expert_prompt,
    build_prompt_calculate,
    build_prompt_interactive,
    build_prompt_new_notebook,
    build_prompt_notebook,
    build_prompt_quickstart,
    build_server_instructions,
    create_notebook_doc,
    evaluate_cell_doc,
    evaluate_selection_doc,
    execute_code_doc,
    legacy_notebook_doc,
    read_notebook_doc,
    write_cell_doc,
)
from .session import (
    clear_raster_cache,
    close_kernel_session,
    cold_execution_count,
    execute_in_kernel,
    get_kernel_session,
    has_existing_kernel_session,
    kernel_idle_timeout,
)
from .telemetry import get_usage_stats, reset_stats
from .transport_classification import (
    ACTIONABLE_ERROR_FAMILIES as _HARD_ERROR_FAMILIES,
)
from .transport_classification import (
    classify_final_transport as _classify_transport,
)
from .transport_classification import (
    extract_error_families as _extract_error_families,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mathematica_mcp")

mcp = FastMCP(
    "mathematica-mcp",
    instructions=build_server_instructions(FEATURES),
)
_CORE_TOOL_REGISTRY: list[tuple[str, Callable[..., Any]]] = []

# --- Execution style presets ---
# style bundles only output_target + mode. sync/max_wait/timeout are orthogonal.
_STYLE_DEFAULTS: dict[str, dict[str, str]] = {
    "compute": {"output_target": "cli", "mode": "kernel"},
    "notebook": {"output_target": "notebook", "mode": "kernel"},
    "interactive": {"output_target": "notebook", "mode": "frontend"},
}

# --- Interactive content auto-routing ---
# A guidance-blind client (Codex does not read MCP server instructions) or the
# lean `evaluate` tool (which has no style/mode param) can send Manipulate/Dynamic
# to a notebook and land in kernel mode, whose output writer renders these heads as
# dead InputForm text instead of a live panel: the user sees frozen text where a
# slider should be. When the caller pinned neither style nor mode, execute_code
# detects these heads and forces frontend mode so the panel actually renders.
# Explicit style/mode always wins. The lookbehind is word-anchored so symbol names
# like "ManipulateData"/"ManipulatePlot" do NOT match; false positives (the word
# inside a string) merely get frontend's pending contract, and false negatives keep
# today's kernel behaviour.
_INTERACTIVE_HEADS = ("Manipulate", "DynamicModule", "Dynamic", "Animate", "ListAnimate")
_INTERACTIVE_CODE_RE = re.compile(r"(?<![A-Za-z0-9$`])(?:" + "|".join(_INTERACTIVE_HEADS) + r")\s*\[")


def _is_interactive_code(code: str) -> bool:
    """True if *code* calls an interactive head (Manipulate/Dynamic/...) at word boundary."""
    return bool(_INTERACTIVE_CODE_RE.search(code))


_INTERACTIVE_AUTO_ROUTE_NOTE = (
    "Interactive content (Manipulate/Dynamic/Animate) was detected and auto-routed to "
    "frontend mode so it renders as a live panel; kernel mode would have written it as "
    "dead InputForm text. Pass style/mode explicitly to override."
)


_VALID_RESPONSE_DETAILS = frozenset({"compact", "standard", "verbose", "diagnostic"})
_RESPONSE_DETAIL_ALIASES = {
    "short": "compact",
    "medium": "standard",
    "long": "verbose",
}


def _resolve_execution_params(
    style: str | None,
    output_target: str | None,
    mode: str | None,
    default_output_target: str,
) -> tuple[str, str]:
    """Resolve effective output_target and mode.

    Priority: explicit param > style defaults > profile defaults.
    Raises ValueError for unknown styles.
    """
    if style is not None and style not in _STYLE_DEFAULTS:
        raise ValueError(f"Unknown style '{style}'. Valid styles: compute, notebook, interactive")
    style_defaults = _STYLE_DEFAULTS.get(style, {}) if style else {}

    eff_output_target = output_target or style_defaults.get("output_target", default_output_target)
    eff_mode = mode or style_defaults.get("mode", "kernel")

    # Normalize: CLI path ignores mode entirely, so canonicalize to kernel
    if eff_output_target == "cli":
        eff_mode = "kernel"

    return eff_output_target, eff_mode


def _normalize_response_detail(response_detail: str) -> str:
    """Accept a few common aliases while keeping the internal detail set small."""
    normalized = _RESPONSE_DETAIL_ALIASES.get(response_detail, response_detail)
    if normalized not in _VALID_RESPONSE_DETAILS:
        valid = ", ".join(sorted(_VALID_RESPONSE_DETAILS | set(_RESPONSE_DETAIL_ALIASES)))
        raise ValueError(f"Unknown response_detail '{response_detail}'. Valid values: {valid}")
    return normalized


def _resolve_lean_response_detail() -> str:
    """Default response_detail for the lean evaluate tool. Compact unless
    MATHEMATICA_RESPONSE_DETAIL overrides it (set it to "standard" for the old
    full-shape behaviour); an invalid value warns and stays compact so a typo
    never silently disables slimming."""
    raw = os.getenv("MATHEMATICA_RESPONSE_DETAIL")
    if not raw:
        return "compact"
    try:
        return _normalize_response_detail(raw.strip())
    except ValueError as e:
        logger.warning("%s Using 'compact'.", e)
        return "compact"


_LEAN_RESPONSE_DETAIL = _resolve_lean_response_detail()

# Set by the lean evaluate wrapper around its execute_code call: its output is
# routed through _lean_paginate (cursor pagination), so compact filtering must
# NOT pre-summarise the large output and drop what the paginator would serve.
_lean_paginated = contextvars.ContextVar("lean_paginated", default=False)


def _tool(group: str):
    """Tag a core tool with its exposure group."""

    def decorator(func):
        _CORE_TOOL_REGISTRY.append((group, func))
        return func

    return decorator


def _register_core_tools() -> None:
    from .telemetry import telemetry_tool

    for group, func in _CORE_TOOL_REGISTRY:
        if FEATURES.tool_group_enabled(group):
            mcp.tool()(telemetry_tool(func.__name__)(func))


def _json_response(payload: Any) -> str:
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Routing memory (lazy init)
# ---------------------------------------------------------------------------

_routing_mem = None  # set during startup if routing_memory != "off"

# Journal singleton — records computation history before response filtering
_journal = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Transport status classification
# ---------------------------------------------------------------------------


_WL_MSG_TAG_RE = re.compile(r"[A-Z]\w*::[a-zA-Z]\w*")


def _messages_from_warnings(warnings: Any) -> list[dict[str, Any]]:
    """Extract WL message tags (Symbol::tag) from the raw $MessageList text the
    kernel/CLI path returns, so the analyzer runs on non-notebook executions too.

    $MessageList tags carry no severity, and the kernel emits this list even when a
    computation *succeeded with warnings* (e.g. Solve::ratnz, NIntegrate::ncvb). So
    default to "warning" rather than "error" — mislabelling a benign warning as an
    error made analyze_messages report severity="error" and could flip should_retry
    on a result that actually succeeded. Only mark a tag "error" when its Symbol
    family is a known hard-error family (reusing the same set transport
    classification treats as a genuine failure)."""
    if not warnings:
        return []
    text = " ".join(warnings) if isinstance(warnings, list) else str(warnings)
    tags = list(dict.fromkeys(_WL_MSG_TAG_RE.findall(text)))
    out: list[dict[str, Any]] = []
    for t in tags:
        family = t.split("::")[0]
        msg_type = "error" if family in _HARD_ERROR_FAMILIES else "warning"
        out.append({"tag": t, "text": t, "type": msg_type})
    return out


def _attach_error_analysis(response: dict) -> None:
    """Attach error_analysis (incl. retry_with) to any evaluate response that
    surfaced messages. Idempotent: the notebook path builds a richer version
    itself, so this only fills in the kernel/CLI paths (plan §5.3)."""
    if not isinstance(response, dict) or response.get("error_analysis"):
        return
    messages = response.get("messages") or _messages_from_warnings(response.get("warnings"))
    if not messages:
        return
    analysis = analyze_messages(messages)
    if analysis.get("total_messages", 0) == 0:
        return
    response["error_analysis"] = {
        "total_errors": analysis["errors"],
        "total_warnings": analysis["warnings"],
        "severity": analysis["severity"],
        "recommendations": analysis["recommendations"],
        "should_retry": analysis["should_retry"],
        "retry_with": analysis.get("retry_with"),
    }


def _finalize_execute_response(
    response: dict,
    *,
    route_variant: str,
    execution_path: str,
    fell_back: bool,
    start_time: float,
    response_detail: str = "standard",
    expression_type: str | None = None,
) -> str:
    """Normalize execute_code response and optionally record routing stats."""
    import time as _time

    response["route_variant"] = route_variant
    response["execution_path"] = execution_path
    # Error analysis + retry_with on every evaluate path (notebook path already
    # built its own; this fills kernel/CLI).
    _attach_error_analysis(response)
    # Diagnostic routing when there is no mechanical fix: point at the message
    # log and recovery guide instead of inventing a canned retry_with.
    ea = response.get("error_analysis")
    if isinstance(ea, dict) and ea.get("total_errors", 0) > 0 and not ea.get("retry_with"):
        ea.setdefault(
            "next_step",
            "kernel(action='messages') for details; guide(topic='errors') for recovery patterns"
            if FEATURES.profile == "lean"
            else "get_messages() for details",
        )
    # Non-obvious follow-up hint: a rendered graphic is best viewed as an image.
    if response.get("is_graphics"):
        response.setdefault("available_followups", ["screenshot(scope='cell') to view the rendered graphic"])
    elif response.get("has_errors") or (isinstance(ea, dict) and ea.get("total_errors", 0) > 0):
        response.setdefault(
            "available_followups",
            ["kernel(action='messages')"] if FEATURES.profile == "lean" else ["get_messages()"],
        )
    response["overall_timing_ms"] = int((_time.monotonic() - start_time) * 1000)
    # Extract families FIRST — used by transport classification
    response["error_families"] = _extract_error_families(response)
    # Then classify — uses error_families to avoid misclassifying semantic failures
    response["transport_status"] = _classify_transport(response, fell_back=fell_back)
    _maybe_record_routing(response, route_variant=route_variant, expression_type=expression_type)

    # Journal recording — BEFORE response filtering (captures raw canonical result)
    if _journal is not None:
        with contextlib.suppress(Exception):
            _journal.record(
                code=response.get("code", ""),
                output=response.get("output", ""),
                success=response.get("success", response.get("status") == "executed_in_notebook"),
                timing_ms=response.get("overall_timing_ms", 0),
                route_variant=route_variant,
                execution_path=execution_path,
                transport_status=response.get("transport_status", ""),
                error_families=response.get("error_families"),
                timed_out=response.get("timed_out", False),
                from_cache=response.get("from_cache", False),
            )

    # Response detail filtering — applied LAST (after routing + journal recording)
    from .cache import get_kernel_epoch
    from .response_filter import _filter_response

    _diag_hints = None
    if response_detail == "diagnostic" and _routing_mem is not None:
        _diag_hints = _routing_mem.get_routing_hints(FEATURES.profile) or None

    filtered = _filter_response(
        response,
        response_detail,
        cache_epoch=get_kernel_epoch() if response_detail == "diagnostic" else None,
        routing_hints=_diag_hints,
        summarize_large=not _lean_paginated.get(),
    )
    return _json_response(filtered)


def _maybe_record_routing(response: dict, *, route_variant: str, expression_type: str | None = None) -> None:
    """Fire-and-forget routing stat recording. Never raises."""
    if _routing_mem is None:
        return
    with contextlib.suppress(Exception):
        _routing_mem.record(
            profile=FEATURES.profile,
            route_variant=route_variant,
            execution_path=response.get("execution_path", "unknown"),
            transport_status=response.get("transport_status", "ok"),
            latency_ms=response.get("overall_timing_ms", 0),
            error_families=response.get("error_families", []),
            expression_type=expression_type,
        )


async def _run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _addon_result(
    command: str,
    params: dict | None = None,
    timeout: float | None = None,
) -> dict:
    return await _run_blocking(_try_addon_command, command, params, timeout)


async def _addon_json(command: str, params: dict | None = None) -> str:
    return _json_response(await _addon_result(command, params))


async def _image_from_result(result: dict) -> Image:
    image_path = result["path"]

    def _read_and_remove() -> bytes:
        if not _is_valid_png(image_path):
            # Clean up invalid file and raise so caller gets a clear error.
            if os.path.exists(image_path):
                with contextlib.suppress(OSError):
                    os.remove(image_path)
            raise ValueError(f"Invalid or corrupt PNG at {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        os.remove(image_path)
        return image_bytes

    image_bytes = await _run_blocking(_read_and_remove)
    return Image(data=image_bytes, format="png")


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_valid_png(path: str) -> bool:
    """Check that *path* exists, is non-empty, and starts with PNG magic bytes."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 8:
            return False
        with open(path, "rb") as f:
            return f.read(8) == _PNG_MAGIC
    except OSError:
        return False


def _attach_image_if_valid(result: dict) -> None:
    """Attach rendered_image to result if image_path exists and is a valid PNG.

    Validates the file on disk before trusting the path.  Strips is_graphics
    and image_path from the result if the file is missing, empty, or not a
    valid PNG so the caller does not hand out a broken path.
    """
    if not result.get("is_graphics") or not result.get("image_path"):
        return
    image_path = result["image_path"]
    if _is_valid_png(image_path):
        result["rendered_image"] = image_path
        result["tip"] = "Use Read tool to view image."
    else:
        # Rasterization claimed success but no valid file — delete the
        # bad file from disk (if it exists) and strip metadata so
        # clients don't receive a broken path.
        if os.path.exists(image_path):
            with contextlib.suppress(OSError):
                os.remove(image_path)
        result.pop("image_path", None)
        result.pop("is_graphics", None)
        result.pop("rendered_image", None)
        result.pop("tip", None)
        # Restore output to the textual form so clients don't see a
        # broken "[Graphics rendered to image: ...]" placeholder.
        if result.get("output_inputform"):
            result["output"] = result["output_inputform"]


def _hydrate_usage(symbols: list[str]) -> dict[str, str]:
    """Batch-fetch usage strings for *symbols* via a single subprocess call.

    Returns a dict mapping symbol name → usage string.  Cached results
    from the symbol index metadata cache are used where available.
    """
    from . import symbol_index

    result: dict[str, str] = {}
    uncached: list[str] = []

    for sym in symbols:
        meta = symbol_index.get_cached_metadata(sym)
        if meta and meta.get("usage"):
            result[sym] = meta["usage"]
        else:
            uncached.append(sym)

    if not uncached:
        return result

    from .lazy_wolfram_tools import _json_wl, _wl_string
    from .session import evaluate_wl

    sym_list = ", ".join(_wl_string(s) for s in uncached)
    # `/. _MessageName -> ""`: an UNDEFINED symbol's usage stays an unevaluated
    # MessageName whose ToString is the literal "Sym::usage" — without the filter
    # that fake text would be cached into the symbol index as real usage.
    code = (
        f"Association[Map[# -> Quiet[Check["
        f'ToString[ToExpression[# <> "::usage"] /. _MessageName -> ""], ""]] &, {{{sym_list}}}]]'
    )

    try:
        # Warm-first (persistent kernel, cold wolframscript fallback), JSON-first
        # parsing so corrupted regex-parsed text is never cached into the symbol
        # index. Read-only usage-string lookup — no isolation needed.
        wl = evaluate_wl(_json_wl(code), 15)
        if wl.success:
            try:
                parsed = json.loads(wl.text)
            except (json.JSONDecodeError, ValueError):
                parsed = _parse_wolfram_association(wl.text)
            if isinstance(parsed, dict):
                for sym in uncached:
                    usage = parsed.get(sym, "")
                    if isinstance(usage, str) and usage:
                        result[sym] = usage
                        symbol_index.cache_metadata(sym, {"usage": usage})
    except Exception:
        pass

    return result


def _lookup_symbols_in_kernel(query: str) -> dict[str, Any]:
    """Search for Wolfram symbols matching *query*.

    Tries the in-memory symbol index first (pure Python, no subprocess).
    Falls back to a wolframscript subprocess if the index is unavailable.
    """
    from . import symbol_index

    # Fast path: use pre-built index for System symbols.
    matches = symbol_index.search(query, max_results=20)
    if matches:
        return {
            "success": True,
            "candidates": [{"symbol": name, "usage": ""} for name in matches],
            "system_only": True,
        }

    # Fallback: warm kernel (persistent session, cold wolframscript inside
    # evaluate_wl). Read-only Names[]/usage/attribute reads — no isolation
    # needed; the warm kernel also means Global` matches now reflect the
    # user's real session symbols instead of a fresh kernel's empty Global`.
    from .lazy_wolfram_tools import _json_wl, _wl_string
    from .session import evaluate_wl

    lookup_code = f"""
Module[{{query, systemMatches, globalMatches, allMatches, getInfo}},
  query = {_wl_string(query)};
  systemMatches = Select[Names["System`*"], StringContainsQ[#, query, IgnoreCase -> True] &];
  globalMatches = Select[Names["Global`*"], StringContainsQ[#, query, IgnoreCase -> True] &];
  allMatches = Take[Join[systemMatches, globalMatches], UpTo[20]];
  getInfo[sym_String] := Module[{{usage, opts, attrs, syntaxInfo}},
    usage = Quiet[Check[ToString[ToExpression[sym <> "::usage"]], ""]];
    opts = Quiet[Check[ToString[Length[Options[ToExpression[sym]]]], "0"]];
    attrs = Quiet[Check[ToString[Attributes[ToExpression[sym]]], "{{}}"]];
    syntaxInfo = Quiet[Check[ToString[SyntaxInformation[ToExpression[sym]]], "{{}}"]];
    <|"symbol" -> sym, "usage" -> usage, "options_count" -> opts, "attributes" -> attrs, "syntax_info" -> syntaxInfo|>
  ];
  <|"success" -> True, "query" -> query, "matches" -> (getInfo /@ allMatches)|>
]
"""

    try:
        wl = evaluate_wl(_json_wl(lookup_code), 30)
        if not wl.success:
            return {"success": False, "error": wl.error or "Lookup failed"}
        try:
            parsed = json.loads(wl.text)
        except (json.JSONDecodeError, ValueError):
            parsed = _parse_wolfram_association(wl.text)
        if isinstance(parsed, dict) and isinstance(parsed.get("matches"), list):
            candidates = [m for m in parsed["matches"] if isinstance(m, dict) and m.get("symbol")]
            return {"success": True, "candidates": candidates}
        return {"success": True, "candidates": []}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_wolfram_association(raw: str) -> dict[str, Any]:
    """Convert Wolfram Association syntax (<|...|>) to Python dict."""
    try:
        s = raw.strip()
        # Remove newlines within the association (multiline Mathematica output)
        s = re.sub(r"\n\s*", " ", s)
        # Remove carriage returns
        s = s.replace("\r", "")
        # Handle fractions like 100584/625 - quote them as strings
        s = re.sub(r":\s*(\d+)/(\d+)\s*([,}])", r': "\1/\2"\3', s)
        # Convert Association delimiters
        s = re.sub(r"<\|", "{", s)
        s = re.sub(r"\|>", "}", s)
        # Convert arrow to colon
        s = re.sub(r"\s*->\s*", ": ", s)
        # Convert Mathematica booleans
        s = s.replace("True", "true").replace("False", "false").replace("Null", "null")
        # Quote unquoted symbols (identifiers starting with letter, may contain ` $ digits)
        # But be careful not to match already quoted strings or numbers
        s = re.sub(r":\s*([A-Za-z][A-Za-z0-9`$]*(?:\s+[A-Za-z][A-Za-z0-9`$]*)*)\s*([,}])", r': "\1"\2', s)
        # Handle special Mathematica output like "100584 kilometers" with line breaks
        s = re.sub(r":\s*(\d+)\s+([A-Za-z]+)\s*([,}])", r': "\1 \2"\3', s)
        return json.loads(s)
    except Exception:
        return {"success": True, "raw": raw, "parse_error": True}


def _extract_short_description(usage: str) -> str:
    """Extract first sentence from Wolfram usage string."""
    if not usage or usage == "Null":
        return "No description available"

    usage = re.sub(r"^[A-Za-z]+\[.*?\]\s*", "", usage, count=1)
    match = re.match(r"^([^.!?]*[.!?])", usage)
    if match:
        return match.group(1).strip()
    return usage[:100].strip() + ("..." if len(usage) > 100 else "")


def _extract_example_signature(usage: str, symbol: str) -> str:
    """Extract usage pattern like Symbol[args] from usage string."""
    if not usage or usage == "Null":
        return f"{symbol}[...]"

    pattern = rf"{re.escape(symbol)}\[[^\]]*\]"
    match = re.search(pattern, usage)
    if match:
        return match.group(0)
    return f"{symbol}[...]"


def _rank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Rank symbol candidates by relevance using exact/prefix/similarity scoring."""
    query_lower = query.lower()
    scored = []

    for c in candidates:
        symbol = c.get("symbol", "")
        symbol_name = symbol.split("`")[-1]
        symbol_lower = symbol_name.lower()

        score = 0.0
        if symbol_lower == query_lower:
            score += 100
        elif symbol_lower.startswith(query_lower):
            score += 50
        elif query_lower in symbol_lower:
            score += 25

        similarity = difflib.SequenceMatcher(None, query_lower, symbol_lower).ratio()
        score += similarity * 20
        score -= len(symbol_name) * 0.1

        if symbol.startswith("System`"):
            score += 5

        c["_score"] = score
        c["symbol_name"] = symbol_name
        scored.append(c)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def _try_addon_command(
    command: str,
    params: dict | None = None,
    timeout: float | None = None,
) -> dict:
    try:
        conn = get_mathematica_connection()
        return conn.send_command(command, params or {}, timeout=timeout)
    except Exception as e:
        return {"success": False, "error": str(e)}


def _check_addon_protocol(result: dict) -> None:
    """Flag a stale addon (protocol skew) with a reinstall hint (plan §3.8).
    Addons don't update with pip, so an old init.m can lag the Python client."""
    from .connection import ADDON_PROTOCOL_VERSION

    reported = result.get("protocol_version")
    if reported is None or (isinstance(reported, (int, float)) and reported < ADDON_PROTOCOL_VERSION):
        result["addon_outdated"] = True
        result["addon_hint"] = (
            f"Addon protocol {reported if reported is not None else 'missing'} is older than "
            f"expected {ADDON_PROTOCOL_VERSION}. Reinstall the addon (run `mathematica-mcp setup`) "
            "and restart Mathematica to pick up new features."
        )


def _warm_path_status() -> dict[str, Any]:
    """Warm-funnel diagnostics for status responses (plan §3.5): cold-execution
    counter (0 on the lean happy path), persistent-session liveness, and the
    idle-shutdown timeout."""
    return {
        "cold_executions": cold_execution_count(),
        "kernel_session_active": has_existing_kernel_session(),
        "idle_timeout_seconds": kernel_idle_timeout(),
    }


@_tool("core")
async def get_mathematica_status() -> str:
    """Get connection status and system info."""
    try:
        result = await _addon_result("get_status")
        if result.get("success") is False and result.get("error"):
            raise RuntimeError(result["error"])
        result["connection_mode"] = "addon"
        result["warm_path"] = _warm_path_status()
        _check_addon_protocol(result)
        return _json_response(result)
    except Exception as e:
        try:
            session = await _run_blocking(get_kernel_session)
            if session is None:
                raise RuntimeError("No kernel session available")
            from wolframclient.language import wlexpr

            version = session.evaluate(wlexpr("$VersionNumber"))
            return _json_response(
                {
                    "connection_mode": "kernel_only",
                    "kernel_version": float(version),
                    "note": "Addon not running - notebook control unavailable. Execute StartMCPServer[] in Mathematica.",
                    "warm_path": _warm_path_status(),
                    "error": str(e),
                }
            )
        except Exception as e2:
            return _json_response(
                {
                    "connection_mode": "disconnected",
                    "error": f"No connection available: {e2}",
                }
            )


@_tool("core")
async def get_session_brief() -> str:
    """Compact session state summary: connection, profile, recent errors, routing advice."""
    from .guidance import build_session_brief
    from .session import has_existing_kernel_session

    # Determine connection mode with short timeout
    connection_mode = "disconnected"
    try:
        result = await _addon_result("get_status", timeout=0.5)
        if result.get("success") is not False:
            connection_mode = "addon"
        else:
            # Addon returned failure — check for kernel session
            if await _run_blocking(has_existing_kernel_session):
                connection_mode = "kernel_only"
    except Exception:
        if await _run_blocking(has_existing_kernel_session):
            connection_mode = "kernel_only"

    # Gather routing hints and recent errors
    routing_hints: list[str] = []
    recent_errors: list[str] = []
    if _routing_mem is not None:
        if _routing_mem.mode == "advise":
            routing_hints = _routing_mem.get_routing_hints(FEATURES.profile)[:2]
        recent_errors = _routing_mem.get_recent_error_families(limit=3, max_age_seconds=86400)

    return build_session_brief(
        FEATURES,
        connection_mode=connection_mode,
        routing_hints=routing_hints,
        recent_errors=recent_errors,
    )


@_tool("notebook_primary")
async def get_notebooks() -> str:
    """List all open Mathematica notebooks. Returns ID, filename, title."""
    result = await _addon_result("get_notebooks")
    if isinstance(result, dict) and result.get("success") is True and isinstance(result.get("notebooks"), list):
        return _json_response(result["notebooks"])
    return _json_response(result)


@_tool("notebook_primary")
async def get_notebook_info(notebook: str | None = None, session_id: str | None = None) -> str:
    """Get details about a notebook (filename, directory, cell count)."""
    result = await _addon_result("get_notebook_info", {"notebook": notebook, "session_id": session_id})
    return _json_response(result)


@_tool("notebook_primary")
async def create_notebook(
    title: str = "Untitled",
    session_id: str | None = None,
    show_chatbar: bool = False,
) -> str:
    """Create a new empty notebook. Returns notebook ID.

    Use when the user explicitly asks for a NEW notebook. Sets the active
    notebook so subsequent execute_code(output_target="notebook") calls target it.
    On Mathematica >=15, show_chatbar=True keeps the chat sidebar that
    agent-created notebooks suppress by default (older addons ignore the param).
    """
    result = await _addon_result(
        "create_notebook",
        {"title": title, "session_id": session_id, "show_chatbar": show_chatbar},
    )
    return _json_response(result)


@_tool("notebook_primary")
async def save_notebook(
    notebook: str | None = None,
    path: str | None = None,
    format: Literal["Notebook", "PDF", "HTML", "TeX"] = "Notebook",
    session_id: str | None = None,
) -> str:
    """Save a notebook to disk."""
    result = await _addon_result(
        "save_notebook",
        {
            "notebook": notebook,
            "path": path,
            "format": format,
            "session_id": session_id,
        },
    )
    return _json_response(result)


@_tool("notebook_primary")
async def close_notebook(notebook: str | None = None, session_id: str | None = None) -> str:
    """Close a notebook."""
    result = await _addon_result("close_notebook", {"notebook": notebook, "session_id": session_id})
    return _json_response(result)


@_tool("notebook_primary")
async def get_cells(
    notebook: str | None = None,
    style: str | None = None,
    session_id: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    include_content: bool = True,
) -> str:
    """Get list of cells in a notebook."""
    result = await _addon_result(
        "get_cells",
        {
            "notebook": notebook,
            "style": style,
            "session_id": session_id,
            "offset": offset,
            "limit": limit,
            "include_content": include_content,
        },
    )
    return _json_response(result)


@_tool("notebook_primary")
async def get_cell_content(
    cell_id: str,
    notebook: str | None = None,
    session_id: str | None = None,
) -> str:
    """Get the full content of a specific cell."""
    result = await _addon_result(
        "get_cell_content",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def write_cell(
    content: str,
    style: str = "Input",
    notebook: str | None = None,
    position: Literal["After", "Before", "End", "Beginning"] = "After",
    session_id: str | None = None,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Write a new cell to a notebook without evaluating it.

    NOTE: For executing code, prefer execute_code(code, output_target="notebook")
    which writes AND evaluates the cell atomically.
    """
    result = await _addon_result(
        "write_cell",
        {
            "notebook": notebook,
            "content": content,
            "style": style,
            "position": position,
            "session_id": session_id,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def delete_cell(
    cell_id: str,
    notebook: str | None = None,
    session_id: str | None = None,
) -> str:
    """Delete a cell from a notebook."""
    result = await _addon_result(
        "delete_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def evaluate_cell(
    cell_id: str,
    notebook: str | None = None,
    session_id: str | None = None,
    max_wait: int = 10,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Evaluate a specific cell."""
    result = await _addon_result(
        "evaluate_cell",
        {
            "cell_id": cell_id,
            "notebook": notebook,
            "session_id": session_id,
            "max_wait": max_wait,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return _json_response(result)


@_tool("core")
async def execute_code(
    code: str,
    format: Literal["text", "latex", "mathematica"] = "text",
    output_target: Literal["cli", "notebook"] | None = None,
    mode: Literal["kernel", "frontend"] | None = None,
    style: Literal["compute", "notebook", "interactive"] | None = None,
    response_detail: Literal["compact", "standard", "verbose", "diagnostic", "short", "medium", "long"] = "standard",
    render_graphics: bool = True,
    deterministic_seed: int | None = None,
    session_id: str | None = None,
    isolate_context: bool = False,
    timeout: int = 300,
    max_wait: int = 30,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Execute Wolfram Language code."""
    import time as _time

    _exec_start = _time.monotonic()
    try:
        response_detail = _normalize_response_detail(response_detail)
    except ValueError as e:
        return _json_response({"success": False, "error": str(e)})
    # Interactive auto-routing: decide BEFORE resolution overwrites `mode`, since
    # the trigger is the caller pinning neither style nor mode (see _is_interactive_code).
    _auto_route_interactive = style is None and mode is None and _is_interactive_code(code)
    try:
        output_target, mode = _resolve_execution_params(style, output_target, mode, FEATURES.default_output_target)
    except ValueError as e:
        return _json_response({"success": False, "error": str(e)})
    # Only meaningful for a notebook target; frontend mode is what renders a live panel.
    auto_routed_interactive = _auto_route_interactive and output_target == "notebook"
    if auto_routed_interactive:
        mode = "frontend"

    # Route variant for routing memory (agent-controlled dimensions only)
    if output_target == "cli":
        route_variant = "compute"
    elif mode == "frontend":
        route_variant = "notebook_frontend"
    else:
        route_variant = "notebook_kernel"

    # Expression classification for routing telemetry
    _expr_type: str | None = None
    if _routing_mem is not None:
        from .routing_memory import _classify_expression

        _expr_type = _classify_expression(code)

    if output_target == "notebook":
        try:
            # Use atomic command that combines find/create notebook + write + evaluate
            # in a single round-trip for better performance
            # mode="kernel" is the new fast path (no polling)
            params = {
                "code": code,
                "max_wait": max_wait,
                "mode": mode,
                "session_id": session_id,
                "isolate_context": isolate_context,
                "sync": sync,
                "sync_wait": sync_wait,
                "timeout": timeout,
            }
            if deterministic_seed is not None:
                params["deterministic_seed"] = deterministic_seed
            # Use timeout + 10s margin for socket so the addon can enforce its own timeout
            socket_timeout = float(timeout) + 10.0
            result = await _addon_result("execute_code_notebook", params, timeout=socket_timeout)

            # Handle timeout: return immediately, do NOT fall through to CLI
            if result.get("timed_out"):
                # kernel mode returns timing_ms, frontend returns waited_seconds
                duration_ms = result.get("timing_ms")
                if duration_ms is None and result.get("waited_seconds") is not None:
                    duration_ms = round(result["waited_seconds"] * 1000)
                response = {
                    "status": "timeout",
                    "code": code,
                    "notebook_id": result.get("notebook_id"),
                    "cell_id": result.get("cell_id"),
                    "evaluated": False,
                    "timed_out": True,
                    "timing_ms": duration_ms,
                    "message": (
                        f"Computation exceeded {timeout}s timeout. "
                        "Increase timeout or use submit_computation() for long tasks."
                    ),
                }
                if result.get("output_preview"):
                    response["output_preview"] = result.get("output_preview")
                response["executed_output_target"] = "notebook"
                response["executed_mode"] = mode
                if style is not None:
                    response["requested_style"] = style
                return _finalize_execute_response(
                    response,
                    route_variant=route_variant,
                    execution_path="addon_notebook",
                    fell_back=False,
                    start_time=_exec_start,
                    response_detail=response_detail,
                    expression_type=_expr_type,
                )

            if result.get("success"):
                # Frontend dispatch that could not observe completion within the
                # in-handler poll: honest pending, not a fabricated empty success.
                # This is NOT a failure. The eval runs in the notebook once the
                # call returns; the agent recovers by re-reading the notebook.
                if result.get("status") == "evaluation_pending":
                    response = {
                        "status": "evaluation_pending",
                        "code": code,
                        "notebook_id": result.get("notebook_id"),
                        "cell_id": result.get("cell_id"),
                        "evaluated": False,
                        "evaluation_complete": False,
                        "waited_seconds": result.get("waited_seconds"),
                        "message": (
                            result.get("message")
                            or "Evaluation dispatched to the notebook; it runs after this call returns."
                        ),
                        "next_step": (
                            "Re-check the notebook with get_cells (or screenshot_notebook) to read the "
                            "output cell once it appears; the evaluation runs after this call returns."
                        ),
                        "executed_output_target": "notebook",
                        "executed_mode": mode,
                    }
                    if style is not None:
                        response["requested_style"] = style
                    if auto_routed_interactive:
                        response["auto_routed"] = _INTERACTIVE_AUTO_ROUTE_NOTE
                        response["next_step"] += " (Interactive content was auto-routed to frontend mode.)"
                    return _finalize_execute_response(
                        response,
                        route_variant=route_variant,
                        execution_path="addon_notebook",
                        fell_back=False,
                        start_time=_exec_start,
                        response_detail=response_detail,
                        expression_type=_expr_type,
                    )

                response = {
                    "status": "executed_in_notebook",
                    "code": code,
                    "notebook_id": result.get("notebook_id"),
                    "cell_id": result.get("cell_id"),
                    "evaluated": True,
                    "evaluation_complete": result.get("evaluation_complete", True),
                    "message": "Executed in notebook.",
                }
                if result.get("created_notebook"):
                    response["note"] = "Created new notebook 'Analysis'."

                # NEW: Process error messages if present
                if result.get("has_errors") or result.get("has_warnings"):
                    messages = result.get("messages", [])
                    response["messages"] = messages
                    response["has_errors"] = result.get("has_errors", False)
                    response["has_warnings"] = result.get("has_warnings", False)

                    # Update status to indicate errors
                    if result.get("has_errors"):
                        response["status"] = "executed_with_errors"
                        response["message"] = (
                            "Code executed in notebook but produced errors. "
                            "See 'error_analysis' field for detailed suggestions."
                        )

                    # Format error summary for easy reading
                    error_msgs = [m for m in messages if m.get("type") == "error"]
                    warning_msgs = [m for m in messages if m.get("type") == "warning"]

                    if error_msgs or warning_msgs:
                        summary_parts = []
                        if error_msgs:
                            summary_parts.append(
                                f"{len(error_msgs)} error(s): "
                                + "; ".join(f"{m.get('tag', 'Unknown')}" for m in error_msgs[:3])
                            )
                        if warning_msgs:
                            summary_parts.append(
                                f"{len(warning_msgs)} warning(s): "
                                + "; ".join(f"{m.get('tag', 'Unknown')}" for m in warning_msgs[:3])
                            )
                        response["error_summary"] = " | ".join(summary_parts)

                        # NEW: Add intelligent error analysis
                        error_analysis = analyze_messages(messages)
                        response["error_analysis"] = {
                            "total_errors": error_analysis["errors"],
                            "total_warnings": error_analysis["warnings"],
                            "severity": error_analysis["severity"],
                            "recommendations": error_analysis["recommendations"],
                            "should_retry": error_analysis["should_retry"],
                            "retry_with": error_analysis.get("retry_with"),
                        }

                        # Include detailed analysis for each error
                        if error_analysis.get("analyses"):
                            response["detailed_analyses"] = [
                                {
                                    "tag": a["original_message"]["tag"],
                                    "description": a.get("description", ""),
                                    "suggested_fix": a.get("suggested_fix", ""),
                                    "example": a.get("example", ""),
                                    "confidence": a.get("confidence", "low"),
                                }
                                for a in error_analysis["analyses"]
                                if a.get("confidence") in ["high", "medium"]
                            ]

                        # Add formatted error message for LLM
                        response["llm_error_report"] = format_error_for_llm(messages, code)

                        # Add output preview if available
                        if result.get("output_preview"):
                            response["output_preview"] = result.get("output_preview")

                response["executed_output_target"] = "notebook"
                response["executed_mode"] = mode
                if style is not None:
                    response["requested_style"] = style
                if auto_routed_interactive:
                    response["auto_routed"] = _INTERACTIVE_AUTO_ROUTE_NOTE
                return _finalize_execute_response(
                    response,
                    route_variant=route_variant,
                    execution_path="addon_notebook",
                    fell_back=False,
                    start_time=_exec_start,
                    response_detail=response_detail,
                    expression_type=_expr_type,
                )
            else:
                raise RuntimeError(result.get("error", "Atomic notebook execution failed"))

        except Exception as e:
            logger.warning(f"Notebook execution failed: {e}. Returning notebook transport failure.")

            # Record addon_notebook failure in attempt telemetry
            if _routing_mem is not None:
                from .constants import AttemptOutcome as _AO

                _routing_mem.record_transport_attempt(
                    FEATURES.profile, route_variant, _EP.ADDON_NOTEBOOK, _AO.INFRA_ERROR
                )

            response = {
                "success": False,
                "status": "notebook_error",
                "code": code,
                "message": "Notebook execution failed before a notebook result was produced.",
                "error": str(e),
                "executed_output_target": "notebook",
                "executed_mode": mode,
            }
            if style is not None:
                response["requested_style"] = style
            return _finalize_execute_response(
                response,
                route_variant=route_variant,
                execution_path=_EP.ADDON_NOTEBOOK,
                fell_back=False,
                start_time=_exec_start,
                response_detail=response_detail,
                expression_type=_expr_type,
            )

    # Check breaker before attempting addon_cli
    _cli_lease = None
    _routing_skipped = False
    if _routing_mem is not None:
        _cli_lease = _routing_mem.begin_transport_attempt(FEATURES.profile, route_variant, _EP.ADDON_CLI)
        if _cli_lease.action == "skip":
            _routing_skipped = True

    if _routing_skipped:
        # Breaker says skip addon_cli — go straight to kernel
        result = await _run_blocking(
            execute_in_kernel,
            code,
            format,
            render_graphics=render_graphics,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            timeout=timeout,
        )
        result["executed_output_target"] = "cli"
        if style is not None:
            result["requested_style"] = style
        _attach_image_if_valid(result)
        return _finalize_execute_response(
            result,
            route_variant=route_variant,
            execution_path=_EP.KERNEL_DIRECT_ROUTING_SKIP,
            fell_back=False,
            start_time=_exec_start,
            response_detail=response_detail,
            expression_type=_expr_type,
        )

    params = {
        "code": code,
        "format": format,
        "render_graphics": render_graphics,
        "session_id": session_id,
        "isolate_context": isolate_context,
        "timeout": timeout,
    }
    if deterministic_seed is not None:
        params["deterministic_seed"] = deterministic_seed
    socket_timeout = float(timeout) + 10.0

    _lease_completed = False
    try:
        result = await _addon_result("execute_code", params, timeout=socket_timeout)

        # Unified lifecycle: records attempt telemetry + handles probe
        if _routing_mem is not None and _cli_lease is not None:
            from .transport_classification import classify_attempt_outcome

            _attempt_outcome = classify_attempt_outcome(result)
            _routing_mem.finish_transport_attempt(_cli_lease, _attempt_outcome, profile=FEATURES.profile)
            _lease_completed = True
    except Exception:
        if _routing_mem is not None and _cli_lease is not None:
            from .constants import AttemptOutcome as _AO

            _routing_mem.finish_transport_attempt(_cli_lease, _AO.INFRA_ERROR, profile=FEATURES.profile)
            _lease_completed = True
        raise
    finally:
        if _cli_lease is not None and not _lease_completed:
            _routing_mem.abort_transport_attempt(_cli_lease)

    # Check if addon command succeeded, otherwise fall back to kernel
    _cli_fell_back = False
    if result.get("success") is False or "error" in result:
        result = await _run_blocking(
            execute_in_kernel,
            code,
            format,
            render_graphics=render_graphics,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            timeout=timeout,
        )
        result["execution_mode"] = "kernel_fallback"
        _cli_fell_back = True
    result["executed_output_target"] = "cli"
    if style is not None:
        result["requested_style"] = style
    _attach_image_if_valid(result)
    return _finalize_execute_response(
        result,
        route_variant=route_variant,
        execution_path=_EP.KERNEL_FALLBACK if _cli_fell_back else _EP.ADDON_CLI,
        fell_back=_cli_fell_back,
        start_time=_exec_start,
        response_detail=response_detail,
        expression_type=_expr_type,
    )


@_tool("admin")
async def batch_commands(commands: list[dict[str, Any]]) -> str:
    """Execute multiple commands in one round-trip."""
    return await _addon_json("batch_commands", {"commands": commands})


@_tool("notebook_advanced")
async def evaluate_selection(
    notebook: str | None = None,
    session_id: str | None = None,
    max_wait: int = 10,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """
    Evaluate the currently selected cell(s) in a notebook.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
    """
    result = await _addon_result(
        "execute_selection",
        {
            "notebook": notebook,
            "session_id": session_id,
            "max_wait": max_wait,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return _json_response(result)


@_tool("notebook_primary")
async def screenshot_notebook(
    notebook: str | None = None,
    max_height: int = 2000,
    session_id: str | None = None,
    use_rasterize: bool = False,
    wait_ms: int = 100,
) -> Image:
    """
    Capture a screenshot of an entire notebook window.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
        max_height: Maximum height in pixels (prevents huge images)

    Returns the screenshot as an image that can be viewed directly.
    """
    result = await _addon_result(
        "screenshot_notebook",
        {
            "notebook": notebook,
            "max_height": max_height,
            "session_id": session_id,
            "use_rasterize": use_rasterize,
            "wait_ms": wait_ms,
        },
    )

    return await _image_from_result(result)


@_tool("notebook_primary")
async def screenshot_cell(
    cell_id: str,
    notebook: str | None = None,
    session_id: str | None = None,
    use_rasterize: bool = False,
) -> Image:
    """
    Capture a screenshot of a specific cell's content and output.

    Useful for seeing plots, graphics, or formatted mathematical output.

    Args:
        cell_id: The cell object ID to screenshot
    """
    result = await _addon_result(
        "screenshot_cell",
        {
            "cell_id": cell_id,
            "notebook": notebook,
            "session_id": session_id,
            "use_rasterize": use_rasterize,
        },
    )

    return await _image_from_result(result)


@_tool("notebook_primary")
async def rasterize_expression(expression: str, image_size: int = 400) -> Image:
    """
    Render a Wolfram Language expression as an image.

    Useful for visualizing plots, matrices, or formatted output without
    modifying any notebook.

    Args:
        expression: Wolfram Language expression to render
        image_size: Size of the resulting image in pixels

    Examples:
        rasterize_expression("Plot[Sin[x], {x, 0, 2 Pi}]")
        rasterize_expression("MatrixForm[{{1, 2}, {3, 4}}]")
        rasterize_expression("Graphics[Circle[]]", image_size=200)
    """
    result = await _addon_result("rasterize_expression", {"expression": expression, "image_size": image_size})
    return await _image_from_result(result)


@_tool("notebook_advanced")
async def select_cell(
    cell_id: str,
    notebook: str | None = None,
    session_id: str | None = None,
) -> str:
    """Select a cell in the notebook (moves cursor to it)."""
    result = await _addon_result(
        "select_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def scroll_to_cell(
    cell_id: str,
    notebook: str | None = None,
    session_id: str | None = None,
) -> str:
    """Scroll the notebook view to make a cell visible."""
    result = await _addon_result(
        "scroll_to_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_primary")
async def export_notebook(
    path: str,
    notebook: str | None = None,
    format: Literal["PDF", "HTML", "TeX", "Markdown"] = "PDF",
    session_id: str | None = None,
) -> str:
    """Export a notebook to PDF, HTML, TeX, or Markdown."""
    result = await _addon_result(
        "export_notebook",
        {"notebook": notebook, "path": path, "format": format, "session_id": session_id},
    )
    return _json_response(result)


@_tool("moat")
async def verify_derivation(
    steps: list[str],
    format: Literal["text", "latex", "mathematica"] = "text",
    timeout: int = 120,
) -> str:
    """Verify a sequence of mathematical expressions steps."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.verify_derivation(
        steps,
        format,
        timeout,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("core")
async def get_kernel_state() -> str:
    """Get current Wolfram kernel session state (memory, uptime, version)."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.get_kernel_state(parse_wolfram_association=_parse_wolfram_association)


@_tool("kernel_tools")
async def load_package(package_name: str) -> str:
    """Load a Mathematica package (e.g., "Developer`")."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    result = await _lazy_wolfram_tools.load_package(
        package_name,
        parse_wolfram_association=_parse_wolfram_association,
    )
    # Only invalidate cache if the package load actually succeeded.
    try:
        if json.loads(result).get("success"):
            bump_kernel_epoch()
    except (json.JSONDecodeError, AttributeError):
        bump_kernel_epoch()  # Can't tell — assume state may have changed.
    return result


@_tool("kernel_tools")
async def list_loaded_packages() -> str:
    """List all currently loaded packages and contexts."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.list_loaded_packages(parse_wolfram_association=_parse_wolfram_association)


# ============================================================================
# TIER 1: Variable Introspection (Session State)
# ============================================================================


@_tool("session")
async def list_variables(include_system: bool = False) -> str:
    """List all user-defined variables in the current Mathematica kernel session."""
    result = await _addon_result("list_variables", {"include_system": include_system})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("session")
async def get_variable(name: str) -> str:
    """Get detailed information about a specific variable."""
    result = await _addon_result("get_variable", {"name": name})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("session")
async def set_variable(name: str, value: str) -> str:
    """
    Set a variable in the Mathematica kernel session.

    Args:
        name: Variable name (e.g., "x", "myData")
        value: Wolfram Language expression to assign (e.g., "5", "{1,2,3}", "Plot[Sin[x],{x,0,Pi}]")

    Returns:
        Confirmation with the assigned value

    Example:
        set_variable("x", "Range[10]") -> {success: true, value: "{1,2,3,4,5,6,7,8,9,10}"}
    """
    result = await _addon_result("set_variable", {"name": name, "value": value})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    bump_kernel_epoch()
    return _json_response(result)


@_tool("session")
async def clear_variables(
    names: list[str] | None = None,
    pattern: str | None = None,
    clear_all: bool = False,
) -> str:
    """
    Clear variables from the Mathematica kernel session.

    Equivalent to Python's 'del' or clearing notebook state.

    Args:
        names: Specific variable names to clear (e.g., ["x", "y", "z"])
        pattern: Wolfram pattern to match (e.g., "temp*" clears temp1, temp2, etc.)
        clear_all: If True, clear ALL Global` variables (use with caution!)

    Returns:
        List of cleared variables

    Example:
        clear_variables(names=["x", "y"]) -> {cleared: ["x", "y"], count: 2}
        clear_variables(pattern="temp*") -> {cleared: ["temp1", "temp2"], count: 2}
    """
    params = {}
    if names:
        params["names"] = names
    if pattern:
        params["pattern"] = pattern
    if clear_all:
        params["clear_all"] = True

    result = await _addon_result("clear_variables", params)

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    bump_kernel_epoch()
    return _json_response(result)


@_tool("session")
async def get_expression_info(expression: str) -> str:
    """
    Get detailed structural information about a Wolfram expression.

    Like Python's type() on steroids - shows Head, FullForm, tree structure,
    depth, leaf count, and type checks (NumericQ, ListQ, etc.)

    Args:
        expression: Wolfram Language expression to analyze

    Returns:
        Structural information: head, full form, depth, leaf count, type flags

    Example:
        get_expression_info("{{1,2},{3,4}}") -> {head: "List", depth: 3, dimensions: [2,2]}
        get_expression_info("Sin[x] + Cos[x]") -> {head: "Plus", leaf_count: 3}
    """
    result = await _addon_result("get_expression_info", {"expression": expression})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


# ============================================================================
# TIER 1: Error Recovery
# ============================================================================


@_tool("core")
async def get_messages(count: int = 10) -> str:
    """
    Get recent Mathematica messages/warnings from the session.

    Like Python's exception traceback - helps debug what went wrong.
    Includes recently captured evaluation and dispatch-level messages.

    Args:
        count: Number of recent messages to retrieve (default 10)

    Returns:
        List of recent messages with timestamps

    Example:
        After a failed computation:
        get_messages() -> [{timestamp: "...", message: "Power::infy: Infinite expression 1/0 encountered."}]
    """
    result = await _addon_result("get_messages", {"count": count})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("core")
async def restart_kernel() -> str:
    """
    Restart the Mathematica kernel, clearing all state.

    This is the nuclear option - clears all variables, definitions, and state.
    Use when the kernel is in a bad state or you need a fresh start.

    Returns:
        Confirmation of kernel restart
    """
    await _run_blocking(close_kernel_session)
    # Invalidate all cached results — kernel state is completely reset.
    _query_cache.clear()
    clear_cache()
    clear_raster_cache()
    bump_kernel_epoch()
    # Force reconnection
    result = await _addon_result("ping")

    return _json_response(
        {
            "success": True,
            "message": "Kernel session cleared. Fresh session will be created on next execution.",
            "ping_result": result,
        }
    )


# ============================================================================
# TIER 2: File Handling (.nb, .wl, .wlnb)
# ============================================================================


def _expand_path(path: str) -> str:
    """Expand ~ and make path absolute."""
    expanded = os.path.expanduser(path)
    return os.path.abspath(expanded)


def _load_cached_notebook(path: str, truncation_threshold: int = 25000):
    from .notebook_parser import parse_notebook_cached

    return parse_notebook_cached(path, truncation_threshold=truncation_threshold)


class _TelemetryMcpWrapper:
    """Proxy that auto-wraps every tool registered via ``@mcp.tool()``
    with the :func:`telemetry_tool` decorator so that optional tool
    modules get instrumentation for free without changing their code."""

    def __init__(self, mcp_instance):
        self._mcp = mcp_instance

    def tool(self, *args, **kwargs):
        from .telemetry import telemetry_tool

        original_decorator = self._mcp.tool(*args, **kwargs)

        def instrumenting_decorator(func):
            instrumented = telemetry_tool(func.__name__)(func)
            return original_decorator(instrumented)

        return instrumenting_decorator

    def __getattr__(self, name):
        return getattr(self._mcp, name)


def _register_optional_tools() -> None:
    instrumented_mcp = _TelemetryMcpWrapper(mcp)

    if FEATURES.symbol_lookup:
        from .optional_symbol_tools import register_symbol_lookup_tools

        register_symbol_lookup_tools(
            instrumented_mcp,
            lookup_symbols_in_kernel=_lookup_symbols_in_kernel,
            hydrate_usage=_hydrate_usage,
            extract_short_description=_extract_short_description,
            extract_example_signature=_extract_example_signature,
            rank_candidates=_rank_candidates,
            parse_wolfram_association=_parse_wolfram_association,
            execute_code=execute_code,
        )

    if FEATURES.math_aliases:
        from .optional_math_aliases import register_math_alias_tools

        register_math_alias_tools(instrumented_mcp, execute_code)

    if FEATURES.function_repository:
        from .optional_repository_tools import register_function_repository_tools

        register_function_repository_tools(instrumented_mcp, parse_wolfram_association=_parse_wolfram_association)

    if FEATURES.data_repository:
        from .optional_repository_tools import register_data_repository_tools

        register_data_repository_tools(instrumented_mcp, parse_wolfram_association=_parse_wolfram_association)

    if FEATURES.async_computation:
        from .optional_async_jobs import register_async_computation_tools

        register_async_computation_tools(instrumented_mcp)

    if FEATURES.cache_tools:
        from .optional_cache_tools import register_cache_tools

        register_cache_tools(
            instrumented_mcp,
            cache_expression_fn=_cache_expr,
            get_cached_expression_fn=_get_cached,
            list_cached_expressions_fn=list_cached_expressions,
            clear_cache_fn=clear_cache,
            execute_code=execute_code,
        )

    if FEATURES.telemetry:
        from .optional_telemetry_tools import register_telemetry_tools

        register_telemetry_tools(
            instrumented_mcp,
            get_usage_stats=get_usage_stats,
            reset_stats=reset_stats,
        )

    if FEATURES.routing_memory != "off" and FEATURES.profile in ("full", "classic"):
        from .optional_routing_tools import register_routing_tools

        register_routing_tools(instrumented_mcp)


# ============================================================================
# LEAN PROFILE — 12 consolidated tools, split by param shape (plan §2).
# Thin dispatchers over the SAME internals the classic tools use; registered
# only when MATHEMATICA_PROFILE=lean (group "lean"). verify_derivation is shared
# via the "moat" group. Docstrings here are <=200 chars and are intentionally
# NOT rewritten by _apply_guidance_docs.
# ============================================================================

_GUIDE_CONTENT: dict[str, str] = {
    "workflow": (
        "Compute: evaluate(code, target='kernel'). Show in a notebook: notebooks(action='create') "
        "then evaluate(code, target='notebook'). Interactive content (Manipulate/Dynamic/Animate) via "
        "target='notebook' is auto-rendered as a live panel by the front end; the response may be "
        "evaluation_pending, so re-check with cells(action='read') once the output cell lands. "
        "Inspect: cells(action='list'), screenshot(scope='cell'). "
        "Verify algebra: verify_derivation(steps). Read a .nb without a kernel: read_notebook_file(path)."
    ),
    "errors": (
        "Failed evaluations return error_analysis with suggested_fix and, when available, retry_with (a "
        "corrected call you can rerun). check syntax first with evaluate(code, dry_run=True). Kernel wedged? "
        "kernel(action='restart')."
    ),
    "notebook_hygiene": (
        "One idea per cell; style='Section'/'Text' for structure, 'Input' for code. Evaluate a specific cell "
        "with evaluate(target='cell', cell_id=...). Save with notebooks(action='save'). Agent-created notebooks "
        "suppress the chat sidebar on v15."
    ),
    "screenshots": (
        "screenshot(scope='notebook'|'cell'|'expression'). Prefer scope='cell' after a plot to capture just that "
        "output. scope='expression' rasterizes a WL expression without touching the notebook."
    ),
    "v15": (
        "On Mathematica >=15 agent-created notebooks set ShowChatbar->False (pass show_chatbar to override). "
        "14.x stays supported behind $VersionNumber guards. The addon advertises a protocol_version so a newer "
        "client can detect a stale addon and prompt a reinstall."
    ),
    "profiles": (
        "lean (default): 12 consolidated tools. classic: the full 82-tool surface (alias 'full'). math/notebook: "
        "legacy curated sets. Set MATHEMATICA_PROFILE to switch."
    ),
    "toolsets": (
        "Add extras to lean via MATHEMATICA_TOOLSETS (comma-separated): data_io, graphics_plus, cloud, debug, "
        "notebook_files, notebook_edit, symbols, math_aliases, repository, async_jobs, cache."
    ),
    "batch": (
        "batch(ops=[{'command': ..., 'params': {...}}, ...]) runs ADDON commands (not MCP tool names). "
        "Commands: ping, get_status, get_notebooks, get_notebook_info, create_notebook, save_notebook, "
        "close_notebook, get_cells, get_cell_content, write_cell, delete_cell, evaluate_cell, execute_code, "
        "execute_code_notebook, execute_selection, batch_commands, screenshot_notebook, screenshot_cell, "
        "rasterize_expression, select_cell, scroll_to_cell, export_notebook, list_variables, get_variable, "
        "set_variable, clear_variables, get_expression_info, get_messages, open_notebook_file, run_script, "
        "trace_evaluation, time_expression, check_syntax, import_data, export_data, list_import_formats, "
        "export_graphics. Example: batch(ops=[{'command': 'execute_code', 'params': {'code': '1+1'}}])."
    ),
}


def _lean_bad(field: str, valid: str, example: str) -> str:
    return _json_response({"success": False, "error": f"invalid {field}. valid: {valid}. e.g. {example}"})


def _lean_annotate(text: str, key: str, value: Any, *, only_on_success: bool = False) -> str:
    """Attach *key* (a note or followups) to a JSON tool response.

    Non-JSON responses pass through untouched; with only_on_success, failed
    responses (error present or success=False) pass through too.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    if not isinstance(data, dict):
        return text
    if only_on_success and (data.get("error") or data.get("success") is False):
        return text
    data.setdefault(key, value)
    return _json_response(data)


def _lean_paginate(text: str) -> str:
    """Cap oversized lean-tool output, stashing the remainder behind a cursor
    the client fetches by passing cursor= back to the same tool (plan §3.7)."""
    from . import cursor_store

    first, info = cursor_store.paginate(text)
    if info is None:
        return text
    return _json_response(
        {
            "truncated": True,
            "preview": first,
            "next_cursor": info["next_cursor"],
            "total_length": info["total_length"],
            "note": "Output truncated. Pass cursor=<next_cursor> to this tool for the next page.",
        }
    )


def _lean_cursor_page(cursor: str) -> str:
    from . import cursor_store

    result = cursor_store.page(cursor)
    if result is None:
        return _json_response(
            {
                "success": False,
                "error": "unknown or expired cursor",
                "next_step": "re-run the original call without cursor= to regenerate",
            }
        )
    return _json_response(result)


@_tool("lean")
async def status() -> str:
    """Connection, kernel version, active profile, features, and warm-path health (cold-execution count, idle timeout). No params."""
    merged: dict[str, Any] = {}
    for getter in (get_mathematica_status, get_feature_status):
        try:
            data = json.loads(await getter())
            if isinstance(data, dict):
                # get_feature_status always reports success=True; it must not
                # overwrite a disconnected/error status from get_mathematica_status,
                # which conveys state via connection_mode/error and omits success on
                # failure. Features stay namespaced under "features".
                if getter is get_feature_status:
                    data.pop("success", None)
                merged.update(data)
        except Exception:
            pass
    return _json_response(merged)


@_tool("lean")
async def notebooks(
    action: Literal["list", "info", "create", "open", "save", "close", "export"],
    notebook: str | None = None,
    title: str = "Untitled",
    path: str | None = None,
    format: Literal["Notebook", "PDF", "HTML", "TeX", "Markdown"] | None = None,
    session_id: str | None = None,
    show_chatbar: bool = False,
) -> str:
    """Manage notebooks. action: list | info | create(title, show_chatbar) | open(path) | save | close | export(path). format: save=Notebook|PDF|HTML|TeX, export=PDF|HTML|TeX|Markdown."""
    if action == "list":
        return await get_notebooks()
    if action == "info":
        return await get_notebook_info(notebook, session_id)
    if action == "create":
        return _lean_annotate(
            await create_notebook(title, session_id, show_chatbar),
            "available_followups",
            ["evaluate(code=..., target='notebook') to run code in it"],
            only_on_success=True,
        )
    if action == "open":
        if not path:
            return _json_response({"success": False, "error": "open requires path"})
        return await open_notebook_file(path, session_id)
    if action == "save":
        fmt = format or "Notebook"
        if fmt not in ("Notebook", "PDF", "HTML", "TeX"):
            return _json_response(
                {
                    "success": False,
                    "error": f"format '{fmt}' not valid for save. valid: Notebook|PDF|HTML|TeX",
                    "next_step": f"notebooks(action='export', format='{fmt}', path=...) for {fmt} export",
                }
            )
        return await save_notebook(notebook, path, fmt, session_id)
    if action == "close":
        return await close_notebook(notebook, session_id)
    if action == "export":
        if not path:
            return _json_response({"success": False, "error": "export requires path"})
        fmt = format or "PDF"
        if fmt not in ("PDF", "HTML", "TeX", "Markdown"):
            return _json_response(
                {
                    "success": False,
                    "error": f"format '{fmt}' not valid for export. valid: PDF|HTML|TeX|Markdown",
                    "next_step": "notebooks(action='save', format='Notebook', path=...) to save a .nb",
                }
            )
        return await export_notebook(path, notebook, fmt, session_id)
    return _lean_bad("action", "list|info|create|open|save|close|export", "notebooks(action='list')")


@_tool("lean")
async def cells(
    action: Literal["list", "read", "select", "scroll"] = "list",
    notebook: str | None = None,
    cell_id: str | None = None,
    style: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    include_content: bool = True,
    session_id: str | None = None,
    cursor: str | None = None,
) -> str:
    """Read notebook cells. action: list | read(cell_id) | select(cell_id) | scroll(cell_id). style/offset/limit filter list; cursor= pages long output."""
    if cursor:
        return _lean_cursor_page(cursor)
    if action == "list":
        return _lean_paginate(await get_cells(notebook, style, session_id, offset, limit, include_content))
    if not cell_id:
        return _json_response({"success": False, "error": f"{action} requires cell_id"})
    if action == "read":
        return _lean_paginate(await get_cell_content(cell_id, notebook, session_id))
    if action == "select":
        return await select_cell(cell_id, notebook, session_id)
    if action == "scroll":
        return await scroll_to_cell(cell_id, notebook, session_id)
    return _lean_bad("action", "list|read|select|scroll", "cells(action='list')")


@_tool("lean")
async def edit_cells(
    action: Literal["write", "delete"],
    content: str | None = None,
    cell_id: str | None = None,
    style: str = "Input",
    notebook: str | None = None,
    position: Literal["After", "Before", "End", "Beginning"] = "After",
    session_id: str | None = None,
) -> str:
    """Edit notebook cells. action: write(content, style, position) | delete(cell_id)."""
    if action == "write":
        if content is None:
            return _json_response({"success": False, "error": "write requires content"})
        return await write_cell(content, style, notebook, position, session_id)
    if action == "delete":
        if not cell_id:
            return _json_response({"success": False, "error": "delete requires cell_id"})
        return await delete_cell(cell_id, notebook, session_id)
    return _lean_bad("action", "write|delete", "edit_cells(action='write', content='1+1')")


@_tool("lean")
async def evaluate(
    code: str | None = None,
    target: Literal["kernel", "notebook", "cell", "selection"] = "kernel",
    file: str | None = None,
    cell_id: str | None = None,
    dry_run: bool = False,
    format: Literal["text", "latex", "mathematica"] = "text",
    notebook: str | None = None,
    session_id: str | None = None,
    timeout: int = 300,
    cursor: str | None = None,
) -> str:
    """Evaluate code. target: kernel | notebook | cell(cell_id) | selection. dry_run=True checks syntax; file= runs a .wl script; cursor= pages long output."""
    if cursor:
        return _lean_cursor_page(cursor)
    if code is not None and file:
        return _json_response(
            {
                "success": False,
                "error": "code and file are both set; pass code= to evaluate code OR file= to run a script, not both",
            }
        )
    if dry_run and file:
        return _json_response(
            {
                "success": False,
                "error": "dry_run checks code, not file; pass code= with dry_run=True, or file= without dry_run",
            }
        )
    if dry_run:
        if code is None:
            return _json_response({"success": False, "error": "dry_run requires code"})
        return _lean_annotate(
            await check_syntax(code),
            "available_followups",
            ["re-run without dry_run to execute"],
            only_on_success=True,
        )
    # ponytail: timeout only applies to target=kernel/notebook (execute_code
    # accepts it). run_script takes no timeout, and evaluate_cell/selection use
    # max_wait (a notebook poll interval, not an execution timeout) — different
    # semantics, so a non-default timeout gets a corrective note, not forwarded.
    note = (
        "timeout applies to target=kernel/notebook only; cell/selection/file evaluation ignores it"
        if timeout != 300
        else None
    )
    if file:
        out = await run_script(file)
        return _lean_paginate(_lean_annotate(out, "note", note) if note else out)
    if target == "cell":
        if not cell_id:
            return _json_response({"success": False, "error": "target=cell requires cell_id"})
        out = await evaluate_cell(cell_id, notebook, session_id)
        return _lean_paginate(_lean_annotate(out, "note", note) if note else out)
    if target == "selection":
        out = await evaluate_selection(notebook, session_id)
        return _lean_paginate(_lean_annotate(out, "note", note) if note else out)
    if code is None:
        return _json_response({"success": False, "error": "evaluate requires code (or file, or target=cell/selection)"})
    output_target = "notebook" if target == "notebook" else "cli"
    # Flag compact filtering to skip large-output summarisation: _lean_paginate
    # below serves the full output via cursor, so summarising here would drop it.
    _token = _lean_paginated.set(True)
    try:
        out = await execute_code(
            code,
            format,
            output_target=output_target,
            # _LEAN_RESPONSE_DETAIL is always a valid detail level (resolver
            # normalizes/falls back); execute_code re-normalizes it regardless.
            response_detail=_LEAN_RESPONSE_DETAIL,  # type: ignore[arg-type]
            session_id=session_id,
            timeout=timeout,
        )
    finally:
        _lean_paginated.reset(_token)
    return _lean_paginate(out)


async def _cached_screenshot(key_parts: tuple, produce) -> Image:
    """Serve a notebook/cell PNG from the opt-in cache or produce and store one.
    Keyed by the current notebook epoch, so any mutating MCP command forces a
    fresh capture; a miss calls through to the real screenshot function."""
    key = (*key_parts, get_notebook_epoch())
    png = get_cached_screenshot(key)
    if png is not None:
        logger.info("screenshot cache hit: %s", key)
        return Image(data=png, format="png")
    logger.info("screenshot cache miss: %s", key)
    image = await produce()
    if image.data is not None:
        put_cached_screenshot(key, image.data)
    return image


@_tool("lean")
async def screenshot(
    scope: Literal["notebook", "cell", "expression"] = "notebook",
    notebook: str | None = None,
    cell_id: str | None = None,
    expression: str | None = None,
    max_height: int = 2000,
    image_size: int = 400,
    session_id: str | None = None,
    cache: bool = False,
) -> Image:
    """Capture a PNG. scope: notebook | cell(cell_id) | expression(expression).
    cache=True reuses the last PNG for notebook/cell scope until an MCP command
    mutates the notebook; may return stale pixels if it changed outside MCP
    (manual edits, Dynamic/Manipulate repaints, scroll/resize). The cache needs a
    stable target: pass notebook= or session_id=; without them the notebook-scope
    cache is skipped (the focused notebook cannot be named reliably). expression
    scope ignores it (already cached at the raster layer)."""
    if scope == "cell":
        if not cell_id:
            raise ValueError("scope=cell requires cell_id")
        if cache:
            return await _cached_screenshot(
                ("cell", notebook or "focused", cell_id, session_id),
                lambda: screenshot_cell(cell_id, notebook, session_id),
            )
        return await screenshot_cell(cell_id, notebook, session_id)
    if scope == "expression":
        if not expression:
            raise ValueError("scope=expression requires expression")
        return await rasterize_expression(expression, image_size)
    if cache:
        if notebook is None and session_id is None:
            # "focused" target anchors on $MCPActiveNotebook, which reads
            # silently rewrite without bumping the epoch, so the cache key
            # cannot name a stable notebook. Capture fresh instead of risking
            # another notebook's pixels under the "focused" key.
            logger.info("screenshot cache skipped: no stable notebook identity (pass notebook= or session_id=)")
            return await screenshot_notebook(notebook, max_height, session_id)
        return await _cached_screenshot(
            ("notebook", notebook or "focused", max_height, session_id),
            lambda: screenshot_notebook(notebook, max_height, session_id),
        )
    return await screenshot_notebook(notebook, max_height, session_id)


@_tool("lean")
async def kernel(
    action: Literal["state", "messages", "restart", "load_package", "packages", "inspect"] = "state",
    package: str | None = None,
    expression: str | None = None,
    count: int = 10,
    cursor: str | None = None,
) -> str:
    """Kernel admin. action: state | messages | restart | load_package(package) | packages | inspect(expression). cursor= pages long output."""
    if cursor:
        return _lean_cursor_page(cursor)
    if action == "state":
        return _lean_paginate(await get_kernel_state())
    if action == "messages":
        return _lean_paginate(await get_messages(count))
    if action == "restart":
        return await restart_kernel()
    if action == "packages":
        return _lean_paginate(await list_loaded_packages())
    if action == "load_package":
        if not package:
            return _json_response({"success": False, "error": "load_package requires package"})
        return await load_package(package)
    if action == "inspect":
        if not expression:
            return _json_response({"success": False, "error": "inspect requires expression"})
        return _lean_paginate(await get_expression_info(expression))
    return _lean_bad("action", "state|messages|restart|load_package|packages|inspect", "kernel(action='state')")


@_tool("lean")
async def vars(
    action: Literal["list", "get", "set", "clear", "clear_all"] = "list",
    name: str | None = None,
    value: str | None = None,
    pattern: str | None = None,
    include_system: bool = False,
    cursor: str | None = None,
) -> str:
    """Manage kernel variables. action: list | get(name) | set(name,value) | clear(name | pattern) | clear_all. cursor= pages long output."""
    if cursor:
        return _lean_cursor_page(cursor)
    if action == "list":
        return _lean_paginate(await list_variables(include_system))
    if action == "get":
        if not name:
            return _json_response({"success": False, "error": "get requires name"})
        return _lean_paginate(await get_variable(name))
    if action == "set":
        if not name or value is None:
            return _json_response({"success": False, "error": "set requires name and value"})
        return await set_variable(name, value)
    if action == "clear":
        # Data-loss guard: a bare clear must never silently wipe the session.
        if not name and not pattern:
            return _json_response(
                {
                    "success": False,
                    "error": "clear requires name= or pattern=",
                    "next_step": "vars(action='clear', name='x')  # or vars(action='clear_all') to wipe everything",
                }
            )
        return await clear_variables(names=[name] if name else None, pattern=pattern, clear_all=False)
    if action == "clear_all":
        return await clear_variables(clear_all=True)
    return _lean_bad("action", "list|get|set|clear|clear_all", "vars(action='list')")


@_tool("lean")
async def read_notebook_file(
    path: str,
    mode: Literal["markdown", "wolfram", "outline", "json", "plain"] = "markdown",
    cursor: str | None = None,
) -> str:
    """Read a .nb/.wl file from disk (works without a kernel via Python fallback). mode: markdown | wolfram | outline | json | plain. cursor= pages long output."""
    if cursor:
        return _lean_cursor_page(cursor)
    return _lean_paginate(await read_notebook(path, output_format=mode))


@_tool("lean")
async def guide(
    topic: Literal[
        "workflow", "errors", "notebook_hygiene", "screenshots", "v15", "profiles", "toolsets", "batch"
    ] = "workflow",
) -> str:
    """On-demand guidance. topic: workflow | errors | notebook_hygiene | screenshots | v15 | profiles | toolsets | batch."""
    return _json_response({"topic": topic, "guidance": _GUIDE_CONTENT.get(topic, "Unknown topic.")})


@_tool("lean")
async def batch(ops: list[dict[str, Any]]) -> str:
    """Run multiple addon ops in one round trip. ops: [{command, params}, ...]. Command vocabulary: guide(topic='batch')."""
    return await batch_commands(ops)


def _apply_guidance_docs() -> None:
    execute_code.__doc__ = execute_code_doc(FEATURES.default_output_target)
    create_notebook.__doc__ = create_notebook_doc()
    write_cell.__doc__ = write_cell_doc()
    evaluate_cell.__doc__ = evaluate_cell_doc()
    evaluate_selection.__doc__ = evaluate_selection_doc()
    read_notebook.__doc__ = read_notebook_doc()
    read_notebook_content.__doc__ = legacy_notebook_doc("Read notebook content as structured text.")
    convert_notebook.__doc__ = legacy_notebook_doc(
        "Convert a notebook into markdown, LaTeX, plain text, or Wolfram code."
    )
    get_notebook_outline.__doc__ = legacy_notebook_doc("Get the notebook's section outline.")
    parse_notebook_python.__doc__ = legacy_notebook_doc("Parse a notebook with the Python-native parser.")
    get_notebook_cell.__doc__ = legacy_notebook_doc("Read a single notebook cell by index.")


@_tool("notebook_primary")
async def open_notebook_file(path: str, session_id: str | None = None) -> str:
    """
    Open an existing Mathematica notebook file (.nb) in the Mathematica frontend.

    Supports:
    - Absolute paths: /Users/foo/notebook.nb
    - Home-relative paths: ~/Documents/notebook.nb
    - Relative paths (resolved from current directory)

    Args:
        path: Path to the .nb file

    Returns:
        Notebook ID and metadata for use with other notebook commands

    Example:
        open_notebook_file("~/Documents/analysis.nb") -> {id: "NotebookObject[...]", cell_count: 15}
    """
    expanded = _expand_path(path)
    result = await _addon_result("open_notebook_file", {"path": expanded, "session_id": session_id})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"], "path": expanded})

    return _json_response(result)


@_tool("file_legacy")
async def run_script(path: str) -> str:
    """
    Execute a Wolfram Language script file (.wl, .m) and return the result.

    This is equivalent to Get[path] - loads and executes the script in the
    current kernel session. Any definitions or side effects persist.

    Args:
        path: Path to the .wl or .m script file

    Returns:
        The result of the last expression in the script, plus timing info

    Example:
        run_script("~/scripts/setup.wl") -> {result: "Null", timing_ms: 150}
    """
    expanded = _expand_path(path)
    result = await _addon_result("run_script", {"path": expanded})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"], "path": expanded})

    bump_kernel_epoch()
    return _json_response(result)


@_tool("file_legacy")
async def read_notebook_content(path: str, include_outputs: bool = False) -> str:
    """
    Read the content of a notebook file as structured text.

    Extracts all cells from a notebook file without opening it in the frontend.
    Useful for understanding what's in a notebook before opening it.

    Args:
        path: Path to the .nb file
        include_outputs: If True, include Output cells (default: only Input/Text)

    Returns:
        Structured list of cells with their content and styles
    """
    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps({"success": False, "error": f"File not found: {expanded}"}, indent=2)

    try:
        from .notebook_parser import CellStyle

        notebook = await _run_blocking(_load_cached_notebook, expanded, truncation_threshold=25000)
        allowed_styles = {
            CellStyle.INPUT,
            CellStyle.CODE,
            CellStyle.TEXT,
            CellStyle.SECTION,
            CellStyle.SUBSECTION,
            CellStyle.TITLE,
        }

        cells = []
        for cell in notebook.cells:
            if include_outputs or cell.style in allowed_styles:
                cells.append({"content": cell.content, "style": cell.style.value})

        return _json_response(
            {
                "success": True,
                "path": expanded,
                "cell_count": len(cells),
                "cells": cells[:50],
            }
        )
    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


@_tool("file_legacy")
async def convert_notebook(
    path: str,
    output_format: Literal["markdown", "latex", "plain", "wolfram"] = "markdown",
) -> str:
    """
    Convert a Mathematica notebook to another format.

    Supported formats:
    - markdown: Readable Markdown with code blocks
    - latex: LaTeX document
    - plain: Plain text
    - wolfram: Pure Wolfram Language code only

    Args:
        path: Path to the .nb file
        output_format: Target format

    Returns:
        Converted content as a string
    """
    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps({"success": False, "error": f"File not found: {expanded}"}, indent=2)

    try:
        from .notebook_parser import NotebookParser

        notebook = await _run_blocking(_load_cached_notebook, expanded, truncation_threshold=25000)
        parser = NotebookParser(truncation_threshold=25000)

        if output_format == "markdown":
            content = parser.to_markdown(notebook)
        elif output_format == "wolfram":
            content = parser.to_wolfram_code(notebook)
        elif output_format == "plain":
            content = parser.to_plain_text(notebook)
        else:
            from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

            # Warm-first + JSON-first via the shared funnel. File-content
            # conversion only: Import[...] of a .nb returns the notebook
            # expression without evaluating it, and everything is Module-local,
            # so this is read-only w.r.t. kernel state — no isolation needed.
            wl_path = _wl_string(expanded.replace("\\", "/"))
            code = f"""
Module[{{nb, content}},
  nb = Import[{wl_path}];
  content = ExportString[nb, "TeX"];
  <|"success" -> True, "format" -> "{output_format}", "content" -> content|>
]
"""
            return await _run_wl_parsed(code, _parse_wolfram_association, timeout=60)

        return _json_response({"success": True, "format": output_format, "content": content})
    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


@_tool("file_legacy")
async def get_notebook_outline(path: str) -> str:
    """
    Get the structural outline of a notebook (sections, subsections, titles).

    Like a table of contents - shows the organization without full content.

    Args:
        path: Path to the .nb file

    Returns:
        Hierarchical outline of notebook sections
    """
    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps({"success": False, "error": f"File not found: {expanded}"}, indent=2)

    try:
        notebook = await _run_blocking(_load_cached_notebook, expanded, truncation_threshold=25000)
        sections = notebook.get_outline()
        return _json_response(
            {
                "success": True,
                "path": expanded,
                "sections": sections,
                "count": len(sections),
            }
        )
    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


@_tool("file_legacy")
async def parse_notebook_python(
    path: str,
    output_format: Literal["markdown", "wolfram", "outline", "json"] = "markdown",
    truncation_threshold: int = 25000,
) -> str:
    """
    Parse a Mathematica notebook using Python-native parser.

    This tool provides offline notebook parsing without requiring wolframscript.
    It extracts clean, readable Wolfram code from complex BoxData structures.

    Args:
        path: Path to the .nb file
        output_format:
            - "markdown": Readable Markdown with code blocks (default)
            - "wolfram": Pure executable Wolfram Language code only
            - "outline": Hierarchical section outline
            - "json": Structured JSON with all cell data
        truncation_threshold: Max chars per cell before truncation (default 25000).
            Set to 0 to disable truncation (may timeout on large notebooks).

    Returns:
        Notebook content in the requested format
    """
    from .notebook_parser import NotebookParser

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps({"success": False, "error": f"File not found: {expanded}"}, indent=2)

    try:
        effective_threshold = truncation_threshold if truncation_threshold > 0 else 10**9
        parser = NotebookParser(truncation_threshold=effective_threshold)
        notebook = await _run_blocking(
            _load_cached_notebook,
            expanded,
            truncation_threshold=effective_threshold,
        )

        if output_format == "markdown":
            content = parser.to_markdown(notebook)
            return json.dumps(
                {
                    "success": True,
                    "format": "markdown",
                    "path": expanded,
                    "cell_count": len(notebook.cells),
                    "code_cells": len(notebook.get_code_cells()),
                    "content": content,
                },
                indent=2,
            )

        elif output_format == "wolfram":
            content = parser.to_wolfram_code(notebook)
            return json.dumps(
                {
                    "success": True,
                    "format": "wolfram",
                    "path": expanded,
                    "code_cells": len(notebook.get_code_cells()),
                    "content": content,
                },
                indent=2,
            )

        elif output_format == "outline":
            outline = notebook.get_outline()
            return json.dumps(
                {
                    "success": True,
                    "format": "outline",
                    "path": expanded,
                    "section_count": len(outline),
                    "sections": outline,
                },
                indent=2,
            )

        elif output_format == "json":
            cells_data = [
                {
                    "index": c.cell_index,
                    "style": c.style.value,
                    "label": c.cell_label,
                    "content": c.content[:500] if len(c.content) > 500 else c.content,
                    "content_length": len(c.content),
                    "truncated_in_json": len(c.content) > 500,
                    "was_truncated": c.was_truncated,
                    "original_length": c.original_length if c.was_truncated else len(c.content),
                }
                for c in notebook.cells
            ]
            return json.dumps(
                {
                    "success": True,
                    "format": "json",
                    "path": expanded,
                    "title": notebook.title,
                    "cell_count": len(notebook.cells),
                    "code_cells": len(notebook.get_code_cells()),
                    "cells": cells_data,
                },
                indent=2,
            )

        else:
            return json.dumps(
                {"success": False, "error": f"Unknown format: {output_format}"},
                indent=2,
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@_tool("file_legacy")
async def get_notebook_cell(
    path: str,
    cell_index: int,
    full: bool = False,
) -> str:
    """
    Get the full content of a specific cell by index.

    Use parse_notebook_python with format="json" to see all cell indices first.

    Args:
        path: Path to the .nb file
        cell_index: Cell index (0-based)
        full: If True, bypass truncation and return complete cell content (may be very large)

    Returns:
        Full cell content and metadata
    """

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps({"success": False, "error": f"File not found: {expanded}"}, indent=2)

    try:
        threshold = 10**9 if full else 25000
        notebook = await _run_blocking(_load_cached_notebook, expanded, truncation_threshold=threshold)

        if cell_index < 0 or cell_index >= len(notebook.cells):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Cell index {cell_index} out of range (0-{len(notebook.cells) - 1})",
                },
                indent=2,
            )

        cell = notebook.cells[cell_index]
        return json.dumps(
            {
                "success": True,
                "index": cell.cell_index,
                "style": cell.style.value,
                "label": cell.cell_label,
                "content": cell.content,
                "content_length": len(cell.content),
                "was_truncated": cell.was_truncated,
                "original_length": cell.original_length if cell.was_truncated else len(cell.content),
                "raw_content_preview": cell.raw_content[:500] if cell.raw_content else "",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ============================================================================
# TIER 2b: Consolidated Notebook Reading (backend-aware)
# ============================================================================


@_tool("notebook_primary")
async def read_notebook(
    path: str,
    output_format: Literal["markdown", "wolfram", "outline", "json", "plain"] = "markdown",
    cell_types: list[str] | None = None,
    include_outputs: bool = True,
    backend: str | None = None,
    view: str = "semantic",
    include_alternates: bool = False,
    truncation_threshold: int = 25000,
) -> str:
    """
    Read a Mathematica notebook with capability-based backend dispatch.

    Consolidates notebook reading into a single tool. Uses the best available
    backend: kernel (accurate, via NotebookImport) or Python parser (offline).

    Args:
        path: Path to the .nb file
        output_format:
            - "markdown": Readable Markdown with code blocks (default)
            - "wolfram": Pure executable Wolfram Language code only
            - "outline": Hierarchical section outline
            - "json": Structured JSON with cell data and metadata
            - "plain": Plain text
        cell_types: Optional filter — list of styles like ["Input", "Text", "Section"].
                    If omitted, all cell types are included.
        include_outputs: If True (default), include Output cells.
                         If False, filters out Output/Message/Print cells.
        backend: Force a specific backend: "python_syntax" or "kernel_semantic".
                 If omitted, auto-selects based on capability and availability.
        view: Primary view mode: "semantic" (default), "display", or "raw"
        include_alternates: If True, include alternate views per cell (JSON only)
        truncation_threshold: Max chars per cell before truncation (0 = no limit)

    Returns:
        Notebook content in the requested format
    """
    from .notebook_backend import CellView, extract_notebook

    expanded = _expand_path(path)
    if not os.path.exists(expanded):
        return json.dumps({"success": False, "error": f"File not found: {expanded}"}, indent=2)

    # Build cell type filter
    effective_types = list(cell_types) if cell_types else None
    if not include_outputs:
        # Remove output-like styles from whatever type list we have
        output_styles = {"Output", "Message", "Print"}
        if effective_types is not None:
            effective_types = [t for t in effective_types if t not in output_styles]
        else:
            # No explicit types: include everything except output styles
            effective_types = [
                "Input",
                "Code",
                "Text",
                "Title",
                "Chapter",
                "Section",
                "Subsection",
                "Subsubsection",
                "Item",
                "ItemNumbered",
            ]

    # Map format to capability hint for dispatch
    capability_map = {
        "wolfram": "code",
        "outline": "outline",
        "plain": "text",
        "markdown": "full",
        "json": "full",
    }
    capability = capability_map.get(output_format, "full")

    view_enum = {
        "semantic": CellView.SEMANTIC,
        "display": CellView.DISPLAY,
        "raw": CellView.RAW,
    }.get(view, CellView.SEMANTIC)

    try:
        result = await extract_notebook(
            expanded,
            capability=capability,
            cell_types=effective_types,
            view=view_enum,
            include_alternates=include_alternates,
            truncation_threshold=truncation_threshold,
            force_backend=backend,
        )

        if result.error:
            return _json_response({"success": False, "error": result.error})

        if output_format == "markdown":
            return _json_response(
                {
                    "success": True,
                    "format": "markdown",
                    "backend": result.backend,
                    "path": expanded,
                    "cell_count": result.cell_count,
                    "code_cells": result.code_cell_count,
                    "content": result.to_markdown(),
                }
            )
        elif output_format == "wolfram":
            return _json_response(
                {
                    "success": True,
                    "format": "wolfram",
                    "backend": result.backend,
                    "path": expanded,
                    "code_cells": result.code_cell_count,
                    "content": result.to_wolfram_code(),
                }
            )
        elif output_format == "outline":
            outline = result.to_outline()
            return _json_response(
                {
                    "success": True,
                    "format": "outline",
                    "backend": result.backend,
                    "path": expanded,
                    "section_count": len(outline),
                    "sections": outline,
                }
            )
        elif output_format == "plain":
            return _json_response(
                {
                    "success": True,
                    "format": "plain",
                    "backend": result.backend,
                    "path": expanded,
                    "cell_count": result.cell_count,
                    "content": result.to_plain_text(),
                }
            )
        elif output_format == "json":
            return _json_response(result.to_dict(include_alternates))
        else:
            return _json_response({"success": False, "error": f"Unknown format: {output_format}"})

    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


# ============================================================================
# TIER 3: Wolfram Alpha & Natural Language
# ============================================================================


@_tool("knowledge")
async def wolfram_alpha(
    query: str,
    return_type: Literal["result", "data", "full"] = "result",
) -> str:
    """
    Query Wolfram Alpha with natural language.

    This gives Mathematica superpowers - ask questions in plain English
    and get computed answers, data, and more.

    Args:
        query: Natural language question (e.g., "population of France",
               "integrate x^2 from 0 to 1", "weather in Tokyo")
        return_type:
            - "result": Simple text result (default)
            - "data": Structured data when available
            - "full": All available pods/information

    Returns:
        Wolfram Alpha response in requested format

    Example:
        wolfram_alpha("population of Tokyo") -> "13.96 million people (2021)"
        wolfram_alpha("derivative of sin(x^2)", "data") -> {result: "2 x cos(x^2)"}
    """
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.wolfram_alpha(
        query,
        return_type,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("knowledge")
async def interpret_natural_language(text: str) -> str:
    """
    Convert natural language mathematical description to Wolfram Language code.

    This is magic - describe what you want in English and get executable code.

    Args:
        text: Natural language description (e.g., "the integral of x squared from 0 to 1",
              "solve x squared equals 4 for x", "plot sine of x from 0 to 2 pi")

    Returns:
        Wolfram Language code and its evaluation result

    Example:
        interpret_natural_language("the derivative of e to the x")
        -> {code: "D[E^x, x]", result: "E^x"}
    """
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.interpret_natural_language(text)


@_tool("knowledge")
async def entity_lookup(
    entity_type: str,
    name: str,
    properties: list[str] | None = None,
) -> str:
    """
    Look up real-world entity data from Wolfram's curated knowledge base.

    Entity types include: "Country", "City", "Chemical", "Planet", "Company",
    "Person", "Movie", "University", "Element", "Star", and many more.

    Args:
        entity_type: Type of entity (e.g., "Country", "City", "Chemical")
        name: Name to look up (e.g., "France", "Tokyo", "Water")
        properties: Specific properties to retrieve (default: common properties)

    Returns:
        Entity data with requested properties

    Example:
        entity_lookup("Country", "Japan", ["Population", "Capital", "GDP"])
        -> {name: "Japan", Population: "125.8 million", Capital: "Tokyo", GDP: "$4.94 trillion"}
    """
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.entity_lookup(
        entity_type,
        name,
        properties,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("knowledge")
async def convert_units(quantity: str, target_unit: str) -> str:
    """
    Convert between units using Wolfram's comprehensive unit system.

    Args:
        quantity: Value with unit (e.g., "5 miles", "100 kg", "25 Celsius")
        target_unit: Target unit (e.g., "kilometers", "pounds", "Fahrenheit")

    Returns:
        Converted quantity

    Example:
        convert_units("100 kilometers", "miles") -> "62.1371 miles"
        convert_units("0 Celsius", "Fahrenheit") -> "32 Fahrenheit"
    """
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.convert_units(
        quantity,
        target_unit,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("knowledge")
async def get_constant(name: str) -> str:
    """
    Get a physical or mathematical constant.

    Args:
        name: Constant name (e.g., "SpeedOfLight", "PlanckConstant", "Pi",
              "EulerGamma", "GoldenRatio", "Avogadro")

    Returns:
        Constant value with unit (if applicable) and numeric approximation

    Example:
        get_constant("SpeedOfLight") -> {value: "299792458 m/s", numeric: "2.998e8"}
    """
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.get_constant(name, parse_wolfram_association=_parse_wolfram_association)


# ============================================================================
# TIER 4: Interactive Debugging & Tracing
# ============================================================================


@_tool("debug")
async def trace_evaluation(expression: str, max_depth: int = 5) -> str:
    """Trace the step-by-step evaluation of an expression."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.trace_evaluation(
        expression,
        max_depth,
        addon_result=_addon_result,
    )


@_tool("debug")
async def time_expression(expression: str) -> str:
    """Time the evaluation of an expression with memory tracking."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.time_expression(expression, addon_result=_addon_result)


@_tool("core")
async def check_syntax(code: str) -> str:
    """Validate Wolfram Language code syntax without executing it."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.check_syntax(code, addon_result=_addon_result)


# ============================================================================
# TIER 5: Data I/O
# ============================================================================


@_tool("data")
async def import_data(
    path: str,
    format: str | None = None,
) -> str:
    """Import data from a file or URL into Mathematica."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.import_data(
        path,
        format,
        addon_result=_addon_result,
        expand_path=_expand_path,
    )


@_tool("data")
async def export_data(
    expression: str,
    path: str,
    format: str | None = None,
) -> str:
    """Export data or graphics to a file."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.export_data(
        expression,
        path,
        format,
        addon_result=_addon_result,
        expand_path=_expand_path,
    )


@_tool("data")
async def list_supported_formats() -> str:
    """List all supported import/export formats."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.list_supported_formats(
        addon_result=_addon_result,
        parse_wolfram_association=_parse_wolfram_association,
    )


# ============================================================================
# TIER 6: Visualization
# ============================================================================


@_tool("graphics")
async def inspect_graphics(expression: str) -> str:
    """Analyze the structure of a graphics object."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.inspect_graphics(expression, parse_wolfram_association=_parse_wolfram_association)


@_tool("graphics")
async def export_graphics(
    expression: str,
    path: str,
    format: Literal["PNG", "PDF", "SVG", "EPS", "JPEG"] = "PNG",
    size: int = 600,
) -> str:
    """Export a graphics expression to an image file."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.export_graphics(
        expression,
        path,
        format,
        size,
        addon_result=_addon_result,
        expand_path=_expand_path,
    )


@_tool("graphics")
async def compare_plots(expressions: list[str], labels: list[str] | None = None) -> str:
    """Generate a side-by-side comparison of multiple plots."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.compare_plots(
        expressions,
        labels,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("graphics")
async def create_animation(
    expression: str,
    parameter: str,
    range_spec: str,
    frames: int = 20,
) -> str:
    """Create an animation by varying a parameter."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.create_animation(
        expression,
        parameter,
        range_spec,
        frames,
        parse_wolfram_association=_parse_wolfram_association,
    )


# ============================================================================
# Feature Flags and Telemetry
# ============================================================================


@_tool("core")
async def get_feature_status() -> str:
    """Get the status of all feature flags."""
    return json.dumps(
        {
            "success": True,
            "features": FEATURES.to_dict(),
        },
        indent=2,
    )


@_tool("debug")
async def get_computation_journal() -> str:
    """Get recent computation history: code previews, outputs, timing, success status."""
    if _journal is None:
        return _json_response({"success": False, "error": "Journal not initialized"})
    return _json_response({"success": True, **_journal.to_dict()})


@_tool("debug")
async def clear_computation_journal() -> str:
    """Clear the computation journal."""
    if _journal is None:
        return _json_response({"success": False, "error": "Journal not initialized"})
    _journal.clear()
    return _json_response({"success": True, "message": "Computation journal cleared."})


_apply_guidance_docs()
_register_core_tools()
_register_optional_tools()

# Initialize routing memory if enabled
if FEATURES.routing_memory != "off":
    from .routing_memory import init as _init_routing

    _routing_mem = _init_routing(FEATURES.routing_memory)

# Initialize journal
from .journal import ComputationJournal  # noqa: E402

_journal = ComputationJournal()


@mcp.prompt()
def mathematica_expert(user_request: str = "") -> str:
    """Expert guidance for using Mathematica tools effectively."""
    return build_mathematica_expert_prompt(user_request, features=FEATURES)


@mcp.prompt()
def calculate(expression: str) -> str:
    """Compute a result inline in chat. Use for quick math, algebra, or any text answer."""
    return build_prompt_calculate(FEATURES, expression)


@mcp.prompt()
def notebook(task: str) -> str:
    """Execute in the current Mathematica notebook. Use for plots, visualizations, or notebook artifacts."""
    return build_prompt_notebook(FEATURES, task)


@mcp.prompt()
def new_notebook(task: str, title: str = "Analysis") -> str:
    """Create a fresh Mathematica notebook and execute there. Use when you want a clean slate."""
    return build_prompt_new_notebook(FEATURES, task, title)


@mcp.prompt()
def interactive(task: str) -> str:
    """Execute with frontend mode for dynamic/interactive content (Manipulate, Animate, sliders)."""
    return build_prompt_interactive(FEATURES, task)


@mcp.prompt()
def quickstart() -> str:
    """Show available execution styles and how to use them."""
    return build_prompt_quickstart(FEATURES)


def main():
    # Kick the ~13s kernel boot in the background so the first warm call finds a
    # ready session (see session.prewarm_kernel). Import-time is untouched: this
    # only fires on the real serve path, never when tests/schema dumps import us.
    from .session import prewarm_kernel

    prewarm_kernel()
    mcp.run()


if __name__ == "__main__":
    main()
