from __future__ import annotations

import json
import re
from typing import Literal

from mcp.server.fastmcp import FastMCP


def register_symbol_lookup_tools(
    mcp: FastMCP,
    *,
    lookup_symbols_in_kernel,
    hydrate_usage,
    extract_short_description,
    extract_example_signature,
    rank_candidates,
    parse_wolfram_association,
    execute_code,
) -> None:
    @mcp.tool()
    async def resolve_function(
        query: str,
        expression: str | None = None,
        auto_execute: bool = True,
        max_candidates: int = 5,
        output_target: Literal["cli", "notebook"] = "cli",
    ) -> str:
        """Search for Wolfram Language functions and optionally auto-execute."""
        score_threshold = 80
        score_gap_threshold = 15

        lookup_result = lookup_symbols_in_kernel(query)

        if not lookup_result.get("success"):
            return json.dumps(
                {
                    "status": "error",
                    "error": lookup_result.get("error", "Lookup failed"),
                    "query": query,
                },
                indent=2,
            )

        # Fast path: structured candidates from symbol index.
        candidates_raw = lookup_result.get("candidates", [])
        system_only = lookup_result.get("system_only", False)

        if not candidates_raw:
            # Legacy path: parse raw WL output from subprocess fallback.
            raw_output = lookup_result.get("raw_output", "")
            if raw_output:
                try:
                    lines = raw_output.split("\n")
                    for line in lines:
                        if '"symbol"' in line and "->" in line:
                            symbol_match = re.search(r'"symbol"\s*->\s*"([^"]+)"', line)
                            usage_match = re.search(r'"usage"\s*->\s*"([^"]*)"', line)
                            if symbol_match:
                                candidates_raw.append(
                                    {
                                        "symbol": symbol_match.group(1),
                                        "usage": usage_match.group(1) if usage_match else "",
                                    }
                                )
                except Exception:
                    pass

        # When the fast index path was used, it only covers System symbols.
        # Also search Global symbols so user-defined names are not lost.
        if system_only or not candidates_raw:
            from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

            # Read-only Names[] lookup — no isolation needed. Warm kernel means
            # Global` actually contains the user's session symbols now (the old
            # cold subprocess spawned a fresh kernel whose Global` was empty, so
            # this fallback could never find user-defined names).
            pattern = f"Global`*{query}*" if system_only else f"*{query}*"
            names_code = f'<|"success" -> True, "names" -> Names[{_wl_string(pattern)}]|>'
            try:
                parsed = json.loads(await _run_wl_parsed(names_code, parse_wolfram_association, timeout=15))
                names = parsed.get("names", []) if parsed.get("success") else []
                existing = {c.get("symbol") for c in candidates_raw}
                for name in names[:20]:
                    if isinstance(name, str) and name not in existing:
                        candidates_raw.append({"symbol": name, "usage": ""})
            except Exception:
                pass

        if not candidates_raw:
            return json.dumps(
                {
                    "status": "not_found",
                    "query": query,
                    "message": f"No functions found matching '{query}'",
                },
                indent=2,
            )

        top_candidates = rank_candidates(query, candidates_raw)[:max_candidates]

        # Lazy metadata hydration: fetch usage for top candidates that lack it.
        symbols_needing_usage = [
            c.get("symbol_name", c.get("symbol", "")) for c in top_candidates if not c.get("usage")
        ]
        if symbols_needing_usage:
            usage_map = hydrate_usage(symbols_needing_usage)
            for c in top_candidates:
                sym = c.get("symbol_name", c.get("symbol", ""))
                if not c.get("usage") and sym in usage_map:
                    c["usage"] = usage_map[sym]

        formatted_candidates = []
        for candidate in top_candidates:
            symbol_name = candidate.get("symbol_name", candidate.get("symbol", ""))
            usage = candidate.get("usage", "")
            formatted_candidates.append(
                {
                    "symbol": symbol_name,
                    "full_name": candidate.get("symbol", symbol_name),
                    "description": extract_short_description(usage),
                    "example": extract_example_signature(usage, symbol_name),
                    "score": round(candidate.get("_score", 0), 2),
                }
            )

        is_resolved = False
        if formatted_candidates:
            top_score = formatted_candidates[0]["score"]
            if top_score >= score_threshold:
                if len(formatted_candidates) == 1:
                    is_resolved = True
                else:
                    second_score = formatted_candidates[1]["score"]
                    if top_score - second_score >= score_gap_threshold:
                        is_resolved = True

        if is_resolved:
            resolved_symbol = formatted_candidates[0]
            response = {
                "status": "resolved",
                "query": query,
                "resolved_symbol": resolved_symbol["symbol"],
                "description": resolved_symbol["description"],
                "example": resolved_symbol["example"],
                "other_candidates": formatted_candidates[1:] if len(formatted_candidates) > 1 else [],
            }

            if auto_execute and expression:
                exec_result = await execute_code(code=expression, format="text", output_target=output_target)
                response["execution"] = {
                    "executed": True,
                    "expression": expression,
                    "result": json.loads(exec_result) if exec_result.startswith("{") else exec_result,
                }

            return json.dumps(response, indent=2)

        return json.dumps(
            {
                "status": "ambiguous",
                "query": query,
                "message": f"Multiple functions match '{query}'. Please clarify which one you need.",
                "candidates": formatted_candidates,
                "hint": "Provide more specific query or select from candidates above",
            },
            indent=2,
        )

    @mcp.tool()
    async def get_symbol_info(symbol: str) -> str:
        """Get comprehensive information about a Wolfram Language symbol."""
        from . import symbol_index

        # Check metadata cache — only use entries that have the full payload
        # (not usage-only entries from hydration).
        cached = symbol_index.get_cached_metadata(symbol)
        if cached and cached.get("success") and cached.get("attributes") is not None:
            return json.dumps(cached, indent=2)

        from .lazy_wolfram_tools import _run_wl_parsed, _scratch_block, _wl_string

        # Scratch-blocked: ToExpression on caller-supplied text must not create or
        # evaluate symbols in the shared kernel's Global` (the old throwaway kernel
        # gave this isolation for free). Unknown names land in MCPScratch` instead.
        wl_sym = _wl_string(symbol)
        info_code = f"""
Module[{{sym, info, usage, opts, attrs, syntaxInfo, relatedSyms, examples}},
  sym = ToExpression[{wl_sym}];
  (* StringQ guard: an UNDEFINED symbol's sym::usage stays an unevaluated
     MessageName whose ToString is the literal "Sym::usage" — without the guard
     that fake text would be returned AND cached into the symbol index. *)
  usage = Quiet[Check[
    With[{{u = sym::usage}}, If[StringQ[u], u, "No usage information available"]],
    "No usage information available"]];
  opts = Quiet[Check[Map[{{ToString[#[[1]]], ToString[#[[2]]]}} &, Options[sym]], {{}}]];
  attrs = Quiet[Check[ToString /@ Attributes[sym], {{}}]];
  syntaxInfo = Quiet[Check[SyntaxInformation[sym], {{}}]];
  relatedSyms = Quiet[Check[Take[ToString /@ WolframLanguageData[{wl_sym}, "RelatedSymbols"], UpTo[10]], {{}}]];
  <|
    "success" -> True,
    "symbol" -> {wl_sym},
    "usage" -> usage,
    "options" -> opts,
    "options_count" -> Length[opts],
    "attributes" -> attrs,
    "syntax_info" -> ToString[syntaxInfo],
    "related_symbols" -> relatedSyms,
    "is_function" -> MemberQ[Attributes[sym], Protected],
    "context" -> Quiet[Check[Context[sym], "Unknown"]]
  |>
]
"""

        raw = await _run_wl_parsed(_scratch_block(info_code), parse_wolfram_association, timeout=30)
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("success") and parsed.get("attributes") is not None:
            formatted = {
                "success": True,
                "symbol": symbol,
                "usage": parsed.get("usage", ""),
                "attributes": parsed.get("attributes", []),
                "options_count": parsed.get("options_count", 0),
                "options": parsed.get("options", [])[:10],
                "related_symbols": parsed.get("related_symbols", []),
                "context": parsed.get("context", "Unknown"),
            }
            # Cache a copy without execution_method: a later cache hit must not
            # claim the execution method of the original lookup.
            symbol_index.cache_metadata(symbol, dict(formatted))
            formatted["execution_method"] = parsed.get("execution_method")
            return json.dumps(formatted, indent=2)
        # Funnel error or unparseable output: return it as-is (carries
        # success/error/execution_method from _run_wl_parsed).
        return raw

    @mcp.tool()
    async def suggest_similar_functions(query: str) -> str:
        """Find Wolfram functions similar to a query using fuzzy matching."""
        from . import symbol_index

        # Fast path: use pre-built index + hydrate usage for top hits.
        matches = symbol_index.search(query, max_results=20)
        if matches:
            usage_map = hydrate_usage(matches[:10])
            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "matches": [{"name": m, "usage": usage_map.get(m, "")} for m in matches],
                },
                indent=2,
            )

        # Fallback: warm kernel. Read-only — Names[] plus usage strings of
        # System` symbols only — so no isolation needed.
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        code = f"""
Module[{{query, matches}},
  query = {_wl_string(query)};
  matches = Select[
    Names["System`*"],
    StringContainsQ[#, query, IgnoreCase -> True] &
  ];
  matches = Take[matches, UpTo[20]];
  <|
    "success" -> True,
    "query" -> query,
    "matches" -> Map[
      <|
        "name" -> #,
        "usage" -> StringTake[
          ToString[ToExpression[# <> "::usage"] /. _MessageName -> ""],
          UpTo[100]
        ]
      |> &,
      matches
    ]
  |>
]
"""
        return await _run_wl_parsed(code, parse_wolfram_association, timeout=30)
