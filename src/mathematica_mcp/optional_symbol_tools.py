from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP


def register_symbol_lookup_tools(
    mcp: FastMCP,
    *,
    lookup_symbols_in_kernel,
    extract_short_description,
    extract_example_signature,
    rank_candidates,
    parse_wolfram_association,
    execute_code,
) -> None:
    @mcp.tool()
    async def resolve_function(
        query: str,
        expression: Optional[str] = None,
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

        raw_output = lookup_result.get("raw_output", "")
        if not raw_output:
            return json.dumps(
                {
                    "status": "not_found",
                    "query": query,
                    "message": f"No functions found matching '{query}'",
                },
                indent=2,
            )

        candidates_raw = []
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

        if not candidates_raw:
            wolframscript = shutil.which("wolframscript")
            if wolframscript:
                try:
                    result = subprocess.run(
                        [wolframscript, "-code", f'Names["*{query}*"]'],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if result.returncode == 0:
                        for name in re.findall(r'"([^"]+)"', result.stdout)[:20]:
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
                "other_candidates": formatted_candidates[1:]
                if len(formatted_candidates) > 1
                else [],
            }

            if auto_execute and expression:
                exec_result = await execute_code(
                    code=expression, format="text", output_target=output_target
                )
                response["execution"] = {
                    "executed": True,
                    "expression": expression,
                    "result": json.loads(exec_result)
                    if exec_result.startswith("{")
                    else exec_result,
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
        wolframscript = shutil.which("wolframscript")
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        info_code = f"""
Module[{{sym, info, usage, opts, attrs, syntaxInfo, relatedSyms, examples}},
  sym = ToExpression["{symbol}"];
  usage = Quiet[Check[ToString[sym::usage], "No usage information available"]];
  opts = Quiet[Check[Map[{{ToString[#[[1]]], ToString[#[[2]]]}} &, Options[sym]], {{}}]];
  attrs = Quiet[Check[ToString /@ Attributes[sym], {{}}]];
  syntaxInfo = Quiet[Check[SyntaxInformation[sym], {{}}]];
  relatedSyms = Quiet[Check[Take[ToString /@ WolframLanguageData["{symbol}", "RelatedSymbols"], UpTo[10]], {{}}]];
  <|
    "success" -> True,
    "symbol" -> "{symbol}",
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

        try:
            result = subprocess.run(
                [wolframscript, "-code", info_code],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return json.dumps(
                    {
                        "success": False,
                        "error": result.stderr or "Symbol lookup failed",
                        "symbol": symbol,
                    },
                    indent=2,
                )

            raw_output = result.stdout.strip()
            parsed = parse_wolfram_association(raw_output)

            if isinstance(parsed, dict) and parsed.get("success"):
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
                return json.dumps(formatted, indent=2)

            return json.dumps(
                {
                    "success": True,
                    "symbol": symbol,
                    "raw_output": raw_output,
                    "note": "Partial parsing - see raw output",
                },
                indent=2,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {"success": False, "error": "Symbol lookup timed out", "symbol": symbol},
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {"success": False, "error": str(e), "symbol": symbol}, indent=2
            )

    @mcp.tool()
    async def suggest_similar_functions(query: str) -> str:
        """Find Wolfram functions similar to a query using fuzzy matching."""
        wolframscript = shutil.which("wolframscript")
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found"}, indent=2
            )

        code = f'''
Module[{{query, matches}},
  query = "{query}";
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
'''
        try:
            result = subprocess.run(
                [wolframscript, "-code", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
