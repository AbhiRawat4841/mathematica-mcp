from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

from mcp.server.fastmcp import FastMCP


def register_function_repository_tools(mcp: FastMCP, *, parse_wolfram_association) -> None:
    @mcp.tool()
    async def search_function_repository(query: str, max_results: int = 10) -> str:
        """Search the Wolfram Function Repository."""
        from .lazy_wolfram_tools import _find_wolframscript
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        safe_query = query.replace('"', '\\"')
        search_code = f"""
Module[{{results, query, clean, fetch, maxRes}},
  query = "{safe_query}";
  maxRes = {max_results};
  fetch[q_, field_] := Quiet[Check[
    Normal@ResourceSearch[{{"ResourceType" -> "Function", field -> q}}, "Associations"],
    {{}}
  ]];
  results = Quiet[Check[Take[fetch[query, "Name"], UpTo[maxRes]], {{}}]];
  If[results === {{}}, results = Quiet[Check[Take[fetch[query, "Keyword"], UpTo[maxRes]], {{}}]]];
  clean[res_] := <|
    "name" -> ToString[Lookup[res, "Name", ""]],
    "short_description" -> ToString[Lookup[res, "ShortDescription", ""]],
    "repository_location" -> "Wolfram Function Repository"
  |>;
  ExportString[<|"success" -> True, "query" -> query, "count" -> Length[results], "results" -> Map[clean, results]|>, "JSON"]
]
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", search_code],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return json.dumps(
                    {
                        "success": False,
                        "error": result.stderr or "Search failed",
                        "query": query,
                    },
                    indent=2,
                )
            raw_output = result.stdout.strip()
            if not raw_output:
                return json.dumps(
                    {"success": False, "error": "Empty search response", "query": query},
                    indent=2,
                )
            return json.dumps(json.loads(raw_output), indent=2)
        except subprocess.TimeoutExpired:
            return json.dumps(
                {"success": False, "error": "Search timed out", "query": query}, indent=2
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e), "query": query}, indent=2)

    @mcp.tool()
    async def get_function_repository_info(function_name: str) -> str:
        """Get details about a Wolfram Function Repository function."""
        from .lazy_wolfram_tools import _find_wolframscript
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        info_code = f"""
Module[{{ro, info}},
  ro = Quiet[Check[ResourceObject["{function_name}"], $Failed]];
  If[ro === $Failed,
    <|"success" -> False, "error" -> "Function not found in repository"|>,
    info = <|
      "success" -> True,
      "name" -> "{function_name}",
      "description" -> Quiet[Check[ro["Description"], ""]],
      "documentation_link" -> Quiet[Check[ro["DocumentationLink"], ""]],
      "version" -> Quiet[Check[ToString[ro["Version"]], ""]],
      "author" -> Quiet[Check[ro["ContributorInformation"], ""]],
      "keywords" -> Quiet[Check[ro["Keywords"], {{}}]],
      "usage_example" -> Quiet[Check[ToString[First[ro["BasicExamples"], ""]], ""]]
    |>;
    info
  ]
]
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", info_code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.dumps(
                parse_wolfram_association(result.stdout.strip())
                if result.stdout.strip()
                else {"success": False, "error": "Empty response"},
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {"success": False, "error": str(e), "function": function_name}, indent=2
            )

    @mcp.tool()
    async def load_resource_function(function_name: str) -> str:
        """Load a function from the Wolfram Function Repository."""
        from .lazy_wolfram_tools import _find_wolframscript
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        load_code = f"""
Module[{{fn, result}},
  fn = Quiet[Check[ResourceFunction["{function_name}"], $Failed]];
  If[fn === $Failed,
    <|"success" -> False, "error" -> "Failed to load function from repository"|>,
    <|
      "success" -> True,
      "function" -> "{function_name}",
      "loaded" -> True,
      "usage" -> "Use ResourceFunction[\\"{function_name}\\"][args] to call the function",
      "message" -> "Function loaded successfully from Wolfram Function Repository"
    |>
  ]
]
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", load_code],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return json.dumps(
                parse_wolfram_association(result.stdout.strip())
                if result.stdout.strip()
                else {"success": False, "error": "Empty response"},
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {"success": False, "error": str(e), "function": function_name}, indent=2
            )


def register_data_repository_tools(mcp: FastMCP, *, parse_wolfram_association) -> None:
    @mcp.tool()
    async def search_data_repository(query: str, max_results: int = 10) -> str:
        """Search the Wolfram Data Repository."""
        from .lazy_wolfram_tools import _find_wolframscript
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        search_code = f"""
Module[{{results}},
  results = Quiet[Check[
    Take[ResourceSearch[{{"ResourceType" -> "DataResource", "Name" -> "{query}"}}, "SnippetData"], UpTo[{max_results}]],
    {{}}
  ]];
  If[results === {{}},
    results = Quiet[Check[
      Take[ResourceSearch[{{"ResourceType" -> "DataResource", "Keyword" -> "{query}"}}, "SnippetData"], UpTo[{max_results}]],
      {{}}
    ]]
  ];
  <|
    "success" -> True,
    "query" -> "{query}",
    "count" -> Length[results],
    "datasets" -> Map[<|"name" -> #["Name"], "description" -> Quiet[Check[#["ShortDescription"], ""]]|> &, results]
  |>
]
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", search_code],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e), "query": query}, indent=2)

    @mcp.tool()
    async def get_dataset_info(dataset_name: str) -> str:
        """Get detailed information about a Wolfram Data Repository dataset."""
        from .lazy_wolfram_tools import _find_wolframscript
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        info_code = f"""
Module[{{rd, info}},
  rd = Quiet[Check[ResourceObject["{dataset_name}"], $Failed]];
  If[rd === $Failed,
    <|"success" -> False, "error" -> "Dataset not found"|>,
    <|
      "success" -> True,
      "name" -> "{dataset_name}",
      "description" -> Quiet[Check[rd["Description"], ""]],
      "content_types" -> Quiet[Check[rd["ContentTypes"], {{}}]],
      "documentation_link" -> Quiet[Check[rd["DocumentationLink"], ""]],
      "keywords" -> Quiet[Check[rd["Keywords"], {{}}]]
    |>
  ]
]
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", info_code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
        except Exception as e:
            return json.dumps(
                {"success": False, "error": str(e), "dataset": dataset_name}, indent=2
            )

    @mcp.tool()
    async def load_dataset(
        dataset_name: str,
        sample_size: Optional[int] = None,
    ) -> str:
        """Load a dataset from the Wolfram Data Repository."""
        from .lazy_wolfram_tools import _find_wolframscript
        wolframscript = _find_wolframscript()
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found in PATH"}, indent=2
            )

        sample_clause = f"Take[#, UpTo[{sample_size}]]&" if sample_size else "Identity"
        load_code = f"""
Module[{{data, info}},
  data = Quiet[Check[ResourceData["{dataset_name}"], $Failed]];
  If[data === $Failed,
    <|"success" -> False, "error" -> "Failed to load dataset"|>,
    <|
      "success" -> True,
      "name" -> "{dataset_name}",
      "loaded" -> True,
      "type" -> Head[data],
      "dimensions" -> If[Head[data] === Dataset,
        Quiet[Check[Dimensions[Normal[data]], "Unknown"]],
        Quiet[Check[Dimensions[data], "Unknown"]]
      ],
      "sample" -> ToString[{sample_clause}[data], InputForm],
      "columns" -> If[Head[data] === Dataset,
        Quiet[Check[Keys[First[Normal[data]]], {{}}]],
        {{}}
      ]
    |>
  ]
]
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", load_code],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return json.dumps(parse_wolfram_association(result.stdout.strip()), indent=2)
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "success": False,
                    "error": "Dataset loading timed out - dataset may be large",
                    "dataset": dataset_name,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {"success": False, "error": str(e), "dataset": dataset_name}, indent=2
            )
