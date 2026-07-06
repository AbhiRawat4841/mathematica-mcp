from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def register_function_repository_tools(mcp: FastMCP, *, parse_wolfram_association) -> None:
    @mcp.tool()
    async def search_function_repository(query: str, max_results: int = 10) -> str:
        """Search the Wolfram Function Repository."""
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        # Read-only ResourceSearch — no isolation needed; the warm kernel's
        # resource-metadata cache now persists across calls (improvement).
        search_code = f"""
Module[{{results, query, clean, fetch, maxRes}},
  query = {_wl_string(query)};
  maxRes = {int(max_results)};
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
  <|"success" -> True, "query" -> query, "count" -> Length[results], "results" -> Map[clean, results]|>
]
"""
        return await _run_wl_parsed(search_code, parse_wolfram_association, timeout=60)

    @mcp.tool()
    async def get_function_repository_info(function_name: str) -> str:
        """Get details about a Wolfram Function Repository function."""
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        # Read-only ResourceObject metadata query — no isolation needed.
        wl_name = _wl_string(function_name)
        info_code = f"""
Module[{{ro, info}},
  ro = Quiet[Check[ResourceObject[{wl_name}], $Failed]];
  If[ro === $Failed,
    <|"success" -> False, "error" -> "Function not found in repository"|>,
    info = <|
      "success" -> True,
      "name" -> {wl_name},
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
        return await _run_wl_parsed(info_code, parse_wolfram_association, timeout=30)

    @mcp.tool()
    async def load_resource_function(function_name: str) -> str:
        """Load a function from the Wolfram Function Repository."""
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        # Deliberately NOT scratch-blocked: loading into the persistent kernel makes
        # the ResourceFunction actually available to later execute_code calls (the
        # old cold path loaded it into a throwaway kernel that exited — only the
        # on-disk resource cache survived). Semantic improvement, same metadata.
        wl_name = _wl_string(function_name)
        load_code = f"""
Module[{{fn, result}},
  fn = Quiet[Check[ResourceFunction[{wl_name}], $Failed]];
  If[fn === $Failed,
    <|"success" -> False, "error" -> "Failed to load function from repository"|>,
    <|
      "success" -> True,
      "function" -> {wl_name},
      "loaded" -> True,
      "usage" -> "Use ResourceFunction[" <> ToString[{wl_name}, InputForm] <> "][args] to call the function",
      "message" -> "Function loaded successfully from Wolfram Function Repository"
    |>
  ]
]
"""
        return await _run_wl_parsed(load_code, parse_wolfram_association, timeout=60)


def register_data_repository_tools(mcp: FastMCP, *, parse_wolfram_association) -> None:
    @mcp.tool()
    async def search_data_repository(query: str, max_results: int = 10) -> str:
        """Search the Wolfram Data Repository."""
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        # Read-only ResourceSearch — no isolation needed.
        wl_query = _wl_string(query)
        max_res = int(max_results)
        search_code = f"""
Module[{{results}},
  results = Quiet[Check[
    Take[ResourceSearch[{{"ResourceType" -> "DataResource", "Name" -> {wl_query}}}, "SnippetData"], UpTo[{max_res}]],
    {{}}
  ]];
  If[results === {{}},
    results = Quiet[Check[
      Take[ResourceSearch[{{"ResourceType" -> "DataResource", "Keyword" -> {wl_query}}}, "SnippetData"], UpTo[{max_res}]],
      {{}}
    ]]
  ];
  <|
    "success" -> True,
    "query" -> {wl_query},
    "count" -> Length[results],
    "datasets" -> Map[<|"name" -> #["Name"], "description" -> Quiet[Check[#["ShortDescription"], ""]]|> &, results]
  |>
]
"""
        return await _run_wl_parsed(search_code, parse_wolfram_association, timeout=60)

    @mcp.tool()
    async def get_dataset_info(dataset_name: str) -> str:
        """Get detailed information about a Wolfram Data Repository dataset."""
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        # Read-only ResourceObject metadata query — no isolation needed.
        wl_name = _wl_string(dataset_name)
        info_code = f"""
Module[{{rd, info}},
  rd = Quiet[Check[ResourceObject[{wl_name}], $Failed]];
  If[rd === $Failed,
    <|"success" -> False, "error" -> "Dataset not found"|>,
    <|
      "success" -> True,
      "name" -> {wl_name},
      "description" -> Quiet[Check[rd["Description"], ""]],
      "content_types" -> Quiet[Check[rd["ContentTypes"], {{}}]],
      "documentation_link" -> Quiet[Check[rd["DocumentationLink"], ""]],
      "keywords" -> Quiet[Check[rd["Keywords"], {{}}]]
    |>
  ]
]
"""
        return await _run_wl_parsed(info_code, parse_wolfram_association, timeout=30)

    @mcp.tool()
    async def load_dataset(
        dataset_name: str,
        sample_size: int | None = None,
    ) -> str:
        """Load a dataset from the Wolfram Data Repository."""
        from .lazy_wolfram_tools import _run_wl_parsed, _wl_string

        # Data stays Module-local (no Global` pollution); ResourceData now also
        # warms the persistent kernel's resource cache, so a later
        # ResourceData[...] in execute_code is instant instead of re-downloading.
        wl_name = _wl_string(dataset_name)
        sample_clause = f"Take[#, UpTo[{int(sample_size)}]]&" if sample_size else "Identity"
        load_code = f"""
Module[{{data, info}},
  data = Quiet[Check[ResourceData[{wl_name}], $Failed]];
  If[data === $Failed,
    <|"success" -> False, "error" -> "Failed to load dataset"|>,
    <|
      "success" -> True,
      "name" -> {wl_name},
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
        return await _run_wl_parsed(load_code, parse_wolfram_association, timeout=120)
