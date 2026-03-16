(* ::Package:: *)
(* notebook_converter.wl — Kernel-native notebook extraction for MCP *)
(*
   Design principles:
   - Use NotebookImport where possible (safe, maintained by Wolfram)
   - Never evaluate user code during extraction
   - Preserve cell identity (CellID, group path, labels, tags)
   - Return structured JSON for Python consumption
   - Mark lossy conversions explicitly
*)

BeginPackage["MCPNotebookConverter`"];

MCPExtractNotebook::usage = "MCPExtractNotebook[file] extracts all cells with metadata as JSON.";
MCPExtractCode::usage = "MCPExtractCode[file] extracts only Input/Code cells as InputText.";

Begin["`Private`"];

(* ========================================================================= *)
(* MAIN EXTRACTION FUNCTION                                                  *)
(* ========================================================================= *)

MCPExtractNotebook[file_String] := Module[
  {nbExpr, topLevel, cells, title, result},

  (* Import notebook — auto-detect format for .nb files.
     Import[file] returns the full Notebook[...] expression. *)
  nbExpr = Quiet[Check[Import[file], $Failed]];
  If[nbExpr === $Failed || !MatchQ[nbExpr, _Notebook],
    Return[ExportString[<|
      "error" -> "Failed to import notebook",
      "path" -> file
    |>, "RawJSON"]]
  ];

  (* Extract top-level cell list *)
  topLevel = First[nbExpr];
  If[!ListQ[topLevel], topLevel = {topLevel}];

  (* Walk the cell tree *)
  cells = Flatten[MapIndexed[processTopLevel, topLevel], 1];

  (* Assign sequential indices *)
  cells = MapIndexed[Append[#1, "index" -> (#2[[1]] - 1)] &, cells];

  (* Find title *)
  title = "";
  Do[
    If[c["style"] === "Title",
      title = c["content"];
      Break[]
    ],
    {c, cells}
  ];

  result = <|
    "path" -> file,
    "title" -> title,
    "cell_count" -> Length[cells],
    "code_cells" -> Count[cells, _?(MemberQ[{"Input", "Code"}, #["style"]] &)],
    "backend" -> "kernel_semantic",
    "cells" -> cells
  |>;

  ExportString[result, "RawJSON"]
];


(* ========================================================================= *)
(* CELL TREE WALKER                                                          *)
(* ========================================================================= *)

(* Real .nb AST shape: top-level elements are Cell[CellGroupData[{...}, Open]]
   — a Cell wrapper around CellGroupData.  Handle this first. *)
processTopLevel[Cell[CellGroupData[cellList_List, ___], ___], {groupIdx_}] :=
  MapIndexed[
    processSingleCell[#1, groupIdx, #2[[1]]] &,
    cellList
  ];

(* Bare CellGroupData without Cell wrapper (some notebooks) *)
processTopLevel[CellGroupData[cellList_List, ___], {groupIdx_}] :=
  MapIndexed[
    processSingleCell[#1, groupIdx, #2[[1]]] &,
    cellList
  ];

(* Standalone cell without grouping *)
processTopLevel[cell_Cell, {groupIdx_}] :=
  {processSingleCell[cell, groupIdx, 1]};

(* Skip non-cell elements (notebook options, etc.) *)
processTopLevel[_, _] := {};


(* ========================================================================= *)
(* SINGLE CELL PROCESSING                                                    *)
(* ========================================================================= *)

processSingleCell[Cell[content_, style_String, opts___], groupId_, posInGroup_] :=
  Module[{cellId, label, tags, isGenerated, converted, lossy},
    cellId = Quiet[CellID /. {opts} /. CellID -> Null];
    label = Quiet[CellLabel /. {opts} /. CellLabel -> ""];
    tags = Quiet[CellTags /. {opts} /. CellTags -> {}];
    isGenerated = TrueQ[Quiet[GeneratedCell /. {opts} /. GeneratedCell -> False]];

    (* Convert content based on cell style *)
    {converted, lossy} = safeConvertContent[content, style];

    <|
      "style" -> style,
      "content" -> converted,
      "cell_id" -> cellId,
      "group_id" -> groupId,
      "position_in_group" -> posInGroup,
      "cell_label" -> If[StringQ[label], label, ""],
      "tags" -> If[ListQ[tags], tags, If[StringQ[tags], {tags}, {}]],
      "is_generated" -> isGenerated,
      "conversion_lossy" -> lossy
    |>
  ];

(* Nested CellGroupData inside a group — recurse *)
processSingleCell[CellGroupData[cellList_List, ___], groupId_, _] :=
  Sequence @@ MapIndexed[
    processSingleCell[#1, groupId, #2[[1]]] &,
    cellList
  ];

(* Cell wrapping CellGroupData inside a group (nested groups) *)
processSingleCell[Cell[CellGroupData[cellList_List, ___], ___], groupId_, _] :=
  Sequence @@ MapIndexed[
    processSingleCell[#1, groupId, #2[[1]]] &,
    cellList
  ];

(* Fallback: skip unrecognized structures *)
processSingleCell[_, _, _] := Nothing;


(* ========================================================================= *)
(* SAFE CONTENT CONVERSION                                                   *)
(* ========================================================================= *)

(* Input/Code cells: convert BoxData to InputForm safely *)
safeConvertContent[BoxData[boxes_], style_String] /;
  MemberQ[{"Input", "Code"}, style] :=
  Module[{held, result},
    held = Quiet[Check[MakeExpression[boxes, StandardForm], $Failed]];
    If[MatchQ[held, HoldComplete[___]],
      (* Safe: ToString of the held wrapper, then strip it at string level *)
      result = ToString[held, InputForm];
      result = StringReplace[result,
        RegularExpression["^HoldComplete\\[(.*)\\]$"] :> "$1"
      ];
      {result, False},
      (* Fallback: flatten boxes to text without MakeExpression *)
      {flattenBoxes[boxes], True}
    ]
  ];

(* Output cells: try same approach, mark as semantic if successful *)
safeConvertContent[BoxData[boxes_], _] :=
  Module[{held, result},
    held = Quiet[Check[MakeExpression[boxes, StandardForm], $Failed]];
    If[MatchQ[held, HoldComplete[___]],
      result = ToString[held, InputForm];
      result = StringReplace[result,
        RegularExpression["^HoldComplete\\[(.*)\\]$"] :> "$1"
      ];
      {result, False},
      (* Lossy fallback for output *)
      {flattenBoxes[boxes], True}
    ]
  ];

(* TextData: extract plain text parts *)
safeConvertContent[TextData[parts_], _] :=
  {StringJoin[extractTextParts[parts]], False};

(* Plain string *)
safeConvertContent[text_String, _] := {text, False};

(* Dynamic content: describe structure *)
safeConvertContent[BoxData[DynamicModuleBox[vars_, body_, ___]], _] :=
  Module[{desc},
    desc = "[Dynamic: " <> ToString[Head[body]] <> " with " <>
      ToString[Length[vars]] <> " variables]";
    {desc, True}
  ];

(* Fallback *)
safeConvertContent[content_, _] :=
  {ToString[content, InputForm], True};


(* ========================================================================= *)
(* TEXT EXTRACTION                                                           *)
(* ========================================================================= *)

extractTextParts[parts_List] := StringJoin[extractTextPart /@ parts];
extractTextParts[part_] := extractTextPart[part];

extractTextPart[s_String] := s;
extractTextPart[StyleBox[s_String, ___]] := s;
extractTextPart[ButtonBox[s_String, ___]] := s;
extractTextPart[Cell[BoxData[FormBox[content_, ___]], "InlineFormula", ___]] :=
  flattenBoxes[content];
extractTextPart[Cell[BoxData[boxes_], "InlineFormula", ___]] :=
  flattenBoxes[boxes];
extractTextPart[Cell[text_String, ___]] := text;
extractTextPart[_] := "";


(* ========================================================================= *)
(* BOX FLATTENER (fallback when MakeExpression fails)                        *)
(* ========================================================================= *)

flattenBoxes[RowBox[items_List]] :=
  StringJoin[flattenBoxes /@ items];

flattenBoxes[SuperscriptBox[base_, exp_]] :=
  flattenBoxes[base] <> "^" <> flattenBoxes[exp];

flattenBoxes[SubscriptBox[base_, sub_]] :=
  flattenBoxes[base] <> "_" <> flattenBoxes[sub];

flattenBoxes[FractionBox[num_, den_]] :=
  "(" <> flattenBoxes[num] <> "/" <> flattenBoxes[den] <> ")";

flattenBoxes[SqrtBox[x_]] :=
  "Sqrt[" <> flattenBoxes[x] <> "]";

flattenBoxes[s_String] := s;

flattenBoxes[other_] := ToString[other, InputForm];


(* ========================================================================= *)
(* CODE-ONLY EXTRACTION                                                      *)
(* ========================================================================= *)

MCPExtractCode[file_String] := Module[
  {nbExpr, topLevel, cells, codeOnly, result},

  nbExpr = Quiet[Check[Import[file], $Failed]];
  If[nbExpr === $Failed || !MatchQ[nbExpr, _Notebook],
    Return[ExportString[<|"error" -> "Failed to import notebook"|>, "RawJSON"]]
  ];

  topLevel = First[nbExpr];
  If[!ListQ[topLevel], topLevel = {topLevel}];

  cells = Flatten[MapIndexed[processTopLevel, topLevel], 1];
  codeOnly = Select[cells, MemberQ[{"Input", "Code"}, #["style"]] &];

  result = StringRiffle[
    Map[#["content"] &, codeOnly],
    "\n\n"
  ];

  ExportString[<|
    "code" -> result,
    "cell_count" -> Length[codeOnly],
    "backend" -> "kernel_semantic"
  |>, "RawJSON"]
];


End[];
EndPackage[];
