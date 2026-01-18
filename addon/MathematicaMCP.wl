(* ::Package:: *)
(* MathematicaMCP.wl - Socket server addon for external control of Mathematica *)
(* Enables LLMs to control Mathematica notebooks and kernel via MCP *)

BeginPackage["MathematicaMCP`"];

StartMCPServer::usage = "StartMCPServer[] starts the MCP socket server on the configured port (default 9881).";
StopMCPServer::usage = "StopMCPServer[] stops the MCP socket server.";
MCPServerStatus::usage = "MCPServerStatus[] returns the current server status.";
RestartMCPServer::usage = "RestartMCPServer[] restarts the MCP socket server.";

Begin["`Private`"];

$MCPPort = 9881;
$MCPListener = None;
$MCPDebug = False;

debugLog[msg_] := If[$MCPDebug, Print["[MCP Debug] ", msg]];

(* ============================================================================ *)
(* SERVER MANAGEMENT                                                            *)
(* ============================================================================ *)

StartMCPServer[] := Module[{},
  If[$MCPListener =!= None,
    Print["[MathematicaMCP] Server already running on port ", $MCPPort];
    Return[$MCPListener]
  ];
  
  $MCPListener = SocketListen[$MCPPort,
    handleConnection,
    HandlerFunctionsKeys -> {"SourceSocket", "DataBytes"}
  ];
  
  If[FailureQ[$MCPListener],
    Print["[MathematicaMCP] Failed to start server: ", $MCPListener];
    $MCPListener = None;
    Return[$Failed]
  ];
  
  Print["[MathematicaMCP] Server started on port ", $MCPPort];
  $MCPListener
];

StopMCPServer[] := Module[{},
  If[$MCPListener =!= None,
    Close[$MCPListener["Socket"]];
    DeleteObject[$MCPListener];
    $MCPListener = None;
    Print["[MathematicaMCP] Server stopped"];
  ];
];

RestartMCPServer[] := (StopMCPServer[]; Pause[0.5]; StartMCPServer[]);

MCPServerStatus[] := <|
  "running" -> ($MCPListener =!= None),
  "port" -> $MCPPort,
  "listener" -> $MCPListener
|>;

(* ============================================================================ *)
(* CONNECTION HANDLER                                                           *)
(* ============================================================================ *)

handleConnection[assoc_Association] := Module[
  {socket, dataBytes, requestStr, request, response, responseStr},
  
  socket = assoc["SourceSocket"];
  dataBytes = assoc["DataBytes"];
  
  (* Debug: show raw bytes *)
  If[$MCPDebug,
    Print["[MCP Debug] Raw bytes (", Length[dataBytes], "): ", 
      If[Length[dataBytes] > 0, dataBytes[[;;Min[50, Length[dataBytes]]]], "empty"]];
  ];
  
  If[Length[dataBytes] == 0, Return[]];
  
  (* Convert bytes to string - handle both ByteArray and List of integers *)
  requestStr = If[ByteArrayQ[dataBytes],
    ByteArrayToString[dataBytes],
    FromCharacterCode[dataBytes]  (* Fallback for list of integers *)
  ];
  
  debugLog["Received string: " <> StringTake[requestStr, UpTo[200]]];
  
  (* Try parsing *)
  request = Quiet[ImportString[requestStr, "RawJSON"]];
  
  If[$MCPDebug,
    Print["[MCP Debug] Parsed result: ", request];
    Print["[MCP Debug] Is Association: ", AssociationQ[request]];
  ];
  
  response = If[!AssociationQ[request],
    <|"status" -> "error", "message" -> "Invalid JSON request", 
      "debug_received" -> StringTake[requestStr, UpTo[100]],
      "debug_bytes" -> Length[dataBytes]|>,
    processCommand[request]
  ];
  
  (* Safely convert response to JSON *)
  responseStr = Quiet[Check[ExportString[response, "RawJSON"], None]];
  
  If[responseStr === None || !StringQ[responseStr],
    (* Fallback: create simple error response *)
    responseStr = "{\"status\":\"error\",\"message\":\"Failed to serialize response\"}";
    If[$MCPDebug, Print["[MCP Debug] ExportString failed, using fallback"]];
  ];
  
  debugLog["Sending: " <> StringTake[responseStr, UpTo[200]]];
  
  BinaryWrite[socket, StringToByteArray[responseStr]];
];

(* ============================================================================ *)
(* COMMAND DISPATCHER                                                           *)
(* ============================================================================ *)

processCommand[request_Association] := Module[
  {id, command, params, result},
  
  id = Lookup[request, "id", CreateUUID[]];
  command = Lookup[request, "command", "unknown"];
  params = Lookup[request, "params", <||>];
  
  debugLog["Processing command: " <> command];
  
  result = Quiet @ Check[
    Switch[command,
      "ping", cmdPing[params],
      "get_status", cmdGetStatus[params],
      
      "get_notebooks", cmdGetNotebooks[params],
      "get_notebook_info", cmdGetNotebookInfo[params],
      "create_notebook", cmdCreateNotebook[params],
      "save_notebook", cmdSaveNotebook[params],
      "close_notebook", cmdCloseNotebook[params],
      
      "get_cells", cmdGetCells[params],
      "get_cell_content", cmdGetCellContent[params],
      "write_cell", cmdWriteCell[params],
      "delete_cell", cmdDeleteCell[params],
      "evaluate_cell", cmdEvaluateCell[params],
      
      "execute_code", cmdExecuteCode[params],
      "execute_selection", cmdExecuteSelection[params],
      
      "screenshot_notebook", cmdScreenshotNotebook[params],
      "screenshot_cell", cmdScreenshotCell[params],
      "rasterize_expression", cmdRasterizeExpression[params],
      
      "select_cell", cmdSelectCell[params],
      "scroll_to_cell", cmdScrollToCell[params],
      
      "export_notebook", cmdExportNotebook[params],
      
      _, <|"error" -> ("Unknown command: " <> command)|>
    ],
    <|"error" -> "Command execution failed"|>
  ];
  
  If[KeyExistsQ[result, "error"],
    <|"id" -> id, "status" -> "error", "message" -> result["error"]|>,
    <|"id" -> id, "status" -> "success", "result" -> result|>
  ]
];

(* ============================================================================ *)
(* HELPER FUNCTIONS                                                             *)
(* ============================================================================ *)

resolveNotebook[spec_] := Which[
  spec === None || spec === Null || spec === "",
    InputNotebook[],
  StringQ[spec] && StringContainsQ[spec, "NotebookObject"],
    ToExpression[spec],
  StringQ[spec],
    SelectFirst[Notebooks[], 
      StringContainsQ[ToString[NotebookFileName[#] /. $Failed -> ""], spec, IgnoreCase -> True] &,
      InputNotebook[]
    ],
  True,
    InputNotebook[]
];

resolveCellObject[spec_] := Which[
  StringQ[spec] && StringContainsQ[spec, "CellObject"],
    ToExpression[spec],
  True,
    spec
];

notebookToAssoc[nb_NotebookObject] := Module[{filename, title, modified, visible},
  filename = Quiet[NotebookFileName[nb] /. $Failed -> Null];
  title = Quiet[CurrentValue[nb, WindowTitle] /. $Failed -> "Unknown"];
  modified = Quiet[CurrentValue[nb, "Modified"] /. $Failed -> False];
  visible = Quiet[CurrentValue[nb, Visible] /. $Failed -> True];
  <|
    "id" -> ToString[nb, InputForm],
    "filename" -> If[StringQ[filename], filename, Null],
    "title" -> If[StringQ[title], title, ToString[title]],
    "modified" -> TrueQ[modified],
    "visible" -> TrueQ[visible]
  |>
];

cellToAssoc[cell_CellObject] := Module[{content, style},
  style = CurrentValue[cell, CellStyle];
  content = Quiet[NotebookRead[cell]];
  <|
    "id" -> ToString[cell, InputForm],
    "style" -> If[ListQ[style], First[style], style],
    "content_preview" -> StringTake[
      ToString[content /. Cell[c_, ___] :> c, InputForm], 
      UpTo[200]
    ]
  |>
];

(* ============================================================================ *)
(* STATUS COMMANDS                                                              *)
(* ============================================================================ *)

cmdPing[_] := <|
  "pong" -> True, 
  "timestamp" -> DateString["ISODateTime"],
  "version" -> "0.1.0"
|>;

cmdGetStatus[_] := <|
  "frontend_version" -> ToString[$FrontEndVersion],
  "kernel_version" -> $VersionNumber,
  "system_id" -> $SystemID,
  "notebooks_open" -> Length[Notebooks[]],
  "mcp_server_running" -> ($MCPListener =!= None),
  "mcp_port" -> $MCPPort
|>;

(* ============================================================================ *)
(* NOTEBOOK COMMANDS                                                            *)
(* ============================================================================ *)

cmdGetNotebooks[_] := Module[{nbs},
  nbs = Notebooks[];
  If[!ListQ[nbs], Return[<|"error" -> "Failed to get notebooks"|>]];
  (* Filter to only NotebookObjects *)
  nbs = Select[nbs, MatchQ[#, _NotebookObject] &];
  Map[notebookToAssoc, nbs]
];

cmdGetNotebookInfo[params_] := Module[{nb},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  If[nb === None || !MatchQ[nb, _NotebookObject],
    Return[<|"error" -> "No notebook found"|>]
  ];
  <|
    "id" -> ToString[nb, InputForm],
    "filename" -> (NotebookFileName[nb] /. $Failed -> None),
    "directory" -> (NotebookDirectory[nb] /. $Failed -> None),
    "title" -> CurrentValue[nb, WindowTitle],
    "cell_count" -> Length[Cells[nb]],
    "cell_styles" -> DeleteDuplicates[
      Flatten[{CurrentValue[#, CellStyle]} & /@ Cells[nb]]
    ],
    "modified" -> CurrentValue[nb, "Modified"],
    "visible" -> CurrentValue[nb, Visible]
  |>
];

cmdCreateNotebook[params_] := Module[{nb, title},
  title = Lookup[params, "title", "Untitled"];
  nb = CreateDocument[{}, WindowTitle -> title];
  <|
    "id" -> ToString[nb, InputForm],
    "title" -> title,
    "created" -> True
  |>
];

cmdSaveNotebook[params_] := Module[{nb, path, format},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  path = Lookup[params, "path", None];
  format = Lookup[params, "format", "Notebook"];
  
  If[path =!= None,
    Export[path, nb, format];
    <|"saved" -> True, "path" -> path, "format" -> format|>,
    NotebookSave[nb];
    <|"saved" -> True, "path" -> NotebookFileName[nb]|>
  ]
];

cmdCloseNotebook[params_] := Module[{nb},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  NotebookClose[nb];
  <|"closed" -> True|>
];

(* ============================================================================ *)
(* CELL COMMANDS                                                                *)
(* ============================================================================ *)

cmdGetCells[params_] := Module[{nb, style, cells},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  style = Lookup[params, "style", None];
  
  cells = If[style === None,
    Cells[nb],
    Cells[nb, CellStyle -> style]
  ];
  
  cellToAssoc /@ cells
];

cmdGetCellContent[params_] := Module[{cell, content},
  cell = resolveCellObject[Lookup[params, "cell_id", None]];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  
  content = NotebookRead[cell];
  <|
    "id" -> ToString[cell, InputForm],
    "style" -> CurrentValue[cell, CellStyle],
    "content" -> ToString[content /. Cell[c_, ___] :> c, InputForm],
    "evaluatable" -> CurrentValue[cell, Evaluatable]
  |>
];

cmdWriteCell[params_] := Module[{nb, content, style, position, cell},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  content = Lookup[params, "content", ""];
  style = Lookup[params, "style", "Input"];
  position = Lookup[params, "position", "After"];
  
  cell = Cell[content, style];
  
  Switch[position,
    "End",
      SelectionMove[nb, After, Notebook];
      NotebookWrite[nb, cell],
    "Beginning",
      SelectionMove[nb, Before, Notebook];
      NotebookWrite[nb, cell],
    "Before",
      NotebookWrite[nb, cell, Before],
    _,
      NotebookWrite[nb, cell, After]
  ];
  
  <|"written" -> True, "style" -> style, "position" -> position|>
];

cmdDeleteCell[params_] := Module[{cell},
  cell = resolveCellObject[Lookup[params, "cell_id", None]];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  NotebookDelete[cell];
  <|"deleted" -> True|>
];

cmdEvaluateCell[params_] := Module[{cell},
  cell = resolveCellObject[Lookup[params, "cell_id", None]];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  SelectionMove[cell, All, CellContents];
  SelectionEvaluate[ParentNotebook[cell]];
  <|"evaluated" -> True, "cell" -> ToString[cell, InputForm]|>
];

(* ============================================================================ *)
(* CODE EXECUTION                                                               *)
(* ============================================================================ *)

cmdExecuteCode[params_] := Module[{code, format, result, output, texOutput},
  code = Lookup[params, "code", ""];
  format = Lookup[params, "format", "text"];
  
  If[code === "",
    Return[<|"error" -> "No code provided"|>]
  ];
  
  result = Quiet @ Check[ToExpression[code], $Failed];
  
  If[result === $Failed,
    Return[<|
      "success" -> False,
      "output" -> "Evaluation failed",
      "error" -> "Syntax or evaluation error"
    |>]
  ];
  
  output = Switch[format,
    "latex", ToString[TeXForm[result]],
    "mathematica", ToString[result, InputForm],
    _, ToString[result]
  ];
  
  texOutput = Quiet[ToString[TeXForm[result]]];
  
  <|
    "success" -> True,
    "output" -> output,
    "output_tex" -> texOutput,
    "output_inputform" -> ToString[result, InputForm]
  |>
];

cmdExecuteSelection[params_] := Module[{nb},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  SelectionEvaluate[nb];
  <|"evaluated" -> True|>
];

(* ============================================================================ *)
(* SCREENSHOT COMMANDS                                                          *)
(* ============================================================================ *)

cmdScreenshotNotebook[params_] := Module[{nb, img, path, maxHeight, format},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  maxHeight = Lookup[params, "max_height", 2000];
  format = Lookup[params, "format", "PNG"];
  
  path = FileNameJoin[{$TemporaryDirectory, "mcp_nb_" <> CreateUUID[] <> ".png"}];
  
  img = Rasterize[nb, ImageResolution -> 144];
  
  If[ImageDimensions[img][[2]] > maxHeight,
    img = ImageResize[img, {Automatic, maxHeight}]
  ];
  
  Export[path, img, format];
  
  <|
    "path" -> path,
    "width" -> ImageDimensions[img][[1]],
    "height" -> ImageDimensions[img][[2]],
    "format" -> format
  |>
];

cmdScreenshotCell[params_] := Module[{cell, content, img, path},
  cell = resolveCellObject[Lookup[params, "cell_id", None]];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  
  content = NotebookRead[cell];
  path = FileNameJoin[{$TemporaryDirectory, "mcp_cell_" <> CreateUUID[] <> ".png"}];
  
  img = Rasterize[content, ImageResolution -> 144];
  Export[path, img, "PNG"];
  
  <|
    "path" -> path,
    "width" -> ImageDimensions[img][[1]],
    "height" -> ImageDimensions[img][[2]]
  |>
];

cmdRasterizeExpression[params_] := Module[{expr, result, img, path, imageSize},
  expr = Lookup[params, "expression", ""];
  imageSize = Lookup[params, "image_size", 400];
  
  If[expr === "",
    Return[<|"error" -> "No expression provided"|>]
  ];
  
  result = Quiet @ Check[ToExpression[expr], $Failed];
  If[result === $Failed,
    Return[<|"error" -> "Failed to evaluate expression"|>]
  ];
  
  path = FileNameJoin[{$TemporaryDirectory, "mcp_expr_" <> CreateUUID[] <> ".png"}];
  img = Rasterize[result, ImageResolution -> 144, ImageSize -> imageSize];
  Export[path, img, "PNG"];
  
  <|
    "path" -> path,
    "width" -> ImageDimensions[img][[1]],
    "height" -> ImageDimensions[img][[2]]
  |>
];

(* ============================================================================ *)
(* NAVIGATION COMMANDS                                                          *)
(* ============================================================================ *)

cmdSelectCell[params_] := Module[{cell},
  cell = resolveCellObject[Lookup[params, "cell_id", None]];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  SelectionMove[cell, All, Cell];
  <|"selected" -> True|>
];

cmdScrollToCell[params_] := Module[{cell, nb},
  cell = resolveCellObject[Lookup[params, "cell_id", None]];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  nb = ParentNotebook[cell];
  SelectionMove[cell, Before, Cell];
  FrontEndTokenExecute[nb, "ScrollNotebookStart"];
  SelectionMove[cell, All, Cell];
  <|"scrolled" -> True|>
];

(* ============================================================================ *)
(* EXPORT COMMANDS                                                              *)
(* ============================================================================ *)

cmdExportNotebook[params_] := Module[{nb, path, format},
  nb = resolveNotebook[Lookup[params, "notebook", None]];
  path = Lookup[params, "path", None];
  format = Lookup[params, "format", "PDF"];
  
  If[path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  
  Export[path, nb, format];
  
  <|
    "exported" -> True,
    "path" -> path,
    "format" -> format
  |>
];

End[];
EndPackage[];

Print["[MathematicaMCP] Package loaded. Run StartMCPServer[] to start the server."];
