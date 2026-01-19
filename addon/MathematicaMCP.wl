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
$MCPHost = "127.0.0.1";
$MCPAuthToken = Quiet[Check[Environment["MATHEMATICA_MCP_TOKEN"], ""]];
$MCPMaxMessageBytes = 5*1024*1024; (* 5MB max request *)
$MCPMaxResponseBytes = 5*1024*1024; (* 5MB max response *)
$MCPBuffers = <||>; (* per-socket input buffers *)

debugLog[msg_] := If[$MCPDebug, Print["[MCP Debug] ", msg]];

(* ============================================================================ *)
(* SERVER MANAGEMENT                                                            *)
(* ============================================================================ *)

StartMCPServer[] := Module[{},
  If[$MCPListener =!= None,
    Print["[MathematicaMCP] Server already running on port ", $MCPPort];
    Return[$MCPListener]
  ];
  
  $MCPListener = SocketListen[{ $MCPHost, $MCPPort },
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
    $MCPBuffers = <||>;
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
  {socket, dataBytes, bufferKey, existing, newChunk, combined, messages, responseStrs},
  
  socket = assoc["SourceSocket"];
  dataBytes = assoc["DataBytes"];
  bufferKey = ToString[socket, InputForm];
  
  (* Debug: show raw bytes *)
  If[$MCPDebug,
    Print["[MCP Debug] Raw bytes (", Length[dataBytes], "): ", 
      If[Length[dataBytes] > 0, dataBytes[[;;Min[50, Length[dataBytes]]]], "empty"]];
  ];
  
  If[Length[dataBytes] == 0, Return[]];
  
  newChunk = If[ByteArrayQ[dataBytes],
    ByteArrayToString[dataBytes, "UTF-8"],
    FromCharacterCode[dataBytes, "UTF-8"]
  ];
  
  existing = Lookup[$MCPBuffers, bufferKey, ""];
  combined = existing <> newChunk;
  
  If[StringLength[combined] > $MCPMaxMessageBytes,
    responseStrs = {"{\"status\":\"error\",\"message\":\"Request too large\"}\n"};
    BinaryWrite[socket, StringToByteArray[StringJoin[responseStrs]]];
    $MCPBuffers = KeyDrop[$MCPBuffers, bufferKey];
    Return[];
  ];
  
  messages = Select[StringSplit[combined, "\n"], StringLength[#] > 0 &];
  If[!StringEndsQ[combined, "\n"],
    $MCPBuffers[bufferKey] = Last[messages];
    messages = Most[messages],
    $MCPBuffers[bufferKey] = "";
  ];
  
  responseStrs = Map[
    Function[msg,
      Module[{request, response, responseStr},
        debugLog["Received message: " <> StringTake[msg, UpTo[200]]];
        request = Quiet[ImportString[msg, "RawJSON"]];
        
        If[$MCPDebug,
          Print["[MCP Debug] Parsed result: ", request];
          Print["[MCP Debug] Is Association: ", AssociationQ[request]];
        ];
        
        response = If[!AssociationQ[request],
          <|"status" -> "error", "message" -> "Invalid JSON request"|>,
          processCommand[request]
        ];
        
        responseStr = Quiet[Check[ExportString[response, "RawJSON"], None]];
        If[responseStr === None || !StringQ[responseStr],
          responseStr = "{\"status\":\"error\",\"message\":\"Failed to serialize response\"}";
          If[$MCPDebug, Print["[MCP Debug] ExportString failed, using fallback"]];
        ];
        
        If[StringLength[responseStr] > $MCPMaxResponseBytes,
          responseStr = "{\"status\":\"error\",\"message\":\"Response too large\"}";
        ];
        responseStr <> "\n"
      ]
    ],
    messages
  ];
  
  If[Length[responseStrs] > 0,
    debugLog["Sending: " <> StringTake[StringJoin[responseStrs], UpTo[200]]];
    BinaryWrite[socket, StringToByteArray[StringJoin[responseStrs]]];
  ];
];

(* ============================================================================ *)
(* COMMAND DISPATCHER                                                           *)
(* ============================================================================ *)

processCommand[request_Association] := Module[
  {id, command, params, token, result},
  
  id = Lookup[request, "id", CreateUUID[]];
  command = Lookup[request, "command", "unknown"];
  params = Lookup[request, "params", <||>];
  token = Lookup[request, "token", ""];
  
  debugLog["Processing command: " <> command];
  
  If[StringLength[$MCPAuthToken] > 0 && token =!= $MCPAuthToken,
    Return[<|"id" -> id, "status" -> "error", "message" -> "Unauthorized"|>]
  ];
  
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
      
      (* TIER 1: Variable Introspection *)
      "list_variables", cmdListVariables[params],
      "get_variable", cmdGetVariable[params],
      "set_variable", cmdSetVariable[params],
      "clear_variables", cmdClearVariables[params],
      "get_expression_info", cmdGetExpressionInfo[params],
      
      (* TIER 1: Error Recovery *)
      "get_messages", cmdGetMessages[params],
      
      (* TIER 2: File Handling *)
      "open_notebook_file", cmdOpenNotebookFile[params],
      "run_script", cmdRunScript[params],
      
      (* TIER 4: Debugging *)
      "trace_evaluation", cmdTraceEvaluation[params],
      "time_expression", cmdTimeExpression[params],
      "check_syntax", cmdCheckSyntax[params],
      
      (* TIER 5: Data I/O *)
      "import_data", cmdImportData[params],
      "export_data", cmdExportData[params],
      "list_import_formats", cmdListImportFormats[params],
      
      (* TIER 6: Visualization *)
      "export_graphics", cmdExportGraphics[params],
      
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
  StringQ[spec],
    Module[{held},
      held = Quiet[Check[ToExpression[spec, InputForm, HoldComplete], $Failed]];
      If[MatchQ[held, HoldComplete[_NotebookObject]],
        ReleaseHold[held],
        SelectFirst[Notebooks[], 
          StringContainsQ[ToString[NotebookFileName[#] /. $Failed -> ""], spec, IgnoreCase -> True] &,
          InputNotebook[]
        ]
      ]
    ],
  True,
    InputNotebook[]
];

resolveCellObject[spec_] := Which[
  StringQ[spec],
    Module[{held},
      held = Quiet[Check[ToExpression[spec, InputForm, HoldComplete], $Failed]];
      If[MatchQ[held, HoldComplete[_CellObject]], ReleaseHold[held], spec]
    ],
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

(* ============================================================================ *)
(* TIER 1: VARIABLE INTROSPECTION                                               *)
(* ============================================================================ *)

cmdListVariables[params_] := Module[{names, includeSystem, variables},
  includeSystem = Lookup[params, "include_system", False];
  
  names = If[TrueQ[includeSystem],
    Names["Global`*"],
    Select[Names["Global`*"], !StringStartsQ[#, "$"] &]
  ];
  
  variables = Map[
    Function[{name},
      Module[{val, head, bytes},
        val = Quiet[Check[ToExpression[name], $Failed]];
        head = If[val === $Failed, "Failed", ToString[Head[val]]];
        bytes = If[val === $Failed, 0, Quiet[Check[ByteCount[val], 0]]];
        <|
          "name" -> StringReplace[name, "Global`" -> ""],
          "head" -> head,
          "bytes" -> bytes,
          "preview" -> If[val === $Failed, 
            "Failed to evaluate",
            ToString[Short[val, 3], InputForm]
          ]
        |>
      ]
    ],
    names
  ];
  
  <|
    "success" -> True,
    "count" -> Length[variables],
    "variables" -> variables
  |>
];

cmdGetVariable[params_] := Module[{name, fullName, val, result},
  name = Lookup[params, "name", None];
  
  If[name === None,
    Return[<|"error" -> "No variable name specified"|>]
  ];
  
  fullName = If[StringContainsQ[name, "`"], name, "Global`" <> name];
  
  If[!NameQ[fullName],
    Return[<|"error" -> ("Variable '" <> name <> "' not found")|>]
  ];
  
  val = Quiet[Check[ToExpression[fullName], $Failed]];
  
  If[val === $Failed,
    Return[<|"error" -> ("Failed to evaluate '" <> name <> "'")|>]
  ];
  
  <|
    "success" -> True,
    "name" -> name,
    "value" -> ToString[val, InputForm],
    "head" -> ToString[Head[val]],
    "bytes" -> ByteCount[val],
    "dimensions" -> If[ArrayQ[val], Dimensions[val], Null],
    "tex" -> Quiet[Check[ToString[TeXForm[val]], Null]]
  |>
];

cmdSetVariable[params_] := Module[{name, value, fullName, result},
  name = Lookup[params, "name", None];
  value = Lookup[params, "value", None];
  
  If[name === None,
    Return[<|"error" -> "No variable name specified"|>]
  ];
  If[value === None,
    Return[<|"error" -> "No value specified"|>]
  ];
  
  fullName = If[StringContainsQ[name, "`"], name, "Global`" <> name];
  
  result = Quiet[Check[
    ToExpression[fullName <> " = " <> value],
    $Failed
  ]];
  
  If[result === $Failed,
    Return[<|"error" -> "Failed to set variable"|>]
  ];
  
  <|
    "success" -> True,
    "name" -> name,
    "value" -> ToString[result, InputForm],
    "head" -> ToString[Head[result]]
  |>
];

cmdClearVariables[params_] := Module[{names, pattern, cleared},
  names = Lookup[params, "names", None];
  pattern = Lookup[params, "pattern", None];
  
  cleared = Which[
    ListQ[names] && Length[names] > 0,
      Map[
        Function[{name},
          Module[{fullName},
            fullName = If[StringContainsQ[name, "`"], name, "Global`" <> name];
            Quiet[ClearAll[fullName]];
            name
          ]
        ],
        names
      ],
    StringQ[pattern],
      Module[{matching},
        matching = Names[pattern];
        Quiet[ClearAll /@ matching];
        matching
      ],
    True,
      Module[{allGlobal},
        allGlobal = Names["Global`*"];
        Quiet[ClearAll /@ allGlobal];
        allGlobal
      ]
  ];
  
  <|
    "success" -> True,
    "cleared" -> cleared,
    "count" -> Length[cleared]
  |>
];

cmdGetExpressionInfo[params_] := Module[{expr, exprStr, result, head, depth, leafCount, atomQ},
  exprStr = Lookup[params, "expression", None];
  
  If[exprStr === None,
    Return[<|"error" -> "No expression specified"|>]
  ];
  
  result = Quiet[Check[ToExpression[exprStr], $Failed]];
  
  If(result === $Failed,
    Return[<|"error" -> "Failed to evaluate expression"|>]
  ];
  
  <|
    "success" -> True,
    "expression" -> exprStr,
    "head" -> ToString[Head[result]],
    "full_form" -> ToString[FullForm[result]],
    "depth" -> Depth[result],
    "leaf_count" -> LeafCount[result],
    "byte_count" -> ByteCount[result],
    "atomic" -> AtomQ[result],
    "numeric" -> NumericQ[result],
    "list" -> ListQ[result],
    "dimensions" -> If[ArrayQ[result], Dimensions[result], Null]
  |>
];

(* ============================================================================ *)
(* TIER 1: ERROR RECOVERY                                                       *)
(* ============================================================================ *)

$MCPMessageLog = {};
$MCPMaxMessages = 50;

(* Hook into message system to capture messages *)
captureMessage[msg_] := Module[{},
  AppendTo[$MCPMessageLog, <|
    "timestamp" -> DateString["ISODateTime"],
    "message" -> ToString[msg]
  |>];
  (* Keep only last N messages *)
  If[Length[$MCPMessageLog] > $MCPMaxMessages,
    $MCPMessageLog = Take[$MCPMessageLog, -$MCPMaxMessages]
  ];
];

cmdGetMessages[params_] := Module[{count, messages},
  count = Lookup[params, "count", 10];
  
  messages = Take[$MCPMessageLog, UpTo[count]];
  
  <|
    "success" -> True,
    "count" -> Length(messages),
    "total_captured" -> Length($MCPMessageLog),
    "messages" -> Reverse[messages]
  |>
];

(* ============================================================================ *)
(* TIER 2: FILE HANDLING                                                        *)
(* ============================================================================ *)

cmdOpenNotebookFile[params_] := Module[{path, expandedPath, nb},
  path = Lookup[params, "path", None];
  
  If[path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  
  (* Expand ~ to home directory *)
  expandedPath = If[StringStartsQ[path, "~"],
    StringReplace[path, StartOfString ~~ "~" -> $HomeDirectory],
    path
  ];
  
  (* Convert to absolute path if relative *)
  If[!FileExistsQ[expandedPath],
    Return[<|"error" -> ("File not found: " <> expandedPath)|>]
  ];
  
  nb = Quiet[Check[NotebookOpen[expandedPath], $Failed]];
  
  If[nb === $Failed || !MatchQ[nb, _NotebookObject],
    Return[<|"error" -> "Failed to open notebook"|>]
  ];
  
  <|
    "success" -> True,
    "id" -> ToString[nb, InputForm],
    "path" -> expandedPath,
    "title" -> CurrentValue[nb, WindowTitle],
    "cell_count" -> Length[Cells[nb]]
  |>
];

cmdRunScript[params_] := Module[{path, expandedPath, result, startTime, timing},
  path = Lookup[params, "path", None];
  
  If(path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  
  expandedPath = If[StringStartsQ[path, "~"],
    StringReplace[path, StartOfString ~~ "~" -> $HomeDirectory],
    path
  ];
  
  If(!FileExistsQ[expandedPath],
    Return[<|"error" -> ("File not found: " <> expandedPath)|>]
  ];
  
  startTime = AbsoluteTime[];
  result = Quiet[Check[Get[expandedPath], $Failed]];
  timing = Round[(AbsoluteTime[] - startTime) * 1000];
  
  If(result === $Failed,
    Return[<|
      "success" -> False,
      "error" -> "Script execution failed",
      "path" -> expandedPath,
      "timing_ms" -> timing
    |>]
  ];
  
  <|
    "success" -> True,
    "path" -> expandedPath,
    "result" -> ToString[result, InputForm],
    "timing_ms" -> timing
  |>
];

(* ============================================================================ *)
(* TIER 4: DEBUGGING                                                            *)
(* ============================================================================ *)

cmdTraceEvaluation[params_] := Module[{expr, maxDepth, trace, result},
  expr = Lookup[params, "expression", None];
  maxDepth = Lookup[params, "max_depth", 5];
  
  If(expr === None,
    Return[<|"error" -> "No expression specified"|>]
  ];
  
  trace = {};
  
  result = Quiet[Check[
    Block[{$Output = {}},
      TraceScan[
        (AppendTo[trace, ToString[#, InputForm]]) &,
        ToExpression[expr],
        TraceDepth -> maxDepth
      ]
    ],
    $Failed
  ]];
  
  If(result === $Failed,
    Return[<|"error" -> "Trace failed"|>]
  ];
  
  <|
    "success" -> True,
    "expression" -> expr,
    "result" -> ToString[result, InputForm],
    "steps" -> Length[trace],
    "trace" -> Take[trace, UpTo[100]]  (* Limit output size *)
  |>
];

cmdTimeExpression[params_] := Module[{expr, result, timing, memBefore, memAfter},
  expr = Lookup[params, "expression", None];
  
  If(expr === None,
    Return[<|"error" -> "No expression specified"|>]
  ];
  
  memBefore = MemoryInUse[];
  timing = Quiet[Check[AbsoluteTiming[ToExpression[expr]], $Failed]];
  memAfter = MemoryInUse[];
  
  If[timing === $Failed,
    Return[<|"error" -> "Timing failed"|>]
  ];
  
  <|
    "success" -> True,
    "expression" -> expr,
    "time_seconds" -> timing[[1]],
    "time_ms" -> Round[timing[[1]] * 1000],
    "result" -> ToString[timing[[2]], InputForm],
    "memory_delta_bytes" -> (memAfter - memBefore)
  |>
];

cmdCheckSyntax[params_] := Module[{code, check},
  code = Lookup[params, "code", None];
  
  If(code === None,
    Return[<|"error" -> "No code specified"|>]
  ];
  
  check = Quiet[SyntaxQ[code]];
  
  <|
    "success" -> True,
    "code" -> StringTake[code, UpTo[200]],
    "valid" -> TrueQ[check],
    "message" -> If[!TrueQ[check], "Syntax error in code", "Valid syntax"]
  |>
];

(* ============================================================================ *)
(* TIER 5: DATA I/O                                                             *)
(* ============================================================================ *)

cmdImportData[params_] := Module[{path, format, expandedPath, data, opts},
  path = Lookup[params, "path", None];
  format = Lookup[params, "format", Automatic];
  
  If[path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  
  expandedPath = If[StringStartsQ[path, "~"],
    StringReplace[path, StartOfString ~~ "~" -> $HomeDirectory],
    path
  ];
  
  If[!FileExistsQ[expandedPath] && !StringStartsQ[expandedPath, "http"],
    Return[<|"error" -> ("File not found: " <> expandedPath)|>]
  ];
  
  data = Quiet[Check[
    If[format === Automatic,
      Import[expandedPath],
      Import[expandedPath, format]
    ],
    $Failed
  ]];
  
  If[data === $Failed,
    Return[<|"error" -> "Import failed"|>]
  ];
  
  <|
    "success" -> True,
    "path" -> expandedPath,
    "format" -> If[format === Automatic, "Auto-detected", format],
    "head" -> ToString[Head[data]],
    "dimensions" -> If[ArrayQ[data], Dimensions[data], None],
    "byte_count" -> ByteCount[data],
    "preview" -> ToString[Short[data, 5], InputForm]
  |>
];

cmdExportData[params_] := Module[{path, expression, format, expandedPath, result},
  path = Lookup[params, "path", None];
  expression = Lookup[params, "expression", None];
  format = Lookup[params, "format", Automatic];
  
  If[path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  If[expression === None,
    Return[<|"error" -> "No expression specified"|>]
  ];
  
  expandedPath = If[StringStartsQ[path, "~"],
    StringReplace[path, StartOfString ~~ "~" -> $HomeDirectory],
    path
  ];
  
  result = Quiet[Check[
    Module[{expr},
      expr = ToExpression[expression];
      If[format === Automatic,
        Export[expandedPath, expr],
        Export[expandedPath, expr, format]
      ]
    ],
    $Failed
  ]];
  
  If[result === $Failed,
    Return[<|"error" -> "Export failed"|>]
  ];
  
  <|
    "success" -> True,
    "path" -> expandedPath,
    "format" -> If[format === Automatic, "Auto-detected", format],
    "bytes" -> FileByteCount[expandedPath]
  |>
];

cmdListImportFormats[params_] := Module[{},
  <|
    "success" -> True,
    "import_formats" -> $ImportFormats,
    "export_formats" -> $ExportFormats
  |>
];

(* ============================================================================ *)
(* TIER 6: VISUALIZATION                                                        *)
(* ============================================================================ *)

cmdExportGraphics[params_] := Module[{expression, path, format, size, expandedPath, result, img},
  expression = Lookup[params, "expression", None];
  path = Lookup[params, "path", None];
  format = Lookup[params, "format", "PNG"];
  size = Lookup[params, "size", 600];
  
  If[expression === None,
    Return[<|"error" -> "No expression specified"|>]
  ];
  If[path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  
  expandedPath = If[StringStartsQ[path, "~"],
    StringReplace[path, StartOfString ~~ "~" -> $HomeDirectory],
    path
  ];
  
  result = Quiet[Check[
    Module[{expr},
      expr = ToExpression[expression];
      Export[expandedPath, expr, format, ImageSize -> size]
    ],
    $Failed
  ]];
  
  If[result === $Failed,
    Return[<|"error" -> "Export failed"|>]
  ];
  
  <|
    "success" -> True,
    "path" -> expandedPath,
    "format" -> format,
    "size" -> size,
    "bytes" -> FileByteCount[expandedPath]
  |>
];

End[];
EndPackage[];

Print["[MathematicaMCP] Package loaded. Run StartMCPServer[] to start the server."];
