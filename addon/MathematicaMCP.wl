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
$MCPMaxResponseBytes = 20*1024*1024; (* 20MB max response *)
$MCPBuffers = <||>; (* per-socket input buffers *)
$MCPActiveNotebook = None;
$MCPSessionNotebooks = <||>;
$MCPSessionContexts = <||>;

debugLog[msg_] := Module[{},
  If[$MCPDebug,
    PutAppend[
      ToString[DateString["ISODateTime"]] <> ": " <> ToString[msg],
      "/tmp/mcp_debug.log"
    ]
  ]
];

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
        
        responseStr = Quiet[Check[ExportString[response, "RawJSON", "Compact" -> True], None]];
        If[responseStr === None || !StringQ[responseStr],
          responseStr = Quiet[Check[ExportString[jsonSanitize[response], "RawJSON", "Compact" -> True], None]];
        ];
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
  {id, command, params, token, result, response},
  
  id = Lookup[request, "id", CreateUUID[]];
  command = Lookup[request, "command", "unknown"];
  params = Lookup[request, "params", <||>];
  token = Lookup[request, "token", ""];
  
  debugLog["Processing command: " <> command];
  
  If[StringLength[$MCPAuthToken] > 0 && token =!= $MCPAuthToken,
    Return[<|"id" -> id, "status" -> "error", "message" -> "Unauthorized"|>]
  ];
  
  result = dispatchCommand[command, params];

  response = If[KeyExistsQ[result, "error"],
    <|"id" -> id, "status" -> "error", "message" -> result["error"]|>,
    <|"id" -> id, "status" -> "success", "result" -> result|>
  ];

  maybeCompressResponse[response, params]
];

(* ============================================================================ *)
(* HELPER FUNCTIONS                                                             *)
(* ============================================================================ *)

validNotebookQ[nb_] := Module[{vis},
  If[!MatchQ[nb, _NotebookObject], Return[False]];
  vis = Quiet[Check[CurrentValue[nb, Visible], $Failed]];
  vis =!= $Failed
];

(* Check if notebook is editable (not Messages window, Welcome Screen, Palette, etc.) *)
editableNotebookQ[nb_] := Module[{title, editable, windowFrame, styleSheet, fn, saveable},
  If[!validNotebookQ[nb], Return[False]];
  
  (* CRITICAL: Check if this is the system Messages notebook *)
  If[nb === $MessagesNotebook, Return[False]];
  If[nb === MessagesNotebook[], Return[False]];
  
  title = Quiet[CurrentValue[nb, WindowTitle] /. $Failed -> ""];
  editable = Quiet[CurrentValue[nb, Editable] /. $Failed -> True];
  saveable = Quiet[CurrentValue[nb, Saveable] /. $Failed -> True];
  windowFrame = Quiet[CurrentValue[nb, WindowFrame] /. $Failed -> "Normal"];
  styleSheet = Quiet[ToString[CurrentValue[nb, StyleDefinitions]] /. $Failed -> ""];
  
  (* Messages window and special windows are not saveable *)
  If[saveable === False, Return[False]];
  
  (* Reject special windows by title *)
  If[StringContainsQ[title, "Messages", IgnoreCase -> True], Return[False]];
  If[StringContainsQ[title, "Welcome", IgnoreCase -> True], Return[False]];
  If[StringMatchQ[windowFrame, "Palette" | "ThinFrame" | "Frameless"], Return[False]];
  If[editable === False, Return[False]];
  If[StringContainsQ[styleSheet, "Palette", IgnoreCase -> True], Return[False]];
  
  (* Additional check: Reject notebooks with non-.nb file extensions (special windows) *)
  fn = Quiet[NotebookFileName[nb] /. $Failed -> None];
  If[StringQ[fn] && !StringEndsQ[fn, ".nb"], Return[False]];
  
  (* Check StyleDefinitions more thoroughly - Messages window uses MessagesNotebook.nb stylesheet *)
  If[StringContainsQ[styleSheet, "MessagesNotebook", IgnoreCase -> True], Return[False]];
  If[StringContainsQ[styleSheet, "WelcomeScreen", IgnoreCase -> True], Return[False]];
  
  True
];

newCellId[] := Module[{uuid, hash},
  uuid = CreateUUID[];
  hash = Hash[uuid, "CRC32"];
  Mod[hash, 2^31 - 1] + 1
];

getSessionContext[sessionId_] := Module[{hash, ctx},
  If[!StringQ[sessionId] || StringLength[sessionId] == 0, Return[None]];
  If[KeyExistsQ[$MCPSessionContexts, sessionId],
    Return[$MCPSessionContexts[sessionId]]
  ];
  hash = IntegerString[Hash[sessionId, "CRC32"], 16];
  ctx = "MCP`" <> hash <> "`";
  $MCPSessionContexts[sessionId] = ctx;
  ctx
];

resolveSyncMode[params_] := Module[{sync, refresh},
  sync = Lookup[params, "sync", None];
  refresh = TrueQ[Lookup[params, "refresh", False]];
  Which[
    StringQ[sync] && MemberQ[{"none", "refresh", "strict"}, sync], sync,
    refresh, "refresh",
    True, "none"
  ]
];

jsonSanitize[value_] := Module[{v = value, sanitizedStr},
  Which[
    AssociationQ[v], AssociationMap[jsonSanitize, v],
    ListQ[v], jsonSanitize /@ v,
    v === None || v === Null, Null,
    StringQ[v], 
      (* Escape special characters that break JSON *)
      StringReplace[v, {"\n" -> "\\n", "\r" -> "", "\t" -> "\\t", "\\" -> "\\\\"}],
    TrueQ[v] || v === False, v,
    NumberQ[v] && !MatchQ[v, _Complex], v,
    (* Handle special Mathematica objects that can't serialize directly *)
    MatchQ[v, _CellObject | _NotebookObject | _FrontEndObject], 
      ToString[v, InputForm],
    MatchQ[v, _BoxData | _Cell | _StyleBox | _RowBox | _Dynamic],
      sanitizedStr = Quiet[Check[ToString[v, InputForm], "<<BoxData>>"]];
      StringTake[sanitizedStr, UpTo[500]],
    MatchQ[v, _RGBColor | _GrayLevel | _Hue | _CMYKColor],
      ToString[v, InputForm],
    True, 
      sanitizedStr = Quiet[Check[ToString[v, InputForm], ToString[Head[v]]]];
      (* Limit very long strings *)
      If[StringLength[sanitizedStr] > 1000,
        StringTake[sanitizedStr, 1000] <> "...",
        sanitizedStr
      ]
  ]
];

resolveNotebook[spec_, sessionId_: None] := Module[{nb, held, candidates, editableNbs},
  If[StringQ[sessionId] && KeyExistsQ[$MCPSessionNotebooks, sessionId],
    nb = $MCPSessionNotebooks[sessionId];
    If[validNotebookQ[nb], Return[nb]];
    $MCPSessionNotebooks = KeyDrop[$MCPSessionNotebooks, sessionId];
  ];

  If[spec === None || spec === Null || spec === "",
    nb = If[editableNotebookQ[$MCPActiveNotebook], $MCPActiveNotebook, InputNotebook[]],
    nb = Which[
      StringQ[spec],
        (held = Quiet[Check[ToExpression[spec, InputForm, HoldComplete], $Failed]];
         If[MatchQ[held, HoldComplete[_NotebookObject]],
           ReleaseHold[held],
           SelectFirst[Notebooks[],
             StringContainsQ[
               ToString[NotebookFileName[#] /. $Failed -> ""],
               spec,
               IgnoreCase -> True
             ] &,
             InputNotebook[]
           ]
         ]),
      True,
        InputNotebook[]
    ]
  ];

  (* If we got a non-editable notebook (Messages, Welcome Screen, etc.), find or create an editable one *)
  If[!editableNotebookQ[nb],
    editableNbs = Select[Notebooks[], editableNotebookQ];
    nb = If[Length[editableNbs] > 0,
      First[editableNbs],
      CreateDocument[{}, WindowTitle -> "Analysis"]
    ]
  ];

  If[validNotebookQ[nb], $MCPActiveNotebook = nb];
  nb
];

resolveCellObject[spec_, nb_: None] := Module[{held, cellId, searchNbs, cells, targetNb},
  If[StringQ[spec],
    held = Quiet[Check[ToExpression[spec, InputForm, HoldComplete], $Failed]];
    If[MatchQ[held, HoldComplete[_CellObject]], Return[ReleaseHold[held]]]
  ];

  cellId = Which[
    IntegerQ[spec], spec,
    NumberQ[spec] && spec =!= Indeterminate && spec =!= Infinity && spec =!= -Infinity, Round[spec],
    StringQ[spec] && StringMatchQ[spec, NumberString], ToExpression[spec],
    True, None
  ];

  If[cellId === None, Return[spec]];

  targetNb = If[nb === None || nb === Null, $MCPActiveNotebook, nb];
  If[validNotebookQ[targetNb],
    cells = Quiet[Cells[targetNb, CellID -> cellId]];
    If[ListQ[cells] && Length[cells] > 0, Return[First[cells]]]
  ];

  searchNbs = Select[Notebooks[], validNotebookQ];
  cells = Quiet @ Flatten[Cells[#, CellID -> cellId] & /@ searchNbs];
  If[ListQ[cells] && Length[cells] > 0, First[cells], spec]
];

notebookToAssoc[nb_NotebookObject] := Module[{filename, title, modified, visible},
  filename = Quiet[NotebookFileName[nb] /. $Failed -> Null];
  title = Quiet[CurrentValue[nb, WindowTitle] /. $Failed -> "Unknown"];
  modified = Quiet[CurrentValue[nb, "Modified"] /. $Failed -> False];
  visible = Quiet[CurrentValue[nb, Visible] /. $Failed -> True];
  <|
    "id" -> ToString[nb, InputForm],
    "filename" -> If[StringQ[filename], filename, Null],
    "title" -> If[StringQ[title], title, ToString[title, InputForm]],
    "modified" -> TrueQ[modified],
    "visible" -> TrueQ[visible]
  |>
];

cellToAssoc[cell_CellObject, includeContent_: True] := Module[{content, style, cellId, styleValue, cellIdValue, contentPreview},
  (* Safely get style with error handling *)
  style = Quiet[Check[CurrentValue[cell, CellStyle], "Unknown"]];
  cellId = Quiet[Check[CurrentValue[cell, CellID], Null] /. $Failed -> Null];
  
  (* Safely get content *)
  content = If[includeContent, 
    Quiet[Check[NotebookRead[cell], Null]], 
    Null
  ];
  
  (* Sanitize style value *)
  styleValue = If[ListQ[style], First[style, "Unknown"], style];
  styleValue = If[StringQ[styleValue], styleValue, ToString[styleValue, InputForm]];
  
  (* Sanitize cell ID *)
  cellIdValue = Which[
    cellId === Null || cellId === $Failed, Null,
    IntegerQ[cellId], cellId,
    StringQ[cellId], cellId,
    True, ToString[cellId, InputForm]
  ];
  
  (* Safely create content preview *)
  contentPreview = If[includeContent && content =!= Null,
    Quiet[Check[
      Module[{extracted, str},
        extracted = content /. Cell[c_, ___] :> c;
        str = ToString[extracted, InputForm];
        (* Limit length and sanitize *)
        str = StringTake[str, UpTo[200]];
        StringReplace[str, {"\n" -> " ", "\r" -> "", "\t" -> " "}]
      ],
      "<<content unavailable>>"
    ]],
    Null
  ];
  
  <|
    "id" -> ToString[cell, InputForm],
    "cell_id" -> cellIdValue,
    "style" -> styleValue,
    "content_preview" -> contentPreview
  |>
];

dispatchCommand[command_, params_] := Quiet @ Check[
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
    "execute_code_notebook", cmdExecuteCodeNotebook[params],
    "execute_selection", cmdExecuteSelection[params],
    "batch_commands", cmdBatchCommands[params],

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

maybeCompressResponse[response_, params_] := Module[
  {compress, minBytes, result, resultJson, compressed, compressedJson},
  compress = TrueQ[Lookup[params, "compress", False]];
  minBytes = Lookup[params, "compression_min_bytes", 16384];
  If[!compress || !KeyExistsQ[response, "result"], Return[response]];

  result = response["result"];
  resultJson = Quiet[Check[ExportString[result, "RawJSON", "Compact" -> True], ""]];
  If[StringLength[resultJson] < minBytes, Return[response]];

  compressed = Quiet[Check[Compress[result], ""]];
  compressedJson = If[StringQ[compressed], StringLength[compressed], Infinity];
  If[compressedJson >= StringLength[resultJson], Return[response]];

  Association[
    KeyDrop[response, "result"],
    <|
      "compressed" -> True,
      "compression" -> "WLCompress",
      "result_compressed" -> compressed,
      "result_size" -> StringLength[resultJson],
      "compressed_size" -> compressedJson
    |>
  ]
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

cmdGetNotebookInfo[params_] := Module[{nb, cells, sessionId, cellStyles, titleVal},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  If[nb === None || !MatchQ[nb, _NotebookObject],
    Return[<|"error" -> "No notebook found"|>]
  ];
  
  (* Wrap in Quiet to handle edge cases *)
  cells = Quiet[Check[Cells[nb], {}]];
  If[!ListQ[cells], cells = {}];
  
  (* Sanitize cell styles - ensure they're all strings *)
  cellStyles = Quiet[Check[
    DeleteDuplicates[
      Map[
        Function[{cell},
          Module[{s},
            s = Quiet[Check[CurrentValue[cell, CellStyle], "Unknown"]];
            If[ListQ[s], s = First[s, "Unknown"]];
            If[StringQ[s], s, ToString[s, InputForm]]
          ]
        ],
        cells
      ]
    ],
    {"Unknown"}
  ]];
  
  (* Sanitize title *)
  titleVal = Quiet[Check[CurrentValue[nb, WindowTitle], "Unknown"]];
  If[!StringQ[titleVal], titleVal = ToString[titleVal, InputForm]];
  
  <|
    "id" -> ToString[nb, InputForm],
    "filename" -> Quiet[Check[NotebookFileName[nb], Null] /. $Failed -> Null],
    "directory" -> Quiet[Check[NotebookDirectory[nb], Null] /. $Failed -> Null],
    "title" -> titleVal,
    "cell_count" -> Length[cells],
    "cell_styles" -> cellStyles,
    "modified" -> TrueQ[Quiet[CurrentValue[nb, "Modified"]]],
    "visible" -> TrueQ[Quiet[CurrentValue[nb, Visible]]]
  |>
];

cmdCreateNotebook[params_] := Module[{nb, title, sessionId},
  title = Lookup[params, "title", "Untitled"];
  sessionId = Lookup[params, "session_id", None];
  nb = CreateDocument[{}, WindowTitle -> title];
  If[StringQ[sessionId] && StringLength[sessionId] > 0,
    $MCPSessionNotebooks[sessionId] = nb;
  ];
  $MCPActiveNotebook = nb;
  <|
    "id" -> ToString[nb, InputForm],
    "title" -> title,
    "created" -> True
  |>
];

cmdSaveNotebook[params_] := Module[{nb, path, format, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  path = Lookup[params, "path", None];
  format = Lookup[params, "format", "Notebook"];
  
  If[path =!= None,
    Export[path, nb, format];
    <|"saved" -> True, "path" -> path, "format" -> format|>,
    NotebookSave[nb];
    <|"saved" -> True, "path" -> NotebookFileName[nb]|>
  ]
];

cmdCloseNotebook[params_] := Module[{nb, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  NotebookClose[nb];
  If[StringQ[sessionId] && KeyExistsQ[$MCPSessionNotebooks, sessionId],
    $MCPSessionNotebooks = KeyDrop[$MCPSessionNotebooks, sessionId]
  ];
  <|"closed" -> True|>
];

(* ============================================================================ *)
(* CELL COMMANDS                                                                *)
(* ============================================================================ *)

cmdGetCells[params_] := Module[{nb, style, cells, sessionId, offset, limit, includeContent, total, start, end, slice, returnMeta, cellAssocs},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  If[nb === None || !validNotebookQ[nb],
    Return[<|"error" -> "No valid notebook found"|>]
  ];
  
  style = Lookup[params, "style", None];
  offset = Max[0, Lookup[params, "offset", 0]];
  limit = Lookup[params, "limit", None];
  includeContent = Lookup[params, "include_content", True];
  returnMeta = KeyExistsQ[params, "offset"] || KeyExistsQ[params, "limit"] || KeyExistsQ[params, "include_content"];

  cells = Quiet[Check[
    If[style === None,
      Cells[nb],
      Cells[nb, CellStyle -> style]
    ],
    {}
  ]];
  If[!ListQ[cells], cells = {}];

  total = Length[cells];
  start = offset + 1;
  end = If[IntegerQ[limit], Min[offset + limit, total], total];
  slice = If[total == 0 || start > total, {}, Take[cells, {start, end}]];

  (* Safely convert cells to associations with error handling *)
  cellAssocs = Map[
    Function[{cell},
      Quiet[Check[cellToAssoc[cell, includeContent], <|"id" -> "error", "style" -> "Unknown", "error" -> "Failed to read cell"|>]]
    ],
    If[returnMeta, slice, cells]
  ];

  If[!returnMeta,
    cellAssocs,
    <|
      "count" -> Length[slice],
      "total" -> total,
      "offset" -> offset,
      "limit" -> If[limit === None, Null, limit],
      "cells" -> cellAssocs
    |>
  ]
];

cmdGetCellContent[params_] := Module[{cell, content, nb, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  cell = resolveCellObject[Lookup[params, "cell_id", None], nb];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  
  content = NotebookRead[cell];
  jsonSanitize @ <|
    "id" -> ToString[cell, InputForm],
    "style" -> CurrentValue[cell, CellStyle],
    "content" -> ToString[content /. Cell[c_, ___] :> c, InputForm],
    "evaluatable" -> CurrentValue[cell, Evaluatable]
  |>
];

cmdWriteCell[params_] := Module[{nb, content, style, position, cell, syncMode, syncWait, cellId, sessionId, cellObj},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  content = Lookup[params, "content", ""];
  style = Lookup[params, "style", "Input"];
  position = Lookup[params, "position", "After"];
  syncMode = resolveSyncMode[params];
  syncWait = Lookup[params, "sync_wait", 2];
  cellId = newCellId[];
  
  cell = Cell[content, style, CellID -> cellId];
  
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
  
  If[syncMode =!= "none",
    FrontEndTokenExecute[nb, "Refresh"];
    Pause[0.05];
  ];
  If[syncMode === "strict",
    Module[{elapsed = 0, interval = 0.05, found = False},
      While[elapsed < syncWait && !found,
        Pause[interval];
        elapsed += interval;
        found = Length[Cells[nb, CellID -> cellId]] > 0;
      ];
    ]
  ];
  
  cellObj = Quiet@Check[First[Cells[nb, CellID -> cellId]], None];

  <|
    "written" -> True,
    "style" -> style,
    "position" -> position,
    "cell_id" -> If[cellObj === None, ToString[cellId], ToString[cellObj, InputForm]],
    "cell_id_numeric" -> cellId
  |>
];

cmdDeleteCell[params_] := Module[{cell, nb, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  cell = resolveCellObject[Lookup[params, "cell_id", None], nb];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  NotebookDelete[cell];
  <|"deleted" -> True|>
];

cmdEvaluateCell[params_] := Module[{cell, nb, maxWait, waitInterval, elapsed, syncMode, sessionId, evaluating, syncWait},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  cell = resolveCellObject[Lookup[params, "cell_id", None], nb];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  nb = ParentNotebook[cell];
  maxWait = Lookup[params, "max_wait", 10];
  waitInterval = 0.1;
  syncMode = resolveSyncMode[params];
  syncWait = Lookup[params, "sync_wait", 2];

  SelectionMove[cell, All, Cell];
  FrontEndTokenExecute["EvaluateCells"];

  elapsed = 0;
  While[elapsed < maxWait,
    Pause[waitInterval];
    elapsed += waitInterval;
    evaluating = Quiet[Check[CurrentValue[nb, Evaluating], False]];
    If[!TrueQ[evaluating], Break[]];
  ];

  If[syncMode =!= "none",
    FrontEndTokenExecute[nb, "Refresh"];
    Pause[0.05];
  ];
  If[syncMode === "strict",
    Module[{elapsedSync = 0, interval = 0.05},
      While[elapsedSync < syncWait,
        Pause[interval];
        elapsedSync += interval;
        evaluating = Quiet[Check[CurrentValue[nb, Evaluating], False]];
        If[!TrueQ[evaluating], Break[]];
      ];
    ]
  ];

  <|"evaluated" -> True, "cell" -> ToString[cell, InputForm], "waited_seconds" -> elapsed|>
];

(* ============================================================================ *)
(* CODE EXECUTION                                                               *)
(* ============================================================================ *)

cmdExecuteCode[params_] := Module[
  {code, format, result, outputInput, outputFull = "", outputTex = "", messages,
   timing, failed, deterministicSeed, timeout, sessionId, isolateContext, ctx, exprToEval,
   formattedMessages, hasErrors, hasWarnings, errorType},

  code = Lookup[params, "code", ""];
  format = Lookup[params, "format", "text"];
  deterministicSeed = Lookup[params, "deterministic_seed", None];
  timeout = Lookup[params, "timeout", None];
  sessionId = Lookup[params, "session_id", None];
  isolateContext = TrueQ[Lookup[params, "isolate_context", False]];

  If[code === "",
    Return[<|"error" -> "No code provided"|>]
  ];

  ctx = If[isolateContext, getSessionContext[sessionId], None];

  exprToEval = ToExpression[code];
  If[ctx =!= None,
    exprToEval = Block[{$Context = ctx, $ContextPath = {ctx, "System`"}}, exprToEval]
  ];
  If[deterministicSeed =!= None && deterministicSeed =!= Null,
    exprToEval = BlockRandom[SeedRandom[deterministicSeed]; exprToEval]
  ];
  If[NumberQ[timeout],
    exprToEval = TimeConstrained[exprToEval, timeout, $Aborted]
  ];

  {timing, {result, messages}} = AbsoluteTiming[
    Block[{$MessageList = {}},
      {Quiet @ Check[exprToEval, $Failed], $MessageList}
    ]
  ];

  failed = (result === $Failed || result === $Aborted);
  errorType = If[result === $Aborted, "timeout", "evaluation_error"];

  outputInput = ToString[result, InputForm];
  If[format === "mathematica", outputFull = ToString[result, FullForm]];
  If[format === "latex", outputTex = ToString[TeXForm[result]]];

  formattedMessages = Map[
    Function[msg,
      Quiet @ Check[
        <|
          "tag" -> ToString[First[msg], OutputForm],
          "text" -> ToString[Last[msg], OutputForm],
          "type" -> If[StringContainsQ[ToString[First[msg]], "::"],
            If[StringEndsQ[ToString[First[msg]], "::warning"], "warning", "error"],
            "message"
          ]
        |>,
        <|"tag" -> "Unknown", "text" -> "Failed to format message", "type" -> "error"|>
      ]
    ],
    Take[messages, UpTo[20]]
  ];
  formattedMessages = Select[formattedMessages, AssociationQ[#] && StringLength[#["text"]] > 0 &];
  hasErrors = Length[Select[formattedMessages, #"type" == "error" &]] > 0;
  hasWarnings = Length[Select[formattedMessages, #"type" == "warning" &]] > 0;

  Module[{response},
    response = <|
      "success" -> Not[failed],
      "output" -> outputInput,
      "output_inputform" -> outputInput,
      "output_fullform" -> outputFull,
      "output_tex" -> outputTex,
      "messages" -> formattedMessages,
      "warnings" -> If[hasWarnings,
        (#["text"] & /@ Select[formattedMessages, #["type"] == "warning" &]),
        {}
      ],
      "has_errors" -> hasErrors,
      "has_warnings" -> hasWarnings,
      "timing_ms" -> Round[timing * 1000],
      "execution_method" -> "addon"
    |>;
    If[failed, response["error"] = errorType];
    response
  ]
];

cmdExecuteSelection[params_] := Module[{nb, maxWait, waitInterval, elapsed, syncMode, sessionId, evaluating, cellsBefore, cellsAfter, syncWait},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  maxWait = Lookup[params, "max_wait", 10];
  waitInterval = 0.1;
  syncMode = resolveSyncMode[params];
  syncWait = Lookup[params, "sync_wait", 2];

  cellsBefore = Quiet[Check[Length[Cells[nb]], 0]];
  FrontEndTokenExecute["EvaluateCells"];

  elapsed = 0;
  While[elapsed < maxWait,
    Pause[waitInterval];
    elapsed += waitInterval;
    evaluating = Quiet[Check[CurrentValue[nb, Evaluating], False]];
    If[!TrueQ[evaluating], Break[]];
  ];

  If[syncMode =!= "none",
    FrontEndTokenExecute[nb, "Refresh"];
    Pause[0.05];
  ];
  If[syncMode === "strict",
    Module[{elapsedSync = 0, interval = 0.05},
      While[elapsedSync < syncWait,
        Pause[interval];
        elapsedSync += interval;
        evaluating = Quiet[Check[CurrentValue[nb, Evaluating], False]];
        If[!TrueQ[evaluating], Break[]];
      ];
    ]
  ];

  cellsAfter = Quiet[Check[Length[Cells[nb]], cellsBefore]];
  <|
    "evaluated" -> True,
    "waited_seconds" -> elapsed,
    "cells_before" -> cellsBefore,
    "cells_after" -> cellsAfter
  |>
];

cmdBatchCommands[params_] := Module[{commands, results},
  commands = Lookup[params, "commands", {}];
  If[!ListQ[commands],
    Return[<|"error" -> "commands must be a list"|>]
  ];

  results = Map[
    Function[cmd,
      Module[{cmdName, cmdParams, res},
        cmdName = Lookup[cmd, "command", "unknown"];
        cmdParams = Lookup[cmd, "params", <||>];
        res = dispatchCommand[cmdName, cmdParams];
        If[KeyExistsQ[res, "error"],
          <|"success" -> False, "command" -> cmdName, "error" -> res["error"]|>,
          <|"success" -> True, "command" -> cmdName, "result" -> res|>
        ]
      ]
    ],
    commands
  ];

  <|"success" -> True, "count" -> Length[results], "results" -> results|>
];

(* Atomic notebook execution with kernel-mode fast path *)
cmdExecuteCodeNotebook[params_] := Module[
  {code, mode, syncMode, nb, createdNew, maxWait, sessionId, deterministicSeed, isolateContext, syncWait, nbSpec},

  code = Lookup[params, "code", ""];
  If[code === "", Return[<|"error" -> "No code provided"|>]];

  mode = Lookup[params, "mode", "kernel"];
  syncMode = resolveSyncMode[params];
  maxWait = Lookup[params, "max_wait", 30];
  sessionId = Lookup[params, "session_id", None];
  deterministicSeed = Lookup[params, "deterministic_seed", None];
  isolateContext = TrueQ[Lookup[params, "isolate_context", False]];
  syncWait = Lookup[params, "sync_wait", 2];
  nbSpec = Lookup[params, "notebook", None];

  createdNew = False;
  nb = resolveNotebook[nbSpec, sessionId];
  If[nb === None || !validNotebookQ[nb],
    createdNew = True;
    nb = CreateDocument[{}, WindowTitle -> "Analysis"]
  ];
  $MCPActiveNotebook = nb;
  If[StringQ[sessionId] && StringLength[sessionId] > 0,
    $MCPSessionNotebooks[sessionId] = nb
  ];

  If[mode === "frontend",
    executeCodeNotebookFrontend[nb, code, maxWait, syncMode, createdNew, sessionId, deterministicSeed, isolateContext, syncWait],
    executeCodeNotebookKernel[nb, code, syncMode, createdNew, sessionId, deterministicSeed, isolateContext, syncWait]
  ]
];

executeCodeNotebookKernel[nb_, code_, syncMode_, createdNew_, sessionId_, deterministicSeed_, isolateContext_, syncWait_] := Module[
  {result, messages, timing, isGraphics, outputCell, inputCellId, inputCellObj, exprToEval, ctx},

  inputCellId = newCellId[];
  SelectionMove[nb, After, Notebook];
  NotebookWrite[nb, Cell[code, "Input", CellID -> inputCellId], After];
  inputCellObj = Quiet@Check[First[Cells[nb, CellID -> inputCellId]], None];

  ctx = If[TrueQ[isolateContext], getSessionContext[sessionId], None];
  exprToEval = ToExpression[code];
  If[ctx =!= None,
    exprToEval = Block[{$Context = ctx, $ContextPath = {ctx, "System`"}}, exprToEval]
  ];
  If[deterministicSeed =!= None && deterministicSeed =!= Null,
    exprToEval = BlockRandom[SeedRandom[deterministicSeed]; exprToEval]
  ];

  {timing, {result, messages}} = AbsoluteTiming[
    Block[{$MessageList = {}},
      {Quiet @ Check[exprToEval, $Failed], $MessageList}
    ]
  ];

  isGraphics = MatchQ[result, _Graphics | _Graphics3D | _Image | _Legended | _Graph];

  outputCell = If[result === $Failed,
    Cell["$Failed", "Output", GeneratedCell -> True],
    If[isGraphics,
      Cell[BoxData[ToBoxes[result]], "Output", GeneratedCell -> True],
      Cell[ToString[result, InputForm], "Output", GeneratedCell -> True]
    ]
  ];

  SelectionMove[nb, After, Notebook];
  NotebookWrite[nb, outputCell, After];

  If[syncMode =!= "none",
    FrontEndTokenExecute[nb, "Refresh"];
    Pause[0.05];
  ];
  If[syncMode === "strict",
    Module[{elapsed = 0, interval = 0.05, found = False},
      While[elapsed < syncWait && !found,
        Pause[interval];
        elapsed += interval;
        found = Length[Cells[nb, CellID -> inputCellId]] > 0;
      ];
    ]
  ];

  Module[{formattedMessages, hasErrors, hasWarnings},
    formattedMessages = Map[
      Function[msg,
        Quiet @ Check[
          <|
            "tag" -> ToString[First[msg], OutputForm],
            "text" -> ToString[Last[msg], OutputForm],
            "type" -> If[StringContainsQ[ToString[First[msg]], "::"],
              If[StringEndsQ[ToString[First[msg]], "::warning"], "warning", "error"],
              "message"
            ]
          |>,
          <|"tag" -> "Unknown", "text" -> "Failed to format message", "type" -> "error"|>
        ]
      ],
      Take[messages, UpTo[20]]
    ];
    formattedMessages = Select[formattedMessages, AssociationQ[#] && StringLength[#["text"]] > 0 &];
    hasErrors = Length[Select[formattedMessages, #["type"] == "error" &]] > 0;
    hasWarnings = Length[Select[formattedMessages, #["type"] == "warning" &]] > 0;

    <|
      "success" -> True,
      "mode" -> "kernel",
      "notebook_id" -> ToString[nb, InputForm],
      "cell_id" -> If[inputCellObj === None, ToString[inputCellId], ToString[inputCellObj, InputForm]],
      "cell_id_numeric" -> inputCellId,
      "timing_ms" -> Round[timing * 1000],
      "created_notebook" -> createdNew,
      "messages" -> formattedMessages,
      "has_errors" -> hasErrors,
      "has_warnings" -> hasWarnings,
      "message_count" -> Length[formattedMessages],
      "output_preview" -> StringTake[ToString[result, InputForm], UpTo[1000]],
      "is_graphics" -> isGraphics
    |>
  ]
];

executeCodeNotebookFrontend[nb_, code_, maxWait_, syncMode_, createdNew_, sessionId_, deterministicSeed_, isolateContext_, syncWait_] := Module[
  {cell, cellsBefore, cellsAfter, waitInterval, elapsed, inputCellId, inputCellObj, execCode, ctx},

  cellsBefore = Length[Cells[nb]];

  execCode = code;
  ctx = If[TrueQ[isolateContext], getSessionContext[sessionId], None];
  If[ctx =!= None,
    execCode = "Block[{$Context = \"" <> ctx <> "\", $ContextPath = {\"" <> ctx <> "\", \"System`\"}}, " <> execCode <> "]"
  ];
  If[deterministicSeed =!= None && deterministicSeed =!= Null,
    execCode = "BlockRandom[SeedRandom[" <> ToString[deterministicSeed] <> "]; " <> execCode <> "]"
  ];

  inputCellId = newCellId[];
  SelectionMove[nb, After, Notebook];
  NotebookWrite[nb, Cell[execCode, "Input", CellID -> inputCellId], After];
  inputCellObj = Quiet@Check[First[Cells[nb, CellID -> inputCellId]], None];
  cell = If[inputCellObj === None, Last[Cells[nb, CellStyle -> "Input"]], inputCellObj];

  waitInterval = 0.05;
  SelectionMove[cell, All, Cell];
  FrontEndTokenExecute["EvaluateCells"];

  elapsed = 0;
  While[elapsed < maxWait,
    Pause[waitInterval];
    elapsed += waitInterval;
    cellsAfter = Length[Cells[nb]];
    If[cellsAfter > cellsBefore + 1,
      Pause[0.1];
      Break[];
    ];
  ];

  If[syncMode =!= "none",
    FrontEndTokenExecute[nb, "Refresh"];
    Pause[0.05];
  ];
  If[syncMode === "strict",
    Module[{elapsedSync = 0, interval = 0.05, found = False},
      While[elapsedSync < syncWait && !found,
        Pause[interval];
        elapsedSync += interval;
        found = Length[Cells[nb, CellID -> inputCellId]] > 0;
      ];
    ]
  ];

  Module[{messages, outputCells, outputContent, outputText, hasErrors, hasWarnings},
    messages = Quiet @ Module[{recent, formatted},
      recent = Take[$MessageList, UpTo[20]];
      formatted = Map[
        Function[msg,
          Quiet @ Check[
            <|
              "tag" -> ToString[First[msg], OutputForm],
              "text" -> ToString[Last[msg], OutputForm],
              "type" -> If[StringContainsQ[ToString[First[msg]], "::"],
                If[StringEndsQ[ToString[First[msg]], "::warning"], "warning", "error"],
                "message"
              ]
            |>,
            <|"tag" -> "Unknown", "text" -> "Failed to format message", "type" -> "error"|>
          ]
        ],
        recent
      ];
      Select[formatted, AssociationQ[#] && StringLength[#["text"]] > 0 &]
    ];

    outputCells = Quiet @ Cells[nb, CellStyle -> "Output"];
    outputContent = If[Length[outputCells] > 0,
      Quiet @ NotebookRead[Last[outputCells]],
      None
    ];

    outputText = Quiet @ Check[
      If[outputContent =!= None, ToString[outputContent, OutputForm], ""],
      ""
    ];

    hasErrors = Length[Select[messages, #["type"] == "error" &]] > 0;
    hasWarnings = Length[Select[messages, #["type"] == "warning" &]] > 0;

    <|
      "success" -> True,
      "mode" -> "frontend",
      "notebook_id" -> ToString[nb, InputForm],
      "cell_id" -> If[inputCellObj === None, ToString[cell, InputForm], ToString[inputCellObj, InputForm]],
      "cell_id_numeric" -> inputCellId,
      "waited_seconds" -> elapsed,
      "created_notebook" -> createdNew,
      "messages" -> messages,
      "has_errors" -> hasErrors,
      "has_warnings" -> hasWarnings,
      "message_count" -> Length[messages],
      "output_preview" -> StringTake[outputText, UpTo[1000]]
    |>
  ]
];

(* ============================================================================ *)
(* SCREENSHOT COMMANDS                                                          *)
(* ============================================================================ *)

cmdScreenshotNotebook[params_] := Module[{nb, img, path, maxHeight, format, waitMs, useRasterize, exportResult, method, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  If[nb === None || !validNotebookQ[nb],
    nb = CreateDocument[{}, WindowTitle -> "MCP Screenshot"];
    If[nb === None || !validNotebookQ[nb], Return[<|"error" -> "No notebook found"|>]];
  ];
  $MCPActiveNotebook = nb;
  maxHeight = Lookup[params, "max_height", 2000];
  format = Lookup[params, "format", "PNG"];
  waitMs = Lookup[params, "wait_ms", 100];
  useRasterize = TrueQ[Lookup[params, "use_rasterize", False]];
  
  FrontEndTokenExecute[nb, "Refresh"];
  Pause[waitMs / 1000.0];
  
  path = FileNameJoin[{$TemporaryDirectory, "mcp_nb_" <> CreateUUID[] <> ".png"}];
  method = "rasterize";
  
  If[!useRasterize,
    exportResult = Quiet @ Check[
      FrontEndExecute[FrontEnd`ExportPacket[nb, "PNG"]],
      $Failed
    ];
    If[MatchQ[exportResult, {_ByteArray, ___}],
      Export[path, ImportByteArray[First[exportResult], "PNG"], format];
      img = Import[path];
      method = "export_packet";
    ]
  ];
  
  If[method === "rasterize",
    img = Rasterize[nb, ImageResolution -> 144];
    If[ImageDimensions[img][[2]] > maxHeight,
      img = ImageResize[img, {Automatic, maxHeight}]
    ];
    Export[path, img, format];
  ];
  
  <|
    "path" -> path,
    "width" -> ImageDimensions[img][[1]],
    "height" -> ImageDimensions[img][[2]],
    "format" -> format,
    "method" -> method
  |>
];

cmdScreenshotCell[params_] := Module[{cell, content, img, path, useRasterize, exportResult, method, nb, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  If[nb === None || !validNotebookQ[nb],
    Return[<|"error" -> "No notebook found"|>]
  ];
  cell = resolveCellObject[Lookup[params, "cell_id", None], nb];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  
  useRasterize = TrueQ[Lookup[params, "use_rasterize", False]];
  content = NotebookRead[cell];
  path = FileNameJoin[{$TemporaryDirectory, "mcp_cell_" <> CreateUUID[] <> ".png"}];
  method = "rasterize";
  
  If[!useRasterize,
    nb = ParentNotebook[cell];
    SelectionMove[cell, All, Cell];
    exportResult = Quiet @ Check[
      FrontEndExecute[FrontEnd`ExportPacket[NotebookSelection[nb], "PNG"]],
      $Failed
    ];
    If[MatchQ[exportResult, {_ByteArray, ___}],
      Export[path, ImportByteArray[First[exportResult], "PNG"], "PNG"];
      img = Import[path];
      method = "export_packet";
    ]
  ];
  
  If[method === "rasterize",
    img = Rasterize[content, ImageResolution -> 144];
    Export[path, img, "PNG"];
  ];
  
  <|
    "path" -> path,
    "width" -> ImageDimensions[img][[1]],
    "height" -> ImageDimensions[img][[2]],
    "method" -> method
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

cmdSelectCell[params_] := Module[{cell, nb, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  cell = resolveCellObject[Lookup[params, "cell_id", None], nb];
  If[!MatchQ[cell, _CellObject],
    Return[<|"error" -> "Invalid cell ID"|>]
  ];
  SelectionMove[cell, All, Cell];
  <|"selected" -> True|>
];

cmdScrollToCell[params_] := Module[{cell, nb, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
  cell = resolveCellObject[Lookup[params, "cell_id", None], nb];
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

cmdExportNotebook[params_] := Module[{nb, path, format, sessionId},
  sessionId = Lookup[params, "session_id", None];
  nb = resolveNotebook[Lookup[params, "notebook", None], sessionId];
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
  
  If[result === $Failed,
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

cmdOpenNotebookFile[params_] := Module[{path, expandedPath, nb, sessionId},
  path = Lookup[params, "path", None];
  sessionId = Lookup[params, "session_id", None];
  
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
  
  If[StringQ[sessionId] && StringLength[sessionId] > 0,
    $MCPSessionNotebooks[sessionId] = nb
  ];
  $MCPActiveNotebook = nb;

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
  
  If[path === None,
    Return[<|"error" -> "No path specified"|>]
  ];
  
  expandedPath = If[StringStartsQ[path, "~"],
    StringReplace[path, StartOfString ~~ "~" -> $HomeDirectory],
    path
  ];
  
  If[!FileExistsQ[expandedPath],
    Return[<|"error" -> ("File not found: " <> expandedPath)|>]
  ];
  
  startTime = AbsoluteTime[];
  result = Quiet[Check[Get[expandedPath], $Failed]];
  timing = Round[(AbsoluteTime[] - startTime) * 1000];
  
  If[result === $Failed,
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
  
  If[expr === None,
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
  
  If[result === $Failed,
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
  
  If[expr === None,
    Return[<|"error" -> "No expression specified"|>]
  ];
  
  memBefore = MemoryInUse[];
  timing = Quiet[Check[AbsoluteTiming[ToExpression[expr]], $Failed]];
  memAfter = MemoryInUse[];
  
  If[timing === $Failed,
    Return[<|"error" -> "Timing failed"|>]
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
  
  If[code === None,
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
