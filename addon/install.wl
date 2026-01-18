(* install.wl - Install MathematicaMCP for auto-loading *)
(* Run this once: wolframscript -file install.wl *)

$MCPPackagePath = DirectoryName[$InputFileName];
If[$MCPPackagePath === "", $MCPPackagePath = Directory[]];

$MCPPackageFile = FileNameJoin[{$MCPPackagePath, "MathematicaMCP.wl"}];
$InitPath = FileNameJoin[{$UserBaseDirectory, "Kernel", "init.m"}];

If[!FileExistsQ[$MCPPackageFile],
  Print["Error: MathematicaMCP.wl not found at: ", $MCPPackageFile];
  Exit[1]
];

initContent = If[FileExistsQ[$InitPath], Import[$InitPath, "Text"], ""];

loadCode = StringJoin[
  "\n\n(* MathematicaMCP - Auto-load for LLM control *)\n",
  "Quiet @ Get[\"", $MCPPackageFile, "\"];\n",
  "MathematicaMCP`StartMCPServer[];\n"
];

If[StringContainsQ[initContent, "MathematicaMCP"],
  Print["MathematicaMCP already configured in init.m"];
  Print["To reinstall, remove the MathematicaMCP section from:"];
  Print["  ", $InitPath];
  ,
  
  If[!DirectoryQ[DirectoryName[$InitPath]],
    CreateDirectory[DirectoryName[$InitPath]]
  ];
  
  Export[$InitPath, initContent <> loadCode, "Text"];
  
  Print["=== MathematicaMCP Installation Complete ==="];
  Print[""];
  Print["Added to: ", $InitPath];
  Print[""];
  Print["The MCP server will now auto-start when Mathematica launches."];
  Print[""];
  Print["To start manually in a running session:"];
  Print["  Get[\"", $MCPPackageFile, "\"];"];
  Print["  StartMCPServer[]"];
  Print[""];
  Print["To uninstall, remove the MathematicaMCP section from init.m"];
];
