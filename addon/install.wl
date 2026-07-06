(* install.wl - Install MathematicaMCP for auto-loading *)
(* Run this once: wolframscript -file install.wl *)
(* Safe to re-run: any previous MathematicaMCP section in init.m is replaced, *)
(* so upgrades re-point the loader at THIS package copy (a stale loader line *)
(* pointing at an old uv/pip cache path was the main cause of protocol-skew *)
(* warnings after upgrading the Python package). *)

$MCPPackagePath = DirectoryName[$InputFileName];
If[$MCPPackagePath === "", $MCPPackagePath = Directory[]];

$MCPPackageFile = FileNameJoin[{$MCPPackagePath, "MathematicaMCP.wl"}];
$InitPath = FileNameJoin[{$UserBaseDirectory, "Kernel", "init.m"}];

If[!FileExistsQ[$MCPPackageFile],
  Print["Error: MathematicaMCP.wl not found at: ", $MCPPackageFile];
  Exit[1]
];

initContent = If[FileExistsQ[$InitPath], Import[$InitPath, "Text"], ""];
If[!StringQ[initContent], initContent = ""];

$MCPPackageFileNormalized = StringReplace[$MCPPackageFile, "\\" -> "/"];
loadCode = StringJoin[
  "\n\n(* MathematicaMCP - Auto-load for LLM control *)\n",
  "Quiet @ Get[\"", $MCPPackageFileNormalized, "\"];\n",
  "MathematicaMCP`StartMCPServer[];\n"
];

(* Idempotent rewrite: drop every line of any previous MathematicaMCP section
   (the comment, Get[...], and StartMCPServer[] lines all contain the string
   "MathematicaMCP"), then append the fresh loader. *)
wasConfigured = StringContainsQ[initContent, "MathematicaMCP"];
cleanedLines = Select[StringSplit[initContent, "\n"], !StringContainsQ[#, "MathematicaMCP"] &];
cleanedContent = StringTrim[StringRiffle[cleanedLines, "\n"], RegularExpression["[\\n\\s]+$"]];

If[!DirectoryQ[DirectoryName[$InitPath]],
  CreateDirectory[DirectoryName[$InitPath]]
];

Export[$InitPath, cleanedContent <> loadCode, "Text"];

If[wasConfigured,
  Print["=== MathematicaMCP Updated ==="];
  Print[""];
  Print["Replaced the existing MathematicaMCP section in: ", $InitPath];
  Print["The loader now points at: ", $MCPPackageFileNormalized];
  ,
  Print["=== MathematicaMCP Installation Complete ==="];
  Print[""];
  Print["Added to: ", $InitPath];
];
Print[""];
Print["The MCP server will now auto-start when Mathematica launches."];
Print[""];
Print["To start manually in a running session:"];
Print["  Get[\"", $MCPPackageFileNormalized, "\"];"];
Print["  StartMCPServer[]"];
Print[""];
Print["To uninstall, remove the MathematicaMCP section from init.m"];
