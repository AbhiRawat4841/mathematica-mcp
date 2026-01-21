# MathematicaMCP Addon

The Wolfram Language package that enables external control of Mathematica via socket connection.

## What It Does

This addon creates a TCP socket server (default port 9881) inside the Mathematica frontend. External programs can connect and send JSON commands to:

- Query and manipulate open notebooks
- Execute Wolfram Language code
- Take screenshots of notebooks and cells
- Navigate within notebooks
- Export to various formats

## Installation

### Auto-Load on Startup (Recommended)

From the **mathematica-mcp** root directory:
```bash
wolframscript -file addon/install.wl
```

This adds the package to your `~/Library/Wolfram/Kernel/init.m` (macOS) or `%APPDATA%\Wolfram\Kernel\init.m` (Windows) so the server starts automatically when Mathematica launches.

**After installation, restart Mathematica** for the changes to take effect.

### Manual Load

In Mathematica:
```mathematica
Get["/path/to/mathematica-mcp/addon/MathematicaMCP.wl"]
StartMCPServer[]
```

## Usage

### Server Control

```mathematica
(* Start the server *)
StartMCPServer[]
(* Output: [MathematicaMCP] Server started on port 9881 *)

(* Check status *)
MCPServerStatus[]
(* Output: <|"running" -> True, "port" -> 9881, "listener" -> ...|> *)

(* Restart *)
RestartMCPServer[]

(* Stop *)
StopMCPServer[]
```

### Configuration

Change port (before starting):
```mathematica
MathematicaMCP`Private`$MCPPort = 9882;
StartMCPServer[]
```

Enable debug logging:
```mathematica
MathematicaMCP`Private`$MCPDebug = True;
```

## Command Protocol

Commands are sent as JSON over TCP:

```json
{
  "command": "execute_code",
  "params": {
    "code": "1 + 1",
    "format": "text"
  }
}
```

Response:
```json
{
  "id": "...",
  "status": "success",
  "result": {
    "success": true,
    "output": "2",
    "output_tex": "2",
    "output_inputform": "2"
  }
}
```

## Available Commands

### Status
- `ping` - Returns `{"pong": true, "timestamp": "...", "version": "..."}`
- `get_status` - Returns frontend/kernel version, open notebook count

### Notebooks
- `get_notebooks` - List all open notebooks
- `get_notebook_info` - Details about a notebook
- `create_notebook` - Create new notebook
- `save_notebook` - Save notebook
- `close_notebook` - Close notebook

### Cells
- `get_cells` - List cells (optional style filter)
- `get_cell_content` - Get cell content
- `write_cell` - Insert new cell
- `delete_cell` - Remove cell
- `evaluate_cell` - Evaluate specific cell

### Execution
- `execute_code` - Run code, return result
- `execute_selection` - Evaluate current selection

### Screenshots
- `screenshot_notebook` - Rasterize entire notebook
- `screenshot_cell` - Rasterize single cell
- `rasterize_expression` - Render expression to image

### Navigation
- `select_cell` - Select a cell
- `scroll_to_cell` - Scroll to cell

### Export
- `export_notebook` - Export to PDF/HTML/TeX/Markdown

## Uninstallation

Remove the auto-load from init.m:
```bash
# Edit ~/Library/Wolfram/Kernel/init.m (macOS)
# Remove the MathematicaMCP section
```

Or in Mathematica:
```mathematica
$initPath = FileNameJoin[{$UserBaseDirectory, "Kernel", "init.m"}];
content = Import[$initPath, "Text"];
newContent = StringReplace[content,
  RegularExpression["\\n*\\(\\* MathematicaMCP.*?StartMCPServer\\[\\];\\n"] -> ""];
Export[$initPath, newContent, "Text"];
```

## Troubleshooting

### Port Already in Use
```mathematica
MathematicaMCP`Private`$MCPPort = 9882;
RestartMCPServer[]
```

Then set `MATHEMATICA_PORT=9882` environment variable for the Python client.

### Server Won't Start
Check for existing socket:
```mathematica
Close /@ Sockets[]  (* Close all sockets *)
StartMCPServer[]
```

### Commands Not Working
Enable debug mode:
```mathematica
MathematicaMCP`Private`$MCPDebug = True;
```
Then watch the Mathematica Messages window for debug output.
