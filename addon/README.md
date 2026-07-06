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

Success responses of notebook-touching commands (`create_notebook`, `write_cell`, `evaluate_cell`, `get_cells`, `batch_commands`, ...) additionally carry a `state_delta` field: `{"notebook": ..., "cell_count": ..., "kernel_busy": ...}` where `kernel_busy` reflects the command's target notebook's `Evaluating` state (protocol 4; it cannot observe an in-flight front-end evaluation from the socket handler - use the `evaluation_pending`/`evaluation_complete` fields for progress). Pure-kernel commands omit it for speed.

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
- `ping` - Returns `{"pong": true, "timestamp": "...", "version": "...", "protocol_version": 4}`
- `get_status` - Returns frontend/kernel version, open notebook count, `mcp_port`, `mcp_server_running`, and `protocol_version` (the Python client compares it against its expected version to detect a stale addon)

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
- `execute_code_notebook` - Run code via notebook evaluation
- `execute_selection` - Evaluate current selection
- `batch_commands` - Execute multiple commands in one request

### Variable Introspection
- `list_variables` - List defined variables
- `get_variable` - Get a variable's value
- `set_variable` - Set a variable's value
- `clear_variables` - Clear variables
- `get_expression_info` - Inspect expression structure

### Error Recovery
- `get_messages` - Retrieve kernel messages

### Screenshots
- `screenshot_notebook` - Rasterize entire notebook
- `screenshot_cell` - Rasterize single cell
- `rasterize_expression` - Render expression to image

### Navigation
- `select_cell` - Select a cell
- `scroll_to_cell` - Scroll to cell

### File Handling
- `open_notebook_file` - Open a notebook file
- `run_script` - Run a Wolfram script

### Debugging
- `trace_evaluation` - Trace expression evaluation
- `time_expression` - Time expression evaluation
- `check_syntax` - Check code syntax

### Data I/O
- `import_data` - Import data from file
- `export_data` - Export data to file
- `list_import_formats` - List supported import formats

### Visualization
- `export_graphics` - Export graphics to file

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
  RegularExpression["(?s)\\n*\\(\\* MathematicaMCP.*?StartMCPServer\\[\\];\\n"] -> ""];
Export[$initPath, newContent, "Text"];
```

## Evaluation Architecture

The addon uses two distinct kernel links for evaluation:

- **Preemptive link** - All `SocketListen` handlers (including `execute_code`, `ping`, and MCP commands) run here. Can interrupt the main link. Fast and independent.
- **Main link** - `evaluate_cell` and `execute_code_notebook` in frontend mode dispatch to this link via `FrontEndTokenExecute["EvaluateCells"]`. Queued and single-threaded. Also used by Shift+Enter in the notebook UI.

`execute_code_notebook` supports two modes:
- **`mode="kernel"`** (default): Evaluates code directly on the preemptive link using `AbsoluteTiming`, writes the output cell manually. No polling. Fastest path (~50ms). Interactive results (`Manipulate`/`Dynamic`/`Animate`) are written as rendered boxes (a live panel); other non-graphics results as `InputForm` text.
- **`mode="frontend"`**: Dispatches via `FrontEndTokenExecute["EvaluateCells"]` to the main link and returns quickly (in-handler wait capped at 0.2s; the front end cannot complete while the handler runs). The normal response is `status: "evaluation_pending"` with `evaluation_complete: false` - the evaluation runs after the call returns and its output cell lands in the notebook (re-check with `get_cells`). Required for `Manipulate`, `Dynamic`, and other FrontEnd-dependent content. `evaluate_cell` and `evaluate_selection` follow the same pending contract (protocol 4).

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

### Notebook Cells Stuck at "Running..."

This means the main link is blocked by a previous computation. The preemptive link (used by `execute_code` and MCP commands) remains fast - this is why `Manipulate` sliders can still respond while cells show "Running...".

**Fix:** Restart the kernel (Evaluation > Quit Kernel > Local), then re-run `StartMCPServer[]`.

**Note:** `RestartMCPServer[]` only restarts the TCP socket listener - it does **not** clear the main link queue. Use it only when the MCP connection itself is broken (timeouts with no response).
