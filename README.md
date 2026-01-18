# Mathematica MCP

Full GUI control of Mathematica notebooks and kernel via Model Context Protocol (MCP).

This MCP server allows LLMs (like Claude) to:
- **Create, read, and manipulate notebooks** - Full programmatic access to Mathematica notebooks
- **Execute Wolfram Language code** - Run computations and get results
- **Take screenshots** - Capture notebook views, cells, or rendered expressions
- **Navigate and edit** - Select cells, scroll, insert content
- **Export** - Save notebooks to PDF, HTML, TeX, and more

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│ Claude/LLM   │ MCP │ mathematica-mcp │ TCP │ MathematicaMCP.wl       │
│              │◄───►│ (Python Server) │◄───►│ (Addon in Mathematica)  │
└──────────────┘     └─────────────────┘     │                         │
                                             │ ┌─────────┐ ┌─────────┐ │
                                             │ │ Kernel  │ │Frontend │ │
                                             │ └─────────┘ └─────────┘ │
                                             └─────────────────────────┘
```

**Two components:**
1. **Python MCP Server** - Exposes tools to LLMs via the MCP protocol
2. **Mathematica Addon** - Runs inside Mathematica, provides socket-based control

---

## Installation

### Step 1: Install the Python Package

```bash
cd ~/mcp/mathematica-mcp
uv sync
```

Or with pip:
```bash
pip install -e ~/mcp/mathematica-mcp
```

### Step 2: Install the Mathematica Addon

**Option A: Auto-load on startup (recommended)**

```bash
cd ~/mcp/mathematica-mcp/addon
wolframscript -file install.wl
```

- Writes an auto-load snippet to `~/Library/Wolfram/Kernel/init.m` (user base) so the addon loads and starts on launch.
- Default port: **9881**. Change at runtime with `MathematicaMCP`Private`$MCPPort = <port>; StartMCPServer[]` before starting the server.

**Option B: Manual load in Mathematica (single session)**

```mathematica
Get["~/mcp/mathematica-mcp/addon/MathematicaMCP.wl"]
MathematicaMCP`Private`$MCPPort = 9881; (* optional: change before start *)
StartMCPServer[]
```

**Troubleshooting ports:** If you see "Address already in use", set a new port (e.g., 9883) with `$MCPPort` before `StartMCPServer[]` and set `MATHEMATICA_PORT` on the client side accordingly.

### Step 3: Configure Your LLM Client

**For Claude Code (`.mcp.json` in project root):**

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/YOUR_USERNAME/mcp/mathematica-mcp",
        "run",
        "mathematica-mcp"
      ]
    }
  }
}
```

**For OpenCode (`~/.config/opencode/opencode.json`):**

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/YOUR_USERNAME/mcp/mathematica-mcp",
        "run",
        "mathematica-mcp"
      ]
    }
  }
}
```

**For Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/Users/YOUR_USERNAME/mcp/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

---

## Quick Start

1. **Start Mathematica** (the addon auto-starts if installed)
2. **Verify the addon is running:**
   ```mathematica
   MCPServerStatus[]
   (* Should show: <|"running" -> True, "port" -> 9881, ...|> *)
   ```
3. **Start your LLM client** (Claude Code, OpenCode, Claude Desktop, etc.)
4. **Ask Claude to interact with Mathematica!**

---

## Available Tools

### Status & Connection

| Tool | Description |
|------|-------------|
| `get_mathematica_status` | Check connection status and system info |

### Notebook Management

| Tool | Description |
|------|-------------|
| `get_notebooks` | List all open notebooks |
| `get_notebook_info` | Get details about a specific notebook |
| `create_notebook` | Create a new empty notebook |
| `save_notebook` | Save notebook to disk |
| `close_notebook` | Close a notebook |
| `export_notebook` | Export to PDF, HTML, TeX, Markdown |

### Cell Operations

| Tool | Description |
|------|-------------|
| `get_cells` | List cells in a notebook (optionally filter by style) |
| `get_cell_content` | Get full content of a cell |
| `write_cell` | Insert a new cell |
| `delete_cell` | Remove a cell |
| `evaluate_cell` | Evaluate a specific cell |

### Code Execution

| Tool | Description |
|------|-------------|
| `execute_code` | Run Wolfram Language code (defaults to `output_target="notebook"` for proper graphics rendering) |
| `evaluate_selection` | Evaluate currently selected cells |

### Screenshots & Visualization

| Tool | Description |
|------|-------------|
| `screenshot_notebook` | Capture full notebook view |
| `screenshot_cell` | Capture a specific cell |
| `rasterize_expression` | Render any expression as image (without modifying notebook) |

### Navigation

| Tool | Description |
|------|-------------|
| `select_cell` | Select a cell (move cursor) |
| `scroll_to_cell` | Scroll to make cell visible |

---

## Understanding Tool Workflows

This section explains **when to use which tool** for different scenarios. Understanding these distinctions is critical for effective notebook automation.

### The Three Ways to Execute Code

| Tool | What It Does | Where Output Appears | Use When |
|------|--------------|---------------------|----------|
| `write_cell` | **Only writes** code to notebook | No output (code not executed) | Building notebooks for later execution |
| `evaluate_cell` | Executes a cell **in the notebook** | Output appears in Mathematica GUI | User needs to see results in Mathematica |
| `execute_code` | Executes code **immediately** | CLI (text) or notebook (graphics) | Quick computations or plotting |
| `rasterize_expression` | Executes and **returns image** | Image returned to LLM/conversation | LLM needs to see/describe plots |

### Key Insight: `write_cell` Does NOT Execute

```python
# This ONLY adds code to the notebook - no plot is generated!
write_cell("Plot[Sin[x], {x, 0, 2 Pi}]", style="Input")

# To see the plot, you must ALSO do one of:
evaluate_cell(cell_id)           # Execute in Mathematica
# OR
rasterize_expression("Plot[...]") # Get image back to LLM
```

### The `output_target` Parameter

The `execute_code` tool has an `output_target` parameter that controls where results go:

#### `output_target="notebook"` (default)

Inserts an **Input cell** into the active notebook and **evaluates it**. The output (including graphics) renders properly in Mathematica. Best for:
- Plots and visualizations
- Building documented notebooks
- When user needs to see results in Mathematica GUI
- **Any graphical output** (this is why it's the default)

```python
# Creates plot IN the Mathematica notebook (default behavior)
execute_code("Plot[Sin[x], {x, 0, 2 Pi}]")

# Adds 3D surface to notebook
execute_code("Plot3D[Sin[x] Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}]")
```

**Note:** With `output_target="notebook"`, the LLM does NOT see the plot directly. To see it, use `screenshot_notebook()`, `screenshot_cell()`, or `rasterize_expression()`.

#### `output_target="cli"`

Returns the result as **text** to the LLM. Use this for:
- Symbolic computations
- Numerical results
- LaTeX-formatted expressions
- Any non-graphical output where you need the result as text

```python
# Returns text: "2"
execute_code("Integrate[x^2 Exp[-x], {x, 0, Infinity}]", output_target="cli")

# Returns LaTeX: "\frac{x^3}{3}"
execute_code("Integrate[x^2, x]", format="latex", output_target="cli")

# Returns: "{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}"
execute_code("Table[Fibonacci[n], {n, 1, 10}]", output_target="cli")
```

### Complete Workflow Examples

#### Example 1: Create Notebook with Plots (User Views in Mathematica)

```python
# Step 1: Create notebook
create_notebook(title="Analysis")

# Step 2: Add content (write_cell does NOT execute)
write_cell("My Analysis", style="Title")
write_cell("Plot of sin(x):", style="Text")
write_cell("Plot[Sin[x], {x, 0, 2 Pi}]", style="Input")

# Step 3: Get cell IDs
cells = get_cells(style="Input")

# Step 4: Evaluate cells (NOW the plot renders in Mathematica)
evaluate_cell(cell_id=cells[0]["id"])

# Step 5: (Optional) Take screenshot to see result
screenshot_notebook()
```

#### Example 2: Quick Plot Visible to LLM

```python
# Use rasterize_expression - LLM sees the plot directly in conversation
rasterize_expression("Plot[Sin[x], {x, 0, 2 Pi}]", image_size=500)
# Returns: Image that LLM can see and describe
```

#### Example 3: Computation + Plot in Notebook

```python
# Compute symbolically (result returned as text)
result = execute_code("Integrate[Sin[x]^2, x]", output_target="cli")
# Returns: "x/2 - Sin[2x]/4"

# Plot in notebook (user sees in Mathematica)
execute_code("Plot[Sin[x]^2, {x, 0, 2 Pi}, PlotLabel -> \"Sin²(x)\"]",
             output_target="notebook")

# Verify by taking screenshot
screenshot_notebook()
```

#### Example 4: Build and Evaluate Entire Notebook

```python
# Create structure
create_notebook(title="Fibonacci Analysis")
write_cell("Fibonacci Sequence", style="Title")
write_cell("Generating the sequence:", style="Text")
write_cell("fib = Table[Fibonacci[k], {k, 1, 20}]", style="Input")
write_cell("Ratio convergence to golden ratio:", style="Text")
write_cell("ListPlot[Table[Fibonacci[k+1]/Fibonacci[k]//N, {k,1,20}]]", style="Input")

# Get all input cells
cells = get_cells(style="Input")

# Evaluate each one
for cell in cells:
    evaluate_cell(cell_id=cell["id"])

# Now ALL plots are visible in Mathematica
screenshot_notebook()
```

#### Example 5: LLM Describes Plot Content

```python
# Render plot as image (LLM receives it)
image = rasterize_expression("""
Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi},
  PlotLegends -> {"Sin", "Cos"},
  PlotStyle -> {Blue, Red}]
""", image_size=500)

# LLM can now describe: "The plot shows two sinusoidal curves -
# a blue sine wave and a red cosine wave, phase-shifted by π/2..."
```

### Decision Tree: Which Tool to Use?

```
Do you need to BUILD a notebook for later use?
├── YES → use write_cell (then evaluate_cell when ready)
└── NO → Do you need the LLM to SEE the result?
         ├── YES → Is it graphical?
         │         ├── YES → use rasterize_expression
         │         └── NO → use execute_code(output_target="cli")
         └── NO → Is it graphical (plot, 3D, etc)?
                  ├── YES → use execute_code() [default renders in notebook]
                  └── NO → use execute_code(output_target="cli") for text result
```

### Common Mistakes and Fixes

| Mistake | Problem | Fix |
|---------|---------|-----|
| `write_cell("Plot[...]")` and expecting to see plot | `write_cell` doesn't execute | Add `evaluate_cell()` afterward OR use `execute_code("Plot[...]")` |
| `execute_code("1+1")` expecting text back | Default now renders in notebook | Add `output_target="cli"` for text results |
| Using `execute_code("Plot[...]")` but LLM can't describe the plot | Plot is in Mathematica, not returned to LLM | Use `rasterize_expression()` instead, OR take `screenshot_notebook()` after |
| `evaluate_cell()` returns nothing useful | It only confirms execution, doesn't return output | Use `screenshot_cell()` to see result, or `execute_code(output_target="cli")` for text output |

### Format Parameter Examples

The `format` parameter works with `output_target="cli"`:

```python
# Plain text (default)
execute_code("Integrate[x^2, x]", format="text")
# Returns: "x^3/3"

# LaTeX (for documents)
execute_code("Integrate[x^2, x]", format="latex")
# Returns: "\frac{x^3}{3}"

# Mathematica InputForm (copy-pasteable)
execute_code("Integrate[x^2, x]", format="mathematica")
# Returns: "x^3/3"

# Complex expression in LaTeX
execute_code("Integrate[1/(x^4 + 1), x]", format="latex")
# Returns: "\frac{-\log(x^2-\sqrt{2}x+1)+\log(x^2+\sqrt{2}x+1)..."
```

---

## Verified Test Results

The following tests have been verified working with Claude Opus 4.5 and Mathematica 14.1 on macOS ARM64:

### Test Summary

| Test | Description | Method | Result |
|------|-------------|--------|--------|
| 1 | Basic 2D Plot (Sin) | `execute_code` → notebook | ✓ |
| 2 | Multiple Functions + State Persistence | `execute_code` → notebook | ✓ |
| 3 | 3D Surface Plot | `execute_code` → notebook | ✓ |
| 4 | Contour Plot | `execute_code` → notebook | ✓ |
| 5 | Parametric Plot (Lissajous) | `execute_code` → notebook | ✓ |
| 6 | Graphics (Color Wheel) | `rasterize_expression` | ✓ |
| 7 | Symbolic Computation | `execute_code` → CLI | ✓ |
| 8 | LaTeX Formatted Output | `execute_code` → CLI | ✓ |
| 9 | State Persistence | `execute_code` → CLI | ✓ |

### Test Details

#### Test 1: Basic 2D Plot

```python
execute_code("Plot[Sin[x], {x, 0, 2 Pi}, PlotLabel -> \"Test 1: Basic Sin Wave\"]",
             output_target="notebook")
```
**Result:** Smooth sine wave from 0 to 2π, oscillating between -1 and 1, with label visible.

#### Test 2: State Persistence + Multiple Functions

```python
execute_code("testVar = 42;", output_target="notebook")

execute_code("""
Plot[{Sin[x], Cos[x], Sin[x] Cos[x]}, {x, 0, 2 Pi},
  PlotLabel -> StringJoin["Test 2: Multiple Functions (testVar = ", ToString[testVar], ")"],
  PlotLegends -> {"Sin", "Cos", "Sin*Cos"},
  PlotStyle -> {Blue, Red, Green}]
""", output_target="notebook")
```
**Result:** Three colored curves (Blue=Sin, Red=Cos, Green=Sin*Cos) with legend. Title shows "testVar = 42" confirming state persistence.

#### Test 3: 3D Surface Plot

```python
execute_code("""
Plot3D[Sin[x] Cos[y], {x, 0, 2 Pi}, {y, 0, 2 Pi},
  PlotLabel -> "Test 3: 3D Surface",
  ColorFunction -> "Rainbow",
  Mesh -> None]
""", output_target="notebook")
```
**Result:** Rainbow-colored 3D surface with characteristic saddle-point geometry.

#### Test 4: Contour Plot

```python
execute_code("""
ContourPlot[Sin[x^2 + y^2], {x, -2, 2}, {y, -2, 2},
  PlotLabel -> "Test 4: Contour Plot",
  ColorFunction -> "TemperatureMap",
  Contours -> 15]
""", output_target="notebook")
```
**Result:** Concentric circular contours (ripple pattern) with temperature color map.

#### Test 5: Parametric Plot (Lissajous Curve)

```python
execute_code("""
ParametricPlot[{Sin[3t], Cos[5t]}, {t, 0, 2 Pi},
  PlotLabel -> "Test 5: Lissajous Curve (3:5)",
  PlotStyle -> {Thick, Purple},
  AspectRatio -> 1]
""", output_target="notebook")
```
**Result:** Purple Lissajous curve with 3:5 frequency ratio, complex interweaving pattern.

#### Test 6: Rasterize Expression (Color Wheel)

```python
rasterize_expression("""
Graphics[{EdgeForm[Black],
  Table[{Hue[i/12], Disk[{Cos[2 Pi i/12], Sin[2 Pi i/12]}, 0.3]}, {i, 0, 11}]},
  PlotLabel -> "Test 6: Rasterize (Color Wheel)"]
""", image_size=400)
```
**Result:** 12 colored disks arranged in a circle showing full Hue spectrum. Rendered directly without modifying notebook.

#### Test 7: Symbolic Computation (CLI Output)

```python
execute_code("""
result = Integrate[x^2 * Exp[-x], {x, 0, Infinity}];
result
""", format="text", output_target="cli")
```
**Result:** Returns `2` (correct: Γ(3) = 2!)

#### Test 8: LaTeX Formatted Output

```python
execute_code("Integrate[1/(x^4 + 1), x]", format="latex", output_target="cli")
```
**Result:** Returns LaTeX-formatted antiderivative:
```latex
\frac{-\log(x^2-\sqrt{2}x+1)+\log(x^2+\sqrt{2}x+1)-2\tan^{-1}(1-\sqrt{2}x)+2\tan^{-1}(\sqrt{2}x+1)}{4\sqrt{2}}
```

#### Test 9: State Persistence Verification

```python
execute_code("{testVar, result}", output_target="cli")
```
**Result:** Returns `{42, 2}` - variables from earlier tests are preserved.

### Verified Capabilities

- **Plots render correctly** in Mathematica notebook GUI
- **LLM can see and describe screenshots** (colors, shapes, labels, legends)
- **State persists** between sequential commands (variables carry through)
- **Both output targets work**: `output_target="notebook"` and `output_target="cli"`
- **Rasterize renders images** directly without modifying notebook
- **Screenshot captures** full notebook state
- **Multiple output formats**: text, latex, mathematica (InputForm)

---

## Example Workflows

### Workflow 1: Interactive Computation

**User:** "Calculate the integral of sin(x)cos(x) from 0 to pi and show me the steps"

**Claude uses:**
```python
execute_code("Integrate[Sin[x] Cos[x], {x, 0, Pi}]", output_target="cli")
# Returns: 0

execute_code("Integrate[Sin[x] Cos[x], x]", format="latex", output_target="cli")
# Returns: "\frac{\sin^2(x)}{2}"

execute_code("Plot[Sin[x] Cos[x], {x, 0, Pi}, PlotLabel -> \"Sin[x]Cos[x]\"]")
# Creates plot in active notebook (default behavior)
```

### Workflow 2: Create a Documented Notebook

**User:** "Create a notebook analyzing the Fibonacci sequence with plots"

**Claude uses:**
```python
create_notebook(title="Fibonacci Analysis")
# Returns notebook ID

write_cell("Fibonacci Sequence Analysis", style="Title")
write_cell("First, let's generate the sequence:", style="Text")
write_cell("fib = RecurrenceTable[{a[n] == a[n-1] + a[n-2], a[1] == 1, a[2] == 1}, a, {n, 20}]",
           style="Input")
write_cell("Plotting the growth:", style="Text")
write_cell("ListPlot[fib, Joined -> True, PlotLabel -> \"Fibonacci Growth\"]", style="Input")

evaluate_selection()  # or evaluate each cell

screenshot_notebook()
# Returns image of the completed notebook
```

### Workflow 3: Debug Mathematical Derivation

**User:** "Check my derivation in the open notebook - are steps 3-5 correct?"

**Claude uses:**
```python
get_notebooks()
# Lists open notebooks

get_cells(notebook="derivation.nb", style="Input")
# Returns list of input cells with IDs and content previews

get_cell_content(cell_id="CellObject[...]")  # for cells 3, 4, 5
# Gets the actual mathematical content

execute_code("Simplify[step3 == step4]")
# Verifies each transition
```

### Workflow 4: Export Research Work

**User:** "Export my analysis notebook to PDF and take a screenshot"

**Claude uses:**
```python
get_notebooks()
# Find the target notebook

save_notebook(notebook="analysis.nb")
# Save current state

export_notebook(path="~/Documents/analysis.pdf", format="PDF")
# Export to PDF

screenshot_notebook(notebook="analysis.nb")
# Returns preview image
```

### Workflow 5: Visualize Mathematical Objects

**User:** "Show me what a torus knot looks like"

**Claude uses:**
```python
rasterize_expression(
    "ParametricPlot3D[{(3 + Cos[3t]) Cos[2t], (3 + Cos[3t]) Sin[2t], Sin[3t]}, {t, 0, 2Pi}]",
    image_size=500
)
# Returns rendered 3D plot image directly
```

### Workflow 6: Batch Cell Evaluation

**User:** "Evaluate all the Input cells in my notebook"

**Claude uses:**
```python
get_cells(style="Input")
# Returns all input cells

# For each cell:
evaluate_cell(cell_id="CellObject[...]")

screenshot_notebook()
# Show the results
```

### Workflow 7: Compare Mathematical Expressions

**User:** "Is sin(2x) equal to 2sin(x)cos(x)?"

**Claude uses:**
```python
execute_code("Simplify[Sin[2x] == 2 Sin[x] Cos[x]]")
# Returns: True

execute_code("""
Plot[{Sin[2x], 2 Sin[x] Cos[x]}, {x, 0, 2 Pi},
  PlotStyle -> {{Blue, Thick}, {Red, Dashed}},
  PlotLegends -> {"Sin[2x]", "2 Sin[x] Cos[x]"}]
""", output_target="notebook")
# Visual confirmation - curves overlap perfectly

screenshot_notebook()
# LLM describes: "Both curves overlap perfectly, confirming the identity"
```

### Workflow 8: Verify Calculation Steps

**User:** "Check if this derivative is correct: d/dx[x^3 e^x] = (3x^2 + x^3)e^x"

**Claude uses:**
```python
execute_code("D[x^3 Exp[x], x]")
# Returns: e^x x^2 (3 + x)

execute_code("Simplify[D[x^3 Exp[x], x] == (3 x^2 + x^3) Exp[x]]")
# Returns: True

execute_code("D[x^3 Exp[x], x]", format="latex")
# Returns: "e^x x^2 (x+3)" in LaTeX
```

---

## Advanced Usage

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHEMATICA_HOST` | `localhost` | Host where Mathematica addon runs |
| `MATHEMATICA_PORT` | `9881` | Port for the socket connection |

### Output Formats

The `execute_code` tool supports three output formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| `text` | Plain text representation | General computation results |
| `latex` | LaTeX formatted output | Mathematical expressions for documents |
| `mathematica` | Mathematica InputForm | Copy-pasteable Wolfram code |

### Output Targets

| Target | Description | Use Case |
|--------|-------------|----------|
| `notebook` (default) | Insert into active notebook and evaluate | Plots, visualizations, any graphics |
| `cli` | Return result as text | Symbolic computations, numerical results |

### Fallback Mode

If the Mathematica addon is not running, some tools (like `execute_code`) will fall back to using `wolframclient` for kernel-only operation. Install the optional dependency:

```bash
pip install wolframclient
```

Note: Fallback mode cannot control notebooks - only execute code.

### Manual Addon Control

In Mathematica:
```mathematica
(* Start the server *)
Get["~/mcp/mathematica-mcp/addon/MathematicaMCP.wl"]
StartMCPServer[]

(* Check status *)
MCPServerStatus[]

(* Restart if needed *)
RestartMCPServer[]

(* Stop the server *)
StopMCPServer[]
```

### Debug Mode

Enable verbose logging in Mathematica:
```mathematica
MathematicaMCP`Private`$MCPDebug = True;
```

---

## Troubleshooting

### "Could not connect to Mathematica addon"

1. Is Mathematica running?
2. Run `MCPServerStatus[]` in Mathematica - does it show `"running" -> True`?
3. Try `RestartMCPServer[]` in Mathematica
4. Check if port 9881 is available: `lsof -i :9881`

### "Timeout waiting for response"

- The computation might be slow. Increase timeout or simplify the request.
- For long computations, break them into smaller steps.

### "Addon loads but server won't start"

- Check for port conflicts: `lsof -i :9881`
- Try a different port:
  ```mathematica
  MathematicaMCP`Private`$MCPPort = 9882;
  StartMCPServer[]
  ```
  Then set `MATHEMATICA_PORT=9882` environment variable.

### Screenshots are blank or wrong size

- Ensure the notebook is visible (not minimized)
- Try scrolling to the target cell first with `scroll_to_cell`
- Use `max_height` parameter to control screenshot size

### Variables not persisting between calls

- This is expected if Mathematica kernel restarts
- Variables persist within a session - verified working in tests
- Use `execute_code` with compound statements if needed

---

## Project Structure

```
mathematica-mcp/
├── README.md
├── pyproject.toml
├── src/
│   └── mathematica_mcp/
│       ├── __init__.py
│       ├── server.py       # MCP server with all tools
│       ├── connection.py   # Socket connection to addon
│       └── session.py      # Kernel fallback (wolframclient)
├── addon/
│   ├── MathematicaMCP.wl   # Main addon package
│   ├── install.wl          # Installation script
│   └── README.md           # Addon-specific docs
└── tests/
    └── ...
```

---

## Compatibility

| Component | Tested Version |
|-----------|----------------|
| Mathematica | 14.1 |
| Python | 3.11+ |
| macOS | ARM64 (Apple Silicon) |
| Claude | Opus 4.5 |
| MCP Protocol | 1.0 |

---

## License

MIT License

---

## Credits

Inspired by [blender-mcp](https://github.com/ahujasid/blender-mcp) for the socket-based addon architecture.

---

*Last tested: January 2026*
