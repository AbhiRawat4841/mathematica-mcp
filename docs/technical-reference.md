# Mathematica MCP

**The most comprehensive Model Context Protocol server for Mathematica** - giving LLMs the same power over Mathematica that they have over Python.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)

> **New to MCP?**
> **[Click here for the Beginner Quick Start Guide](quick-start.md)** to get up and running in 5 minutes!

---

## Why This MCP?

### The Problem

Other Mathematica MCPs treat Mathematica as a **stateless calculator** - each command runs in isolation, variables don't persist, and there's no way to inspect session state. This is like trying to code Python without being able to see your variables or debug errors.

### Our Solution

This MCP gives LLMs **full session control** with persistent state, variable introspection, debugging tools, and natural language integration - making Mathematica as easy to work with as Python.

---

## Key Advantages vs Other Mathematica MCPs

**Why we’re different (at a glance):** Persistent state, full variable introspection, natural language to code, deep debugging, rich file I/O, knowledge/units, and graphics export. Think “Mathematica with Python-like ergonomics” instead of “stateless calculator.”

- Persistent session state: variables/definitions survive across calls via the addon (not stateless evals)
- Full visibility: list/get/clear variables, inspect expressions (Head, depth, size)
- Natural language + Wolfram Alpha: NL→code, Alpha queries, entities, units, constants
- Real debugging: trace evaluation, timing with memory, syntax check, fuzzy function search
- **Error analysis**: Notebook errors are pattern-matched against 10+ error types with confidence-scored fix suggestions for the AI
- File and notebooks: open/run .nb/.wl/.wlnb, read/convert notebooks (MD/LaTeX), outlines, scripts
- **Offline notebook parsing**: Python-native parser extracts clean Wolfram code from complex BoxData without wolframscript
- Data I/O: import/export 250+ formats (CSV/JSON/Excel/URLs), list supported formats
- Knowledge/units: entity lookup (countries/chemicals/planets…), convert_units (thousands), get_constant
- Graphics: export PNG/PDF/SVG/EPS, inspect graphics, compare plots, animations
- GUI control: create/edit notebooks and cells, evaluate, screenshot/rasterize

---

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
1. **Python MCP Server** - Exposes 65+ tools to LLMs via MCP protocol
2. **Mathematica Addon** - Runs inside Mathematica with persistent session state

**Performance:** Notebook execution uses an atomic command that combines notebook lookup, cell creation, and evaluation into a single round-trip (vs. 4 separate calls), resulting in 3-4x faster plot rendering.

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

**Option B: Manual load in Mathematica**

```mathematica
Get["~/mcp/mathematica-mcp/addon/MathematicaMCP.wl"]
StartMCPServer[]
```

### Step 3: Configure Your LLM Client

Use the same MCP server definition across clients; only the config file location differs.

**Claude app (Claude Desktop)**

Config file (macOS): `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

**Claude Code (`.mcp.json` in project root)**

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

**Cursor**

Add the same JSON to your Cursor MCP config (typically `~/.cursor/mcp.json`).

**VSCode**

If you are using an MCP-capable extension (e.g., Continue), add the same MCP server definition in the extension settings.

**OpenCode**

Add the same MCP server definition in your OpenCode MCP config (project or global config). Refer to your OpenCode config location and include `mathematica` under `mcpServers`.

### Make the AI Use It Naturally

To nudge assistants to reach for this MCP without explicit tool calls:

- Add an `agents.md` entry with triggers and example prompts.
- Include a few concrete task examples in your README (below).
- Mention the keywords users are likely to say: Mathematica, Wolfram Language, WolframScript, Wolfram Engine, Wolfram|Alpha, Wolfram Cloud, notebooks, .nb, notebook programming, symbolic computation, algebra, calculus, integrals, derivatives, equation solving, optimization, linear algebra, matrices, plots/graphics/visualization, data import/export, datasets, time series, image processing, signal processing, geodata, units, entities, machine learning, neural networks, graphs, plus action verbs like integrate, differentiate, expand, simplify, factor, solve, plot, calculate, evaluate, optimize, minimize, maximize, sum, series, limit.
- Add intent gating: only activate the MCP when the user asks to compute/evaluate/plot/export/solve or references a concrete target (expression, dataset, notebook file). If it is just casual mention, ask a quick confirmation.

**Suggested prompts (copy/paste for users):**

- "Open this notebook and summarize the methods section."
- "Compute this integral and plot the result in Mathematica."
- "Parse the .nb file and extract only the code."
- "Use Wolfram Alpha to look up the GDP of Japan and convert to USD." 
- "Generate a 3D plot and export it as PNG."
- "Check the steps of this derivation for correctness."

**agents.md template (if your AI tool supports skills/agents metadata):**

```md
## Mathematica MCP

**Purpose**: Use Mathematica for symbolic math, notebooks, plots, and Wolfram knowledge queries.

**Triggers**:
- Mathematica, Wolfram Language, WolframScript, Wolfram Engine
- Wolfram|Alpha, Wolfram Cloud
- notebooks, .nb, notebook programming, notebook parsing, export to Markdown/LaTeX
- symbolic computation, algebra, calculus, integrals, derivatives, limits, equation solving, optimization
- integrate, differentiate, expand, simplify, factor, solve, plot, calculate, evaluate, optimize, minimize, maximize, sum, series, limit
- linear algebra, matrices, tensors
- plots, graphics, visualization, animations
- data import/export, datasets, time series
- image processing, signal processing, geodata
- units, entities, knowledgebase
- machine learning, neural networks, graphs
- BCH, commutators, Lie algebra

**Intent gating**:
- Only activate when the user expresses action intent (compute/evaluate/plot/export/solve) or provides a concrete target (expression, dataset, notebook file).
- If a trigger word appears in casual chat, do not activate the tool; ask for confirmation.

**Example prompts**:
- "Convert this Mathematica notebook to Markdown and summarize key results."
- "Evaluate this Wolfram expression and show the output."
- "Find all cells in the notebook that contain plots."
```

---

## Quick Start

1. **Start Mathematica** (addon auto-starts if installed)
2. **Verify addon is running:** `MCPServerStatus[]`
3. **Start your LLM client**
4. **Ask Claude to interact with Mathematica!**

---

## Complete Tool Reference

### Execution Output Fields

`execute_code` and kernel fallbacks return a structured payload with multiple output representations so clients can choose a stable format for parsing:

- `output`: defaults to InputForm (stable, human-readable)
- `output_inputform`: explicit InputForm string
- `output_fullform`: explicit FullForm string (lossless, machine-friendly)
- `output_tex`: TeXForm string (for rendering)

### TIER 1: Session State & Variable Introspection

*Like Python's `dir()`, `type()`, and `del` - but for Mathematica*

| Tool | Description |
|------|-------------|
| `list_variables` | List all user-defined variables with types and sizes |
| `get_variable` | Get detailed info: value, Head, dimensions, TeX form |
| `set_variable` | Set a variable in the kernel session |
| `clear_variables` | Clear specific variables or all Global` symbols |
| `get_expression_info` | Analyze expression structure: Head, depth, leaf count |
| `get_messages` | Get recent Mathematica messages/warnings |
| `restart_kernel` | Clear all state and restart fresh |

**Examples:**

```python
# Set some variables
set_variable("x", "42")
set_variable("matrix", "{{1,2},{3,4}}")

# List all variables (like Python's dir())
list_variables()
# Returns:
# {
#   "count": 2,
#   "variables": [
#     {"name": "x", "head": "Integer", "bytes": 16, "preview": "42"},
#     {"name": "matrix", "head": "List", "bytes": 232, "preview": "{{1,2},{3,4}}"}
#   ]
# }

# Get detailed variable info (like Python's type() + repr())
get_variable("matrix")
# Returns:
# {
#   "value": "{{1, 2}, {3, 4}}",
#   "head": "List",
#   "dimensions": [2, 2],
#   "tex": "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}"
# }

# Analyze any expression
get_expression_info("Sin[x] + Cos[x]")
# Returns:
# {
#   "head": "Plus",
#   "depth": 3,
#   "leaf_count": 3,
#   "full_form": "Plus[Sin[x], Cos[x]]"
# }

# Clear specific variables
clear_variables(names=["x", "matrix"])

# Clear ALL variables (like restarting Python kernel)
clear_variables(clear_all=True)
```

---

### TIER 2: File Handling (.nb, .wl, .wlnb)

*Work with Mathematica files as easily as Python scripts*

| Tool | Description |
|------|-------------|
| `open_notebook_file` | Open a .nb file in Mathematica frontend |
| `run_script` | Execute a .wl script file (like Python's `exec()`) |
| `read_notebook_content` | Extract cells as structured text |
| `convert_notebook` | Convert to Markdown, LaTeX, or plain text |
| `get_notebook_outline` | Get table of contents (sections/subsections) |

**Examples:**

```python
# Open an existing notebook
open_notebook_file("~/Documents/analysis.nb")
# Returns: {"id": "NotebookObject[...]", "cell_count": 25}

# Run a Wolfram Language script
run_script("~/scripts/setup.wl")
# Returns: {"result": "Setup complete", "timing_ms": 150}
# Any definitions in the script persist in the session!

# Read notebook without opening in GUI
read_notebook_content("~/Documents/analysis.nb", include_outputs=False)
# Returns structured list of all Input/Text cells

# Convert notebook to Markdown for documentation
convert_notebook("~/Documents/analysis.nb", output_format="markdown")
# Returns: "# Analysis\n\n## Introduction\n..."

# Get notebook outline (table of contents)
get_notebook_outline("~/Documents/analysis.nb")
# Returns:
# {
#   "sections": [
#     {"level": "Title", "title": "My Analysis"},
#     {"level": "Section", "title": "Introduction"},
#     {"level": "Subsection", "title": "Background"}
#   ]
# }
```

### Python-Native Notebook Parsing (Offline)

*Parse complex notebooks without wolframscript - works offline*

| Tool | Description |
|------|-------------|
| `parse_notebook_python` | Parse .nb files with Python (markdown/wolfram/outline/json output) |
| `get_notebook_cell` | Get full content of a specific cell by index |

These tools use a pure Python parser that extracts clean, readable Wolfram code from complex BoxData structures. Ideal for:
- Offline notebook analysis (no Mathematica license needed)
- Understanding notebooks with complex mathematical notation
- Extracting executable code from research notebooks
- **Large notebook support**: Automatic truncation prevents timeouts on notebooks with huge symbolic expressions

**Parameters:**

| Tool | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `parse_notebook_python` | `path` | (required) | Path to .nb file |
| | `output_format` | `"markdown"` | Output: markdown, wolfram, outline, json |
| | `truncation_threshold` | `25000` | Max chars per cell before truncation (0 = disable) |
| `get_notebook_cell` | `path` | (required) | Path to .nb file |
| | `cell_index` | (required) | Cell index (0-based) |
| | `full` | `false` | Bypass truncation for complete content |

**Examples:**

```python
# Parse notebook to readable Markdown
parse_notebook_python("~/research/quantum_nmr.nb", output_format="markdown")
# Returns:
# {
#   "success": true,
#   "cell_count": 142,
#   "code_cells": 48,
#   "content": "## Auxiliary functions\n\n### Formatting\n\n```wolfram\nClear[Ix,Iy,Iz];\nIx[j_]:=Subsuperscript[...];\n```\n..."
# }

# Extract only executable Wolfram code
parse_notebook_python("~/research/quantum_nmr.nb", output_format="wolfram")
# Returns pure Wolfram Language code ready to execute:
# (* In[1]:= *)
# Clear[R]
# R[{beta_,phi_}]:=RotationMatrix[phi,{0,0,1}].RotationMatrix[beta,{1,0,0}]...

# Get hierarchical outline
parse_notebook_python("~/research/quantum_nmr.nb", output_format="outline")
# Returns:
# {
#   "sections": [
#     {"level": "Section", "title": "Auxiliary functions", "index": 0},
#     {"level": "Subsection", "title": "Formatting", "index": 1},
#     {"level": "Subsection", "title": "Toggling frame functions", "index": 5}
#   ]
# }

# Get structured cell data (shows truncation info)
parse_notebook_python("~/research/quantum_nmr.nb", output_format="json")
# Returns all cells with: index, style, label, content preview,
# was_truncated (bool), original_length (int)

# Get full content of a specific cell (bypasses truncation)
get_notebook_cell("~/research/quantum_nmr.nb", cell_index=10, full=True)
# Returns complete cell content even if it was truncated in parse output

# Disable truncation for small notebooks (may timeout on large ones)
parse_notebook_python("~/small_notebook.nb", truncation_threshold=0)
```

**Truncation Behavior:**

Cells with content exceeding `truncation_threshold` (default 25KB) are automatically truncated to prevent timeouts on notebooks with extremely large symbolic expressions (e.g., deeply nested BCH commutators). Truncated cells:
- Show a notice in markdown output with the original size
- Include `was_truncated: true` and `original_length` in JSON output
- Can be retrieved in full using `get_notebook_cell(..., full=True)`

**Supported BoxData structures:**
- RowBox, FractionBox, SuperscriptBox, SubscriptBox, SubsuperscriptBox
- SqrtBox, RadicalBox, GridBox (matrices)
- Greek letters: α, β, γ, φ, π, θ, ω, etc.
- Special operators: →, :>, [[, ]], <|, |>, etc.

---

### TIER 3: Natural Language & Knowledge Base

*Ask questions in English, access Wolfram's curated knowledge*

| Tool | Description |
|------|-------------|
| `wolfram_alpha` | Query Wolfram Alpha with natural language |
| `interpret_natural_language` | Convert English to Wolfram code |
| `entity_lookup` | Look up countries, chemicals, planets, etc. |
| `convert_units` | Convert between any units |
| `get_constant` | Get physical/mathematical constants |

**Examples:**

```python
# Ask Wolfram Alpha anything
wolfram_alpha("population of France")
# Returns: "66,438,822 people"

wolfram_alpha("integrate x^2 from 0 to 1")
# Returns: "1/3"

wolfram_alpha("weather in Tokyo")
# Returns: current weather data

# Convert natural language to Wolfram code
interpret_natural_language("the derivative of x squared")
# Returns:
# {
#   "wolfram_code": "D[x^2, x]",
#   "result": "2 x"
# }

# Look up real-world entities
entity_lookup("Country", "Japan", properties=["Population", "Capital", "GDP"])
# Returns:
# {
#   "name": "Japan",
#   "Population": "124,370,947 people",
#   "Capital": "Tokyo",
#   "GDP": "$4.94 trillion"
# }

entity_lookup("Chemical", "Water")
# Returns molecular weight, structure, properties...

entity_lookup("Planet", "Mars")
# Returns orbital period, mass, moons...

# Convert units (1000s of units supported)
convert_units("100 kilometers", "miles")
# Returns: {"result": "62.1371 miles", "numeric": "62.1371"}

convert_units("0 Celsius", "Fahrenheit")
# Returns: {"result": "32 Fahrenheit"}

convert_units("1 lightyear", "kilometers")
# Returns: {"result": "9.461×10^12 kilometers"}

# Get physical constants
get_constant("SpeedOfLight")
# Returns: {"exact": "299792458 m/s", "numeric": "2.998×10^8"}

get_constant("Pi")
# Returns: {"exact": "Pi", "numeric": "3.14159265358979..."}

get_constant("PlanckConstant")
# Returns the value with proper units
```

---

### TIER 4: Debugging & Development Tools

*Debug Mathematica code like a professional*

| Tool | Description |
|------|-------------|
| `trace_evaluation` | Step-by-step evaluation trace (like a debugger) |
| `time_expression` | Measure execution time and memory |
| `check_syntax` | Validate syntax without executing |
| `suggest_similar_functions` | Fuzzy search for function names |

**Examples:**

```python
# Trace how an expression is evaluated (like a debugger)
trace_evaluation("Expand[(x+1)^2]", max_depth=5)
# Returns:
# {
#   "result": "1 + 2x + x^2",
#   "steps": 12,
#   "trace": ["(x+1)^2", "x^2 + 2*x*1 + 1^2", "1 + 2x + x^2"]
# }

# Time an expression (like Python's timeit)
time_expression("Table[Prime[n], {n, 10000}]")
# Returns:
# {
#   "time_seconds": 0.523,
#   "time_ms": 523,
#   "result": "{2, 3, 5, 7, ...}",
#   "memory_delta_bytes": 123456
# }

# Check syntax before running (like a linter)
check_syntax("Plot[Sin[x], {x, 0, 2 Pi}]")
# Returns: {"valid": true, "message": "Valid syntax"}

check_syntax("Plot[Sin[x], {x, 0, 2 Pi")  # Missing ]
# Returns: {"valid": false, "message": "Syntax error in code"}

# Can't remember exact function name?
suggest_similar_functions("Integr")
# Returns:
# {
#   "matches": [
#     {"name": "Integrate", "usage": "Integrate[f, x] gives..."},
#     {"name": "NIntegrate", "usage": "NIntegrate[f, {x, ...}] gives..."},
#     {"name": "FourierIntegral", "usage": "..."}
#   ]
# }
```

---

### TIER 5: Data Import/Export

*Work with 250+ file formats*

| Tool | Description |
|------|-------------|
| `import_data` | Import from file or URL (CSV, JSON, Excel, SQL...) |
| `export_data` | Export expressions to any format |
| `list_supported_formats` | List all 250+ supported formats |

**Examples:**

```python
# Import CSV data
import_data("~/data/sales.csv")
# Returns:
# {
#   "head": "List",
#   "dimensions": [1000, 5],
#   "preview": "{{Date, Product, ...}, ...}"
# }

# Import from URL
import_data("https://example.com/data.json")
# Returns parsed JSON as Mathematica Association

# Import Excel
import_data("~/data/report.xlsx", format="XLSX")

# Export computation results
export_data("Table[{x, Sin[x]}, {x, 0, 2 Pi, 0.1}]", "~/output.csv")
# Returns: {"path": "~/output.csv", "bytes": 1234}

# Export a plot
export_data("Plot[Sin[x], {x, 0, 2 Pi}]", "~/plot.pdf", format="PDF")

# List all supported formats
list_supported_formats()
# Returns:
# {
#   "import_formats": ["CSV", "JSON", "XLSX", "PDF", "PNG", ...],  # 256 formats
#   "export_formats": ["CSV", "JSON", "PDF", "SVG", "HTML", ...]   # 264 formats
# }
```

---

### TIER 6: Visualization & Graphics

*Create, analyze, and export graphics*

| Tool | Description |
|------|-------------|
| `export_graphics` | Export plots to PNG, PDF, SVG, EPS |
| `inspect_graphics` | Analyze graphics structure |
| `compare_plots` | Side-by-side plot comparison |
| `create_animation` | Generate animated plots |
| `rasterize_expression` | Render any expression as image |

**Examples:**

```python
# Export a plot as PNG
export_graphics(
    "Plot[Sin[x], {x, 0, 2 Pi}]",
    "~/plot.png",
    format="PNG",
    size=800
)
# Returns: {"path": "~/plot.png", "bytes": 16425}

# Export as vector graphics
export_graphics(
    "Plot3D[Sin[x y], {x, -3, 3}, {y, -3, 3}]",
    "~/surface.svg",
    format="SVG"
)

# Inspect graphics structure
inspect_graphics("Plot[Sin[x], {x, 0, 2 Pi}]")
# Returns:
# {
#   "head": "Graphics",
#   "primitives": ["Line", "Directive"],
#   "plot_range": "{{0, 6.28}, {-1, 1}}",
#   "options": {...}
# }

# Compare multiple plots side-by-side
compare_plots(
    ["Plot[Sin[x], {x,0,2Pi}]", "Plot[Cos[x], {x,0,2Pi}]"],
    labels=["Sine", "Cosine"]
)
# Returns combined GraphicsRow expression

# Create animation
create_animation(
    "Plot[Sin[n*x], {x, 0, 2Pi}]",
    parameter="n",
    range_spec="1, 5",
    frames=20
)
# Returns animation expression (can export as GIF)
```

---

### Core Tools (Existing)

#### Notebook Management

| Tool | Description |
|------|-------------|
| `get_notebooks` | List all open notebooks |
| `get_notebook_info` | Get details about a notebook |
| `create_notebook` | Create a new notebook |
| `save_notebook` | Save notebook to disk |
| `close_notebook` | Close a notebook |
| `export_notebook` | Export to PDF, HTML, TeX |

#### Cell Operations

| Tool | Description |
|------|-------------|
| `get_cells` | List cells in a notebook |
| `get_cell_content` | Get cell content |
| `write_cell` | Insert a new cell |
| `delete_cell` | Remove a cell |
| `evaluate_cell` | Evaluate a cell |

#### Code Execution

| Tool | Description |
|------|-------------|
| `execute_code` | Run Wolfram Language code |
| `evaluate_selection` | Evaluate selected cells |

#### Mathematical Operations

| Tool | Description |
|------|-------------|
| `mathematica_integrate` | Compute integrals |
| `mathematica_solve` | Solve equations |
| `mathematica_simplify` | Simplify expressions |
| `mathematica_differentiate` | Compute derivatives |
| `mathematica_expand` | Expand expressions |
| `mathematica_factor` | Factor expressions |
| `mathematica_limit` | Compute limits |
| `mathematica_series` | Taylor series |

#### Verification & Introspection

| Tool | Description |
|------|-------------|
| `verify_derivation` | Verify mathematical derivation steps |
| `resolve_function` | Search for functions by name |
| `get_symbol_info` | Get symbol documentation |

#### Repository Integration

| Tool | Description |
|------|-------------|
| `search_function_repository` | Search 2900+ community functions |
| `load_resource_function` | Load a repository function |
| `search_data_repository` | Search curated datasets |
| `load_dataset` | Load a dataset |

#### Async & Caching

| Tool | Description |
|------|-------------|
| `submit_computation` | Submit long-running computation |
| `poll_computation` | Check job status |
| `get_computation_result` | Get completed result |
| `cache_expression` | Cache expensive computations |
| `get_cached` | Retrieve cached result |

---

## Real-World Use Cases

### Use Case 1: Data Science Workflow

```python
# Import your data
import_data("~/data/experiment.csv")

# Analyze it
execute_code("""
data = Import["~/data/experiment.csv"];
{Mean[data], StandardDeviation[data], Correlation[data]}
""")

# Visualize
export_graphics(
    "ListPlot[data, PlotLabel -> \"Experiment Results\"]",
    "~/results.png"
)

# Export processed data
export_data("processedData", "~/processed.json", format="JSON")
```

### Use Case 2: Teaching Mathematics

```python
# Ask a natural language question
wolfram_alpha("solve quadratic equation x^2 + 5x + 6 = 0")

# Show step-by-step
trace_evaluation("Solve[x^2 + 5x + 6 == 0, x]")

# Create visual explanation
export_graphics(
    "Plot[x^2 + 5x + 6, {x, -5, 0}, Epilog -> {Red, PointSize[Large], Point[{-2, 0}], Point[{-3, 0}]}]",
    "~/quadratic.png"
)
```

### Use Case 3: Physics Calculations

```python
# Get physical constants
get_constant("GravitationalConstant")
get_constant("SpeedOfLight")

# Unit conversions
convert_units("1 astronomical unit", "kilometers")

# Entity data
entity_lookup("Planet", "Jupiter", ["Mass", "Radius", "OrbitalPeriod"])

# Symbolic calculation
execute_code("Solve[F == G m1 m2 / r^2, r]")
```

### Use Case 4: Document Processing

```python
# Read a notebook
outline = get_notebook_outline("~/research/paper.nb")

# Convert to LaTeX for publication
convert_notebook("~/research/paper.nb", output_format="latex")

# Extract just the code
read_notebook_content("~/research/paper.nb", include_outputs=False)
```

---

## Error Detection and Analysis

The Mathematica MCP includes error analysis for notebook execution that pattern-matches errors and provides fix suggestions to help the AI understand and resolve common issues.

### How It Works

When code is executed in notebook mode (`execute_code` with `output_target="notebook"`):

1. **Error Capture**: The addon captures messages from `$MessageList` after cell evaluation
2. **Pattern Matching**: Captured errors are matched against a knowledge base of 10+ common error patterns
3. **Confidence Scoring**: Matches are scored as high/medium/low confidence
4. **Suggestion Generation**: Actionable fixes are suggested based on the error type
5. **LLM Formatting**: Errors are formatted with analysis, causes, fixes, and examples

**Note**: Error analysis is currently active for notebook execution only. CLI mode (`output_target="cli"`) returns raw errors without analysis.

### Supported Error Types

| Error Pattern | Description | Example Fix |
|---------------|-------------|-------------|
| `UnitConvert::compat` | Incompatible units | Use `QuantityMagnitude[]` to extract numeric values |
| `Part::partw` | Index out of range | Check with `Length[list]` before accessing |
| `Part::partd` | Structure depth issue | Verify with `Dimensions[]` or `Depth[]` |
| `Syntax::sntxi` | Syntax error | Check matching brackets and operators |
| `Syntax::tsntxi` | Extra input | Remove extra characters or terminate properly |
| `Divide::infy` | Division by zero | Add domain checks or use `Limit[]` |
| `Power::infy` | Invalid power operation | Use `Assuming[]` to constrain domain |
| `Set::write` | Protected symbol | Choose different variable names |
| `Recursion::reclim` | Recursion limit exceeded | Add base case or increase `$RecursionLimit` |
| `General::stop` | Output suppression | Fix underlying repeated errors |

### Example Error Analysis

**Code with error:**
```mathematica
CountryData["USA", "GDP"] + Quantity[1, "Hours"]
```

**Error Analysis Output:**
```
============================================================
EXECUTION ERRORS DETECTED
============================================================

Summary: Found 1 error(s) and 0 warning(s)

--- ERRORS ---

UnitConvert::compat
  Incompatible units: USD and Hours

  Analysis: Incompatible units error
  Likely cause: Attempting to convert between incompatible unit types
  Suggested fix: Use QuantityMagnitude[] to extract numeric values
  Example: QuantityMagnitude[CountryData["USA", "GDP"], "USDollars"]

--- RECOMMENDATIONS ---
High-confidence fixes available for the following errors:
  • UnitConvert::compat: Use QuantityMagnitude[]
TIP: When working with Entity data, use QuantityMagnitude[]
============================================================
```

### API Usage

The error analyzer is automatically invoked during notebook code execution. It can also be used directly for testing or analysis:

```python
from src.mathematica_mcp.error_analyzer import (
    analyze_error,
    analyze_messages,
    format_error_for_llm
)

# Analyze a single error
result = analyze_error("UnitConvert::compat", "Incompatible units")
print(result['suggested_fix'])

# Analyze multiple messages
messages = [
    {'tag': 'Part::partw', 'text': 'Part out of range', 'type': 'error'}
]
analysis = analyze_messages(messages)

# Format for LLM
formatted = format_error_for_llm(messages, code)
```

### Testing

Comprehensive test suite available in `tests/test_error_detection.py`:
- 33 test cases covering all error pattern types
- Tests for error analyzer module (pattern matching, message analysis, LLM formatting)
- Tests using inline execution to trigger real Mathematica errors
- Pattern validation and real-world scenario tests

Run tests:
```bash
python -m pytest tests/test_error_detection.py -v
```

---

## Feature Flags

Control features via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHEMATICA_ENABLE_FUNCTION_REPO` | `true` | Function Repository integration |
| `MATHEMATICA_ENABLE_DATA_REPO` | `true` | Data Repository integration |
| `MATHEMATICA_ENABLE_ASYNC` | `true` | Async computation workflow |
| `MATHEMATICA_ENABLE_LOOKUP` | `true` | Symbol lookup/introspection |
| `MATHEMATICA_ENABLE_MATH_ALIASES` | `true` | Named math operations |
| `MATHEMATICA_ENABLE_CACHE` | `true` | Expression caching |
| `MATHEMATICA_ENABLE_TELEMETRY` | `false` | Usage telemetry |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHEMATICA_HOST` | `localhost` | Addon host |
| `MATHEMATICA_PORT` | `9881` | Addon port |

---

## Troubleshooting

### "Could not connect to Mathematica addon"

1. Is Mathematica running?
2. Run `MCPServerStatus[]` in Mathematica
3. Try `RestartMCPServer[]`
4. Check port: `lsof -i :9881`

### "Address already in use" on port 9881

This happens when a zombie kernel process is holding the port, even after restarting Mathematica:

```bash
# Find the process using port 9881
lsof -i :9881

# Kill the zombie process (use the PID from above)
kill -9 <PID>
```

Common causes:
- A `WolframKernel` process from `wolframclient` that didn't terminate properly
- A previous MCP server session that crashed without cleanup

After killing the process, run `StartMCPServer[]` in Mathematica again.

### "Timeout waiting for response"

- Use `submit_computation` for long operations
- Break complex computations into steps

### Variables not persisting

- Ensure you're using the **addon connection** (not wolframscript fallback)
- Check `get_mathematica_status()` shows `connection_mode: "addon"`

### Output parsing issues (older Mathematica versions)

The MCP server uses `ExportString[..., "RawJSON"]` for reliable JSON output from Mathematica. For older versions that don't support this, a robust regex-based fallback parser handles Association output with properly escaped quotes.

---

## Compatibility

| Component | Tested Version |
|-----------|----------------|
| Mathematica | 14.1 |
| Python | 3.10+ |
| macOS | ARM64 (Apple Silicon) |
| MCP Protocol | 1.0 |

---

## Project Structure

```
mathematica-mcp/
├── src/mathematica_mcp/
│   ├── server.py          # 65+ MCP tools
│   ├── notebook_parser.py # Python-native .nb parser (offline)
│   ├── connection.py      # Socket connection to addon
│   ├── session.py         # Kernel fallback (wolframscript)
│   ├── config.py          # Feature flags
│   ├── cache.py           # Expression caching
│   └── telemetry.py       # Usage tracking
├── addon/
│   ├── MathematicaMCP.wl   # Main addon (persistent session)
│   ├── install.wl          # Auto-install script
│   └── README.md
└── tests/
    ├── test_session.py              # 49 tests: session, parsing, math operations
    └── test_derivation_verification.py  # 14 tests: algebraic/trig identity verification
```

---

## License

MIT License

---

*Last updated: January 2026 (v1.3 - Performance optimization: atomic notebook execution reduces 4 round-trips to 1; Graphics rendering fix for proper plot display)*
