# Mathematica MCP

Full GUI control of Mathematica notebooks and kernel via Model Context Protocol (MCP).

This MCP server allows LLMs (like Claude) to:
- **Create, read, and manipulate notebooks** - Full programmatic access to Mathematica notebooks
- **Execute Wolfram Language code** - Run computations and get results
- **Verify mathematical derivations** - Check if derivation steps are mathematically valid
- **Access Wolfram repositories** - Search and load functions/datasets from Wolfram repositories
- **Take screenshots** - Capture notebook views, cells, or rendered expressions
- **Run long computations** - Submit async jobs for computations >60 seconds

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

**Option B: Manual load in Mathematica (single session)**

```mathematica
Get["~/mcp/mathematica-mcp/addon/MathematicaMCP.wl"]
StartMCPServer[]
```

### Step 3: Configure Your LLM Client

**For Claude Code (`.mcp.json` in project root):**

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
3. **Start your LLM client** (Claude Code, Claude Desktop, etc.)
4. **Ask Claude to interact with Mathematica!**

---

## Available Tools

### Status & Connection

| Tool | Description |
|------|-------------|
| `get_mathematica_status` | Check connection status and system info |
| `get_kernel_state` | Get memory usage, loaded packages, session uptime |
| `get_feature_status` | Show which features are enabled/disabled |

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
| `execute_code` | Run Wolfram Language code |
| `evaluate_selection` | Evaluate currently selected cells |
| `rasterize_expression` | Render expression as image |

### Mathematical Operations (Named Aliases)

| Tool | Description |
|------|-------------|
| `mathematica_integrate` | Compute integrals (definite or indefinite) |
| `mathematica_solve` | Solve equations |
| `mathematica_simplify` | Simplify expressions |
| `mathematica_differentiate` | Compute derivatives |
| `mathematica_expand` | Expand expressions |
| `mathematica_factor` | Factor expressions |
| `mathematica_limit` | Compute limits |
| `mathematica_series` | Taylor/power series expansion |

### Derivation Verification

| Tool | Description |
|------|-------------|
| `verify_derivation` | Check if mathematical derivation steps are valid |

### Symbol Introspection

| Tool | Description |
|------|-------------|
| `resolve_function` | Search for functions by name, get syntax/usage |
| `get_symbol_info` | Get comprehensive symbol information (usage, options, attributes) |

### Wolfram Repository Integration

| Tool | Description |
|------|-------------|
| `search_function_repository` | Search Wolfram Function Repository (2900+ functions) |
| `get_function_repository_info` | Get details about a repository function |
| `load_resource_function` | Load a function from the repository |
| `search_data_repository` | Search Wolfram Data Repository |
| `get_dataset_info` | Get dataset metadata |
| `load_dataset` | Load a dataset from the repository |

### Package Management

| Tool | Description |
|------|-------------|
| `load_package` | Load a Mathematica package |
| `list_loaded_packages` | List all loaded package contexts |

### Long Computation (Async)

| Tool | Description |
|------|-------------|
| `submit_computation` | Submit long-running computation (returns job_id) |
| `poll_computation` | Check status of a submitted job |
| `get_computation_result` | Retrieve result of completed job |

### Expression Caching

| Tool | Description |
|------|-------------|
| `cache_expression` | Evaluate and cache an expression |
| `get_cached` | Retrieve a cached expression |
| `list_cache` | List all cached expressions |
| `clear_expression_cache` | Clear the expression cache |

### Screenshots & Visualization

| Tool | Description |
|------|-------------|
| `screenshot_notebook` | Capture full notebook view |
| `screenshot_cell` | Capture a specific cell |
| `rasterize_expression` | Render any expression as image |

---

## Usage Examples

### Example 1: Verify a Mathematical Derivation

```python
# Check if x² - y² = (x-y)(x+y)
verify_derivation(["x^2 - y^2", "(x-y)*(x+y)"])
# Returns: Step 1 → 2: ✓ VALID

# Check a multi-step derivation with an error
verify_derivation(["Sin[x]^2 + Cos[x]^2", "1", "2"])
# Returns:
#   Step 1 → 2: ✓ VALID (Pythagorean identity)
#   Step 2 → 3: ✗ INVALID (1 ≠ 2)
```

### Example 2: Use Named Math Aliases

```python
# Indefinite integral
mathematica_integrate("x^2", "x")
# Returns: x^3/3

# Definite integral
mathematica_integrate("x^2", "x", "0", "1")
# Returns: 1/3

# Solve equation
mathematica_solve("x^2 - 4 == 0", "x")
# Returns: {{x -> -2}, {x -> 2}}

# Simplify with assumptions
mathematica_simplify("Sqrt[x^2]", assumptions="x > 0")
# Returns: x

# Compute limit
mathematica_limit("Sin[x]/x", "x", "0")
# Returns: 1

# Taylor series
mathematica_series("Exp[x]", "x", "0", 5)
# Returns: 1 + x + x^2/2 + x^3/6 + x^4/24 + O[x]^5
```

### Example 3: Get Symbol Information

```python
# Deep introspection of a function
get_symbol_info("Plot")
# Returns:
# {
#   "symbol": "Plot",
#   "usage": "Plot[f, {x, xmin, xmax}] generates a plot...",
#   "attributes": ["Protected", "ReadProtected"],
#   "options_count": 67,
#   "related_symbols": ["ListPlot", "Plot3D", "ParametricPlot"]
# }
```

### Example 4: Search Wolfram Function Repository

```python
# Search for graph-related functions
search_function_repository("graph algorithms")
# Returns list of community functions like:
#   - RandomGraph
#   - GraphDistance
#   - FindHamiltonianPath

# Load and use a repository function
load_resource_function("RandomGraph")
execute_code('ResourceFunction["RandomGraph"][{10, 20}]')
```

### Example 5: Long Computation Workflow

```python
# Submit a computation that takes >60 seconds
result = submit_computation(
    "FactorInteger[2^256 - 1]",
    name="Large factorization"
)
# Returns: {"job_id": "abc123", "status": "submitted"}

# Poll for completion
poll_computation("abc123")
# Returns: {"status": "running", "elapsed_seconds": 45.2}

# Get result when done
poll_computation("abc123")
# Returns: {"status": "completed", "has_result": true}

get_computation_result("abc123")
# Returns the factorization result
```

### Example 6: Expression Caching

```python
# Cache an expensive computation
cache_expression("my_integral", "Integrate[Sin[x]^10, x]")
# Returns: result + confirmation cached

# Retrieve later (instant, no recomputation)
get_cached("my_integral")
# Returns the cached result

# List all cached expressions
list_cache()
# Returns: {"my_integral": {"access_count": 2, "age_seconds": 120}}
```

### Example 7: Package Management

```python
# Load a package
load_package("Developer`")
# Returns: {"success": true, "new_contexts": ["Developer`"]}

# List loaded packages
list_loaded_packages()
# Returns list of all loaded package contexts
```

### Example 8: Get Kernel State

```python
get_kernel_state()
# Returns:
# {
#   "kernel_version": 14.1,
#   "memory_in_use_mb": 256.4,
#   "session_time": 3600.5,
#   "global_symbol_count": 42,
#   "loaded_packages": ["Developer`", "GeneralUtilities`", ...]
# }
```

### Example 9: Create Documented Notebook

```python
create_notebook(title="Analysis")
write_cell("My Analysis", style="Title")
write_cell("Computing the integral:", style="Text")
write_cell("Integrate[Sin[x]^2, x]", style="Input")

# Evaluate all input cells
cells = get_cells(style="Input")
for cell in cells:
    evaluate_cell(cell_id=cell["id"])

screenshot_notebook()
```

### Example 10: Visualize and Return to LLM

```python
# Render a plot as image (LLM sees it directly)
rasterize_expression("Plot[Sin[x], {x, 0, 2 Pi}]", image_size=500)
# Returns: Image that LLM can see and describe
```

---

## Feature Flags

Control features via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHEMATICA_ENABLE_FUNCTION_REPO` | `true` | Wolfram Function Repository integration |
| `MATHEMATICA_ENABLE_DATA_REPO` | `true` | Wolfram Data Repository integration |
| `MATHEMATICA_ENABLE_ASYNC` | `true` | Async long computation workflow |
| `MATHEMATICA_ENABLE_LOOKUP` | `true` | Symbol lookup/introspection |
| `MATHEMATICA_ENABLE_MATH_ALIASES` | `true` | Named math operation shortcuts |
| `MATHEMATICA_ENABLE_CACHE` | `true` | Expression caching |
| `MATHEMATICA_ENABLE_TELEMETRY` | `false` | Usage telemetry (opt-in) |

Check current status:
```python
get_feature_status()
# Returns all feature flag states
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHEMATICA_HOST` | `localhost` | Host where Mathematica addon runs |
| `MATHEMATICA_PORT` | `9881` | Port for the socket connection |

---

## Output Formats

The `execute_code` tool supports three output formats:

| Format | Description | Example |
|--------|-------------|---------|
| `text` | Plain text | `x^3/3` |
| `latex` | LaTeX formatted | `\frac{x^3}{3}` |
| `mathematica` | InputForm | `x^3/3` |

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
│       ├── session.py      # Kernel fallback (wolframscript)
│       ├── config.py       # Feature flags
│       ├── telemetry.py    # Usage tracking (opt-in)
│       └── cache.py        # Expression caching
├── addon/
│   ├── MathematicaMCP.wl   # Main addon package
│   ├── install.wl          # Installation script
│   └── README.md
└── tests/
```

---

## Troubleshooting

### "Could not connect to Mathematica addon"

1. Is Mathematica running?
2. Run `MCPServerStatus[]` in Mathematica
3. Try `RestartMCPServer[]` in Mathematica
4. Check if port 9881 is available: `lsof -i :9881`

### "Timeout waiting for response"

- Use `submit_computation` for long-running computations
- Break complex computations into smaller steps

### Port conflicts

```mathematica
MathematicaMCP`Private`$MCPPort = 9882;
StartMCPServer[]
```
Then set `MATHEMATICA_PORT=9882` environment variable.

---

## Compatibility

| Component | Tested Version |
|-----------|----------------|
| Mathematica | 14.1 |
| Python | 3.10+ |
| macOS | ARM64 (Apple Silicon) |
| MCP Protocol | 1.0 |

---

## Credits

Inspired by:
- [blender-mcp](https://github.com/ahujasid/blender-mcp) - Socket-based addon architecture
- [texra-ai/mcp-server-mathematica](https://github.com/texra-ai/mcp-server-mathematica) - Derivation verification
- [paraporoco/Wolfram-MCP](https://github.com/paraporoco/Wolfram-MCP) - Named math aliases

---

## License

MIT License

---

*Last updated: January 2026*
