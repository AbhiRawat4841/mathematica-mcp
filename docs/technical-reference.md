# Mathematica MCP Technical Reference

**Advanced documentation for developers and power users.**

> **Just want to get started?**
> **Start with the installation steps in the main README.**

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
1. **Python MCP Server** - Exposes MCP tools to LLMs: 12 tools in the default `lean` profile (11 consolidated dispatchers + `verify_derivation`), up to ~82 in `classic` (varies by profile and feature flags)
2. **Mathematica Addon** - Runs inside Mathematica with persistent session state

**Performance:** Notebook execution uses an atomic command that combines notebook lookup, cell creation, and evaluation into a single round-trip (vs. 4 separate calls), reducing latency from multiple socket round-trips to one.

**Warm compute funnel:** When the addon is unavailable, kernel evaluation goes through a persistent `WolframLanguageSession` (`session.py`); a cold `wolframscript` subprocess is only a flagged fallback, counted per process and surfaced in the lean `status()` tool. An idle reaper shuts the kernel down after `MATHEMATICA_KERNEL_IDLE_TIMEOUT` seconds of inactivity (default 1800; `0` disables), and the session is closed via `atexit` on interpreter shutdown. At server startup a background daemon thread prewarms this session (`MATHEMATICA_PREWARM`, default on) so the ~13s boot overlaps setup rather than blocking the first warm call (a call landing inside the boot window takes the addon rung below when eligible, or waits for the boot to finish otherwise). During that boot window (or whenever the warm session is unavailable), an opt-in middle rung lets *kernel-independent* pure-math evaluations (`get_constant`, `convert_units`, `verify_derivation`) run on the already-connected addon kernel (~30ms round trip, `execution_method='addon'`) instead of a cold subprocess. Kernel-identity-sensitive calls (variables, packages, kernel state) never take this rung - the addon is the user's front-end kernel - and any parse ambiguity falls through to the always-correct cold path.

**Protocol:** The Python client and the addon share a `protocol_version` handshake (currently `4`); a version skew (e.g. addon installed in `init.m` not updated after `pip install --upgrade`) is detected and reported. Success responses of notebook-touching addon commands carry a `state_delta` (the command's **target** notebook, its cell count, and `kernel_busy` via `CurrentValue[nb, Evaluating]`); pure-kernel commands skip it so trivial `execute_code` responses stay fast. Note that `kernel_busy` cannot observe a *front-end* evaluation in progress (`CurrentValue[nb, Evaluating]` does not resolve from the socket handler - see the preemptive-vs-main-link section); rely on the frontend `evaluation_pending` / `evaluation_complete` contract instead.

**Error guidance:** Failed evaluations carry a structured `error_analysis` (`suggested_fix`, `next_step`, and a `retry_with` corrected call when one can be derived from context) on all evaluate paths; see [Error Detection and Analysis](#error-detection-and-analysis).

**Mathematica 15:** On kernels ≥15, agent-created notebooks (`notebooks(action="create")` / `create_notebook`) set `ShowChatbar -> False` so the built-in AI sidebar does not open in automated notebooks; pass `show_chatbar=True` to keep it. 14.x is supported behind `$VersionNumber >= 15.` guards in the addon (`mcpVersionAtLeast15[]`); set `MMCP_FORCE_V14=1` in the Mathematica kernel's environment to force the pre-15 branches for testing (see [V14_VALIDATION.md](../V14_VALIDATION.md)).

**Security:** See [SECURITY.md](../SECURITY.md) for the full threat model, permissions matrix, and vulnerability reporting process.

---

## Tool Profiles

The server supports four profiles (plus `full` as an alias of `classic`) that control which tools are exposed. This lets you tune the tool surface for your use case, reducing noise for LLMs that don't need notebook or legacy features.

| Profile | Tools | Use Case |
|---------|-------|----------|
| `lean` (default) | 12 | Consolidated tools, ~11.5 KB (~2.9k tokens) of schema (measure: `uv run python benchmarks/profile_surface.py`); extend with `MATHEMATICA_TOOLSETS` |
| `classic` | ~82 | The complete pre-1.0 surface (alias: `full`) - legacy, admin, and all optional tools |
| `math` | ~28 | Pure computation, no notebook tools |
| `notebook` | ~48 | Computation + notebook reading/management + `create_notebook` |

> **Changed in 1.0:** the default profile is `lean` (was `full`). Set `MATHEMATICA_PROFILE=classic` to keep the pre-1.0 surface. See the [Migration Guide](MIGRATION.md).

### Selecting a Profile

**Via CLI flag** (recommended for setup):
```bash
uvx mathematica-mcp-full --profile notebook
uvx mathematica-mcp-full setup claude-desktop --profile math
```

**Via environment variable**:
```bash
export MATHEMATICA_PROFILE=notebook
```

**Via MCP client config**:
```json
{
  "mcpServers": {
    "mathematica": {
      "command": "/ABSOLUTE/PATH/TO/uvx",
      "args": ["mathematica-mcp-full"],
      "env": {"MATHEMATICA_PROFILE": "notebook"}
    }
  }
}
```

The setup command resolves `uv` and `uvx` to absolute paths automatically so GUI clients do not depend on shell `PATH` inheritance.

### What Each Profile Includes

| Profile | Tool Groups |
|---------|-------------|
| `lean` | 12 tools: 11 consolidated action-enum tools (`@_tool("lean")`) + `verify_derivation` (shared `moat` group); opt-in extras via `MATHEMATICA_TOOLSETS` (data_io, graphics_plus, cloud, debug, notebook_files, notebook_edit, symbols, math_aliases, repository, async_jobs, cache) |
| `classic` / `full` | Everything: core, session, knowledge, debug, kernel_tools, notebook_primary, notebook_advanced, file_legacy, data, graphics, admin + all optional tool groups (math aliases, repository, async, cache) |
| `math` | core, session, knowledge, debug, moat (`verify_derivation`), kernel_tools + symbol lookup |
| `notebook` | Everything in math + notebook_primary, data, graphics |

### Lean Opt-in Toolsets

Add extra tool groups to the lean profile with `MATHEMATICA_TOOLSETS` (comma-separated). They can only enable tools, never remove the 12 lean core tools:

```bash
export MATHEMATICA_TOOLSETS=data_io,graphics_plus,cloud,debug
```

| Name | Adds |
|------|------|
| `data_io` | Data import/export tools. |
| `graphics_plus` | Graphics inspection, export, plot comparison, animation. |
| `cloud` | Wolfram Alpha, natural-language interpretation, entities, units, constants. |
| `debug` | Trace / timing / journal tools. |
| `notebook_files` | Legacy notebook file tools. |
| `notebook_edit` | Advanced notebook editing tools. |
| `symbols` | Symbol lookup / documentation tools. |
| `math_aliases` | `mathematica_integrate`, `mathematica_solve`, etc. |
| `repository` | Function + data repository search. |
| `async_jobs` | Async computation submit/poll. |
| `cache` | Expression cache management tools. |

Feature flags (environment variables) can further enable or disable individual tool groups regardless of profile. See [Feature Flags](#feature-flags) below.

Use `get_feature_status()` (classic/math/notebook profiles) or `status()` (lean) to inspect the active profile and enabled features at runtime.

---

## LLM Guidance System

The server includes a multi-layer guidance system (`src/mathematica_mcp/guidance.py`) that ensures agents use MCP tools directly instead of falling back to shell commands or manual file operations. **All layers are profile-aware**: every guidance surface (server instructions, prompts, project guidance files) is generated from the active profile's `FeatureFlags`, so the default `lean` profile speaks the 12-tool lean vocabulary (`evaluate`, `notebooks`, `cells`, `kernel`, `vars`, `guide`, ...) while the `classic` profile keeps the legacy `execute_code(style=...)` layers.

### Layer 1: Server Instructions (all clients)

The FastMCP `instructions` field is sent to every connected client. It establishes the core operating model:
1. **MCP-first**: Always use MCP tools, never `wolframscript`, shell commands, or manual `.nb` file creation.
2. **Profile-aware notebook model**: Notebook guidance appears only when notebook tools are exposed; a "notebook" is a live frontend window, not a file on disk.
3. **Quick defaults** (lean): prefer one compound Wolfram expression over several sequential `evaluate` calls; reuse `session_id`; use `batch(ops)` for multi-call round-trips. (classic): prefer the right `style`, keep `sync="none"`, and consider `response_detail="compact"` (alias: `"short"`) for long multi-step flows.
4. **Recovery defaults** (lean): `status()`, `kernel(action="messages")`, and `guide(topic="errors")`. (classic): `get_session_brief()`, `get_computation_journal()`, and `get_messages()`.

### Layer 2: Intent Keywords

Users can steer routing with natural language keywords. What the keywords map to depends on the profile.

**Lean profile (default):**

| Keyword | Tool Call |
|---------|-----------|
| "calculate", "compute", "solve", "evaluate" | `evaluate(code)` - kernel, result in chat |
| "plot", "show", "in notebook" | `evaluate(code, target="notebook")` - in the live notebook |
| "new notebook" | `notebooks(action="create", title=...)` then `evaluate(code, target="notebook")` |
| "verify", "check derivation" | `verify_derivation(steps)` |

**Classic profile:** tool callers can use the `style` parameter directly. `style` is a high-level preset that bundles `output_target` + `mode`; individual parameters still work and override style.

| Keyword | Style | Tool Call |
|---------|-------|-----------|
| "calculate", "compute", "what is", "evaluate", "solve" | `"compute"` | `execute_code(style="compute")` |
| "plot", "show", "graph", "visualize", "in notebook" | `"notebook"` | `execute_code(style="notebook")` |
| "new notebook", "fresh notebook", "create notebook" | N/A | `create_notebook()` then `execute_code(style="notebook")` |
| "interactive", "manipulate", "slider", "dynamic", "animate" | `"interactive"` | `execute_code(style="interactive")` |

> **Note:** There is no `style="new_notebook"`. Creating a fresh notebook is a two-step workflow: `create_notebook(title="...")` then `execute_code(style="notebook")` (classic), or `notebooks(action="create")` then `evaluate(..., target="notebook")` (lean).

### Layer 3: Tool Docstrings

- **Lean**: the 12 tools ship short (≤200 char) action-enum docstrings that are *not* rewritten at startup; deeper guidance is available on demand via `guide(topic="workflow" | "errors" | "notebook_hygiene" | "screenshots" | "v15" | "profiles" | "toolsets" | "batch")`.
- **Classic**: tool descriptions are rewritten at startup to include `[PRIMARY]`, `[ADVANCED]`, or `[LEGACY]` labels, steering LLMs toward the preferred tool for each task. `create_notebook` is marked as the correct tool when the user explicitly requests a new notebook.

### Layer 4: MCP Prompts

Six MCP prompts (`mathematica_expert`, `calculate`, `notebook`, `new_notebook`, `interactive`, `quickstart`) can be surfaced by clients that support MCP prompts. Their text is profile-conditioned: in lean they instruct `evaluate(code)` / `notebooks(action="create")`; in classic they instruct `execute_code(style=...)` / `create_notebook()`.

### Layer 5: Project Guidance Files

Client-specific project guidance is installed via `--project-dir` and rendered for the active profile:
- **Claude Code**: `.claude/commands/mathematica.md` + `CLAUDE.md` hint block
- **Codex**: `AGENTS.md` with additive project guidance: MCP-first rules, keyword table, notebook-file routing, and workflow examples

```bash
uvx mathematica-mcp-full setup claude-code --project-dir .
uvx mathematica-mcp-full setup codex --project-dir .
```

The layering is intentional: server instructions carry universal defaults and anti-patterns; project guidance files stay shorter and add only client-specific formatting, notebook-file routing, and examples.

---

## Notebook Backend Abstraction

Notebook reading is handled by a capability-based dispatch system (`notebook_backend.py`) that selects the best available backend:

| Backend | Name | Requires | Strengths |
|---------|------|----------|-----------|
| Kernel Semantic | `kernel_semantic` | `wolframscript` or `wolframclient` | Accurate code extraction via `NotebookImport` |
| Python Syntax | `python_syntax` | Nothing (offline) | Fast, no dependencies, handles BoxData |

The kernel backend runs through the warm persistent kernel session (`session.py` / `execute_in_kernel`), not a bare `wolframscript` subprocess - a cold `wolframscript` call is only a flagged-and-counted fallback (see `status()`'s cold-execution count in the lean profile).

### `read_notebook` (Primary Tool)

The consolidated `read_notebook` tool (classic/`notebook` profiles) replaces individual notebook reading tools; the lean profile exposes the same dispatch as `read_notebook_file(path, mode=...)`:

```python
read_notebook("path/to/notebook.nb", output_format="markdown")
read_notebook("path/to/notebook.nb", output_format="wolfram")   # executable code only
read_notebook("path/to/notebook.nb", output_format="outline")   # section hierarchy
read_notebook("path/to/notebook.nb", output_format="json")      # structured cell data
```

Parameters include `backend` (force a specific backend), `view` (semantic/display/raw), `cell_types` filter, `include_outputs`, `truncation_threshold`, and `include_alternates`.

The legacy tools (`read_notebook_content`, `convert_notebook`, `get_notebook_outline`, `parse_notebook_python`, `get_notebook_cell`) remain available in the `classic`/`full` profile (or via `MATHEMATICA_TOOLSETS=notebook_files` in lean) but their docstrings steer LLMs toward `read_notebook`.

Results are cached to disk at `~/.cache/mathematica-mcp/notebooks/` and invalidated when the source file changes.

---

## Installation

Installation lives in one place: the **[Installation Guide](installation.md)** (one-command setup via `uvx mathematica-mcp-full setup <client>`, manual setup, client configs for Claude Desktop / Claude Code / Cursor / VS Code / Codex / Gemini, and troubleshooting). Verify an install with `uvx mathematica-mcp-full doctor`.

---

## Complete Tool Reference

### Lean Profile Tools (default)

The default `lean` profile exposes exactly 12 tools. Eleven are consolidated action-enum dispatchers over the same internals as the classic tools; the twelfth is `verify_derivation`, shared with all profiles.

| Tool | Actions / Key Params | Description |
|------|---------------------|-------------|
| `status()` | - | Connection, kernel version, profile, features, warm-path health (cold-execution count, idle timeout) |
| `evaluate` | `target=kernel\|notebook\|cell\|selection`, `code`, `file`, `dry_run`, `timeout` | Primary execution tool. `dry_run=True` checks syntax; `file=` runs a `.wl` script |
| `notebooks` | `action=list\|info\|create\|open\|save\|close\|export`, `format` (save: `Notebook\|PDF\|HTML\|TeX`; export: `PDF\|HTML\|TeX\|Markdown`), `show_chatbar` | Notebook lifecycle management; on Mathematica ≥15 created notebooks suppress the chat sidebar unless `show_chatbar=True` |
| `cells` | `action=list\|read\|select\|scroll` | Read/navigate notebook cells (supports `style`/`offset`/`limit` filters) |
| `edit_cells` | `action=write\|delete` | Write or delete notebook cells |
| `screenshot` | `scope=notebook\|cell\|expression` | Capture a PNG |
| `kernel` | `action=state\|messages\|restart\|load_package\|packages\|inspect` | Kernel admin and message inspection |
| `vars` | `action=list\|get\|set\|clear\|clear_all` | Kernel variable management (bare `clear` requires `name=` or `pattern=`) |
| `read_notebook_file` | `mode=markdown\|wolfram\|outline\|json\|plain` | Read a `.nb`/`.wl` file from disk |
| `guide` | `topic=workflow\|errors\|notebook_hygiene\|screenshots\|v15\|profiles\|toolsets\|batch` | On-demand guidance |
| `batch` | `ops=[{command, params}, ...]` | Run multiple addon ops in one round-trip |
| `verify_derivation` | `steps` | Verify mathematical derivation steps |

Long outputs from lean tools are paginated: responses over the output cap (`MATHEMATICA_MAX_OUTPUT_CHARS`, default 4000) return a first page plus a `next_cursor`; pass `cursor=` back to the same tool to fetch the remainder.

### Classic Profile Tools

Everything below this point documents the `classic`/`full` surface (~82 tools). The `math` and `notebook` profiles expose curated subsets, and lean can opt individual groups back in via `MATHEMATICA_TOOLSETS`.

### Execution Output Fields

`execute_code` compute-path responses return a structured payload with multiple output representations so clients can choose a stable format for parsing:

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
| `check_syntax` | Validate syntax without executing (core group; in lean use `evaluate(code, dry_run=True)`) |
| `suggest_similar_functions` | Fuzzy search for function names (symbol-lookup feature, not the `debug` toolset; enable via `MATHEMATICA_TOOLSETS=symbols`) |

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
#   "import_formats": ["CSV", "JSON", "XLSX", "PDF", "PNG", ...],  # 250+ formats (kernel-reported, version-dependent)
#   "export_formats": ["CSV", "JSON", "PDF", "SVG", "HTML", ...]   # 250+ formats (kernel-reported, version-dependent)
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

#### Server & Kernel Status

| Tool | Description |
|------|-------------|
| `get_mathematica_status` | Connection status, kernel/frontend version, and system info |
| `get_kernel_state` | Memory usage, uptime, version, loaded packages |
| `get_feature_status` | Show active profile, enabled tool groups, and feature flags |
| `get_session_brief` | Compact session state summary: connection mode, recent errors, routing advice |
| `get_computation_journal` | Recent computation history (in-memory ring buffer) |
| `clear_computation_journal` | Reset the computation journal |

#### Notebook Management

| Tool | Description |
|------|-------------|
| `get_notebooks` | List all open notebooks |
| `get_notebook_info` | Get details about a notebook |
| `create_notebook` | Create a new live notebook window (use when user says "new notebook") |
| `save_notebook` | Save notebook to disk |
| `close_notebook` | Close a notebook |
| `export_notebook` | Export to PDF, HTML, TeX |
| `read_notebook` | Read notebook with backend-aware dispatch (see [Notebook Backend](#notebook-backend-abstraction)) |
| `screenshot_notebook` | Capture entire notebook as PNG image |
| `screenshot_cell` | Capture a single cell's output as PNG image |

#### Cell Operations

| Tool | Description |
|------|-------------|
| `get_cells` | List cells in a notebook |
| `get_cell_content` | Get cell content |
| `write_cell` | Insert a new cell |
| `delete_cell` | Remove a cell |
| `evaluate_cell` | Evaluate a cell |
| `select_cell` | Move cursor to a cell |
| `scroll_to_cell` | Scroll view to show a cell |

#### Code Execution

| Tool | Description |
|------|-------------|
| `execute_code` | Run Wolfram Language code (primary tool: use `style="compute"`, `"notebook"`, or `"interactive"`) |
| `evaluate_selection` | Evaluate selected cells |

#### Kernel Tools

| Tool | Description |
|------|-------------|
| `load_package` | Load Wolfram packages (e.g., `"Developer\`"`) |
| `list_loaded_packages` | Show all active contexts |

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
| `search_function_repository` | Search thousands of community functions |
| `get_function_repository_info` | Get details about a repository function |
| `load_resource_function` | Load a repository function |
| `search_data_repository` | Search curated datasets |
| `get_dataset_info` | Get metadata about a dataset |
| `load_dataset` | Load a dataset |

#### Admin (classic/full profile only)

| Tool | Description |
|------|-------------|
| `batch_commands` | Execute multiple addon commands in a single round-trip |

#### Async & Caching

| Tool | Description |
|------|-------------|
| `submit_computation` | Submit long-running computation |
| `poll_computation` | Check job status |
| `get_computation_result` | Get completed result |
| `cache_expression` | Cache expensive computations |
| `get_cached` | Retrieve cached result |
| `list_cache` | List all cached expressions |
| `clear_expression_cache` | Clear expression cache |

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

When code is executed in notebook mode (`execute_code` with `style="notebook"` or `style="interactive"` in classic; `evaluate(code, target="notebook")` in lean):

1. **Error Capture**: The addon captures messages from `$MessageList` after cell evaluation without discarding valid message-bearing results such as `ComplexInfinity`
2. **Pattern Matching**: Captured errors are matched against a knowledge base of 10+ common error patterns
3. **Confidence Scoring**: Matches are scored as high/medium/low confidence
4. **Suggestion Generation**: Actionable fixes are suggested based on the error type; when a corrected call can be derived from the failing input, a runnable `retry_with` is included (otherwise it is `null` rather than a canned template), and a `next_step` names the literal follow-up call (e.g. `kernel(action='messages')`)
5. **LLM Formatting**: Errors are formatted with analysis, causes, fixes, and examples

**Note**: Error analysis attaches on **all** evaluate paths (notebook, kernel/CLI, cell, selection). Notebook requests do not silently reroute to CLI when the notebook transport fails; they return a notebook-targeted error.

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
from mathematica_mcp.error_analyzer import (
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

Control features via environment variables. Defaults shown are for the `classic`/`full` profile. The default `lean` profile turns **all** optional features off except the internal expression cache - extend it with `MATHEMATICA_TOOLSETS` (e.g. `symbols`, `repository`, `async_jobs`, `cache`) rather than these flags; `math` and `notebook` default most optional features to `false` (only `LOOKUP` and `CACHE` are enabled). Environment variables override profile defaults.

| Variable | Default (`classic`) | Description |
|----------|------------------|-------------|
| `MATHEMATICA_ENABLE_FUNCTION_REPO` | `true` | Function Repository integration |
| `MATHEMATICA_ENABLE_DATA_REPO` | `true` | Data Repository integration |
| `MATHEMATICA_ENABLE_ASYNC` | `true` | Async computation workflow |
| `MATHEMATICA_ENABLE_LOOKUP` | `true` | Symbol lookup/introspection |
| `MATHEMATICA_ENABLE_MATH_ALIASES` | `true` | Named math operations |
| `MATHEMATICA_ENABLE_CACHE` | `true` | Expression caching |
| `MATHEMATICA_ENABLE_TELEMETRY` | `false` | Usage telemetry |
| `MATHEMATICA_ROUTING_MEMORY` | `off` | Routing memory: `off`, `observe`, or `advise` |
| `MATHEMATICA_ROUTING_ACTION` | `off` | Routing action: `off` or `compute_cli_skip` (requires `advise`) |

> **Routing memory** collects aggregate routing statistics for `execute_code` (transport success rates, latency histograms, error families) to improve observability. It stores no raw code or expressions: only cohort counters. In `observe` mode, stats are persisted to `~/.cache/mathematica-mcp/routing_memory.json`. In `advise` mode, the system additionally generates routing hints and enables the optional adaptive routing action. See [Intelligent Routing & Observability](#intelligent-routing--observability) for details.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHEMATICA_HOST` | `localhost` | Host the Python client connects to (addon binds via `$MCPHost`) |
| `MATHEMATICA_PORT` | `9881` | Port the Python client connects to (addon binds via `$MCPPort`) |
| `MATHEMATICA_PROFILE` | `lean` | Tool profile: `lean`, `classic`, `math`, `notebook`, or `full` |
| `MATHEMATICA_TOOLSETS` | *(none)* | Comma-separated opt-in extras for the `lean` profile: `data_io`, `graphics_plus`, `cloud`, `debug`, `notebook_files`, `notebook_edit`, `symbols`, `math_aliases`, `repository`, `async_jobs`, `cache` |
| `MATHEMATICA_KERNEL_IDLE_TIMEOUT` | `1800` | Seconds of kernel inactivity before idle shutdown (`0` disables) |
| `MATHEMATICA_PREWARM` | `on` | Boot the persistent kernel in the background at server startup so the ~13s boot overlaps setup rather than blocking the first warm call; `0`/`false`/`no`/`off` disables |
| `MATHEMATICA_KERNEL_PATH` | *(auto-detect)* | Explicit path to the Wolfram kernel binary for the fallback session |
| `MATHEMATICA_MAX_OUTPUT_CHARS` | `4000` | Hard output cap in characters (minimum 500); oversized output is paginated via cursors |
| `MATHEMATICA_MCP_TOKEN` | *(none)* | Authentication token for secure connections |
| `MATHEMATICA_MCP_CACHE_DIR` | `~/.cache/mathematica-mcp/notebooks` | Disk cache directory for notebook extraction |
| `MATHEMATICA_RETRY_BACKOFF` | `2.0` | Seconds between addon connection retry attempts |
| `MATHEMATICA_ROUTING_MEMORY` | `off` | Routing memory: `off`, `observe`, or `advise` |
| `MATHEMATICA_ROUTING_ACTION` | `off` | Routing action: `off` or `compute_cli_skip` (requires `advise` mode) |
| `MATHEMATICA_ROUTING_MEMORY_DIR` | `~/.cache/mathematica-mcp` | Directory for the routing memory JSON file |
| `MMCP_FORCE_V14` | *(unset)* | Addon-side: set to `1` to force the Mathematica <15 code path (for guard testing) |

Feature-flag environment variables (`MATHEMATICA_ENABLE_FUNCTION_REPO`, `MATHEMATICA_ENABLE_DATA_REPO`, `MATHEMATICA_ENABLE_ASYNC`, `MATHEMATICA_ENABLE_LOOKUP`, `MATHEMATICA_ENABLE_MATH_ALIASES`, `MATHEMATICA_ENABLE_CACHE`, `MATHEMATICA_ENABLE_TELEMETRY`) are documented in [Feature Flags](#feature-flags) above.

---

## Intelligent Routing & Observability

The server includes a multi-layer intelligence system for routing optimization, payload efficiency, session awareness, and smart caching.

### Payload Shaping (`response_detail`)

The `execute_code` tool (classic/math/notebook profiles) accepts a `response_detail` parameter to control response size. The lean `evaluate` tool does not expose the parameter, but it shapes its responses as `compact` by default. Set `MATHEMATICA_RESPONSE_DETAIL=standard` to restore the full backward-compatible shape (an invalid value warns and stays `compact`); the same env var accepts any level or alias below.

| Level | Behavior |
|-------|----------|
| `"standard"` (default for `execute_code`) | Exact backward-compatible response: no fields added or removed |
| `"compact"` (default for lean `evaluate`) | Essential fields only: `success`, `status`, `message`, `output`, `timing_ms`, notebook IDs, transport fields (and `error_analysis` guidance is kept). Strips other verbose metadata plus any remaining empty fields (`""`, `[]`, `{}`, `null`), always keeping `success` and `output` (an empty output is itself a result). Auto-summarizes outputs > 4000 chars with balanced-brace list element counting. Swaps graphics placeholders to `output_inputform`. Failing responses (`success` is `false`, or a truthy `error` key) are exempt: they pass through as the full `standard` shape so transport, message, and error-analysis fields survive for recovery. |
| `"verbose"` | Full response + `detail_level` marker |
| `"diagnostic"` | Full response + `detail_level`, `cache_epoch`, and `routing_hints` (if available) |

Accepted aliases:

- `"short"` → `"compact"`
- `"medium"` → `"standard"`
- `"long"` → `"verbose"`

The filter is a pure function: `standard` is guaranteed to be byte-for-byte identical to the unfiltered response.

### Session Brief

`get_session_brief()` returns a compact ~100-token snapshot:

```
## Session Brief
- **Profile**: full | **Connection**: addon | **Default**: notebook
- **Recent errors**: Syntax::sntxi, Part::partw
- **Routing advice**: addon_cli (compute): 40% infra error rate
```

Uses a 500ms addon timeout and never starts a fresh kernel session. Recent errors are sorted by actual recency (last_seen), not frequency, with a 24-hour age cutoff.

### Computation Journal

`get_computation_journal()` returns recent computation history as an in-memory ring buffer (10 entries). Each entry includes:

- Code preview (first 100 chars) and output preview (first 100 chars)
- `success`, `timing_ms`, `route_variant`, `execution_path`
- `transport_status`, `error_families`, `timed_out`, `from_cache`

The journal records raw canonical results **before** response-detail filtering, so it always captures full output previews regardless of `response_detail` setting. Use `clear_computation_journal()` to reset.

### Smart Caching

Pure-System expressions (those referencing only built-in Wolfram functions with no user-defined symbols) are now **epoch-insensitive**: they survive kernel state mutations (`set_variable`, `clear_variables`, etc.) without re-evaluation.

Examples of epoch-insensitive expressions: `Sin[Pi]`, `Integrate[x^2, x]`, `1 + 1`

Examples of epoch-sensitive expressions (correctly invalidated): `f[3]` (user symbol `f`), `Sin[x]` (user symbol `x`), `Names["MyPkg\`*"]` (session-sensitive introspection), `$Packages` (session-sensitive global)

The analysis is memoized (`lru_cache(1024)`) with a single-pass Wolfram scanner that handles nested comments and string literals. Malformed input safely falls back to epoch-sensitive.

### Screenshot Cache (opt-in)

`screenshot(scope="notebook"|"cell", cache=True)` reuses the last PNG for that scope instead of re-exporting from the front end (a ~400ms operation). It is keyed by a separate notebook mutation epoch that `connection.send_command` bumps after any successful notebook-mutating command (`write_cell`, `delete_cell`, `evaluate_cell`, `execute_code_notebook`, `execute_selection`, `create_notebook`, `close_notebook`, `save_notebook`, `open_notebook_file`, `batch_commands`), so any such command forces a fresh capture. Kernel-only work (`execute_code`, `set_variable`, ...) and reads/navigation (`get_notebook_info`, `select_cell`, `scroll_to_cell`) correctly do not bump the epoch, but a read can still silently repoint `$MCPActiveNotebook` at a different open notebook, so the epoch alone does not make the *focused* notebook a safe cache target. The cache therefore requires a **stable target**: pass `notebook=` (an explicit id) or `session_id=` (session-pinned notebooks resolve independently of the focused notebook). With neither, notebook-scope `cache=True` is skipped and captures fresh every call, because the "focused" key could otherwise serve another notebook's pixels. The cache is a small LRU (8 entries) of PNG bytes.

It is opt-in and off by default because pixels can change without the epoch moving: manual edits in the front end, `Dynamic`/`Manipulate` repaints, scrolling, or window resizing. Two more staleness sources to know about: (a) the epoch is **per server process** (module-level state), so if two MCP server processes share one addon they do not see each other's invalidations, and one can serve a PNG the other's mutation already invalidated; and (b) `execute_code` running raw front-end writes (`NotebookWrite`, `CellPrint`, ...) changes the notebook without going through a `MUTATING_COMMANDS` command, so it does **not** bump the epoch, use `execute_code_notebook`/`write_cell` instead when you want the cache to notice. So `cache=True` may return stale pixels if the notebook changed outside these tracked commands; pass it only when you know nothing but your own tracked MCP commands has touched the notebook. `scope="expression"` ignores the flag (it is already cached at the raster layer).

### Routing Memory

Routing memory is an opt-in observability layer that learns aggregate routing statistics from `execute_code` calls. No Mathematica code or expressions are stored.

**Modes:**

| Mode | Behavior |
|------|----------|
| `off` (default) | No recording, no file I/O, zero overhead |
| `observe` | Records aggregate counters, persists to disk periodically |
| `advise` | Observe + routing hints + enables routing action (if gated) |

Enable with: `MATHEMATICA_ROUTING_MEMORY=observe` (or `advise`)

**What it tracks:**

End-to-end cohort stats grouped by `(profile, route_variant)`, plus optional `expression_type`:
- **Transport outcomes:** ok, degraded fallback, timeout, infrastructure error
- **Semantic errors:** error family frequencies (Syntax, Part, Set, etc.)
- **Latency histogram:** 7 buckets from <50ms to >5s
- **Execution path breakdown:** addon_notebook, addon_cli, kernel_fallback, kernel_direct_routing_skip

Attempt-level transport telemetry per transport leg:
- **Per-path outcomes:** `path_transport_outcomes` tracks typed outcomes (OK, INFRA_ERROR, TIMEOUT, SEMANTIC_ERROR) for each transport path independently
- **Expression classification:** code classified into routing-relevant categories (plot, frontend_dynamic, symbolic_heavy, numeric_heavy, io, general) for fine-grained cohort analysis

**Transport Leg Matrix:**

| Leg | Attempt telemetry | Final telemetry | Notes |
|-----|:-:|:-:|-------|
| addon_notebook | Yes | Yes | Primary notebook path |
| addon_cli | Yes | Yes | Primary CLI path |
| kernel_fallback | No | Yes | Last-resort compute fallback only |
| kernel_direct_routing_skip | No | Yes | Routing decision, not transport |

**Advisory hints** (advise mode only): Built from two sources: transport-path hints (per-path infra/timeout rates) and end-to-end hints (timeout/fallback rates). Structured with severity ordering, deduplication, and 5-hint cap. Available via `get_routing_memory_stats(include_hints=True)`.

### Adaptive Routing Action (opt-in)

When enabled, the server proactively skips transport legs that are persistently failing:

```
MATHEMATICA_ROUTING_MEMORY=advise
MATHEMATICA_ROUTING_ACTION=compute_cli_skip
```

**Both** flags are required. Scoped to compute-only addon_cli bypass (notebook routing is untouched).

**Circuit breaker behavior:**
- Trips after 5 consecutive `INFRA_ERROR` outcomes on a transport path (timeouts and semantic errors do NOT trip)
- 60-second cooldown window during which the path is skipped
- Half-open probe after cooldown: exactly one request is allowed through (concurrency-safe with lock-protected probe-in-flight state)
- Probe success closes the breaker; probe failure retrips with fresh cooldown
- Abort (exception before transport attempt) reverts to open without recording a failure or resetting cooldown

When the breaker skips addon_cli, execution goes directly to kernel with a truthful `kernel_direct_routing_skip` execution path (not mislabeled as `kernel_fallback`).

**Observability:** Skip count, last skip reason, and last skip timestamp are available via `get_routing_memory_stats()` (runtime-only, not persisted).

### Privacy

- The in-memory journal stores 100-char code/output previews (not persisted)
- Routing memory persists only aggregate counters and known system error families; user-defined error tags are mapped to `"other"`
- Routing storage: `~/.cache/mathematica-mcp/routing_memory.json` (~2-5 KB)
- Notebook extraction results are cached to `~/.cache/mathematica-mcp/notebooks/` with mtime-based invalidation

### Tools

| Tool | Profile | Description |
|------|---------|-------------|
| `get_session_brief()` | classic, math, notebook | Compact session state summary (lean: use `status()`) |
| `get_computation_journal()` | classic, math, notebook (lean: `MATHEMATICA_TOOLSETS=debug`) | Recent computation history |
| `clear_computation_journal()` | classic, math, notebook (lean: `MATHEMATICA_TOOLSETS=debug`) | Reset journal |
| `get_routing_memory_stats(include_hints=False)` | classic/full only (opt-in) | Routing stats + optional hints. Requires `MATHEMATICA_ROUTING_MEMORY=observe` or `advise`. |
| `clear_routing_memory()` | classic/full only (opt-in) | Reset all routing stats and breaker state. Requires `MATHEMATICA_ROUTING_MEMORY=observe` or `advise`. |

### Data lifecycle

- **Exponential decay:** counters are aged with a 7-day half-life so old patterns fade
- **Zero-value pruning:** decayed counters and empty nested maps are pruned during serialization
- **Persistence:** atomic JSON writes every 60s (or 200 records), plus on shutdown
- **Fail-open:** any storage failure disables persistence silently; tools continue working
- **Schema migration:** v1 files (pre-0.9.0) load cleanly with missing fields defaulting to empty

---

## Troubleshooting

### "Could not connect to Mathematica addon"

1. Is Mathematica running?
2. Run `MCPServerStatus[]` in Mathematica
3. Try `RestartMCPServer[]`
4. Check port: `lsof -i :9881`

### Addon changes not taking effect after update

Mathematica caches loaded packages in memory. After updating the package (e.g., `pip install --upgrade` or `git pull`), the running Mathematica session still serves the old addon code. Symptoms include missing new features, old error formats, or `timing_ms` always reporting 0.

**Fix:** Reload the addon in your Mathematica session:
```mathematica
RestartMCPServer[]
```

Or for a full reload:
```mathematica
Get["~/mcp/mathematica-mcp/addon/MathematicaMCP.wl"]
StartMCPServer[]
```

This applies any time `addon/MathematicaMCP.wl` changes: the Python server picks up changes automatically, but the Mathematica side requires an explicit reload.

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

- Use `submit_computation` for long operations (classic, or `MATHEMATICA_TOOLSETS=async_jobs` on lean); on lean you can also raise `evaluate(..., timeout=...)`
- Break complex computations into steps

### Variables not persisting

- Ensure you're using the **addon connection** (not wolframscript fallback)
- Check `get_mathematica_status()` (classic) or `status()` (lean) shows `connection_mode: "addon"`

### Output parsing issues (older Mathematica versions)

The MCP server uses `ExportString[..., "RawJSON"]` for reliable JSON output from Mathematica. For older versions that don't support this, a robust regex-based fallback parser handles Association output with properly escaped quotes.

---

## Compatibility

| Component | Tested Version |
|-----------|----------------|
| Mathematica | 14.0+ (15+ first-class; 14.x kept working behind `$VersionNumber >= 15.` guards) |
| Python | 3.10+ |
| macOS | ARM64 (Apple Silicon) |
| Linux | x86_64 (POSIX) |
| Windows | Community-tested (not in official classifiers) |
| MCP Python SDK | `mcp>=1.0.0` (see pyproject.toml) |

---

## Project Structure

```
mathematica-mcp/
├── src/mathematica_mcp/
│   ├── server.py              # Profile-gated tools: classic surface + 12-tool lean dispatchers + MCP prompts
│   ├── config.py              # Profiles, feature flags, tool groups, MATHEMATICA_TOOLSETS
│   ├── guidance.py            # Profile-aware LLM guidance; feeds server instructions, AGENTS.md, CLAUDE.md, and prompts
│   ├── notebook_backend.py    # Notebook extraction backend abstraction
│   ├── notebook_parser.py     # Python-native .nb parser (offline)
│   ├── connection.py          # Socket connection to addon (protocol_version handshake)
│   ├── session.py             # Warm persistent kernel session + cold wolframscript fallback, idle reaper
│   ├── cli.py                 # Setup commands for 6 clients + project guidance (CLAUDE.md, AGENTS.md)
│   ├── cache.py               # In-memory expression caching (epoch-aware)
│   ├── disk_cache.py          # Persistent notebook extraction cache
│   ├── cursor_store.py        # Continuation cursors for oversized tool output
│   ├── response_filter.py     # response_detail payload shaping (pure function)
│   ├── journal.py             # In-memory computation journal (ring buffer, 10 entries)
│   ├── routing_memory.py      # Opt-in aggregate routing stats + circuit breaker
│   ├── transport_classification.py    # Typed transport outcomes (OK/INFRA_ERROR/TIMEOUT/SEMANTIC_ERROR)
│   ├── wl_scan.py             # Wolfram source scanner (epoch analysis, brace counting)
│   ├── constants.py           # Shared execution-path / outcome labels
│   ├── symbol_index.py        # Version-scoped symbol index with disk persistence
│   ├── lazy_wolfram_tools.py  # Async helpers (verify_derivation etc.)
│   ├── error_analyzer.py      # Error pattern matching & LLM formatting
│   ├── telemetry.py           # Usage tracking
│   ├── optional_math_aliases.py       # 8 named math operations
│   ├── optional_repository_tools.py   # 6 repository search/load tools
│   ├── optional_symbol_tools.py       # 3 symbol lookup tools
│   ├── optional_async_jobs.py         # 3 async computation tools
│   ├── optional_cache_tools.py        # 4 expression cache tools
│   ├── optional_telemetry_tools.py    # 2 telemetry tools
│   ├── optional_routing_tools.py      # 2 routing memory tools
│   └── helpers/
│       └── notebook_converter.wl      # WL kernel helper for NotebookImport
├── addon/
│   ├── MathematicaMCP.wl   # Main addon (persistent session)
│   ├── install.wl          # Auto-install script
│   └── README.md
└── tests/                  # 50+ test modules: sessions, profiles, lean tools,
                            # guidance, routing memory, notebook backends,
                            # transport classification, corpus runner, and more
```

---

## Known Issues & Technical Limitations

### Notebook Frontend vs Kernel Timing

Notebook operations (`style="notebook"`) use a kernel-mode fast path (`executeCodeNotebookKernel`) that combines notebook lookup, cell creation, and evaluation into a single atomic round-trip. This is the default and is reliable for standard use.

Frontend mode (`style="interactive"`) dispatches evaluation through the Mathematica FrontEnd via `FrontEndTokenExecute["EvaluateCells"]`, then polls for completion. This path is required for `Manipulate`, `Dynamic`, and other interactive content that relies on the FrontEnd evaluation pipeline.

When performing rapid sequences of individual cell operations (`write_cell` followed immediately by `get_cells`), the Mathematica frontend may not have finished updating its internal state. This can cause:

- Cells appearing empty when queried immediately after creation
- Stale cell counts from `get_cells` after rapid mutations

**Mitigations:**
- Prefer `execute_code(style="notebook")` (kernel mode, default) over individual `write_cell` + `evaluate_cell` sequences
- For interactive/dynamic content (`Manipulate`, `Animate`), `style="interactive"` uses frontend evaluation by design - results appear in the Mathematica window
- Use `style="compute"` when you only need the result as text, not a notebook artifact

### Evaluation Architecture: Preemptive vs Main Link

Understanding the two kernel links is essential for diagnosing notebook evaluation latency:

```
Claude Code → Python MCP → TCP :9881 → MathematicaMCP.wl (SocketListen)
                                             │
                             ┌────────────────┴────────────────┐
                             │                                 │
                       PREEMPTIVE LINK                    MAIN LINK
                       • execute_code                    • evaluate_cell (FrontEnd dispatch)
                       • kernel-mode notebook            • frontend-mode notebook
                       • Dynamic/Manipulate updates      • Shift+Enter in notebook UI
                       • All MCP socket handlers         • Queued, single-threaded
```

**Key behaviors:**
- **SocketListen handlers** occupy the kernel while they run. A handler that `Pause`-polls **blocks** the front-end (main-link) evaluation it dispatched: the dispatched cell cannot progress until the handler returns. Measured (kernel 15.0): the output cell of a `Pause[3]; 42` frontend eval does not appear until after the handler that dispatched it returns.
- **`execute_code`** (kernel mode) evaluates synchronously inside the handler and returns its result directly - no front-end round trip.
- **`evaluate_cell`, `evaluate_selection`, and frontend-mode `execute_code_notebook`** dispatch to the front end via `FrontEndTokenExecute["EvaluateCells"]`. Because that evaluation shares the kernel with the handler, its completion is **not observable in-handler**: `CurrentValue[nb, Evaluating]` returns an *unevaluated* `FrontEnd`CurrentValue[...]` from this context, and no output cell appears during the poll. All three therefore return an honest `evaluation_pending` shape (`evaluated: false`, `evaluation_complete: false`, `waited_seconds`) after a bounded 0.2s grace poll; re-read the notebook with `get_cells`/`get_cell_content` (or `screenshot_notebook`) to collect the result once it appears.
- **`Manipulate`/`Dynamic`** slider updates use the preemptive link - they stay responsive even when cells show "Running...".

**Blocking scenarios and remediation:**

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Notebook cell shows "Running..." for simple expressions | Main link queue blocked by a previous computation | Restart kernel |
| All MCP commands slow (including ping) | Preemptive link blocked by a long `execute_code` | Wait for it to finish, or restart kernel |
| MCP timeout, but manual notebook evaluation works | TCP socket stale or addon connection dropped | `RestartMCPServer[]` (preserves kernel state) |
| `Manipulate` sliders work, but typed cells stuck | Preemptive link free, main link blocked | Restart kernel to clear main link queue |

> **Note:** `RestartMCPServer[]` only restarts the TCP socket listener - it does **not** restart the kernel. It is only useful when the socket connection itself is broken. For evaluation-related slowness, restart the kernel instead.

### Frontend Completion Detection (protocol 4)

The v0.9.4 loop combined `CurrentValue[nb, Evaluating]` polling with a `Break[]` after 0.5s when nothing had been observed. As measured above, neither signal actually resolves from the socket handler - `Evaluating` returns unevaluated and the output cell does not appear during the poll - so what really made the loop "fast" (the reported 129ms) was the 0.5s give-up returning early. That early return reported `{success: true, output_preview: ""}` while the notebook was still computing: any frontend eval longer than ~0.5s falsely looked finished with no output.

Protocol 4 replaces the give-up with an honest contract. `executeCodeNotebookFrontend` still polls for an output cell (breaking immediately if one appears, which would only cover front-end configurations that yield mid-`Pause`), but the poll is capped at `Min[max_wait, 0.2s]`: completion is unobservable in-handler on current Mathematica, so a longer poll buys nothing and would only stall every other command - and the user's own evaluations - on the single-threaded handler. When the poll ends without an observed output cell (the expected outcome) it returns:

```
{ "success": true, "status": "evaluation_pending", "evaluation_complete": false,
  "waited_seconds": <elapsed>, "message": "...re-check with get_cells..." }
```

Pending is **not** a failure and is the **expected** outcome on current Mathematica - dispatch succeeded and the evaluation runs (and its `Out` cell appears in the target notebook) once the call returns. Agents collect the result by re-reading the notebook. The `evaluation_complete: true` branch (with a real `output_preview`) fires only if the front end can yield during the handler, which was **not observed on 14.x or 15.0**; treat it as a best-effort fast path, not a normal return.

> **Warning:** Closing a notebook with a pending evaluation does **not** cancel it. The evaluation still finishes, but with the notebook gone its output is dumped into the system Messages window instead of a cell. Re-check pending work (via `get_cells`/`get_cell_content`) and let the `Out` cell land before closing the notebook.

`evaluate_cell` and `evaluate_selection` carry the **same** honest contract: a 0.2s-capped poll, then `evaluated: false` + `status: "evaluation_pending"` + `evaluation_complete: false` + `waited_seconds` and a re-check message when no output cell is observed (the normal case). Re-read the notebook after calling them, or use frontend-mode `execute_code` which carries the identical contract.

---

## License

MIT License

---

*Last updated: July 2026 (v1.0.0: lean profile default, profile-aware guidance, consolidated action-enum tools)*
