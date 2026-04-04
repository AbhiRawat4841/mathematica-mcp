# Mathematica MCP

**Give your AI Agent the power of Wolfram Language.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)
[![CI](https://github.com/AbhiRawat4841/mathematica-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhiRawat4841/mathematica-mcp/actions/workflows/ci.yml)
[![Repo](https://img.shields.io/github/v/tag/AbhiRawat4841/mathematica-mcp?label=repo)](https://github.com/AbhiRawat4841/mathematica-mcp/releases)
[![Published](https://img.shields.io/pypi/v/mathematica-mcp-full?label=published&cacheSeconds=300)](https://pypi.org/project/mathematica-mcp-full/)

---

## Who This Is For

| Audience | Use Case |
|----------|----------|
| Researchers using LLM coding assistants | Run Mathematica from Claude/Cursor/VS Code without leaving your editor |
| Data scientists | Import, transform, and visualize data through natural language |
| Educators | Create interactive Mathematica notebooks through AI conversation |
| **Not for** | Production web services, untrusted multi-tenant environments |

---

## What is this?

An **MCP Server** that gives AI agents a direct interface to your local **Wolfram Engine**. 79 tools across configurable profiles for symbolic reasoning, visualization, and notebook control.

### Watch it in action

[![Mathematica MCP Demo](https://img.youtube.com/vi/TjGSkvVyc1Y/0.jpg)](https://www.youtube.com/watch?v=TjGSkvVyc1Y)

---

## Quick Start

### Prerequisites

Before installing, you need:

1. **Mathematica 14.0+** with `wolframscript` in your PATH
   - [Download Mathematica](https://www.wolfram.com/mathematica/)
   - macOS: Add to `~/.zshrc`: `export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"`

2. **uv package manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### One-Command Setup (Recommended)

```bash
# For Claude Desktop
uvx mathematica-mcp-full setup claude-desktop

# For Cursor
uvx mathematica-mcp-full setup cursor

# For VS Code (requires GitHub Copilot Chat extension)
uvx mathematica-mcp-full setup vscode

# For OpenAI Codex CLI
uvx mathematica-mcp-full setup codex

# For Google Gemini CLI
uvx mathematica-mcp-full setup gemini

# For Claude Code CLI
uvx mathematica-mcp-full setup claude-code

# Optional: select a tool profile (default is "full")
uvx mathematica-mcp-full setup claude-desktop --profile notebook
```

Then restart Mathematica and your editor. Done!

<details>
<summary>VS Code: Alternative setup via Command Palette</summary>

> **Prerequisite:** [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) extension must be installed - MCP support is built into Copilot.

1. Press `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows)
2. Type "MCP" → Select **"MCP: Add Server"**
3. Choose **"Command (stdio)"** — *not "pip"*
4. Enter command: `uvx`
5. Enter args: `mathematica-mcp-full`
6. Name it: `mathematica`
7. Choose scope: Workspace or User

</details>

### Alternative: Interactive Installer

```bash
bash <(curl -sSL https://raw.githubusercontent.com/AbhiRawat4841/mathematica-mcp/main/install.sh)
```

### Verify Installation

```bash
uvx mathematica-mcp-full doctor
```

> **Tip:** If you encounter errors after updating, clear the cache:
> ```bash
> uv cache clean mathematica-mcp-full && uvx mathematica-mcp-full setup <client>
> ```

### Tool Profiles

Choose how many tools to expose: `math` (~25 tools), `notebook` (~45), or `full` (~79, default).
Pass `--profile` during setup or set `MATHEMATICA_PROFILE` env var. See the **[Technical Reference](docs/technical-reference.md#tool-profiles)** for details.

---

## Manual Installation

<details>
<summary>Click to expand manual setup instructions</summary>

1.  **Clone & Install**:
    ```bash
    git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
    cd mathematica-mcp
    uv sync
    ```

2.  **Install Mathematica Addon**:
    ```bash
    wolframscript -file addon/install.wl
    ```
    *Restart Mathematica after this step.*

3.  **Configure your editor** (replace path with your actual path):

    **Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
    ```json
    {
      "mcpServers": {
        "mathematica": {
          "command": "uv",
          "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp-full"]
        }
      }
    }
    ```

    **Cursor** (`~/.cursor/mcp.json`):
    ```json
    {
      "mcpServers": {
        "mathematica": {
          "command": "uv",
          "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp-full"]
        }
      }
    }
    ```

    **VS Code** (`~/.vscode/mcp.json`):
    > **Note:** VS Code MCP requires [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) extension.

    ```json
    {
      "servers": {
        "mathematica": {
          "type": "stdio",
          "command": "uv",
          "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp-full"]
        }
      }
    }
    ```

</details>

📖 See the **[Installation Guide](docs/installation.md)** for troubleshooting and advanced setup.

---

## What You Can Do

### 1. Solve and Verify a Calculus Problem

> "Integrate x^2 sin(x) from 0 to pi, then verify by differentiating."

```text
Agent calls: execute_code("Integrate[x^2 Sin[x], {x, 0, Pi}]")
=> -4 + Pi^2

Agent calls: verify_derivation(
  steps=["Integrate[x^2 Sin[x], {x, 0, Pi}]", "-4 + Pi^2"]
)
=> {"success": true, "report": "Step 1 → 2: ✓ VALID\n...\n**Summary**: All steps are valid!", "raw_data": {...}, "format": "text"}
```

### 2. Generate a 3D Plot

> "Plot the sombrero function and export it."

```text
Agent calls: execute_code("Plot3D[Sinc[Sqrt[x^2 + y^2]], {x, -4, 4}, {y, -4, 4}]")
=> [3D surface plot rendered as image]

Agent calls: export_graphics("Plot3D[Sinc[Sqrt[x^2+y^2]], {x,-4,4}, {y,-4,4}]", "/tmp/sombrero.png", "PNG")
```

### 3. Read and Analyze a Notebook

> "Show me the outline of SinPlot.nb, then extract the code cells."

```text
Agent calls: read_notebook("SinPlot.nb", output_format="outline")
=> {"success": true, "format": "outline", "section_count": 0, "sections": []}

Agent calls: read_notebook("SinPlot.nb", output_format="json")
=> {"success": true, "cell_count": 2, "code_cells": 1, "cells": [{"style": "Input", "content": "Plot[Sin[x], {x, 0, 2 Pi}]"}, ...]}
```

Beyond these workflows: **symbolic computation**, **2D/3D visualization**, **notebook operations**, **Wolfram Alpha queries**, **data import/export** (250+ formats), and **debugging tools**. See the [Technical Reference](docs/technical-reference.md) for the full tool list.

---

## Execution Styles

### For chat users — use keywords in your prompt

| Say this...                                        | What happens                                       |
|----------------------------------------------------|----------------------------------------------------|
| **"calculate ..."**, **"compute ..."**, **"what is ..."** | Result appears as text in chat              |
| **"plot ..."**, **"show ..."**, **"in notebook ..."**     | Executes in the current Mathematica notebook |
| **"in new notebook: ..."**                         | Creates a fresh notebook, then executes there      |
| **"interactive ..."**, **"manipulate ..."**, **"dynamic ..."** | Live front-end evaluation (sliders, animations) |

### For tool callers — use the `style` parameter

| `style=`        | `output_target` | `mode`     | Best for                           |
|-----------------|-----------------|------------|------------------------------------|
| `"compute"`     | cli             | kernel     | Math, algebra, parsing results     |
| `"notebook"`    | notebook        | kernel     | Plots, visual artifacts            |
| `"interactive"` | notebook        | frontend   | Manipulate, Dynamic, animations    |

`style` is a high-level shortcut for `output_target` + `mode`. Individual params still work and override style.

> **Note:** There is no `style="new_notebook"`. Creating a fresh notebook is a two-step workflow:
> `create_notebook(title="...")` then `execute_code(style="notebook")`.

### Examples

```text
"Calculate the integral of x^3 from 0 to 1"
  → Result appears inline in chat

"Plot Sin[x] from 0 to 2π"
  → Plot appears in current Mathematica notebook

"In new notebook: integrate 1/x^5 + x^7 and plot the integration region"
  → Fresh notebook is created with the work

"Interactive: Manipulate a slider for Plot[Sin[n x], {x, 0, 2π}]"
  → Dynamic UI with sliders in notebook
```

If you don't include a keyword, the default mode depends on your [tool profile](#tool-profiles): `notebook` profiles default to notebook output, `math` profile defaults to inline.

> **Tip:** These styles are also available as MCP prompts (`calculate`, `notebook`, `new_notebook`, `interactive`) in clients that support prompt selection. Use the `quickstart` prompt to see this reference at any time.

---

## Documentation

*   **[Technical Reference](docs/technical-reference.md)** — Architecture, tools, and configuration
*   **[Security Model](SECURITY.md)** — Threat model, permissions, and vulnerability reporting
*   **[Benchmarks](docs/benchmarks.md)** — Performance data and reproduction steps
*   **[Contributing](CONTRIBUTING.md)** — Development setup, testing, and PR process
*   **[Changelog](CHANGELOG.md)** — Version history
*   **[Examples](docs/examples/)** — Polished agent session walkthroughs

---

## License
MIT License
