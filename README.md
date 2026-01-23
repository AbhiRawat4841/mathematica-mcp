# Mathematica MCP

**Give your AI Agent the power of Wolfram Language.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)

---

## What is this?

This **MCP Server** empowers **AI Agents & IDEs** (like Claude Desktop, Cursor, or OpenCode) with a direct interface to your local **Wolfram Engine**. It enables your agent to perform **symbolic reasoning, precise calculation, and interactive visualization** natively.

**Capabilities:**
*   **Execute Code**: Run Wolfram Language expressions in a secure sandbox.
*   **Self-Correct**: Diagnose and fix syntax errors automatically.
*   **Visualize**: Generate high-fidelity plots and export them as images.
*   **Analyze Notebooks**: Parse and manipulate `.nb` files contextually.
*   **Persist State**: Maintain a stateful session (e.g. `x = 5`) across interactions.

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

# For VS Code
uvx mathematica-mcp-full setup vscode

# For OpenAI Codex CLI
uvx mathematica-mcp-full setup codex

# For Google Gemini CLI
uvx mathematica-mcp-full setup gemini
```

Then restart Mathematica and your editor. Done!

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
          "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp"]
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
          "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp"]
        }
      }
    }
    ```

    **VS Code** (`~/.vscode/mcp.json`):
    ```json
    {
      "servers": {
        "mathematica": {
          "type": "stdio",
          "command": "uv",
          "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp"]
        }
      }
    }
    ```

</details>

📖 See the **[Installation Guide](docs/installation.md)** for troubleshooting and advanced setup.

---

## Why use it?

### For Students & Researchers
*   **Symbolic Math**: "Integrate `Sin[x]^4 Cos[x]^2`, simplify it, and show the steps."
*   **Visualization**: "Render a 3D Sombrero surface and export as SVG."

### For Developers
*   **Debugging**: "Trace the evaluation of `MyCustomFunction[x]`."
*   **Parsing**: "Extract only the Wolfram code from this `.nb` file."

### For Data Scientists
*   **Real Data**: "Compare GDP for US, China, and Japan."
*   **Import**: "Load `data.csv` and plot the distribution."

---

## Documentation

*   **[Technical Reference](docs/technical-reference.md)**: Deep dive into architecture, tools, and configuration.

---

## License
MIT License
