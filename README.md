# Mathematica MCP

**Give your AI the power of Wolfram Language.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)

> **Looking for deep technical details?**
> Check out the [Technical Reference](docs/technical-reference.md) for architecture, tool definitions, and advanced configuration.

---

## What is this?

This tool connects AI models (like Claude, Cursor, or OpenCode) directly to your local Mathematica installation. It transforms Mathematica from a passive calculation engine into an active **pair programmer** that can read your notebooks, debug your math, and visualize your data.

Instead of just generating code that *might* work, the AI can **run it, fix it, and show you the result**.

It can also:

- Control notebooks: create, edit, evaluate, and export cells.
- Answer inline: symbolic math, plots, and real-world data lookups.
- Read and parse `.nb` files: extract code, convert to Markdown/LaTeX.
- Keep state: variables persist across calls.
- Expose 65+ tools for math, data, graphics, and debugging.
- **Error analysis**: When code executes in notebooks, errors are captured, pattern-matched, and enriched with fix suggestions.

---

## Why use it? (Real-World Examples)

### 🎓 For Students & Researchers
*Stop copy-pasting equations.*
- **Fibonacci + Golden Ratio**: "Plot Fibonacci growth, ratio convergence to the Golden Ratio, and the Golden Spiral in one notebook."
- **Symbolic Integration**: "Integrate `Sin[x]^4 Cos[x]^2`, simplify it, and show the steps."
- **High-Quality Visualization**: "Render a 3D Sombrero surface and export as SVG for my paper."

### 💻 For Developers
*Treat notebooks like code repositories.*
- **Notebook Parsing**: "Extract only the Wolfram code from this `.nb` file and summarize the structure."
- **Debugging**: "Trace the evaluation of `MyCustomFunction[x]` to see why it returns `Indeterminate`."
- **Test Generation**: "Create tests for my Wolfram package that cover numeric, symbolic, and complex inputs."
- **Error Analysis**: When notebook code produces errors, the AI receives pattern-matched suggestions (e.g., for `UnitConvert::compat`: "Use QuantityMagnitude[] to extract numeric values").

### 📊 For Data Scientists
*Access the world's largest knowledgebase.*
- **Real-World Entities**: "Compare GDP for US, China, Japan, Germany and plot a bar chart."
- **Time Series Forecasting**: "Fetch last 20 years of GDP for G7, smooth it, and forecast 5 years ahead."
- **Signal Cleanup**: "Import a noisy CSV, apply a low-pass filter, and plot before/after."

---

## Installation

### Prerequisites
*   **Mathematica 14.0+** (or Wolfram Engine)
*   **Python 3.10+**

### 1. Install the Server
```bash
# Clone the repository
git clone https://github.com/psi-server/mathematica-mcp.git
cd mathematica-mcp

# Install dependencies
uv sync  # OR: pip install -e .
```

### 2. Install the Mathematica Addon
This allows the AI to maintain a persistent session (remembering variables between messages).

```bash
cd addon
wolframscript -file install.wl
```
*(Alternatively, open `install.wl` in Mathematica and click "Run All Code")*

---

## Quick Start

1. **Start Mathematica** (addon auto-starts if installed)
2. **Verify addon is running:** `MCPServerStatus[]`
3. **Start your AI client**
4. **Ask:** "Plot `Sin[x]` from 0 to `2 Pi` and save it as PNG"

See the full beginner walkthrough in `docs/quick-start.md`.

---

## Client Configuration

Choose your AI client below to configure the connection.

### 🤖 Claude Desktop App
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### 💻 VS Code (via Continue or similar)
Add to your extension configuration (User Settings JSON):

```json
"mcpServers": {
  "mathematica": {
    "command": "uv",
    "args": ["--directory", "/path/to/mathematica-mcp", "run", "mathematica-mcp"]
  }
}
```

### ⌨️ Claude Code (CLI)
Create or edit `.mcp.json` in your project root:

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

### 🧪 Codex CLI
Add to your MCP configuration file (e.g., `~/.mcp.json`):

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

### 🛠️ OpenCode
Add to your MCP configuration file (project or global):

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

---

## Troubleshooting

*   **"Connection Refused"**: Ensure Mathematica is running. If you closed it, run `StartMCPServer[]` in a new notebook.
*   **"Port 9881 Busy"**: An old kernel might be stuck. Run `lsof -i :9881` in your terminal to find and kill the process.
*   **Variable Persistence**: Make sure you see "Connection Mode: Addon" in the tool output. If it says "Script", variables will not be saved between messages.

---

## License
MIT License
