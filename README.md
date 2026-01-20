# Mathematica MCP

**Give your AI the power of Wolfram Language.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)

---

## What is this?

This tool connects AI models (like Claude, Cursor, or OpenCode) directly to your local Mathematica installation. It transforms Mathematica from a passive calculator into an active **pair programmer**.

**Your AI can now:**
*   **Run Code**: It writes and executes Wolfram Language code for you.
*   **Fix Errors**: If code fails, it sees the error and suggests fixes automatically.
*   **Create Visualizations**: It can generate plots and export them as images.
*   **Read Notebooks**: It can open your `.nb` files and understand their content.
*   **Remember Variables**: It remembers your definitions (`x = 5`) throughout the conversation.

### Watch it in action

[![Mathematica MCP Demo](https://img.youtube.com/vi/TjGSkvVyc1Y/0.jpg)](https://www.youtube.com/watch?v=TjGSkvVyc1Y)

---

## Installation

### 0. Prerequisites

- Mathematica 14.0+ installed
- Python 3.10+ installed
- `wolframscript` available (Mathematica menu: Help -> Install wolframscript)

### 🤖 For LLM Agents

If you are an AI assistant (like Claude, Cursor, or OpenCode) installing this for a user, execute the following sequence:

```bash
# Fail fast on any error
set -e

# 1. Install Python dependencies (prefer uv)
if command -v uv >/dev/null 2>&1; then 
    echo "Installing with uv..."
    uv sync
else 
    echo "Installing with pip..."
    pip install -e .
fi

# 2. Install the Mathematica Addon (CRITICAL)
if ! command -v wolframscript >/dev/null 2>&1; then
    echo "Error: wolframscript not found. Please ask user to install it."
    exit 1
fi
# Run without changing directory to maintain path safety
wolframscript -file addon/install.wl

# 3. Report Success
echo "Installation complete. Server path: $(pwd)"
```

**Next Steps for Agent:**
1.  Read the output path from step 3.
2.  Generate the JSON config for the user (using `uv` or `python` as appropriate).
3.  Ask the user to paste it into their client config.

### 1. Install the server and addon

```bash
# 1. Clone the repo
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp

# 2. Install the Python package
pip install -e .
# Or with uv:
# uv sync

# 3. Install the Mathematica addon
cd addon
wolframscript -file install.wl
```

### 2. Configure your client

Use this MCP server definition everywhere (only the config file location changes):

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

If you do not use `uv`, use Python directly:

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "python3",
      "args": ["-m", "mathematica_mcp"]
    }
  }
}
```

#### Claude Desktop

- macOS config: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows config: `%APPDATA%\Claude\claude_desktop_config.json`
- Paste the JSON above and restart the app.

#### Claude Code

- Create `.mcp.json` in your project root and paste the JSON above.

#### Cursor

- Settings -> Features -> MCP -> Add New MCP Server
- Name: `mathematica`
- Type: `command`
- Command: `uv --directory /ABSOLUTE/PATH/TO/mathematica-mcp run mathematica-mcp`

#### VSCode (Continue or other MCP-capable extension)

- Add the same MCP server definition in your extension settings.

#### OpenCode

- Add the same MCP server definition in your OpenCode config (project or global).

### 3. Verify it works

1. Start Mathematica.
2. In a notebook, run `MCPServerStatus[]` and confirm `running -> True`.
3. In your AI client, ask: "What is the value of Pi in Mathematica?".

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
