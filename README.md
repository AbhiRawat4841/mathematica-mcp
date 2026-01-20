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

---

## Installation (The Easy Way)

**[➡️ Click here for the Step-by-Step Beginner Guide](docs/quick-start.md)**

If you are a developer and just want the commands:

```bash
# 1. Clone the repo
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp

# 2. Install Python package
pip install -e .

# 3. Install Mathematica Addon
cd addon
wolframscript -file install.wl
```

Then add this to your MCP config (e.g., `claude_desktop_config.json`):

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

*   **[Beginner Quick Start](docs/quick-start.md)**: The "Happy Path" to get running in 5 minutes.
*   **[Technical Reference](docs/technical-reference.md)**: Deep dive into architecture, tools, and configuration.

---

## License
MIT License
