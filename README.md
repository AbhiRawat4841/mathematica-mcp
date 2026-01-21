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

## Installation

```bash
# 1. Clone and install
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp
uv sync

# 2. Install the Mathematica addon
wolframscript -file addon/install.wl

# 3. Restart Mathematica (required for addon to load)
```

**Verify:** After restarting Mathematica, check the Messages window for `[MathematicaMCP] Server started on port 9881`

📖 **[Full Installation Guide](docs/installation.md)** — Prerequisites, uv installation, client integration (Claude, Cursor, VS Code), and troubleshooting.

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
