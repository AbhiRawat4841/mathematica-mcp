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

### For Humans

Copy and paste this prompt to your LLM agent (Claude Code, AmpCode, Cursor, etc.):

```
Install and configure mathematica-mcp by following the instructions here:
https://raw.githubusercontent.com/AbhiRawat4841/mathematica-mcp/main/docs/installation.md
```

Or read the [Installation Guide](docs/installation.md) directly.

### For LLM Agents

Fetch the installation guide and follow it:

```bash
curl -s https://raw.githubusercontent.com/AbhiRawat4841/mathematica-mcp/main/docs/installation.md
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

*   **[Technical Reference](docs/technical-reference.md)**: Deep dive into architecture, tools, and configuration.

---

## License
MIT License
