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

### Prerequisites

- **Mathematica 14.0+** installed
- **Python 3.10+** installed
- **wolframscript** available in your PATH
- **uv** package manager (Recommended)

### 1. Server Setup

**Step 1: Install the Package**
```bash
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp
uv sync
```

**Step 2: Install Mathematica Addon**
*Required for the server to communicate with the Wolfram Kernel.*
```bash
wolframscript -file addon/install.wl
```

### 2. Client Integration

You need the **Absolute Path** to this repository. Run `pwd` to find it.
*(Replace `/YOUR/PATH/TO/mathematica-mcp` in the examples below with this output)*

#### Claude for Desktop

Go to **Claude > Settings > Developer > Edit Config**.
Add `mathematica` to your `mcpServers` list:

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": [
        "--directory",
        "/YOUR/PATH/TO/mathematica-mcp",
        "run",
        "mathematica-mcp"
      ]
    }
  }
}
```

<details>
<summary>Claude Code (CLI)</summary>

Create a `.mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": [
        "--directory",
        "/YOUR/PATH/TO/mathematica-mcp",
        "run",
        "mathematica-mcp"
      ]
    }
  }
}
```
</details>

#### Cursor Integration

[**Download Cursor**](https://cursor.com)

1. Go to **Settings > Features > MCP**.
2. Click **Add New MCP Server**.
3. Enter the following details:

| Field | Value |
|-------|-------|
| **Name** | `mathematica` |
| **Type** | `command` |
| **Command** | `uv` |
| **Args** | `--directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp` |

**Note**: Paste the *entire* string above into the "Args" field.

#### Visual Studio Code Integration

*Prerequisites*: Install the **[Roo Code](https://marketplace.visualstudio.com/items?itemName=RooVeterinaryInc.roo-cline)** extension.

Add this to your `settings.json` (or the extension's config file):

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": [
        "--directory",
        "/YOUR/PATH/TO/mathematica-mcp",
        "run",
        "mathematica-mcp"
      ]
    }
  }
}
```

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
