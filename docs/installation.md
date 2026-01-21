# Installation Guide

## Prerequisites

- **Mathematica 14.0+** installed
- **Python 3.10+** installed
- **wolframscript** available in your PATH
- **uv** package manager (Recommended)

### Install uv
**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 1. Server Setup

### Step 1: Install the Package
```bash
# Clone the repository
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp

# Install dependencies
uv sync
```

### Step 2: Install Mathematica Addon
**CRITICAL**: This addon allows the server to talk to the Wolfram Kernel.
```bash
wolframscript -file addon/install.wl
```

### Step 3: Restart Mathematica
**Close and reopen Mathematica** for the addon to load automatically.

### Step 4: Verify Installation
After restarting Mathematica, check the **Messages** window (⌘+Shift+M on macOS). You should see:
```
[MathematicaMCP] Server started on port 9881
```

If the server didn't start, you can manually start it in any Mathematica notebook:
```mathematica
Needs["MathematicaMCP`"]
StartMCPServer[]
```

---

## 2. Client Integration

Choose your editor below. You need the **Absolute Path** to this repository.
Run this command to get your path:
```bash
pwd
```
*(Replace `/YOUR/PATH/TO/mathematica-mcp` in the examples below with this output)*

### Claude for Desktop

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

### Cursor Integration

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

**Note**: Paste the *entire* string above into the "Args" field (or add them as separate arguments if the UI provides a list).

### Visual Studio Code Integration

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

---

## 3. Troubleshooting

### Server didn't start automatically
Manually start it in any Mathematica notebook:
```mathematica
Needs["MathematicaMCP`"]
StartMCPServer[]
```

### Port already in use
Change the port in Mathematica:
```mathematica
MathematicaMCP`Private`$MCPPort = 9882;
RestartMCPServer[]
```
Then set the environment variable for the Python client:
```bash
export MATHEMATICA_PORT=9882
```

### wolframscript not found
Ensure Mathematica is installed and wolframscript is in your PATH:
```bash
# macOS - add to ~/.zshrc or ~/.bashrc
export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"
```

### MCP client can't connect
1. Verify Mathematica is running with the addon loaded
2. Check the absolute path in your client config is correct
3. Ensure no firewall is blocking port 9881
