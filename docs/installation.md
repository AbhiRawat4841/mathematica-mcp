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

## 🤖 For LLM Agents (Automated Setup)

If you are an AI assistant installing this for a user, run this script:

```bash
set -e
# 1. Install dependencies
if command -v uv >/dev/null 2>&1; then uv sync; else pip install -e .; fi

# 2. Install Addon
if ! command -v wolframscript >/dev/null 2>&1; then echo "Error: wolframscript missing"; exit 1; fi
wolframscript -file addon/install.wl

# 3. Report
echo "Success! Config path: $(pwd)"
```
