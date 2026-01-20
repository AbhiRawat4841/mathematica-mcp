# Installation Guide

## 0. Prerequisites

- **Mathematica 14.0+** installed
- **Python 3.10+** installed
- **`wolframscript`** available in your PATH (Mathematica menu: Help -> Install wolframscript)

---

## 🤖 For LLM Agents

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

---

## 👤 For Humans (Manual Installation)

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
wolframscript -file addon/install.wl
```

### 2. Configure your client

Use this MCP server definition everywhere (only the config file location changes):

**Using uv (Recommended):**
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

**Using Python directly:**
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

**Config Locations:**

-   **Claude Desktop**:
    -   macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
    -   Windows: `%APPDATA%\Claude\claude_desktop_config.json`
-   **Claude Code**: Create `.mcp.json` in your project root.
-   **Cursor**: Settings -> Features -> MCP -> Add New MCP Server.
-   **OpenCode**: Add to project or global config.

### 3. Verify it works

1.  Start Mathematica.
2.  In a notebook, run `MCPServerStatus[]` and confirm `running -> True`.
3.  In your AI client, ask: "What is the value of Pi in Mathematica?".
