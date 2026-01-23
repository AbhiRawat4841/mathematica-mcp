# Installation Guide

There are two ways to install mathematica-mcp:

| Method | Best For | Time |
|--------|----------|------|
| [Quick Start](#quick-start-recommended) | Most users | ~2 minutes |
| [Manual Installation](#manual-installation) | Developers, custom setups | ~10 minutes |

---

## Quick Start (Recommended)

### Prerequisites

Before running the setup command, ensure you have:

1. **Mathematica 14.0+** with `wolframscript` in your PATH
   ```bash
   # Verify wolframscript is available
   wolframscript -version
   ```
   
   If not found, add to your PATH:
   - **macOS**: Add to `~/.zshrc`: `export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"`
   - **Linux**: Add to `~/.bashrc`: `export PATH="/usr/local/Wolfram/Mathematica/14.0/Executables:$PATH"`
   - **Windows**: Add `C:\Program Files\Wolfram Research\Mathematica\14.0\` to system PATH

2. **uv package manager**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

### One-Command Setup

Once prerequisites are installed, run **one** of these commands based on your editor:

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

# For Claude Code CLI
uvx mathematica-mcp-full setup claude-code
```

Then **restart Mathematica** and **restart your editor**. Done!

### Verify Installation

```bash
uvx mathematica-mcp-full doctor
```

---

## Manual Installation

Use this method if you want to:
- Modify or extend the MCP server code
- Use a development version
- Have more control over the installation

### Prerequisites

- **Mathematica 14.0+** — [Download](https://www.wolfram.com/mathematica/)
- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **wolframscript in your PATH** — See [below](#add-wolframscript-to-path)
- **uv package manager** — [Docs](https://docs.astral.sh/uv/)

#### Install uv

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Add wolframscript to PATH

After installing Mathematica, ensure `wolframscript` is accessible from your terminal.

**macOS** — add to `~/.zshrc` or `~/.bashrc`:
```bash
export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"
```

**Linux** — The installer typically creates symlinks in `/usr/local/bin`. If not:
```bash
export PATH="/usr/local/Wolfram/Mathematica/14.0/Executables:$PATH"
```

**Windows** — Mathematica usually adds it automatically. If not, add to your system PATH:
```
C:\Program Files\Wolfram Research\Mathematica\14.0\
```

**Verify installation:**
```bash
wolframscript -version
```

---

### Step 1: Clone and Install the Package

```bash
# Clone the repository
git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
cd mathematica-mcp

# Install dependencies
uv sync
```

### Step 2: Install Mathematica Addon

**CRITICAL**: This addon allows the server to communicate with the Wolfram Kernel.

```bash
wolframscript -file addon/install.wl
```

### Step 3: Restart Mathematica

**Close and reopen Mathematica** for the addon to load automatically.

After restarting, check the **Messages** window (⌘+Shift+M on macOS). You should see:
```
[MathematicaMCP] Server started on port 9881
```

If the server didn't start, manually start it in any Mathematica notebook:
```mathematica
Needs["MathematicaMCP`"]
StartMCPServer[]
```

### Step 4: Configure Your Editor

Choose your editor below. First, get the **absolute path** to the repository:
```bash
pwd
```
*(Replace `/YOUR/PATH/TO/mathematica-mcp` in the examples below with this output)*

#### Claude Desktop

**Config file location:**
| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

Add to your config:
```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

#### Cursor

**Config file:** `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

Or use the UI: **Settings > Features > MCP > Add New MCP Server**

#### VS Code

**Config file:** `~/.vscode/mcp.json`

```json
{
  "servers": {
    "mathematica": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

> **Note**: VS Code uses `"servers"` (not `"mcpServers"`) and requires `"type": "stdio"`.

#### OpenAI Codex CLI

**Config file:** `~/.codex/config.toml`

```toml
[mcp_servers.mathematica]
command = "uv"
args = ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
```

Or use the CLI:
```bash
codex mcp add mathematica -- uv --directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp
```

#### Google Gemini CLI

**Config file:** `~/.gemini/settings.json`

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

Or use the CLI:
```bash
gemini mcp add mathematica -- uv --directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp
```

#### Claude Code (CLI)

**Config file:** `~/.claude.json`

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "uv",
      "args": ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
    }
  }
}
```

Or use the CLI:
```bash
claude mcp add mathematica --scope user -- uv --directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp
```

### Step 5: Restart Your Editor

Restart your editor to load the MCP server. Look for the MCP indicator (e.g., 🔨 hammer icon in Claude Desktop).

---

## Troubleshooting

### "Addon directory not found" error with uvx
If you see this error after updating, clear the uvx cache:
```bash
uv cache clean mathematica-mcp-full
uvx mathematica-mcp-full setup <your-client>
```

Or force reinstall:
```bash
uvx --reinstall mathematica-mcp-full setup <your-client>
```

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
See [Add wolframscript to PATH](#add-wolframscript-to-path) above.

### MCP client can't connect
1. Verify Mathematica is running with the addon loaded
2. Check the absolute path in your client config is correct
3. Ensure no firewall is blocking port 9881

---

## Advanced Configuration

### Session Isolation

For multi-session use (e.g., multiple notebooks), use `session_id` parameter:
```python
execute_code(code="x = 5", session_id="notebook1")
execute_code(code="x = 10", session_id="notebook2")  # Independent variable
```

### Context Isolation

Use `isolate_context=True` to keep variables separate per session:
```python
execute_code(code="myVar = 42", session_id="session1", isolate_context=True)
```

### Authentication Token (Optional)

Set `MATHEMATICA_MCP_TOKEN` environment variable for secure connections:
```bash
export MATHEMATICA_MCP_TOKEN="your-secret-token"
```

### Deterministic Execution

For reproducible random results:
```python
execute_code(code="RandomReal[]", deterministic_seed=12345)
```
