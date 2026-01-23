# Installation Guide

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

Once prerequisites are installed, run:

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

Then restart Mathematica and your editor. **Done!**

To verify your installation:
```bash
uvx mathematica-mcp-full doctor
```

---

## Prerequisites

- **Mathematica 14.0+** — [Download](https://www.wolfram.com/mathematica/)
- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **wolframscript in your PATH** — See [below](#add-wolframscript-to-path)
- **uv package manager** (Recommended) — [Docs](https://docs.astral.sh/uv/)

### Install uv

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Add wolframscript to PATH

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

**Config file location:**
| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

1. Open Claude Desktop → **Settings** (gear icon) → **Developer** → **Edit Config**
2. Add `mathematica` to your `mcpServers`:

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

3. Save and **restart Claude Desktop**
4. Look for the 🔨 hammer icon in the chat input to confirm MCP is loaded

### Visual Studio Code

> Requires **VS Code 1.102+** with GitHub Copilot

Create `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "mathematica": {
      "type": "stdio",
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

Alternatively, run **MCP: Add Server** from the Command Palette and select **Workspace**.

> **Note**: VS Code uses `"servers"` (not `"mcpServers"`) and requires `"type": "stdio"`.
> 
> See [VS Code MCP documentation](https://code.visualstudio.com/docs/copilot/customization/mcp-servers) for advanced options.

### OpenAI Codex CLI

Codex stores MCP configuration in `~/.codex/config.toml`.

```bash
codex mcp add mathematica -- uv --directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp
```

<details>
<summary>Alternative: Edit config.toml directly</summary>

```toml
[mcp_servers.mathematica]
command = "uv"
args = ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
```

</details>

Verify with `/mcp` in the Codex TUI.

> See [Codex MCP documentation](https://developers.openai.com/codex/mcp/) for authentication and advanced options.

### Claude Code (CLI)

```bash
claude mcp add mathematica --scope user -- uv --directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp
```

<details>
<summary>Alternative: Add via JSON</summary>

```bash
claude mcp add-json mathematica --scope user '{
  "command": "uv",
  "args": ["--directory", "/YOUR/PATH/TO/mathematica-mcp", "run", "mathematica-mcp"]
}'
```

</details>

<details>
<summary>Alternative: Edit ~/.claude.json directly</summary>

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

</details>

Verify with `/mcp` inside Claude Code.

**Scopes:**
| Scope | Flag | Use case |
|-------|------|----------|
| Local | (default) | Current project only |
| User | `--scope user` | All projects on your machine |
| Project | `--scope project` | Shared via `.mcp.json` in repo |

> See [Claude Code MCP documentation](https://docs.anthropic.com/en/docs/claude-code/mcp) for remote servers and OAuth.

### Cursor Integration

[**Download Cursor**](https://cursor.com)

1. Go to **Settings > Features > MCP**.
2. Click **Add New MCP Server**.
3. Enter the following details:

| Field | Value |
|-------|-------|
| **Name** | `mathematica` |
| **Type** | `stdio` |
| **Command** | `uv` |
| **Args** | `--directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp` |

**Note**: Paste the *entire* string above into the "Args" field (or add them as separate arguments if the UI provides a list).

### Google Gemini CLI

Gemini CLI stores MCP configuration in `~/.gemini/settings.json`.

```bash
gemini mcp add mathematica -- uv --directory /YOUR/PATH/TO/mathematica-mcp run mathematica-mcp
```

<details>
<summary>Alternative: Edit settings.json directly</summary>

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

</details>

Verify with `/mcp` in Gemini CLI.

> See [Gemini CLI MCP documentation](https://geminicli.com/docs/tools/mcp-server/) for advanced options.

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
See [Add wolframscript to PATH](#add-wolframscript-to-path) in Prerequisites.

### MCP client can't connect
1. Verify Mathematica is running with the addon loaded
2. Check the absolute path in your client config is correct
3. Ensure no firewall is blocking port 9881

---

## 4. Advanced Configuration

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
