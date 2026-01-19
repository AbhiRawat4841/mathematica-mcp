# Beginner Quick Start Guide

Get Mathematica controlled by AI in 5 minutes.

## Prerequisites

1.  **Mathematica** (Version 14.1 recommended) installed and running.
2.  **Python** (Version 3.10+) installed.
    *   *Check by opening Terminal and typing:* `python3 --version`
3.  **This Repository** downloaded to your Mac.

---

## Step 1: Install the Software

Open your **Terminal** app and run these 3 commands one by one:

```bash
# 1. Go to the downloaded folder (adjust path if needed)
cd ~/Downloads/mathematica-mcp

# 2. Install the Python connector
pip install -e .

# 3. Install the Mathematica Addon
cd addon
wolframscript -file install.wl
```

*If you see "Success" or "Server started", you're good!*

---

## Step 2: Connect Your AI App

Choose your app below.

### Claude for Desktop (Recommended)

1.  Download the [Claude Desktop App](https://claude.ai/download).
2.  Open the config file:
    *   **Mac**: Open Terminal and type `open ~/Library/Application\ Support/Claude/claude_desktop_config.json`
    *   *If the file doesn't exist, create it with a text editor.*
3.  Paste this code inside the `{ }` brackets:

    ```json
    "mcpServers": {
      "mathematica": {
        "command": "python3",
        "args": ["-m", "mathematica_mcp", "run"]
      }
    }
    ```
4.  Restart Claude Desktop. Look for a plug icon.

### Cursor Editor

1.  Open Cursor.
2.  Go to **Settings** (Gear icon) > **Features** > **MCP**.
3.  Click **+ Add New MCP Server**.
4.  Enter:
    *   **Name**: `mathematica`
    *   **Type**: `command`
    *   **Command**: `python3 -m mathematica_mcp run`
5.  Click **Save**. A green light means it's working.

### VS Code

1.  Install the **"MCP Server"** extension (by Robby or similar).
2.  Open your User Settings JSON (`Cmd + Shift + P` -> "Open Settings (JSON)").
3.  Add this to your configuration:

    ```json
    "mcpServers": {
      "mathematica": {
        "command": "python3",
        "args": ["-m", "mathematica_mcp", "run"]
      }
    }
    ```

### Claude Web / ChatGPT App / Gemini
*Advanced Setup Required*

The web versions of these apps cannot connect directly to your Mac ("localhost") for security reasons. To use them, you must expose your server to the internet using a bridge (like ngrok or cloudflared), which is recommended for advanced users only.

**We strongly recommend using Claude Desktop or Cursor for the best experience.**

---

## Step 3: Verify It Works

1.  Open **Mathematica**.
2.  Type `MCPServerStatus[]` and hit Shift+Enter.
3.  You should see: `<| "running" -> True, "port" -> 9881 ... |>`
4.  Go to your AI app and ask: **"What is the value of Pi in Mathematica?"**

---

## Troubleshooting

*   **"Command not found: wolframscript"**: Open Mathematica, go to the menu **Help > Install wolframscript**.
*   **"Connection refused"**: Make sure Mathematica is running and the addon is loaded. Run `StartMCPServer[]` in a notebook to force start it.
