# Beginner Quick Start Guide

**Get Mathematica controlled by AI in 5 minutes.**

This guide assumes you have **no prior experience** with MCP or terminal commands. We will walk you through every step.

---

## Prerequisites (What you need)

Before starting, make sure you have:

1.  **Mathematica** (Version 14.0 or newer) installed and running on your computer.
2.  **Python** (Version 3.10 or newer) installed.
    *   *How to check:* Open your **Terminal** (on Mac) or **Command Prompt** (on Windows) and type `python3 --version`. If you see a number like `3.11.x`, you are ready.
3.  **This Folder** downloaded to your computer.
    *   *Note the path:* We'll assume you downloaded it to your `Downloads` folder.

---

## Step 1: Install the Software

We need to install two things: a **Python connector** (talks to the AI) and a **Mathematica addon** (talks to Mathematica).

1.  Open your **Terminal** app.
2.  Copy and paste the following commands one by one, pressing **Enter** after each line:

    ```bash
    # 1. Go to the downloaded folder
    cd ~/Downloads/mathematica-mcp

    # 2. Install the Python connector
    pip install -e .
    ```

    *(If `pip` fails with "command not found", try `pip3 install -e .`)*

3.  Now, install the Mathematica addon:

    ```bash
    # 3. Go to the addon folder
    cd addon

    # 4. Run the installer script
    wolframscript -file install.wl
    ```

    *If you see "Success" or "Server started", you are done with installation!*

    > **"Command not found: wolframscript"?**
    > If step 4 failed, open the Mathematica app, go to the menu bar, click **Help > Install wolframscript**, and follow the instructions. Then try step 4 again.

---

## Step 2: Connect Your AI App

Now we need to tell your AI app where to find this tool. Choose your app below.

### Option A: Claude for Desktop (Recommended)

1.  Download and install the [Claude Desktop App](https://claude.ai/download).
2.  Open the configuration file:
    *   **On Mac**: Open Terminal and paste this command:
        `open ~/Library/Application\ Support/Claude/claude_desktop_config.json`
    *   **On Windows**: Open `%APPDATA%\Claude\claude_desktop_config.json` in Notepad.
    *   *(If the file is blank or doesn't exist, just create a new text file with that name).*
3.  Delete everything in the file and paste this exact code:

    ```json
    {
      "mcpServers": {
        "mathematica": {
          "command": "uv",
          "args": ["--directory", "/Users/YOUR_USERNAME/Downloads/mathematica-mcp", "run", "mathematica-mcp"]
        }
      }
    }
    ```

    **CRITICAL**: Replace `/Users/YOUR_USERNAME/Downloads/mathematica-mcp` with the **real path** to the folder you downloaded.
    *   *Tip:* In Terminal, type `pwd` while inside the folder to see the real path.

4.  Save the file and **completely restart** the Claude Desktop app.
5.  Look for a **plug icon** 🔌 in the Claude input box. If it's there, you're connected!

### Option B: Cursor Editor

1.  Open Cursor.
2.  Click the **Gear icon** (Settings) in the top right.
3.  Scroll down to **Features** > **MCP**.
4.  Click **+ Add New MCP Server**.
5.  Fill in these details:
    *   **Name**: `mathematica`
    *   **Type**: `command`
    *   **Command**: `uv --directory /Users/YOUR_USERNAME/Downloads/mathematica-mcp run mathematica-mcp`
    *(Remember to replace the path with your real path!)*
6.  Click **Save**. If the light turns green, it works.

---

## Step 3: Verify It Works

1.  Make sure **Mathematica is open**.
2.  Open a new notebook in Mathematica, type `MCPServerStatus[]`, and press **Shift+Enter**.
    *   You should see `running -> True`.
3.  Go to your AI app (Claude or Cursor) and ask:
    > **"What is the value of Pi in Mathematica?"**
4.  The AI should reply with the correct value (`3.14159...`).

**Congratulations! You now have a super-powered AI pair programmer.**

---

## Troubleshooting (Common Problems)

*   **"Connection refused"**:
    This usually means Mathematica isn't running the server.
    *   **Fix**: Open Mathematica and run `StartMCPServer[]` in a notebook.

*   **"Port 9881 Busy"**:
    An old session might be stuck.
    *   **Fix**: Open Terminal and run `lsof -i :9881` to find the Process ID (PID), then run `kill -9 PID` (replace PID with the number).

*   **"Command not found: uv"**:
    If your computer doesn't have `uv` installed, you can change the command in your config file to use python directly:
    *   **Command**: `python3`
    *   **Args**: `["-m", "mathematica_mcp"]`
    *(Make sure you are pointing to the installed package location)*.
