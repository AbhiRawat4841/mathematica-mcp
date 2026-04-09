# Mathematica MCP

**Turn Mathematica into a first-class tool for AI agents.**

A local MCP server that lets AI agents run Mathematica, control notebooks, and verify results. Works with Claude, Cursor, VS Code, Codex, and Gemini.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)
[![CI](https://github.com/AbhiRawat4841/mathematica-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhiRawat4841/mathematica-mcp/actions/workflows/ci.yml)
[![Repo](https://img.shields.io/github/v/tag/AbhiRawat4841/mathematica-mcp?label=repo)](https://github.com/AbhiRawat4841/mathematica-mcp/releases)
[![Published](https://img.shields.io/pypi/v/mathematica-mcp-full?label=published&cacheSeconds=300)](https://pypi.org/project/mathematica-mcp-full/)

---

### Watch it in action

[![Mathematica MCP Demo](https://img.youtube.com/vi/TjGSkvVyc1Y/0.jpg)](https://www.youtube.com/watch?v=TjGSkvVyc1Y)

*An AI agent solving math, generating plots, and controlling a live Mathematica notebook. Errors are returned directly to the agent, no copy-pasting notebook output back into chat.*

---

## Why This Exists

LLMs can write Mathematica code, but they can't run it, verify it, or interact with live notebooks. This MCP server bridges that gap:

- **Live notebook control**: create, edit, evaluate, and screenshot Mathematica notebooks directly from your AI agent
- **Symbolic + numeric + visual in one MCP**: ~79 tools covering algebra, calculus, plotting, data import/export, Wolfram Alpha, and interactive UIs
- **Agent-optimized**: compact response shaping, session state tools, and computation journaling designed for how LLM agents actually work
- **Error-aware execution**: Mathematica errors and warnings are returned directly to the agent, so it can debug without you manually copying notebook output back into chat
- **Local and private**: everything runs on your machine, no code leaves your environment

> Ask your agent for a derivation, a 3D plot, a notebook edit, or a verification step, and it can actually do it.

---

## Who This Is For

| Audience | Use Case |
|----------|----------|
| Researchers using LLM coding assistants | Run Mathematica from Claude/Cursor/VS Code without leaving your editor |
| Data scientists | Import, transform, and visualize data through natural language |
| Educators | Create interactive Mathematica notebooks through AI conversation |
| **Not for** | Production web services, untrusted multi-tenant environments |

---

## What You Can Ask For

**"Integrate x^2 sin(x) from 0 to pi, then verify the result."**

```text
execute_code("Integrate[x^2 Sin[x], {x, 0, Pi}]")  =>  -4 + Pi^2
verify_derivation(steps=["Integrate[...", "-4 + Pi^2"])  =>  All steps valid
```

**"Plot the sombrero function in a new notebook."**

```text
create_notebook(title="Sombrero")
execute_code("Plot3D[Sinc[Sqrt[x^2+y^2]], {x,-4,4}, {y,-4,4}]", style="notebook")
=> [3D surface plot rendered in live notebook]
```

**"Interactive: slider for Sin[n x]"**

```text
execute_code("Manipulate[Plot[Sin[n x],{x,0,2Pi}],{n,1,10}]", style="interactive")
=> [Live slider UI in Mathematica frontend]
```

Beyond these: **data import/export** (hundreds of formats), **Wolfram Alpha queries**, **notebook reading/analysis**, **symbolic debugging**, and more. See the [Technical Reference](docs/technical-reference.md) for the full tool list.

---

## How It Compares

| Capability | Plain LLM | Copy-paste to Mathematica | **This MCP** |
|------------|:---------:|:-------------------------:|:------------:|
| Write Mathematica code | Yes | Yes | Yes |
| Execute and return results | No | Manual | **Automatic** |
| Generate plots/images | No | Manual | **Inline in chat** |
| Control live notebooks | No | No | **Yes** |
| Verify derivations | No | Manual | **One tool call** |
| Interactive UIs (sliders) | No | Manual | **Yes** |
| Error-aware debugging | No | Manual copy-paste | **Yes** |
| Session state awareness | No | No | **Yes** |
| Private / local execution | N/A | Yes | **Yes** |

---

## Quick Start

From install to first working notebook plot in under 2 minutes.

### Prerequisites

1. **Mathematica 14.0+** with `wolframscript` in your PATH
   - [Download Mathematica](https://www.wolfram.com/mathematica/)
   - macOS: Add to `~/.zshrc`: `export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"`

2. **uv package manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### One-Command Setup

```bash
# For Claude Desktop
uvx mathematica-mcp-full setup claude-desktop

# For Cursor
uvx mathematica-mcp-full setup cursor

# For VS Code (requires GitHub Copilot Chat extension)
uvx mathematica-mcp-full setup vscode

# For OpenAI Codex CLI
uvx mathematica-mcp-full setup codex

# For Google Gemini CLI
uvx mathematica-mcp-full setup gemini

# For Claude Code CLI
uvx mathematica-mcp-full setup claude-code

# Optional: select a tool profile (default is "full")
uvx mathematica-mcp-full setup claude-desktop --profile notebook
```

Then restart Mathematica and your editor. Done!

<details>
<summary>VS Code: Alternative setup via Command Palette</summary>

> **Prerequisite:** [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) extension must be installed - MCP support is built into Copilot.

1. Press `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows)
2. Type "MCP" -> Select **"MCP: Add Server"**
3. Choose **"Command (stdio)"**: *not "pip"*
4. Enter command: `uvx`
5. Enter args: `mathematica-mcp-full`
6. Name it: `mathematica`
7. Choose scope: Workspace or User

</details>

<details>
<summary>Alternative: Interactive Installer</summary>

```bash
bash <(curl -sSL https://raw.githubusercontent.com/AbhiRawat4841/mathematica-mcp/main/install.sh)
```

</details>

### Verify Installation

```bash
uvx mathematica-mcp-full doctor
```

> **Tip:** If you encounter errors after updating, clear the cache:
> ```bash
> uv cache clean mathematica-mcp-full && uvx mathematica-mcp-full setup <client>
> ```

---

## Execution Styles

Control where results appear with natural language or the `style` parameter:

| Say this... | `style=` | What happens |
|-------------|----------|-------------|
| "calculate ...", "compute ...", "what is ..." | `"compute"` | Result appears as text in chat |
| "plot ...", "show ...", "in notebook ..." | `"notebook"` | Executes in the current Mathematica notebook |
| "in new notebook: ..." | *two-step* | `create_notebook()` then `execute_code(style="notebook")` |
| "interactive ...", "manipulate ..." | `"interactive"` | Live front-end evaluation (sliders, animations) |

If you don't include a keyword, the default depends on your [tool profile](#tool-profiles).

---

## Tool Profiles

Choose how many tools to expose:

| Profile | Tools | Best for |
|---------|-------|----------|
| `math` | ~25 | Pure computation, no notebook UI |
| `notebook` | ~45 | + notebook read/write/screenshot |
| `full` (default) | ~79 | + advanced notebook ops, repositories, admin |

Pass `--profile` during setup or set `MATHEMATICA_PROFILE` env var.

---

## Built for Agent Workflows

The server is designed for how LLM agents actually work: long conversations with context limits, intermittent failures, and token budgets:

| Feature | What it does | How to use |
|---------|-------------|------------|
| **Compact Responses** | Strip verbose metadata, keep essentials. Saves tokens. | `response_detail="compact"` on `execute_code` |
| **Session Brief** | One-call snapshot: connection status, recent errors, routing advice | `get_session_brief()` |
| **Computation Journal** | Ring buffer of recent computations that helps agents recover context across long conversations | `get_computation_journal()` |
| **Smart Caching** | Pure expressions (`Sin[Pi]`) survive variable mutations without re-evaluation | Always on |
| **Diagnostic Mode** | Full response + cache epoch + routing hints for debugging | `response_detail="diagnostic"` |

### Routing Intelligence (opt-in)

For power users, the server can learn from transport outcomes and adapt:

```bash
# Observe mode: collect stats, no behavior change
export MATHEMATICA_ROUTING_MEMORY=observe

# Advise mode: + routing hints + enables adaptive routing
export MATHEMATICA_ROUTING_MEMORY=advise
export MATHEMATICA_ROUTING_ACTION=compute_cli_skip  # optional: skip failing transport
```

The adaptive routing circuit-breaker automatically skips persistently failing compute CLI transport with half-open probe recovery. See the [Technical Reference](docs/technical-reference.md#intelligent-routing--observability) for details.

> **Privacy:** No full Mathematica code or notebook content is persisted to disk. Routing memory stores only aggregate counters; the in-memory journal stores short code/output previews (not persisted).

---

## Manual Installation

For full details, troubleshooting, and advanced configuration, see the **[Installation Guide](docs/installation.md)**.

<details>
<summary>Click to expand quick manual setup</summary>

1.  **Clone & Install**:
    ```bash
    git clone https://github.com/AbhiRawat4841/mathematica-mcp.git
    cd mathematica-mcp
    uv sync
    ```

2.  **Install Mathematica Addon**:
    ```bash
    wolframscript -file addon/install.wl
    ```
    *Restart Mathematica after this step.*

3.  **Configure your editor**: add the MCP server to your client's config file. See the **[Installation Guide](docs/installation.md#step-4-configure-your-editor)** for Claude Desktop, Cursor, VS Code, and other client configs.

</details>

---

## Documentation

*   **[Technical Reference](docs/technical-reference.md)**: Architecture, tools, and configuration
*   **[Security Model](SECURITY.md)**: Threat model, permissions, and vulnerability reporting
*   **[Benchmarks](docs/benchmarks.md)**: Performance data and reproduction steps
*   **[Contributing](CONTRIBUTING.md)**: Development setup, testing, and PR process
*   **[Changelog](CHANGELOG.md)**: Version history
*   **[Examples](docs/examples/)**: Polished agent session walkthroughs

---

## License
MIT License
