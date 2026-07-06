# Mathematica MCP

**A front-end / notebook automation layer for Mathematica, built for AI agents.**

A local MCP server that lets an AI agent drive a live Mathematica session: run code, create and edit notebooks, capture screenshots, verify derivations, and read `.nb` files without a kernel. Works with Claude, Cursor, VS Code, Codex, and Gemini.

It is designed to run **beside** the official [Wolfram Local MCP](https://www.wolfram.com/artificial-intelligence/mcp/local), not to replace it: Wolfram's server is the reference Wolfram-Language evaluator and documentation surface; this one is the notebook / front-end automation layer. See [How it compares](#how-it-compares).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mathematica 14+](https://img.shields.io/badge/Mathematica-14+-red.svg)](https://www.wolfram.com/mathematica/)
[![CI](https://github.com/AbhiRawat4841/mathematica-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhiRawat4841/mathematica-mcp/actions/workflows/ci.yml)
[![Repo](https://img.shields.io/github/v/tag/AbhiRawat4841/mathematica-mcp?label=repo)](https://github.com/AbhiRawat4841/mathematica-mcp/releases)
[![Published](https://img.shields.io/pypi/v/mathematica-mcp-full?label=published&cacheSeconds=300)](https://pypi.org/project/mathematica-mcp-full/)

> **v1.0 is a breaking release.** The default profile is now `lean` (12 tools) instead of `full` (82 tools). Set `MATHEMATICA_PROFILE=classic` (or `full`) to keep the old surface, and reinstall the Mathematica addon. See the [Migration Guide](docs/MIGRATION.md).

---

### Watch it in action

[![Mathematica MCP Demo](https://img.youtube.com/vi/TjGSkvVyc1Y/0.jpg)](https://www.youtube.com/watch?v=TjGSkvVyc1Y)

*An AI agent solving math, generating plots, and controlling a live Mathematica notebook. Errors are returned directly to the agent, no copy-pasting notebook output back into chat.*

---

## Why This Exists

LLMs can write Mathematica code, but they can't run it, control a live notebook, or verify their own results. This MCP server bridges that gap:

- **Live notebook control**: create, edit, evaluate, and screenshot Mathematica notebooks directly from your AI agent.
- **License-free notebook reading**: `read_notebook_file` parses `.nb` / `.wl` files with a Python-native parser — no kernel and no Mathematica license needed.
- **Warm execution**: computation runs on a persistent headless kernel session, so the agent's calls return in sub-second time instead of paying a cold `wolframscript` start-up on every request.
- **Error-aware execution**: Mathematica messages are fed back to the agent with a `suggested_fix` and, where a correction can be derived, a concrete `retry_with` call, so it can debug without you copying notebook output into chat.
- **Local and private**: core execution runs on your machine. Optional tools like `wolfram_alpha` and repository search contact Wolfram's cloud services only when invoked.

---

## The lean default (12 tools)

v1.0 ships a consolidated **12-tool** surface as the default profile. It exposes ~11.5 KB of tool schema (~2.9k tokens) versus ~61 KB / ~15k tokens for the old 82-tool surface — roughly a 5x cut in the context the agent pays before it does any work.

| Tool | What it does |
|------|--------------|
| `status()` | Connection, kernel version, active profile, features, and warm-path health (cold-execution count, kernel liveness, idle timeout). No params. |
| `notebooks(action, …)` | Manage notebooks: `list \| info \| create \| open \| save \| close \| export`. |
| `cells(action, …)` | Read notebook cells: `list \| read \| select \| scroll`. |
| `edit_cells(action, …)` | Edit cells: `write` (content/style/position) `\| delete`. |
| `evaluate(code, target, …)` | Run code on `kernel \| notebook \| cell \| selection`; `dry_run=True` checks syntax; `file=` runs a `.wl` script; `cursor=` pages long output. |
| `screenshot(scope, …)` | Capture a PNG of a `notebook \| cell \| expression`. |
| `verify_derivation(steps, …)` | Check a sequence of expressions step-by-step (runs warm on the persistent session). |
| `kernel(action, …)` | Kernel admin: `state \| messages \| restart \| load_package \| packages \| inspect`. |
| `vars(action, …)` | Kernel variables: `list \| get \| set \| clear \| clear_all`. `clear` requires a `name` or `pattern`; wiping everything is the explicit `clear_all`. |
| `read_notebook_file(path, mode, …)` | Parse a `.nb` / `.wl` file **without a kernel or license**: `markdown \| wolfram \| outline \| json \| plain`. `cursor=` pages long output. |
| `guide(topic)` | On-demand help: `workflow \| errors \| notebook_hygiene \| screenshots \| v15 \| profiles \| toolsets \| batch`. |
| `batch(ops)` | Run multiple addon ops in a single round trip. |

Each consolidated tool is a thin wrapper over the exact internals the classic surface uses — the same code, just fewer schemas in the agent's context.

---

## Tool Profiles

Set the profile with `--profile` at setup time or the `MATHEMATICA_PROFILE` env var.

| Profile | Tools | Best for |
|---------|-------|----------|
| `lean` **(default)** | 12 | Consolidated surface for everyday agent use; extras opt-in via `MATHEMATICA_TOOLSETS`. |
| `classic` (alias `full`) | 82 | The complete legacy surface, byte-identical to pre-1.0. |
| `math` | ~28 | Pure computation, no notebook UI. |
| `notebook` | ~48 | Notebook read/write/screenshot. |

### Opt-in extras for `lean`

Add extra tool groups to the lean profile with `MATHEMATICA_TOOLSETS` (comma-separated). They can only enable tools, never remove lean tools:

```bash
export MATHEMATICA_TOOLSETS=data_io,graphics_plus,cloud,debug
```

| Name | Adds |
|------|------|
| `data_io` | Data import/export tools. |
| `graphics_plus` | Graphics inspection, export, plot comparison, animation. |
| `cloud` | Wolfram Alpha, natural-language interpretation, entities, units, constants. |
| `debug` | Trace / timing / journal tools. |
| `notebook_files` | Legacy notebook file tools. |
| `notebook_edit` | Advanced notebook editing tools. |
| `symbols` | Symbol lookup / documentation tools. |
| `math_aliases` | `mathematica_integrate`, `mathematica_solve`, etc. |
| `repository` | Function + data repository search. |
| `async_jobs` | Async computation submit/poll. |
| `cache` | Expression cache management tools. |

---

## Warm execution, guidance, and V15

- **Warm funnel.** The 12 tools that used to spawn a cold `wolframscript` subprocess (including `verify_derivation`) — plus the symbol-index build — now run on the persistent `WolframLanguageSession`, returning warm in sub-second time; the cold subprocess remains as a flagged fallback. Every response carries an `execution_method`, and `status()` surfaces a **cold-execution counter** (0 on the lean happy path), kernel-session liveness, and the idle-shutdown timeout (env `MATHEMATICA_KERNEL_IDLE_TIMEOUT`, default `1800`s; `0` disables).
- **Guidance v2.** Failed evaluations carry `error_analysis` with a `suggested_fix`, a `next_step`, and — when a corrected call can actually be derived from context — a concrete `retry_with` you can rerun, on **all** evaluate paths. Oversized output is capped (env `MATHEMATICA_MAX_OUTPUT_CHARS`, default `4000`) and the remainder is stashed behind a continuation `cursor` you pass back to the same tool. Notebook-touching addon responses carry a `state_delta` (`notebook` / `cell_count` / `kernel_busy`; pure-kernel calls skip it for speed), `guide(topic)` gives targeted help on demand, and the server ships profile-aware instructions plus 6 MCP prompts.
- **Mathematica 15 first-class.** On Mathematica ≥15, agent-created notebooks set `ShowChatbar->False` (pass `show_chatbar=True` to override). 14.x stays supported behind `$VersionNumber >= 15.` guards. The Python client and addon share a `protocol_version` handshake (currently `3`): because the addon lives in `$UserBaseDirectory/Kernel/init.m` and does **not** update with `pip`, a version skew is detected and `status()` tells you to reinstall.

---

## How it compares

This server runs **alongside** the official Wolfram Local MCP — `setup <client> --with-official` writes the official server's config next to this one so they run side by side. Overlap is deliberate where it helps agents; the differentiator is notebook / front-end automation that runs without a license round trip.

| Capability | Official Wolfram Local MCP | **This MCP** |
|------------|:--------------------------:|:------------:|
| Wolfram-Language evaluation | `WolframLanguageEvaluator` | `evaluate` (warm persistent kernel) |
| Wolfram Alpha | `WolframAlpha` | `wolfram_alpha` (opt-in `cloud`) |
| Symbol docs / definitions | `SymbolDefinition`, `CreateSymbolDoc` | `kernel(action="inspect")`, `symbols` extra |
| Read a notebook file | `ReadNotebook` (needs kernel) | **`read_notebook_file` — Python-native, no kernel / license** |
| Write a notebook file | `WriteNotebook` | `notebooks`, `edit_cells` (live front-end) |
| Live notebook control (create/edit/eval/screenshot) | No | **Yes** |
| Interactive UIs (sliders, `Manipulate`) | No | **Yes, in the live front-end** |
| Derivation verification | No | **`verify_derivation`** |
| Doc search / code inspection / test reports | `CodeInspector`, `TestReport` | Deliberately **not duplicated** — use the official server |

`ReadNotebook` / `WriteNotebook` overlap the notebook tools here, but the official `ReadNotebook` runs through a kernel; `read_notebook_file` parses the `.nb` directly in Python, so an agent can read notebooks with no license consumed and no kernel start-up.

---

## Quick Start

From install to first working notebook plot in under 2 minutes.

### Prerequisites

1. **Mathematica 14.0+** (15+ recommended) with `wolframscript` in your PATH
   - [Download Mathematica](https://www.wolfram.com/mathematica/)
   - macOS: add to `~/.zshrc`: `export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"`

2. **uv package manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### One-Command Setup

> The PyPI package and CLI are named **`mathematica-mcp-full`** (unchanged in 1.0 — the name predates the lean default).

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

# Optional: pick a profile (default is "lean")
uvx mathematica-mcp-full setup claude-desktop --profile classic
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

## What You Can Ask For

**"Integrate x^2 sin(x) from 0 to pi, then verify the result."**

```text
evaluate("Integrate[x^2 Sin[x], {x, 0, Pi}]")   =>  -4 + Pi^2
verify_derivation(steps=["Integrate[x^2 Sin[x], {x, 0, Pi}]", "-4 + Pi^2"])
=> Step 1 → 2: ✓ VALID
   All steps are valid!
```

**"Plot the sombrero function in a new notebook."**

```text
notebooks(action="create", title="Sombrero")
evaluate("Plot3D[Sinc[Sqrt[x^2+y^2]], {x,-4,4}, {y,-4,4}]", target="notebook")
=> [3D surface plot rendered in the live notebook]
```

**"Read the derivation in this notebook without opening Mathematica."**

```text
read_notebook_file("paper/derivation.nb", mode="markdown")
=> [structured markdown, no kernel or license used]
```

---

## Who This Is For

| Audience | Use Case |
|----------|----------|
| Researchers using LLM coding assistants | Run Mathematica from Claude/Cursor/VS Code without leaving your editor |
| Data scientists | Import, transform, and visualize data through natural language |
| Educators | Create interactive Mathematica notebooks through AI conversation |
| **Not for** | Production web services, untrusted multi-tenant environments |

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

*   **[Migration Guide (0.9.x → 1.0)](docs/MIGRATION.md)**: breaking default-profile change and old→new tool mapping
*   **[Technical Reference](docs/technical-reference.md)**: Architecture, tools, and configuration
*   **[Security Model](SECURITY.md)**: Threat model, permissions, and vulnerability reporting
*   **[Benchmarks](docs/benchmarks.md)**: Performance data and reproduction steps
*   **[Contributing](CONTRIBUTING.md)**: Development setup, testing, and PR process
*   **[Changelog](CHANGELOG.md)**: Version history
*   **[Examples](docs/examples/)**: Polished agent session walkthroughs

---

## License
MIT License
