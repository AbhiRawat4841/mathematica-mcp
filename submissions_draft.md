# Registry Submissions & Posts — Drafts for Review

---

## 1. Official MCP Registry (modelcontextprotocol/registry)

**Method:** `mcp-publisher` CLI tool (PRs not accepted)

**Steps to execute:**
1. Add verification string to README.md (first line after title):
   ```
   <!-- mcp-name: io.github.abhirawat4841/mathematica-mcp -->
   ```
2. Push to GitHub and publish to PyPI (already done — v0.9.0 is live)
3. Install mcp-publisher:
   ```bash
   brew install mcp-publisher
   # or download binary from https://github.com/modelcontextprotocol/registry/releases
   ```
4. Create `server.json` in repo root:
   ```json
   {
     "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
     "name": "io.github.abhirawat4841/mathematica-mcp",
     "description": "Full Wolfram Language integration for AI agents — 82 tools across 3 profiles for symbolic computation, notebook control, visualization, and data workflows",
     "repository": {
       "url": "https://github.com/AbhiRawat4841/mathematica-mcp",
       "source": "github"
     },
     "version": "0.9.0",
     "packages": [
       {
         "registryType": "pypi",
         "identifier": "mathematica-mcp-full",
         "version": "0.9.0",
         "transport": {
           "type": "stdio"
         }
       }
     ]
   }
   ```
5. Authenticate and publish:
   ```bash
   mcp-publisher login github
   mcp-publisher publish
   ```

**Note:** A different project (`lars20070/mathematica-mcp`, PyPI: `mathematica_mcp`, 4 tools) is already registered. Our project is distinct: PyPI package `mathematica-mcp-full`, 82 tools.

---

## 2. PulseMCP

**Method:** Web form at https://www.pulsemcp.com/submit

**Submit URL:** `https://github.com/AbhiRawat4841/mathematica-mcp`

PulseMCP also auto-ingests from the Official MCP Registry weekly, so submitting to #1 first will eventually propagate here. The web form is a backup.

---

## 3. Glama

**Method:** Add server via https://glama.ai/mcp/servers + claim ownership

**Step 1:** Click "Add Server" and submit `https://github.com/AbhiRawat4841/mathematica-mcp`

**Step 2:** Add `glama.json` to repo root:
```json
{
  "$schema": "https://glama.ai/mcp/schemas/server.json",
  "maintainers": ["AbhiRawat4841"]
}
```

**Step 3:** Commit and push, then claim ownership through Glama's web flow.

---

## 4. awesome-mcp-servers (punkpeye/awesome-mcp-servers — 84K stars)

**Method:** PR to README.md

**Category:** `### 🧮 Data Science Tools` (where existing Wolfram Alpha entries live)

**Entry to add (alphabetical order, after existing entries):**

```markdown
- [AbhiRawat4841/mathematica-mcp](https://github.com/AbhiRawat4841/mathematica-mcp) 🐍 🏠 🍎 — Full Wolfram Language integration with 82 tools across 3 profiles: symbolic computation, notebook GUI control, visualization, data import/export, and debugging.
```

**PR title:** `Add AbhiRawat4841/mathematica-mcp`
**PR body:**
```
Adds mathematica-mcp — a comprehensive MCP server for Wolfram Language with 82 tools across 3 configurable profiles (math/notebook/full).

Unlike existing Wolfram Alpha API wrappers in this list, this server connects to a local Mathematica kernel and provides:
- Symbolic computation (integrate, solve, verify derivations)
- Notebook GUI control (create, read, evaluate, screenshot, export)
- Visualization (Plot, Plot3D, export, animation)
- Data import/export (250+ formats)
- Debugging tools (trace, time, syntax check)

MIT licensed, Python 3.10+, supports Claude Desktop, Cursor, VS Code, Codex, Gemini, and Claude Code.
```

---

## 5. awesome-mcp-servers (appcypher/awesome-mcp-servers — 5K stars)

**Method:** PR to README.md

**Category:** `🧬 Research & Data`

**Entry:**
```markdown
- [mathematica-mcp](https://github.com/AbhiRawat4841/mathematica-mcp) - Full Wolfram Language MCP server with 82 tools for symbolic computation, notebook control, and visualization. Python.
```

---

## 6. Wolfram Community Post

**Thread to reply to:** https://community.wolfram.com/groups/-/m/t/3444798
("The pizza agent: How can I use MCP to build agents with Mathematica?")

**Draft reply:**

---

**Subject:** Open-source MCP server with 82 tools for Mathematica

I've been working on an open-source MCP server that gives AI agents deep access to a local Mathematica installation: [mathematica-mcp](https://github.com/AbhiRawat4841/mathematica-mcp).

Unlike minimal wrappers that expose a single `execute_code` tool, this server provides **82 tools across 3 configurable profiles** (math/notebook/full), covering:

- **Symbolic computation** — Integrate, differentiate, solve, simplify, verify derivations step-by-step
- **Notebook GUI control** — Create, open, read, evaluate, screenshot, and export live .nb files from outside the Mathematica frontend
- **Visualization** — Plot, Plot3D, export graphics, create animations, compare plots
- **Data workflows** — Import/export 250+ formats, dataset operations, entity lookup
- **Knowledge base** — Wolfram Alpha queries, physical constants, unit conversions
- **Debugging** — Trace evaluation, time expressions, check syntax, inspect kernel state

It works with Claude Desktop, Cursor, VS Code, OpenAI Codex, Google Gemini, and Claude Code via one-command setup:

```
uvx mathematica-mcp-full setup claude-desktop
```

**Architecture:** A Python MCP server communicates with a Mathematica addon over TCP on localhost. For headless environments, it falls back to wolframscript subprocesses. Notebook parsing has three backends: offline Python parser (fast, no kernel needed), kernel semantic parser (full fidelity), and live addon (GUI control).

**Performance highlights:**
- Symbol index lookup in ~1.2ms vs. ~14s per subprocess (11,700x speedup)
- Atomic notebook execution: single round-trip for create + write + evaluate
- Raster cache and epoch-based invalidation to avoid redundant computation

It's MIT licensed, requires Mathematica 14+ and Python 3.10+, and has CI with a 6-job test matrix (351 tests).

I'd be very interested to hear from this community about use cases, missing features, or integration ideas. The technical reference is at: [docs/technical-reference.md](https://github.com/AbhiRawat4841/mathematica-mcp/blob/main/docs/technical-reference.md)

---

## 7. Show HN Post

**Title:** `Show HN: Mathematica MCP – 82-tool MCP server for Wolfram Language`

**Body:**

---

I built an MCP server that gives AI agents comprehensive access to a local Mathematica installation — 82 tools across 3 configurable profiles.

Most Wolfram/Mathematica MCP servers ship 1-5 tools (typically just "execute code"). This one provides specialized tools for symbolic computation, notebook GUI control, visualization, data workflows, and debugging, so the LLM can reason at a higher abstraction level instead of packing everything into monolithic Wolfram expressions.

**Key capabilities:**
- Symbolic math: integrate, solve, verify multi-step derivations
- Full notebook control: create, read, evaluate, screenshot, export .nb files
- Graphics: Plot, Plot3D, export, animations, side-by-side comparison
- Data: import/export 250+ formats, dataset operations
- 3 profiles: math (28 tools), notebook (48), full (82) — right-size for the task

**How it works:** Python MCP server + Mathematica addon communicating over TCP on localhost. Three execution paths: live addon (GUI control), wolframscript subprocess (headless), wolframclient kernel session (persistent). Offline notebook parsing via pure Python for fast reads without launching Mathematica.

**Performance:** Symbol index search in 1.2ms vs 14s per subprocess call. Atomic notebook execution. Raster cache. Epoch-based cache invalidation.

One-command setup for Claude Desktop, Cursor, VS Code, Codex, Gemini, Claude Code:
```
uvx mathematica-mcp-full setup claude-desktop
```

MIT licensed. Requires Mathematica 14+ and Python 3.10+.

GitHub: https://github.com/AbhiRawat4841/mathematica-mcp
PyPI: https://pypi.org/project/mathematica-mcp-full/
Technical reference: https://github.com/AbhiRawat4841/mathematica-mcp/blob/main/docs/technical-reference.md

---
