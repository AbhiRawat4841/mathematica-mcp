# Why 82 Tools? Building a Comprehensive Mathematica MCP Server

*Most Mathematica MCP servers ship 1-5 tools. We shipped 82. Here's why that matters.*

---

## The Problem with Thin Wrappers

The typical Mathematica MCP server looks like this: one `execute_code` tool, maybe a `wolfram_alpha` query tool, and that's it. You get a pipe to wolframscript and the LLM has to figure out the rest.

This works for demos. It doesn't work for real workflows.

When a researcher asks their AI agent to "check my derivation, plot the result, and export it to my notebook," that's not one tool call -- it's a workflow that spans symbolic computation, verification, graphics rendering, and notebook manipulation. With a 1-tool server, the LLM has to pack everything into a single monolithic Wolfram expression. With 82 tools, it can reason about each step independently.

## The Landscape Today

Here's what exists as of March 2026:

| Server | Tools | What it does |
|--------|-------|-------------|
| **Wolfram MCPServer** (official) | ~13 | Code execution, notebook read/write, doc search |
| **Wolfram MCP Service** | Cloud | $5/month hosted, no local install needed |
| **paraporoco/Wolfram-MCP** | 11 | Dedicated math operations via wolframscript |
| **texra-ai/mcp-server-mathematica** | 2 | Execute code + verify derivation |
| **lars20070/mathematica-mcp** | 4 | Basic wolframscript wrapper |
| **Various Wolfram Alpha wrappers** | 1 | API query only, no local kernel |
| **mathematica-mcp (ours)** | **82** | Full kernel + notebook + graphics + data + debug |

The gap isn't incremental. It's structural.

## What 82 Tools Gets You

### Tool Profiles: Right-size for the Task

Not every conversation needs 82 tools. Three profiles let you choose:

- **math** (~28 tools) -- Pure computation. No file or notebook access.
- **notebook** (~48 tools) -- Math + notebook reading and conversion.
- **full** (~82 tools) -- Everything, including notebook GUI control, data import/export, and admin tools.

Fewer tools means a smaller schema payload during MCP negotiation. The `math` profile is ideal for quick symbolic calculations; `full` is for deep research sessions.

### Atomic Operations vs. Monolithic Expressions

Consider verifying a multi-step derivation. With a 1-tool server:

```
execute_code("Module[{...}, (* 50 lines of verification logic *)]")
```

With mathematica-mcp:

```
verify_derivation(steps=["Integrate[x^2 Sin[x], x]", "2x Sin[x] - (x^2-2) Cos[x]"])
```

The agent doesn't need to know Mathematica's `Module` syntax, exception handling patterns, or output formatting. The tool handles it. This matters because LLMs make fewer errors when they can express intent at a higher abstraction level.

### Notebook Control That Actually Works

Most MCP servers can't touch notebooks. The ones that can typically offer "read" and "write" -- two tools for an application that has cells, sections, styles, evaluation states, screenshots, and export formats.

mathematica-mcp provides:

- **read_notebook** with 5 output formats (markdown, wolfram, outline, json, plain)
- **Three parsing backends**: offline Python parser (fast, no kernel), kernel semantic parser (accurate), and live addon (interactive GUI control)
- Cell-level operations: create, write, evaluate, screenshot, delete, scroll
- Notebook lifecycle: create, open, save, export, close
- **Disk caching** with mtime-based invalidation for repeated reads

This means an agent can analyze a notebook's structure without launching Mathematica, then switch to the kernel backend when it needs full semantic fidelity.

### Performance by Design

The server isn't just wide -- it's fast:

- **Symbol index**: Pure Python lookup in ~1.2ms vs. ~14s per subprocess call (11,700x faster)
- **Raster cache**: 50-entry LRU avoids re-rendering unchanged plots
- **Cache epoch**: Automatic invalidation when kernel state changes (set_variable, clear, restart)
- **Atomic notebook execution**: Single round-trip for create + write + evaluate (vs. 4 separate calls)

These aren't premature optimizations. They're responses to actual latency pain points measured via the built-in telemetry system.

## Architecture: Two Components, Three Execution Paths

```
LLM Client  --(MCP stdio)-->  Python Server  --(TCP localhost:9881)-->  Mathematica Addon
                                    |
                                    +--(subprocess)-->  wolframscript (headless fallback)
                                    |
                                    +--(wolframclient)--> Kernel session (persistent)
```

The Python server handles MCP protocol, caching, and tool routing. The Mathematica addon runs inside the Mathematica GUI and provides live notebook control. When no GUI is available, the server falls back to wolframscript subprocesses or the wolframclient library.

This dual-mode design means the server works everywhere -- headless CI environments, SSH sessions, or full desktop Mathematica.

## Security: Honest About What This Is

mathematica-mcp is a local development tool, not a sandbox. The Wolfram kernel runs with user privileges and has access to the file system, network, and shell. We document this explicitly in [SECURITY.md](https://github.com/AbhiRawat4841/mathematica-mcp/blob/main/SECURITY.md) rather than claiming protections that don't exist.

What we do provide: localhost-only binding, optional auth tokens, 5MB/20MB request/response limits, per-tool timeouts (15-300s), and the `math` profile that disables all file and notebook operations.

## Getting Started

One command, six supported clients:

```bash
# Claude Desktop, Cursor, VS Code, Codex, Gemini, or Claude Code
uvx mathematica-mcp-full setup claude-desktop
```

Requires Mathematica 14+ and Python 3.10+. MIT licensed.

- **GitHub**: [AbhiRawat4841/mathematica-mcp](https://github.com/AbhiRawat4841/mathematica-mcp)
- **PyPI**: [mathematica-mcp-full](https://pypi.org/project/mathematica-mcp-full/)
- **Docs**: [Technical Reference](https://github.com/AbhiRawat4841/mathematica-mcp/blob/main/docs/technical-reference.md)

## The Thesis

A thin MCP wrapper says "here's a pipe to Mathematica, good luck." A comprehensive MCP server says "here are 82 things you can do, each with clear inputs and outputs, and I'll handle the kernel state, caching, timeouts, and error recovery."

The LLM works better when it can reason at the right level of abstraction. That's why 82 tools beats 1.
