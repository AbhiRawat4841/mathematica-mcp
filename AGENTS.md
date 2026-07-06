<!-- mathematica-mcp:start -->
# Mathematica MCP - Agent Instructions

This project has a Mathematica MCP server connected. It gives you direct
control of a running Mathematica instance through MCP tools.

## Key concept

A "notebook" here means a LIVE WINDOW inside the Mathematica frontend, not a `.nb` file on disk.
The default `lean` profile exposes 12 consolidated tools.

## Rules

1. **ALWAYS use MCP tools** for Mathematica work.
2. **NEVER** use `wolframscript` CLI, shell commands, `mkdir`, or manual `.nb` file creation.
3. For notebook files on disk, prefer `read_notebook_file(path)` - no kernel needed. Use `notebooks(action="open"/"save"/"export")` only when the user explicitly wants a live window or disk output.

## Routing (lean profile, default)

| User says | What to do |
|-----------|------------|
| "calculate", "compute", "solve", "evaluate" | `evaluate(code)` - kernel, result in chat |
| "plot", "show", "in notebook" | `evaluate(code, target="notebook")` - in the live notebook |
| "new notebook" | `notebooks(action="create", title=...)` then `evaluate(code, target="notebook")` |
| "screenshot", "what does it look like" | `screenshot(scope="notebook")` (or `scope="cell"` / `"expression"`) |
| "verify", "check derivation" | `verify_derivation(steps)` |
| something failed / warnings | `kernel(action="messages")` then `guide(topic="errors")` |

Default when ambiguous: `evaluate(code)` (kernel). More guidance on demand: `guide(topic="workflow")`.

## Typical workflow

```
# User: "integrate x^2 in new notebook and plot it"
1. notebooks(action="create", title="Integration")   # opens live notebook window
2. evaluate("Integrate[x^2, x]", target="notebook")   # writes + evaluates there
3. evaluate("Plot[x^3/3, {x, -2, 2}]", target="notebook")
```

That's it. No mkdir, no export, no file search.

> With `MATHEMATICA_PROFILE=classic`, the legacy names apply instead:
> `execute_code(style=...)`, `create_notebook(title=...)`, `screenshot_notebook`, `get_messages`.
<!-- mathematica-mcp:end -->
