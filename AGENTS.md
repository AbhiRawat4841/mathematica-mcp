<!-- mathematica-mcp:start -->
# Mathematica MCP — Agent Instructions

This project has a Mathematica MCP server connected. It gives you direct
control of a running Mathematica instance through MCP tools.

## Key concept

A "notebook" here means a LIVE WINDOW inside the Mathematica frontend, not a `.nb` file on disk.
You never need to touch the filesystem unless the user explicitly asks.

## Rules

1. **ALWAYS use MCP tools** for Mathematica work.
2. **NEVER** use `wolframscript` CLI, shell commands, `mkdir`, or manual `.nb` file creation.
3. For notebook files on disk, prefer `read_notebook()` first. Use `open_notebook_file()`, `save_notebook()`, or export only when the user explicitly wants a live window or disk output.

## Style keywords

| User says | What to do |
|-----------|------------|
| "calculate", "compute", "what is", "evaluate", "solve" | `execute_code(style="compute")` — answer inline in chat |
| "plot", "show", "graph", "visualize", "in notebook" | `execute_code(style="notebook")` — in current live notebook |
| "new notebook", "fresh notebook", "create notebook" | `create_notebook(title=...)` first, then `execute_code(style="notebook")` |
| "interactive", "manipulate", "slider", "dynamic", "animate" | `execute_code(style="interactive")` |

Default when ambiguous: `notebook`

## Typical workflow

```
# User: "integrate x^2 in new notebook and plot it"
1. create_notebook(title="Integration")       # opens live notebook window
2. execute_code("Integrate[x^2, x]",          # writes + evaluates in that notebook
     style="notebook")
3. execute_code("Plot[x^3/3, {x, -2, 2}]",   # plot appears in same notebook
     style="notebook")
```

That's it. No mkdir, no export, no file search.
<!-- mathematica-mcp:end -->
