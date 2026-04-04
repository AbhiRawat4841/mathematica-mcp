<!-- mathematica-mcp:start -->
# Mathematica MCP — Agent Instructions

This project has a **live Mathematica MCP server** connected. It gives you direct
control of a running Mathematica instance through MCP tools.

## Key concept: live notebook ≠ .nb file

A "notebook" here means a **live window inside the Mathematica frontend**, not a
`.nb` file on disk. The MCP tools create and manipulate these live notebooks
directly. You never need to touch the filesystem.

## Rules

1. **ALWAYS use MCP tools** (`execute_code`, `create_notebook`, etc.) for ALL
   Mathematica work.
2. **NEVER** use `wolframscript` CLI, shell commands, `mkdir`, or manual `.nb`
   file creation. The MCP tools replace all of that.
3. **NEVER** search for `.nb` files, export notebooks, or save to disk unless
   the user explicitly asks for it.
4. When the user says **"new notebook"**, call `create_notebook(title="...")`
   to open a live notebook window, then call
   `execute_code(code, style="notebook")` to run code in it.

## Style keywords

| User says                              | What to do                                                                 |
|----------------------------------------|----------------------------------------------------------------------------|
| "calculate", "compute", "what is"      | `execute_code(style="compute")` — answer inline in chat                   |
| "plot", "show", "in notebook"          | `execute_code(style="notebook")` — in current live notebook              |
| "new notebook", "fresh notebook"       | `create_notebook(title=...)` first, then `execute_code(style="notebook")` |
| "interactive", "manipulate", "dynamic" | `execute_code(style="interactive")`                                      |

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
