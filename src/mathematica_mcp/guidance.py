"""Centralized routing guidance for prompts, docs, and client hints."""

from __future__ import annotations

from .config import FeatureFlags


def _profile_summary(features: FeatureFlags) -> str:
    if features.profile == "math":
        return (
            "This profile is compute-first. Prefer CLI-style execution and avoid "
            "notebook-specific flows unless the user explicitly needs a notebook."
        )
    if features.profile == "notebook":
        return (
            "This profile is notebook-aware but curated. Prefer high-level notebook "
            "operations and avoid low-level cell-by-cell construction."
        )
    return (
        "This profile exposes the full tool surface, including advanced and legacy "
        "tools. Prefer the primary high-level tools unless there is a specific need "
        "for lower-level control."
    )


def build_mathematica_expert_prompt(
    user_request: str,
    *,
    features: FeatureFlags,
) -> str:
    syntax_line = (
        "- Syntax uncertainty -> `resolve_function()` or `check_syntax()`"
        if features.symbol_lookup
        else "- Syntax uncertainty -> `check_syntax()`"
    )
    knowledge_line = (
        "- Knowledge or entity query -> `wolfram_alpha()`, `entity_lookup()`, `convert_units()`, or `get_constant()`"
    )
    async_line = (
        "- Long computation (>5min) -> `submit_computation()`"
        if features.async_computation
        else "- Long computation (>5min) -> prefer `execute_code(..., timeout=...)` (default 300s)"
    )
    notebook_primary = ""
    notebook_antipattern = ""
    intent_block = ""
    if features.tool_group_enabled("notebook_primary") or features.profile == "full":
        notebook_primary = (
            "- Notebook read/open/save/verify -> `read_notebook()`, "
            "`open_notebook_file()`, `save_notebook()`, `screenshot_notebook()`\n"
        )
        intent_block = f"""
USER INTENT KEYWORDS — detect these in the request and route accordingly:
| Keyword / phrase                                        | Mode             | What to do                                                        |
|---------------------------------------------------------|------------------|-------------------------------------------------------------------|
| "calculate", "compute", "what is", "evaluate", "solve" | **inline**       | `execute_code(output_target="cli")` — answer in chat              |
| "plot", "show", "graph", "visualize", "in notebook"    | **notebook**     | `execute_code(output_target="notebook")` — in active notebook     |
| "new notebook", "fresh notebook", "create notebook"    | **new notebook** | `create_notebook(title=...)` first, then `execute_code(output_target="notebook")` |
| "interactive", "manipulate", "slider", "dynamic", "animate" | **interactive** | `execute_code(output_target="notebook", mode="frontend")`    |

When the request is ambiguous (none of the keywords above), fall back to the profile default output target (`{features.default_output_target}`).
"""
        notebook_antipattern = (
            "NEVER: `create_notebook` -> `write_cell` -> `evaluate_cell` for fresh code execution\n"
            'INSTEAD: `execute_code(code, output_target="notebook")`\n'
            "\n"
            "HOWEVER: when the user explicitly asks for a **new notebook**, call\n"
            '`create_notebook(title="...")` first, then `execute_code(code, output_target="notebook")`.\n'
            "`execute_code` alone reuses whichever notebook is already open.\n"
            "`create_notebook` sets the active notebook so subsequent `execute_code` calls target it.\n"
        )
    else:
        intent_block = f"""
USER INTENT KEYWORDS — detect these in the request and route accordingly:
| Keyword / phrase                                        | Mode       | What to do                                           |
|---------------------------------------------------------|------------|------------------------------------------------------|
| "calculate", "compute", "what is", "evaluate", "solve" | **inline** | `execute_code(output_target="cli")` — answer in chat |

When the request is ambiguous (none of the keywords above), fall back to the profile default output target (`{features.default_output_target}`).
"""

    return f"""You are a Mathematica expert with access to a Wolfram Engine MCP server.

CRITICAL: ALWAYS use the MCP tools for ALL
Mathematica work. NEVER use wolframscript CLI, shell commands, mkdir, or manual .nb
file creation. The MCP tools talk directly to the running Mathematica instance.

USER REQUEST: {user_request}

PROFILE: `{features.profile}`
{_profile_summary(features)}

{intent_block}

TASK -> TOOL
- Pure computation, formulas, numbers, symbolic algebra -> `execute_code(..., output_target="cli", mode="kernel")`
- Plot, image, notebook-visible output, or notebook artifact -> `execute_code(..., output_target="notebook", mode="kernel", sync="none")`
{knowledge_line}
{syntax_line}
- Verification or debugging -> `verify_derivation()`, `trace_evaluation()`, `time_expression()`
{async_line}
{notebook_primary}- Multi-turn context -> always reuse `session_id`

ANTI-PATTERNS
{notebook_antipattern}NEVER: use multiple `execute_code` calls for a single sequential derivation when one Wolfram expression can do it
INSTEAD: use a compound Wolfram expression and return the final result

NEVER: use `mode="frontend"` unless the task truly requires dynamic notebook content such as `Manipulate`
INSTEAD: use `mode="kernel"`

NEVER: use `sync="refresh"` or `sync="strict"` by default
INSTEAD: use `sync="none"`

QUICK DEFAULTS
- `output_target="cli"` for results you need to parse or reason over
- `output_target="notebook"` for visuals or saved notebook artifacts
- `mode="kernel"` unless dynamic frontend behavior is required
- `session_id` for any multi-step interaction
"""


def execute_code_doc(default_output_target: str) -> str:
    return f"""[PRIMARY] Execute Wolfram Language code. Prefer this for nearly all computation and plotting.

Use this tool for algebra, calculus, plotting, symbolic work, notebook-visible
results, and most general Mathematica execution.

Prefer this over `write_cell` + `evaluate_cell` for running code.
With `output_target="notebook"`, it reuses the active notebook (or creates one
if none exists), writes the code, and evaluates it in one call.
NOTE: if the user asks for a NEW notebook, call `create_notebook` first.

Quick guide:
- Pure math or machine-readable result -> `output_target="cli"`
- Plot, image, or notebook artifact -> `output_target="notebook", mode="kernel"`
- Dynamic frontend content only -> `mode="frontend"`

Profile default when `output_target` is omitted: `{default_output_target}`.
"""


def create_notebook_doc() -> str:
    return """[ADVANCED] Create a new empty notebook.

Use when the user explicitly asks for a NEW notebook. This sets the active
notebook so subsequent `execute_code(..., output_target="notebook")` calls
write into it. Without this, `execute_code` reuses the currently open notebook.

For code execution in whatever notebook is already open, use
`execute_code(..., output_target="notebook")` directly instead.
"""


def write_cell_doc() -> str:
    return """[ADVANCED] Write a cell without evaluating it.

Prefer `execute_code(..., output_target="notebook")` for code execution.
Use `write_cell` only for deliberate manual notebook authoring, such as text,
section headers, or carefully controlled cell-by-cell construction.
"""


def evaluate_cell_doc() -> str:
    return """[ADVANCED] Re-evaluate an existing cell in a notebook.

Prefer `execute_code(..., output_target="notebook")` for fresh execution.
Use `evaluate_cell` only when you already have a notebook cell and need to
re-run that specific cell.
"""


def evaluate_selection_doc() -> str:
    return """[ADVANCED] Evaluate the currently selected notebook cell or cells.

Prefer `execute_code(..., output_target="notebook")` for fresh execution.
Use `evaluate_selection` only for explicit notebook-selection workflows.
"""


def read_notebook_doc() -> str:
    return """[PRIMARY] Read a Mathematica notebook with backend-aware dispatch.

Prefer this over `read_notebook_content`, `convert_notebook`,
`get_notebook_outline`, `parse_notebook_python`, and `get_notebook_cell`
unless you need a specific legacy backend or narrow low-level operation.
"""


def legacy_notebook_doc(summary: str) -> str:
    return f"""[LEGACY] {summary}

Prefer `read_notebook()` unless you specifically need this older or narrower
workflow.
"""


def build_claude_hint(features: FeatureFlags) -> str:
    return f"""# Mathematica MCP Usage

Primary execution tool: `execute_code()`
Current profile default output target: `{features.default_output_target}`

**IMPORTANT**: ALWAYS use the Mathematica MCP tools for ALL Mathematica work.
NEVER use `wolframscript` CLI, shell commands, `mkdir`, or manual `.nb` file creation.
The MCP tools talk directly to the running Mathematica instance — just call them.

## Quick Mode Guide
Use these keywords to control where results appear:
- **"calculate"** / **"compute"** / **"what is"** — answer inline in chat
- **"plot"** / **"show"** / **"in notebook"** — execute in current Mathematica notebook
- **"new notebook"** — `create_notebook(title=...)` first, then `execute_code(output_target="notebook")`
- **"interactive"** / **"manipulate"** / **"dynamic"** — notebook with frontend mode (for sliders, animations)

Default when ambiguous: `{features.default_output_target}`

## Routing Rules
- Pure math or parseable result: `execute_code(..., output_target="cli")`
- Plot, image, or notebook artifact: `execute_code(..., output_target="notebook", mode="kernel", sync="none")`
- New notebook requested: `create_notebook(title="...")` first, then `execute_code(..., output_target="notebook")`
- Interactive/dynamic content: `execute_code(..., output_target="notebook", mode="frontend")`
- Reuse `session_id` for multi-step workflows
"""


def build_claude_command(features: FeatureFlags) -> str:
    syntax_line = (
        "- syntax help -> `resolve_function()` or `check_syntax()`"
        if features.symbol_lookup
        else "- syntax help -> `check_syntax()`"
    )
    return f"""# Mathematica MCP Command Guide

Profile: `{features.profile}`

**IMPORTANT**: ALWAYS use MCP tools for Mathematica work. NEVER use wolframscript CLI,
shell commands, or manual .nb file creation. The MCP tools control Mathematica directly.

## Mode Keywords
Users can steer execution by including these keywords in their request:
- **"calculate"** / **"compute"** / **"what is"** → inline result in chat (`output_target="cli"`)
- **"plot"** / **"show"** / **"in notebook"** → execute in current notebook (`output_target="notebook"`)
- **"new notebook"** → create fresh notebook first, then execute there
- **"interactive"** / **"manipulate"** / **"dynamic"** → notebook with frontend mode (`mode="frontend"`)

Default when no keyword matches: `{features.default_output_target}`

## Tool Routing
- computation -> `execute_code(..., output_target="cli", mode="kernel")`
- visual or notebook artifact -> `execute_code(..., output_target="notebook", mode="kernel", sync="none")`
- new notebook -> `create_notebook(title="...")` then `execute_code(..., output_target="notebook")`
- interactive/dynamic -> `execute_code(..., output_target="notebook", mode="frontend")`
{syntax_line}
- derivation or debug -> `verify_derivation()`, `trace_evaluation()`, `time_expression()`

## Avoid
- `create_notebook -> write_cell -> evaluate_cell` for fresh execution (use `execute_code` instead)
- `mode="frontend"` unless dynamic content is required
- `sync="refresh"` or `sync="strict"` by default
- dropping `session_id` in a multi-step workflow
"""


def build_codex_guidance(features: FeatureFlags) -> str:
    return f"""# Mathematica MCP — Agent Instructions

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
   `execute_code(code, output_target="notebook")` to run code in it.

## Mode keywords

| User says                              | What to do                                                                 |
|----------------------------------------|----------------------------------------------------------------------------|
| "calculate", "compute", "what is"      | `execute_code(output_target="cli")` — answer inline in chat               |
| "plot", "show", "in notebook"          | `execute_code(output_target="notebook")` — in current live notebook       |
| "new notebook", "fresh notebook"       | `create_notebook(title=...)` first, then `execute_code(output_target="notebook")` |
| "interactive", "manipulate", "dynamic" | `execute_code(output_target="notebook", mode="frontend")`                 |

Default when ambiguous: `{features.default_output_target}`

## Typical workflow

```
# User: "integrate x^2 in new notebook and plot it"
1. create_notebook(title="Integration")       # opens live notebook window
2. execute_code("Integrate[x^2, x]",          # writes + evaluates in that notebook
     output_target="notebook")
3. execute_code("Plot[x^3/3, {{x, -2, 2}}]",  # plot appears in same notebook
     output_target="notebook")
```

That's it. No mkdir, no export, no file search.
"""
