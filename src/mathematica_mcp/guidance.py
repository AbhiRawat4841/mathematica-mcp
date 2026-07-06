"""Centralized routing guidance for prompts, docs, and client hints."""

from __future__ import annotations

from .config import FeatureFlags


def _profile_summary(features: FeatureFlags) -> str:
    if features.profile == "lean":
        return (
            "This lean profile exposes 12 consolidated tools. `evaluate(code)` is the "
            'primary execution tool (kernel by default); use `target="notebook"` to '
            "show results in the live notebook."
        )
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


def _has_notebook(features: FeatureFlags) -> bool:
    """Check if notebook tools are available in the current profile."""
    return features.tool_group_enabled("notebook_primary") or features.profile == "full"


def _default_target_label(features: FeatureFlags) -> str:
    if features.profile == "lean":
        return "`evaluate(code)` (kernel)"
    return f"`{features.default_output_target}`"


def _ambiguous_fallback_sentence(features: FeatureFlags) -> str:
    if features.profile == "lean":
        return "When the request is ambiguous (none of the keywords above), default to\n`evaluate(code)` (kernel)."
    return (
        "When the request is ambiguous (none of the keywords above), fall back to the\n"
        f"profile default output target (`{features.default_output_target}`)."
    )


_LEAN_STYLE_ROWS = [
    ('"calculate", "compute", "solve", "evaluate"', "`evaluate(code)` — kernel, result in chat"),
    ('"plot", "show", "in notebook"', '`evaluate(code, target="notebook")` — in the live notebook'),
    (
        '"new notebook"',
        '`notebooks(action="create", title=...)` then `evaluate(code, target="notebook")`',
    ),
    ('"verify", "check derivation"', "`verify_derivation(steps)`"),
]


def _style_keyword_table(features: FeatureFlags) -> str:
    if features.profile == "lean":
        rows = [f"| {says} | {does} |" for says, does in _LEAN_STYLE_ROWS]
        header = """| User says | What to do |
|-----------|------------|"""
        return "\n".join([header, *rows])
    rows = [
        '| "calculate", "compute", "what is", "evaluate", "solve" | `execute_code(style="compute")` — answer inline in chat |'
    ]
    if _has_notebook(features):
        rows.extend(
            [
                '| "plot", "show", "graph", "visualize", "in notebook" | `execute_code(style="notebook")` — in current live notebook |',
                '| "new notebook", "fresh notebook", "create notebook" | `create_notebook(title=...)` first, then `execute_code(style="notebook")` |',
                '| "interactive", "manipulate", "slider", "dynamic", "animate" | `execute_code(style="interactive")` |',
            ]
        )
    header = """| User says | What to do |
|-----------|------------|"""
    return "\n".join([header, *rows])


def _style_keyword_bullets(features: FeatureFlags) -> list[str]:
    if features.profile == "lean":
        return [f"**{says}** -> {does}" for says, does in _LEAN_STYLE_ROWS]
    bullets = ['**"calculate"** / **"compute"** / **"what is"** / **"evaluate"** / **"solve"** -> `style="compute"`']
    if _has_notebook(features):
        bullets.extend(
            [
                '**"plot"** / **"show"** / **"graph"** / **"visualize"** / **"in notebook"** -> `style="notebook"`',
                '**"new notebook"** / **"fresh notebook"** / **"create notebook"** -> `create_notebook(title=...)` then `execute_code(style="notebook")`',
                '**"interactive"** / **"manipulate"** / **"slider"** / **"dynamic"** / **"animate"** -> `style="interactive"`',
            ]
        )
    return bullets


def _profile_intro(features: FeatureFlags) -> str:
    if features.profile == "lean":
        return (
            'A "notebook" here means a LIVE WINDOW inside the Mathematica frontend, not a '
            "`.nb` file on disk. This lean profile exposes 12 consolidated tools."
        )
    if _has_notebook(features):
        return 'A "notebook" here means a LIVE WINDOW inside the Mathematica frontend, not a `.nb` file on disk.'
    return "This profile is compute-first. Notebook tools are not exposed in this configuration."


def _routing_lines(features: FeatureFlags) -> list[str]:
    if features.profile == "lean":
        return [
            "Pure computation or symbolic algebra -> `evaluate(code)` (kernel default)",
            'Plot or notebook-visible output -> `evaluate(code, target="notebook")`',
            'New notebook -> `notebooks(action="create", title=...)` then `evaluate(code, target="notebook")`',
            'Inspect notebook state -> `cells(action="list"|"read")` or `screenshot(scope="cell")`',
            'Existing `.nb` file on disk -> `read_notebook_file(path)`; `notebooks(action="open")` only for a live window',
            "Verify algebra steps -> `verify_derivation(steps)`",
            "Syntax uncertainty -> `evaluate(code, dry_run=True)`",
            'Errors or lost context -> `status()`, `kernel(action="messages")`, `guide(topic="errors")`',
        ]
    lines = ['Pure computation, formulas, numbers, or symbolic algebra -> `execute_code(style="compute")`']
    if _has_notebook(features):
        lines.extend(
            [
                'Plot, image, notebook-visible output, or notebook artifact -> `execute_code(style="notebook")`',
                'New notebook explicitly requested -> `create_notebook(title="...")` then `execute_code(style="notebook")`',
                'Interactive or dynamic content -> `execute_code(style="interactive")`',
                "Existing `.nb` file on disk -> `read_notebook()` first; `open_notebook_file()` only if the user wants a live window",
                "Persist or visually verify notebook state -> `save_notebook()`, `screenshot_notebook()`, or `screenshot_cell()`",
            ]
        )
    lines.append(
        "Knowledge or entity query -> `wolfram_alpha()`, `entity_lookup()`, `convert_units()`, or `get_constant()`"
    )
    if features.symbol_lookup:
        lines.append("Syntax uncertainty -> `resolve_function()` or `check_syntax()`")
    else:
        lines.append("Syntax uncertainty -> `check_syntax()`")
    lines.append(
        "Verification or debugging -> `verify_derivation()`, `trace_evaluation()`, `time_expression()`, or `get_messages()`"
    )
    if features.async_computation:
        lines.append("Long computation (>5 min) or repeated timeout -> `submit_computation()`")
    else:
        lines.append("Long computation (>5 min) -> prefer `execute_code(..., timeout=...)` and break work into steps")
    return lines


def _quick_defaults(features: FeatureFlags) -> list[str]:
    if features.profile == "lean":
        return [
            "Prefer one compound Wolfram expression over several sequential `evaluate` calls.",
            "Reuse `session_id` for multi-step workflows; `batch(ops)` runs several calls in one round-trip.",
        ]
    lines = [
        "Prefer one compound Wolfram expression over several sequential `execute_code` calls when only the final result matters.",
        "Reuse `session_id` for any multi-step workflow.",
        'Consider `response_detail="compact"` in long or multi-step workflows to reduce token usage; `"short"` is accepted as an alias. Use `"diagnostic"` only for debugging.',
    ]
    if _has_notebook(features):
        lines.append(
            '`execute_code(style="notebook")` reuses the active notebook; call `create_notebook()` first only when the user asks for a new or different notebook.'
        )
    if features.async_computation:
        lines.append("Use `submit_computation()` instead of retrying the same long-running foreground call.")
    return lines


def _recovery_defaults(features: FeatureFlags) -> list[str]:
    if features.profile == "lean":
        return [
            "Call `status()` before resuming after failures or long context gaps.",
            'Use `kernel(action="messages")` when Mathematica errors are unclear.',
            'Use `guide(topic="errors")` for troubleshooting recipes.',
        ]
    return [
        "Call `get_session_brief()` before resuming after failures or long context gaps.",
        "Use `get_computation_journal()` to recover recent code/output history after context compaction.",
        'Use `get_messages()` or `response_detail="diagnostic"` when Mathematica errors are unclear. `get_messages()` reflects recent captured evaluation and dispatch messages.',
    ]


def _avoid_lines(features: FeatureFlags) -> list[str]:
    lines = [
        "NEVER: split a simple sequential derivation across many MCP round-trips when one Wolfram expression can do it."
    ]
    if features.profile == "lean":
        lines.append(
            'NEVER: `edit_cells(action="write")` then `evaluate(target="cell")` for fresh code; '
            'INSTEAD: `evaluate(code, target="notebook")`.'
        )
        return lines
    if _has_notebook(features):
        lines.extend(
            [
                'NEVER: set `mode="frontend"` directly; use `style="interactive"` instead.',
                'NEVER: `style="new_notebook"` does not exist; create a notebook first, then execute into it.',
                'NEVER: `create_notebook` -> `write_cell` -> `evaluate_cell` for fresh execution; INSTEAD: `execute_code(code, style="notebook")`.',
                'NEVER: use `sync="refresh"` or `sync="strict"` by default; prefer `sync="none"`.',
                'NEVER: use `style="interactive"` for non-dynamic content; it is slower and more timing-sensitive than kernel notebook execution.',
                "NEVER: reach for legacy notebook readers when `read_notebook()` is sufficient.",
            ]
        )
    return lines


def _bullet_block(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def build_server_instructions(features: FeatureFlags) -> str:
    style_lines = _bullet_block(_style_keyword_bullets(features))
    quick_lines = _bullet_block(_quick_defaults(features))
    recovery_lines = _bullet_block(_recovery_defaults(features))
    avoid_lines = _bullet_block(_avoid_lines(features))
    return f"""You have access to a live Mathematica instance through this MCP server.
ALWAYS use the provided MCP tools for ALL Mathematica tasks.

IMPORTANT: {_profile_intro(features)}
Use MCP tools directly; do not touch the filesystem unless the user explicitly asks.

Intent routing:
{style_lines}

Default when ambiguous: {_default_target_label(features)}

Quick defaults:
{quick_lines}

Recovery tools:
{recovery_lines}

Avoid:
{avoid_lines}
"""


def build_mathematica_expert_prompt(
    user_request: str,
    *,
    features: FeatureFlags,
) -> str:
    intent_block = _style_keyword_table(features)
    routing_block = _bullet_block(_routing_lines(features))
    quick_block = _bullet_block(_quick_defaults(features))
    recovery_block = _bullet_block(_recovery_defaults(features))
    avoid_block = _bullet_block(_avoid_lines(features))

    return f"""You are a Mathematica expert with access to a Wolfram Engine MCP server.

CRITICAL: ALWAYS use the MCP tools for ALL
Mathematica work. NEVER use wolframscript CLI, shell commands, mkdir, or manual .nb
file creation. The MCP tools talk directly to the running Mathematica instance.

USER REQUEST: {user_request}

PROFILE: `{features.profile}`
{_profile_summary(features)}

USER INTENT KEYWORDS — detect these in the request and route accordingly:
{intent_block}

{_ambiguous_fallback_sentence(features)}

TASK -> TOOL
{routing_block}

QUICK DEFAULTS
{quick_block}

RECOVERY & INSPECTION
{recovery_block}

ANTI-PATTERNS
{avoid_block}
"""


def execute_code_doc(default_output_target: str) -> str:
    return f"""[PRIMARY] Execute Wolfram Language code. Prefer this for nearly all computation and plotting.

Choose a style:
- `style="compute"` — fast kernel evaluation, result in chat
- `style="notebook"` — evaluate in kernel, show in notebook cell
- `style="interactive"` — front-end evaluation (required for Manipulate/Dynamic)

`style` is a high-level preset for `output_target` + `mode`. Individual params still
work and override style. When `style` and `output_target` are both omitted,
`output_target` defaults to the profile's default (`{default_output_target}`)
and `mode` defaults to `kernel`. When neither `style` nor `mode` is set and the code
calls an interactive head (Manipulate/Dynamic/Animate) into a notebook, it is
auto-routed to frontend mode so it renders as a live panel; the response may then be
`evaluation_pending` (re-read the notebook once the output cell lands).

`response_detail` accepts the canonical levels `compact`, `standard`, `verbose`,
and `diagnostic`, plus the aliases `short`, `medium`, and `long`.

Prefer this over `write_cell` + `evaluate_cell` for running code.
With `output_target="notebook"`, it reuses the active notebook (or creates one
if none exists), writes the code, and evaluates it in one call.
If notebook transport fails, the request returns a notebook-targeted error
instead of silently rerunning through CLI fallback.
NOTE: if the user asks for a NEW notebook, call `create_notebook` first.
"""


def create_notebook_doc() -> str:
    return """[ADVANCED] Create a new empty notebook.

Use when the user explicitly asks for a NEW notebook. This sets the active
notebook so subsequent `execute_code(style="notebook")` calls write into it.
Without this, `execute_code` reuses the currently open notebook.

For code execution in whatever notebook is already open, use
`execute_code(style="notebook")` directly instead.
"""


def write_cell_doc() -> str:
    return """[ADVANCED] Write a cell without evaluating it.

Prefer `execute_code(style="notebook")` for code execution.
Use `write_cell` only for deliberate manual notebook authoring, such as text,
section headers, or carefully controlled cell-by-cell construction.
"""


def evaluate_cell_doc() -> str:
    return """[ADVANCED] Re-evaluate an existing cell in a notebook.

Prefer `execute_code(style="notebook")` for fresh execution.
Use `evaluate_cell` only when you already have a notebook cell and need to
re-run that specific cell.
"""


def evaluate_selection_doc() -> str:
    return """[ADVANCED] Evaluate the currently selected notebook cell or cells.

Prefer `execute_code(style="notebook")` for fresh execution.
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
    style_block = _bullet_block(_style_keyword_bullets(features))

    if features.profile == "lean":
        primary_line = "Primary execution tool: `evaluate()`"
        steer = "Use the `target` parameter or these keywords to control where results appear:"
    else:
        primary_line = (
            "Primary execution tool: `execute_code()`\n"
            f"Current profile default output target: `{features.default_output_target}`"
        )
        steer = "Use the `style` parameter or these keywords to control where results appear:"
    hint = f"""# Mathematica MCP Usage

{primary_line}

**IMPORTANT**: ALWAYS use the Mathematica MCP tools for ALL Mathematica work.
NEVER use `wolframscript` CLI, shell commands, `mkdir`, or manual `.nb` file creation.
The MCP tools talk directly to the running Mathematica instance — just call them.

## Execution Styles
{steer}
{style_block}
Default when ambiguous: {_default_target_label(features)}
"""
    if features.profile == "lean":
        hint += """
## Notebook Files
- For `.nb` files on disk, prefer `read_notebook_file(path)`.
- Use `notebooks(action="open"|"save"|"export")` only when the user explicitly wants a live window or disk output.
"""
        return hint
    if _has_notebook(features):
        hint += """
## Notebook Files
- For `.nb` files on disk, prefer `read_notebook()` first.
- Use `open_notebook_file()`, `save_notebook()`, or export only when the user explicitly wants a live window or disk output.
"""
    return hint


def build_claude_command(features: FeatureFlags) -> str:
    keyword_block = _bullet_block(_style_keyword_bullets(features))
    routing_block = _bullet_block(_routing_lines(features))
    quick_block = _bullet_block(_quick_defaults(features))
    recovery_block = _bullet_block(_recovery_defaults(features))
    avoid_block = _bullet_block(_avoid_lines(features))

    return f"""# Mathematica MCP Command Guide

Profile: `{features.profile}`

**IMPORTANT**: ALWAYS use MCP tools for Mathematica work. NEVER use wolframscript CLI,
shell commands, or manual .nb file creation. The MCP tools control Mathematica directly.

## Style Keywords
Users can steer execution by including these keywords in their request:
{keyword_block}
Default when no keyword matches: {_default_target_label(features)}

## Tool Routing
- Prefer the high-level tool route below unless the user explicitly needs low-level control:
{routing_block}

## Quick Defaults
{quick_block}

## Recovery Tools
{recovery_block}

## Avoid
{avoid_block}
"""


def build_codex_guidance(features: FeatureFlags) -> str:
    rules = [
        "**ALWAYS use MCP tools** for Mathematica work.",
        "**NEVER** use `wolframscript` CLI, shell commands, `mkdir`, or manual `.nb` file creation.",
    ]
    if _has_notebook(features):
        rules.extend(
            [
                "For notebook files on disk, prefer `read_notebook()` first. Use `open_notebook_file()`, `save_notebook()`, or export only when the user explicitly wants a live window or disk output.",
            ]
        )
    keyword_table = _style_keyword_table(features)
    workflow_example = ""

    if features.profile == "lean":
        workflow_example = """
## Typical workflow

```
# User: "integrate x^2 in new notebook and plot it"
1. notebooks(action="create", title="Integration")   # opens live notebook window
2. evaluate("Integrate[x^2, x]", target="notebook")   # writes + evaluates there
3. evaluate("Plot[x^3/3, {x, -2, 2}]", target="notebook")
```

That's it. No mkdir, no export, no file search.
"""
    elif _has_notebook(features):
        workflow_example = """
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
"""
    else:
        workflow_example = """
## Typical workflow

```
# User: "integrate x^2"
1. execute_code("Integrate[x^2, x]", style="compute")   # result in chat
```
"""

    guidance = """# Mathematica MCP — Agent Instructions

This project has a Mathematica MCP server connected. It gives you direct
control of a running Mathematica instance through MCP tools.
"""
    if _has_notebook(features):
        guidance += f"""
## Key concept

{_profile_intro(features)}
You never need to touch the filesystem unless the user explicitly asks.
"""
    else:
        guidance += f"\n{_profile_intro(features)}\n"
    guidance += f"""
## Rules

1. {rules[0]}
2. {rules[1]}
"""
    if _has_notebook(features):
        guidance += f"""3. {rules[2]}
"""
    guidance += f"""
## Style keywords

{keyword_table}
Default when ambiguous: {_default_target_label(features)}
{workflow_example}"""
    return guidance


def build_prompt_calculate(features: FeatureFlags, expression: str) -> str:
    if features.profile == "lean":
        return f"Calculate the following and return the result inline (use evaluate(code)):\n\n{expression}"
    return f"Calculate the following and return the result inline (use style='compute'):\n\n{expression}"


def build_prompt_notebook(features: FeatureFlags, task: str) -> str:
    if features.profile == "lean":
        return f"Execute the following in the current Mathematica notebook (use evaluate(code, target='notebook')):\n\n{task}"
    return f"Execute the following in the current Mathematica notebook (use style='notebook'):\n\n{task}"


def build_prompt_new_notebook(features: FeatureFlags, task: str, title: str = "Analysis") -> str:
    if features.profile == "lean":
        return (
            f"Create a new Mathematica notebook titled '{title}' using notebooks(action='create', title=...), "
            f"then execute the following in it (use evaluate(code, target='notebook')):\n\n"
            f"{task}"
        )
    return (
        f"Create a new Mathematica notebook titled '{title}' using create_notebook(), "
        f"then execute the following in it (use style='notebook'):\n\n"
        f"{task}"
    )


def build_prompt_interactive(features: FeatureFlags, task: str) -> str:
    if features.profile == "lean":
        return (
            f"Execute the following in the live notebook (use evaluate(code, target='notebook')). "
            f"Interactive content (Manipulate/Dynamic/Animate) is auto-detected and rendered as a "
            f"live panel via the front end; the response may be evaluation_pending, so re-check "
            f"with cells(action='read') once the output cell lands:\n\n"
            f"{task}"
        )
    return (
        f"Execute the following in the notebook using frontend mode for dynamic interaction "
        f"(use style='interactive'; interactive heads also auto-route to frontend without it):\n\n"
        f"{task}"
    )


def build_prompt_quickstart(features: FeatureFlags) -> str:
    if features.profile == "lean":
        return """Show the user this quick reference for Mathematica MCP:

## Mathematica MCP — Quick Reference

| Say this...                     | What happens                                              |
|---------------------------------|-----------------------------------------------------------|
| **"calculate ..."**             | `evaluate(code)` — result inline in chat                  |
| **"plot ..."** / **"show ..."** | `evaluate(code, target="notebook")` — in the live notebook |
| **"in new notebook: ..."**      | `notebooks(action="create")` then `evaluate(code, target="notebook")` |
| **"screenshot ..."**            | `screenshot(scope="notebook")` or `screenshot(scope="cell")` |
| **"verify ..."**                | `verify_derivation(steps)`                                |

Recovery: `status()`, `kernel(action="messages")`, `guide(topic="errors")`.
Default when ambiguous: `evaluate(code)` (kernel).
"""
    return f"""Show the user this quick reference for Mathematica MCP styles:

## Mathematica MCP — Execution Styles

### For chat users — use keywords in your prompt

| Say this...                     | What happens                                    |
|---------------------------------|-------------------------------------------------|
| **"calculate ..."**             | Result appears inline in chat                   |
| **"plot ..."** / **"show ..."** | Executes in current Mathematica notebook         |
| **"in new notebook: ..."**      | Creates a fresh notebook, then executes there    |
| **"interactive ..."**           | Notebook with sliders/Manipulate (frontend mode) |

### For tool callers — use the `style` parameter

| `style=`          | What happens                                    |
|-------------------|-------------------------------------------------|
| `"compute"`       | Fast kernel evaluation, result in chat          |
| `"notebook"`      | Evaluate in kernel, show in notebook cell       |
| `"interactive"`   | Front-end evaluation (Manipulate/Dynamic)       |

> There is no `style="new_notebook"`. Create a fresh notebook with
> `create_notebook(title="...")` then `execute_code(style="notebook")`.

### Examples
- "Calculate the integral of x^3 from 0 to 1"  →  answer in chat
- "Plot Sin[x] from 0 to 2π"  →  plot appears in notebook
- "In new notebook: integrate 1/x^5 + x^7 and plot the region"  →  fresh notebook
- "Interactive: Manipulate a slider for Plot[Sin[n x], {{x, 0, 2π}}]"  →  dynamic UI

If you don't use a keyword, the default is: `{features.default_output_target}`
"""


def build_session_brief(
    features: FeatureFlags,
    *,
    connection_mode: str = "unknown",
    routing_hints: list[str] | None = None,
    recent_errors: list[str] | None = None,
) -> str:
    """Build a compact session state brief (~80-100 tokens).

    Pure function — all state passed as parameters for testability.
    """
    lines = ["## Session Brief"]
    lines.append(
        f"- **Profile**: {features.profile} | "
        f"**Connection**: {connection_mode} | "
        f"**Default**: {features.default_output_target}"
    )

    if recent_errors:
        lines.append(f"- **Recent errors**: {', '.join(recent_errors[:3])}")
    else:
        lines.append("- **Recent errors**: none")

    if routing_hints:
        lines.append(f"- **Routing advice**: {'; '.join(routing_hints[:2])}")

    return "\n".join(lines)
