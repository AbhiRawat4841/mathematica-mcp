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
        "- Knowledge or entity query -> `wolfram_alpha()`, `entity_lookup()`, "
        "`convert_units()`, or `get_constant()`"
    )
    async_line = (
        "- Long computation (>60s) -> `submit_computation()`"
        if features.async_computation
        else "- Long computation (>60s) -> prefer `execute_code(..., timeout=...)`"
    )
    notebook_primary = ""
    notebook_antipattern = ""
    if features.tool_group_enabled("notebook_primary") or features.profile == "full":
        notebook_primary = (
            "- Notebook read/open/save/verify -> `read_notebook()`, "
            "`open_notebook_file()`, `save_notebook()`, `screenshot_notebook()`\n"
        )
        notebook_antipattern = (
            "NEVER: `create_notebook` -> `write_cell` -> `evaluate_cell` for fresh code execution\n"
            "INSTEAD: `execute_code(code, output_target=\"notebook\")`\n"
        )

    return f"""You are a Mathematica expert with access to a Wolfram Engine MCP server.

USER REQUEST: {user_request}

PROFILE: `{features.profile}`
{_profile_summary(features)}

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

Prefer this over `create_notebook` + `write_cell` + `evaluate_cell`.
With `output_target="notebook"`, it atomically finds or creates a notebook,
writes the code, and evaluates it in one call.

Quick guide:
- Pure math or machine-readable result -> `output_target="cli"`
- Plot, image, or notebook artifact -> `output_target="notebook", mode="kernel"`
- Dynamic frontend content only -> `mode="frontend"`

Profile default when `output_target` is omitted: `{default_output_target}`.
"""


def create_notebook_doc() -> str:
    return """[ADVANCED] Create a new empty notebook.

Prefer `execute_code(..., output_target="notebook")` for fresh code execution.
Use `create_notebook` only when you specifically need an empty notebook before
manual authoring or layout operations.
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
- Primary execution tool: `execute_code()`
- Pure math or parseable result: `execute_code(..., output_target="cli")`
- Plot, image, or notebook artifact: `execute_code(..., output_target="notebook", mode="kernel", sync="none")`
- Reuse `session_id` for multi-step workflows
- Avoid `mode="frontend"` unless the task needs dynamic notebook behavior
- Avoid `create_notebook -> write_cell -> evaluate_cell` for fresh code execution
- Current profile default output target: `{features.default_output_target}`
"""


def build_claude_command(features: FeatureFlags) -> str:
    syntax_line = (
        "- syntax help -> `resolve_function()` or `check_syntax()`"
        if features.symbol_lookup
        else "- syntax help -> `check_syntax()`"
    )
    return f"""# Mathematica MCP Command Guide

Profile: `{features.profile}`

Use `execute_code()` as the default path.

Routing:
- computation -> `execute_code(..., output_target="cli", mode="kernel")`
- visual or notebook artifact -> `execute_code(..., output_target="notebook", mode="kernel", sync="none")`
{syntax_line}
- derivation or debug -> `verify_derivation()`, `trace_evaluation()`, `time_expression()`

Avoid:
- `create_notebook -> write_cell -> evaluate_cell` for fresh execution
- `mode="frontend"` unless dynamic content is required
- `sync="refresh"` or `sync="strict"` by default
- dropping `session_id` in a multi-step workflow

This guide is generated from the MCP's internal routing rules so it stays in sync
with the server's exposed profile and defaults.
"""
