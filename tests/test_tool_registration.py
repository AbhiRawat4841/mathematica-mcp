from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
LIST_TOOLS_SCRIPT = """
import asyncio, json
from mathematica_mcp.server import mcp
tools = asyncio.run(mcp.list_tools())
print(json.dumps(sorted(tool.name for tool in tools)))
"""

FULL_TOOLS = {
    "batch_commands",
    "cache_expression",
    "check_syntax",
    "clear_expression_cache",
    "clear_variables",
    "close_notebook",
    "compare_plots",
    "convert_notebook",
    "convert_units",
    "create_animation",
    "create_notebook",
    "delete_cell",
    "entity_lookup",
    "evaluate_cell",
    "evaluate_selection",
    "execute_code",
    "export_data",
    "export_graphics",
    "export_notebook",
    "get_cached",
    "get_cell_content",
    "get_cells",
    "get_computation_result",
    "get_constant",
    "get_dataset_info",
    "get_expression_info",
    "get_feature_status",
    "get_function_repository_info",
    "get_kernel_state",
    "get_mathematica_status",
    "get_messages",
    "get_notebook_cell",
    "get_notebook_info",
    "get_notebook_outline",
    "get_notebooks",
    "get_symbol_info",
    "get_variable",
    "import_data",
    "inspect_graphics",
    "interpret_natural_language",
    "list_cache",
    "list_loaded_packages",
    "list_supported_formats",
    "list_variables",
    "load_dataset",
    "load_package",
    "load_resource_function",
    "mathematica_differentiate",
    "mathematica_expand",
    "mathematica_factor",
    "mathematica_integrate",
    "mathematica_limit",
    "mathematica_series",
    "mathematica_simplify",
    "mathematica_solve",
    "open_notebook_file",
    "parse_notebook_python",
    "poll_computation",
    "rasterize_expression",
    "read_notebook",
    "read_notebook_content",
    "resolve_function",
    "restart_kernel",
    "run_script",
    "save_notebook",
    "screenshot_cell",
    "screenshot_notebook",
    "scroll_to_cell",
    "search_data_repository",
    "search_function_repository",
    "select_cell",
    "set_variable",
    "submit_computation",
    "suggest_similar_functions",
    "time_expression",
    "trace_evaluation",
    "verify_derivation",
    "wolfram_alpha",
    "write_cell",
}

MATH_TOOLS = {
    "check_syntax",
    "clear_variables",
    "convert_units",
    "entity_lookup",
    "execute_code",
    "get_constant",
    "get_expression_info",
    "get_feature_status",
    "get_kernel_state",
    "get_mathematica_status",
    "get_messages",
    "get_symbol_info",
    "get_variable",
    "interpret_natural_language",
    "list_loaded_packages",
    "list_variables",
    "load_package",
    "resolve_function",
    "restart_kernel",
    "set_variable",
    "suggest_similar_functions",
    "time_expression",
    "trace_evaluation",
    "verify_derivation",
    "wolfram_alpha",
}

NOTEBOOK_TOOLS = {
    "check_syntax",
    "clear_variables",
    "close_notebook",
    "compare_plots",
    "convert_units",
    "create_animation",
    "entity_lookup",
    "execute_code",
    "export_data",
    "export_graphics",
    "export_notebook",
    "get_cell_content",
    "get_cells",
    "get_constant",
    "get_expression_info",
    "get_feature_status",
    "get_kernel_state",
    "get_mathematica_status",
    "get_messages",
    "get_notebook_info",
    "get_notebooks",
    "get_symbol_info",
    "get_variable",
    "import_data",
    "inspect_graphics",
    "interpret_natural_language",
    "list_loaded_packages",
    "list_supported_formats",
    "list_variables",
    "load_package",
    "open_notebook_file",
    "rasterize_expression",
    "read_notebook",
    "resolve_function",
    "restart_kernel",
    "save_notebook",
    "screenshot_cell",
    "screenshot_notebook",
    "set_variable",
    "suggest_similar_functions",
    "time_expression",
    "trace_evaluation",
    "verify_derivation",
    "wolfram_alpha",
}


def _tool_names(env_overrides: dict[str, str]) -> list[str]:
    env = os.environ.copy()
    for key in list(env):
        if key == "MATHEMATICA_PROFILE" or key.startswith("MATHEMATICA_ENABLE_"):
            env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", LIST_TOOLS_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    return json.loads(result.stdout)


def test_default_profile_matches_full_baseline():
    default_tools = set(_tool_names({}))
    full_tools = set(_tool_names({"MATHEMATICA_PROFILE": "full"}))

    assert default_tools == FULL_TOOLS
    assert full_tools == FULL_TOOLS


def test_math_profile_registers_curated_subset():
    math_tools = set(_tool_names({"MATHEMATICA_PROFILE": "math"}))

    assert math_tools == MATH_TOOLS
    assert "create_notebook" not in math_tools
    assert "mathematica_integrate" not in math_tools
    assert "cache_expression" not in math_tools


def test_notebook_profile_registers_curated_subset():
    notebook_tools = set(_tool_names({"MATHEMATICA_PROFILE": "notebook"}))

    assert notebook_tools == NOTEBOOK_TOOLS
    assert "create_notebook" not in notebook_tools
    assert "evaluate_cell" not in notebook_tools
    assert "convert_notebook" not in notebook_tools
    assert "read_notebook" in notebook_tools


def test_explicit_feature_override_still_works_in_math_profile():
    tools = set(
        _tool_names(
            {
                "MATHEMATICA_PROFILE": "math",
                "MATHEMATICA_ENABLE_MATH_ALIASES": "true",
                "MATHEMATICA_ENABLE_ASYNC": "true",
            }
        )
    )

    assert "mathematica_integrate" in tools
    assert "submit_computation" in tools


def test_cache_tools_only_register_in_full_profile():
    math_tools = set(_tool_names({"MATHEMATICA_PROFILE": "math"}))
    full_tools = set(_tool_names({"MATHEMATICA_PROFILE": "full"}))

    assert "cache_expression" not in math_tools
    assert "list_cache" not in math_tools
    assert "cache_expression" in full_tools
    assert "list_cache" in full_tools


def test_telemetry_tools_only_register_when_enabled():
    disabled = set(_tool_names({"MATHEMATICA_ENABLE_TELEMETRY": "false"}))
    enabled = set(_tool_names({"MATHEMATICA_ENABLE_TELEMETRY": "true"}))

    assert "get_telemetry_stats" not in disabled
    assert "reset_telemetry" not in disabled
    assert "get_telemetry_stats" in enabled
    assert "reset_telemetry" in enabled
