# Migration Guide: 0.9.x → 1.0

## Breaking change: the default profile is now `lean`

Before 1.0 the default profile was `full` (82 tools). **In 1.0 the default is `lean` (12 consolidated tools).** If you do nothing, your agent will see the 12 lean tools instead of the 82 it saw before.

To keep the full pre-1.0 surface, set the profile back:

```bash
export MATHEMATICA_PROFILE=classic   # or: full  (alias)
```

or pass it at setup time:

```bash
uvx mathematica-mcp-full setup <client> --profile classic
```

`classic` (and its alias `full`) is byte-identical to the pre-1.0 surface — same 82 tools, same implementations, no shims. The `math` (~28) and `notebook` (~48) profiles are unchanged.

Why the change: the lean surface is ~11.5 KB of tool schema (~2.9k tokens) versus ~61 KB / ~15k tokens for the 82-tool surface. Most agents route better and burn far less context with the smaller set.

## Reinstall the Mathematica addon

1.0 adds a `protocol_version` handshake (currently `3`) between the Python client and the Mathematica addon. The addon lives in `$UserBaseDirectory/Kernel/init.m` and **does not update when you `pip`/`uvx` upgrade the Python package.** After upgrading, reinstall the addon:

```bash
uvx mathematica-mcp-full setup <client>
# or, manual: wolframscript -file addon/install.wl
```

Then restart Mathematica. If the addon is stale, `status()` reports the version skew and asks you to reinstall.

## Missing a tool under `lean`?

Every classic capability still exists. If a specific tool you relied on isn't one of the 12 lean tools, either:

- switch to `classic` (see above), or
- turn on just the group you need with `MATHEMATICA_TOOLSETS` (lean only, comma-separated):

  ```bash
  export MATHEMATICA_TOOLSETS=data_io,graphics_plus,cloud,debug,notebook_files,notebook_edit,symbols,math_aliases,repository,async_jobs,cache
  ```

  Toolsets can only *add* tools to lean; they never remove the 12 core tools. See the [README](../README.md#opt-in-extras-for-lean) for what each name adds.

## Old tool → lean equivalent

The 12 lean tools are thin wrappers over the same internals. Map your old calls like this:

| Old tool(s) | Lean equivalent |
|-------------|-----------------|
| `get_mathematica_status`, `get_session_brief`, `get_feature_status` | `status()` |
| `execute_code` | `evaluate(code, target=...)` — `target="kernel"` (default) or `"notebook"` |
| `evaluate_cell` | `evaluate(target="cell", cell_id=...)` |
| `evaluate_selection` | `evaluate(target="selection")` |
| `run_script` | `evaluate(file=...)` |
| `check_syntax` | `evaluate(code, dry_run=True)` |
| `create_notebook`, `get_notebooks`, `get_notebook_info`, `open_notebook_file`, `save_notebook`, `close_notebook`, `export_notebook` | `notebooks(action="create\|list\|info\|open\|save\|close\|export", ...)` |
| `get_cells`, `get_cell_content`, `select_cell`, `scroll_to_cell` | `cells(action="list\|read\|select\|scroll", ...)` |
| `write_cell`, `delete_cell` | `edit_cells(action="write\|delete", ...)` |
| `screenshot_notebook`, `screenshot_cell`, `rasterize_expression` | `screenshot(scope="notebook\|cell\|expression", ...)` |
| `verify_derivation` | `verify_derivation(steps=...)` — unchanged name, now runs warm |
| `get_kernel_state`, `get_messages`, `restart_kernel`, `load_package`, `list_loaded_packages`, `get_expression_info` | `kernel(action="state\|messages\|restart\|load_package\|packages\|inspect", ...)` |
| `list_variables`, `get_variable`, `set_variable`, `clear_variables` | `vars(action="list\|get\|set\|clear\|clear_all", ...)` |
| `read_notebook`, `read_notebook_content`, `convert_notebook`, `get_notebook_outline`, `parse_notebook_python`, `get_notebook_cell` | `read_notebook_file(path, mode="markdown\|wolfram\|outline\|json\|plain")` |
| `batch_commands` | `batch(ops=[...])` |
| (anything else) | run under `classic`, or enable the matching `MATHEMATICA_TOOLSETS` group |

## Behavior changes in 1.0

- **`vars(action='clear')` requires `name=` or `pattern=`.** A bare `clear` no longer wipes the session — it returns an error. To clear everything, call `vars(action='clear_all')` explicitly. (Classic `clear_variables` is unchanged.)
- **`evaluate` rejects ambiguous calls.** Passing both `code=` and `file=`, or `dry_run=True` with `file=`, returns an error instead of guessing which you meant.
- **`notebooks` validates `format` per action.** `save` accepts `Notebook|PDF|HTML|TeX`; `export` accepts `PDF|HTML|TeX|Markdown`. Invalid combinations return an error with a `next_step` pointing at the right call.

## Behavioral notes for 1.0

- **Warm execution.** `verify_derivation` and the other previously-cold `wolframscript` tools now run on the persistent kernel session, returning warm in sub-second time. Every response carries an `execution_method`, and `status()` reports a cold-execution counter (0 on the lean happy path).
- **Error recovery.** Failed evaluations now carry `error_analysis` with a `suggested_fix` and `next_step` on **all** evaluate paths (previously the notebook path only); `retry_with` is a concrete corrected call when one can be derived from context, otherwise `null`.
- **Long output.** Output over `MATHEMATICA_MAX_OUTPUT_CHARS` (default `4000`) is capped; pass the returned `cursor` back to the same tool to page the rest.
- **Idle kernel.** The persistent kernel shuts down after `MATHEMATICA_KERNEL_IDLE_TIMEOUT` seconds of inactivity (default `1800`; `0` disables) and restarts on the next call.
- **`state_delta` is scoped.** Addon responses carry a `state_delta` (`notebook` / `cell_count` / `kernel_busy`) only on notebook-touching commands (plus `batch_commands`), not on every response; `kernel_busy` reports the front end's actual `Evaluating` state.
