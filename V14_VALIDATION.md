# V14 Validation Checklist (pre-release)

Mathematica 15 is the primary development target. Every `$VersionNumber >= 15.` branch has a `<15` fallback that must be validated before each release, either on a real 14.x install (or the free **Wolfram Engine 14.x**) or by forcing the guarded branch on a 15 machine.

## Forcing the `<15` branch on a 15 machine

The addon gate is:

```wolfram
mcpVersionAtLeast15[] := TrueQ[$VersionNumber >= 15.] && Environment["MMCP_FORCE_V14"] =!= "1";
```

Set `MMCP_FORCE_V14=1` in the environment the **Mathematica kernel** runs in (not just the Python process) to force every `<15` branch:

```bash
export MMCP_FORCE_V14=1
```

Then restart the kernel / front-end so the addon re-reads the environment. Unset it to return to real ≥15 behavior. Automated mocked coverage for the guarded branches lives under `tests/` and exercises `MMCP_FORCE_V14=1`.

## Guarded features

- [ ] **`ShowChatbar->False` suppression** (`addon/MathematicaMCP.wl`, `mcpNotebookOptions`)
  - ≥15: agent-created notebooks (`notebooks(action="create")`) omit the chat sidebar; `show_chatbar=True` re-enables it.
  - <15: `ShowChatbar` option is **not** added at all - verify created notebooks open normally with no error about an unknown option.

- [ ] **`mcpVersionAtLeast15[]` gate honors `MMCP_FORCE_V14`**
  - With `MMCP_FORCE_V14=1` on a 15 machine, `mcpVersionAtLeast15[]` returns `False` and the notebook is created **without** `ShowChatbar`.
  - Without the flag on 15, it returns `True` and the option is present.

- [ ] **Addon `protocol_version` handshake** (`$MCPProtocolVersion = 4`; `cmdPing`, `cmdGetStatus`)
  - `status()` should report the addon `protocol_version` on both 14.x and 15.
  - A stale addon (older `protocol_version` than the Python client's `ADDON_PROTOCOL_VERSION`, currently `4`) must trigger the reinstall message from `status()`. Confirm the handshake is version-independent (works on 14.x, not gated behind ≥15).

- [ ] **Core notebook + kernel flows on 14.x**
  - `evaluate` (kernel + notebook target), `notebooks`, `cells`, `edit_cells`, `screenshot`, `verify_derivation`, `read_notebook_file`, `vars`, `kernel` all succeed on a 14.x kernel.
  - Warm-path smoke: `status()` shows `cold_executions = 0` after a warm `verify_derivation` on 14.x.

- [ ] **`read_notebook_file` is license-independent on 14.x**
  - Parses a `.nb` with no kernel running (Python-native parser); confirm no license is consumed.

## Coverage note

`guide('v15')` describes exactly the three version-sensitive behaviors that exist: `ShowChatbar` suppression on agent-created notebooks (override with `show_chatbar=True` on `notebooks(action="create")` / `create_notebook`), the `$VersionNumber >= 15.` guards, and the addon `protocol_version` handshake - all covered by the checklist above. Theme-pinned screenshots and chat-cell filtering are **not implemented** (no `theme` param on `screenshot`, no chat-cell filter in `cells`) and are intentionally not mentioned in the guide; they remain roadmap items.
