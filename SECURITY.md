# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.7.x   | :white_check_mark: |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Threat Model

### What mathematica-mcp IS

A **local development tool** that bridges MCP-compatible AI agents to a user's own Mathematica installation. It is designed for **single-user, local-machine use**.

### What mathematica-mcp IS NOT

- A multi-tenant service
- A sandboxed execution environment
- Suitable for running untrusted code

### Trust Boundaries

```
LLM Client  ──(MCP stdio)──>  Python Server  ──(TCP localhost:9881)──>  Mathematica Kernel
```

1. **LLM Client → Python Server**: MCP stdio protocol. Optional auth token via `MATHEMATICA_MCP_TOKEN` environment variable.
2. **Python Server → Mathematica Addon**: TCP socket on `127.0.0.1:9881` (local-only binding). Not accessible from the network.
3. **Mathematica Kernel**: Full Wolfram Language execution with the user's privileges. No kernel-level sandboxing.

## Security Controls

### Network Binding

The Mathematica addon binds to `127.0.0.1` only — not `0.0.0.0`. The server is not accessible from other machines on the network by default. The port is configurable via `MATHEMATICA_PORT`.

### Authentication

An optional `MATHEMATICA_MCP_TOKEN` environment variable can be set. When configured, every request from the Python server must include a matching token. **Not enabled by default** (local-only access is assumed).

### Size Limits

- **Request**: 5 MB maximum (`MAX_REQUEST_BYTES`)
- **Response**: 20 MB maximum (`MAX_RESPONSE_BYTES`)

These prevent memory exhaustion from oversized payloads.

### Timeouts

- **Socket timeout**: 180 seconds (configurable)
- **Per-tool timeouts**: 15–300 seconds depending on the operation
- **Subprocess timeout**: Enforced via `subprocess.run(..., timeout=T)`

These prevent hung connections and runaway computations.

## Known Input Handling Gaps

The symbol lookup fallback path (`_lookup_symbols_in_kernel` in `server.py`) interpolates query strings directly into Wolfram Language code via Python f-strings when the fast in-memory index is unavailable. The fast path (pure Python symbol index) avoids this entirely.

**Primary mitigation**: the server binds to localhost only and is not network-exposed by default, limiting the attack surface to the local machine. However, prompt-injected or copied user text can flow through this path. This is documented here for transparency.

## Dangerous Operations

The following tools execute arbitrary Wolfram Language code with the **user's full OS privileges**:

- `execute_code` — runs any Wolfram expression
- `run_script` — runs a `.wl` script file
- `evaluate_cell` / `evaluate_selection` — evaluates notebook cells

Wolfram Language has access to:

- **File system**: read, write, delete files
- **Network**: HTTP requests, socket connections
- **Shell commands**: `Run[]`, `RunProcess[]`
- **System information**: environment variables, process details

**There is no kernel-level sandbox.** The Wolfram kernel runs with the same permissions as the user who started it.

## Safe Defaults

- The `math` profile exposes only ~25 computation tools (no file or notebook operations)
- Telemetry is disabled by default
- Expression cache is memory-only
- Raster cache: 50 entries max, cleaned on restart

## Reporting a Vulnerability

Please report security issues via [GitHub Security Advisories](https://github.com/AbhiRawat4841/mathematica-mcp/security/advisories/new).

**Do not** open a public GitHub issue for security vulnerabilities.

## Recommended Practices

1. Use the `math` profile when notebook/file access is unnecessary
2. Enable `MATHEMATICA_MCP_TOKEN` when sharing a machine with other users
3. Run Mathematica as a non-privileged user
4. Keep mathematica-mcp and Mathematica up to date
