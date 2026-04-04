#!/usr/bin/env python3
"""
CLI for mathematica-mcp-full setup and diagnostics.

Usage:
    uvx mathematica-mcp-full setup claude-desktop
    uvx mathematica-mcp-full setup cursor
    uvx mathematica-mcp-full setup vscode
    uvx mathematica-mcp-full doctor
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import VALID_PROFILES, FeatureFlags
from .guidance import build_claude_command, build_claude_hint, build_codex_guidance

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color(text: str, c: str) -> str:
    """Apply color if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{c}{text}{RESET}"
    return text


def success(msg: str) -> None:
    print(f"{color('✓', GREEN)} {msg}")


def error(msg: str) -> None:
    print(f"{color('✗', RED)} {msg}")


def warn(msg: str) -> None:
    print(f"{color('!', YELLOW)} {msg}")


def info(msg: str) -> None:
    print(f"{color('→', BLUE)} {msg}")


# Client configuration definitions
CLIENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude-desktop": {
        "name": "Claude Desktop",
        "config_paths": {
            "Darwin": "~/Library/Application Support/Claude/claude_desktop_config.json",
            "Linux": "~/.config/Claude/claude_desktop_config.json",
            "Windows": "%APPDATA%/Claude/claude_desktop_config.json",
        },
        "format": "json",
        "key": "mcpServers",
        "server_name": "mathematica",
    },
    "cursor": {
        "name": "Cursor",
        "config_paths": {
            "Darwin": "~/.cursor/mcp.json",
            "Linux": "~/.cursor/mcp.json",
            "Windows": "%USERPROFILE%/.cursor/mcp.json",
        },
        "format": "json",
        "key": "mcpServers",
        "server_name": "mathematica",
    },
    "vscode": {
        "name": "VS Code",
        "config_paths": {
            "Darwin": "~/.vscode/mcp.json",
            "Linux": "~/.vscode/mcp.json",
            "Windows": "%USERPROFILE%/.vscode/mcp.json",
        },
        "format": "json",
        "key": "servers",
        "server_name": "mathematica",
        "extra_fields": {"type": "stdio"},
    },
    "claude-code": {
        "name": "Claude Code (CLI)",
        "config_paths": {
            "Darwin": "~/.claude.json",
            "Linux": "~/.claude.json",
            "Windows": "%USERPROFILE%/.claude.json",
        },
        "format": "json",
        "key": "mcpServers",
        "server_name": "mathematica",
    },
    "codex": {
        "name": "OpenAI Codex CLI",
        "config_paths": {
            "Darwin": "~/.codex/config.toml",
            "Linux": "~/.codex/config.toml",
            "Windows": "%USERPROFILE%/.codex/config.toml",
        },
        "format": "toml",
        "server_name": "mathematica",
    },
    "gemini": {
        "name": "Gemini CLI",
        "config_paths": {
            "Darwin": "~/.gemini/settings.json",
            "Linux": "~/.gemini/settings.json",
            "Windows": "%USERPROFILE%/.gemini/settings.json",
        },
        "format": "json",
        "key": "mcpServers",
        "server_name": "mathematica",
    },
}


def get_system() -> str:
    """Get the current operating system."""
    return platform.system()


def expand_path(path: str) -> Path:
    """Expand environment variables and ~ in path."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    return Path(expanded)


def get_config_path(client: str) -> Path | None:
    """Get the config file path for a client on the current system."""
    if client not in CLIENT_CONFIGS:
        return None
    system = get_system()
    paths = CLIENT_CONFIGS[client]["config_paths"]
    if system not in paths:
        return None
    return expand_path(paths[system])


def get_package_dir() -> Path:
    """Get the project root when running from source, otherwise site-packages."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "addon").exists():
            return parent
    return current.parent.parent


def resolve_launcher(command: str) -> str:
    """Resolve *command* to an absolute executable path when possible.

    GUI MCP clients do not always inherit the same PATH as an interactive shell.
    Writing the absolute launcher path into generated configs makes startup
    resilient to those PATH differences. Fall back to the bare command so manual
    edits and unusual installs still work.
    """
    resolved = shutil.which(command)
    return resolved or command


def get_addon_dir() -> Path:
    """Get the addon directory."""
    # First check inside the package (when installed via pip/uvx)
    pkg_addon = Path(__file__).parent / "addon"
    if pkg_addon.exists():
        return pkg_addon
    # Fallback: check project root (when running from source)
    source_addon = Path(__file__).parent.parent.parent / "addon"
    if source_addon.exists():
        return source_addon
    return pkg_addon


def find_wolframscript() -> Path | None:
    """Find wolframscript executable."""
    # Check PATH first
    ws = shutil.which("wolframscript")
    if ws:
        return Path(ws)

    # Common locations
    system = get_system()
    if system == "Darwin":
        candidates = [
            "/Applications/Mathematica.app/Contents/MacOS/wolframscript",
            "/Applications/Wolfram Engine.app/Contents/MacOS/wolframscript",
        ]
    elif system == "Linux":
        candidates = [
            "/usr/local/bin/wolframscript",
            "/usr/bin/wolframscript",
            "/opt/Wolfram/Mathematica/14.0/Executables/wolframscript",
            "/usr/local/Wolfram/Mathematica/14.0/Executables/wolframscript",
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Program Files\Wolfram Research\Mathematica\14.0\wolframscript.exe",
            r"C:\Program Files\Wolfram Research\Wolfram Engine\14.0\wolframscript.exe",
        ]
    else:
        candidates = []

    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p

    return None


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is sufficient."""
    version = sys.version_info
    if version >= (3, 10):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (need 3.10+)"


def check_wolframscript() -> tuple[bool, str]:
    """Check if wolframscript is available."""
    ws = find_wolframscript()
    if ws:
        try:
            result = subprocess.run([str(ws), "-version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip() or "found"
            return True, f"wolframscript {version} at {ws}"
        except Exception:
            return True, f"wolframscript at {ws}"
    return False, "wolframscript not found in PATH"


def check_mathematica_addon() -> tuple[bool, str]:
    """Check if Mathematica addon is installed."""
    system = get_system()
    init_paths = get_addon_init_paths(Path.home(), system)
    if not init_paths:
        return False, "Unknown OS"

    for init_path in init_paths:
        if init_path.exists():
            content = init_path.read_text()
            if "MathematicaMCP" in content:
                return True, f"Addon configured in {init_path}"

    return False, "Addon not found in any known init.m location"


def get_addon_init_paths(home: Path, system: str) -> list[Path]:
    """Return likely Mathematica/Wolfram init.m locations for the OS."""
    if system == "Darwin":
        return [
            home / "Library/Wolfram/Kernel/init.m",
            home / "Library/Mathematica/Kernel/init.m",
        ]
    if system == "Linux":
        return [
            home / ".Wolfram/Kernel/init.m",
            home / ".Mathematica/Kernel/init.m",
        ]
    if system == "Windows":
        roaming = home / "AppData/Roaming"
        return [
            roaming / "Wolfram/Kernel/init.m",
            roaming / "Mathematica/Kernel/init.m",
        ]
    return []


def check_mcp_server_port() -> tuple[bool, str]:
    """Check if MCP server is listening on expected port."""
    import socket

    port = int(os.environ.get("MATHEMATICA_PORT", 9881))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result == 0:
            return True, f"MCP server responding on port {port}"
        return False, f"MCP server not responding on port {port} (is Mathematica running?)"
    except Exception as e:
        return False, f"Could not check port {port}: {e}"


def check_client_config(client: str) -> tuple[bool, str]:
    """Check if client config exists and contains mathematica server."""
    config_path = get_config_path(client)
    if not config_path:
        return False, f"Unknown client: {client}"

    if not config_path.exists():
        return False, f"Config file not found: {config_path}"

    client_info = CLIENT_CONFIGS[client]
    config_format = client_info.get("format", "json")
    server_name = client_info["server_name"]

    try:
        content = config_path.read_text()

        if config_format == "toml":
            # For TOML (Codex CLI), check for [mcp_servers.mathematica] section
            if f"[mcp_servers.{server_name}]" in content:
                return True, f"mathematica server configured in {config_path}"
            return False, f"mathematica server not in {config_path}"
        else:
            # JSON format
            config = json.loads(content)
            key = client_info["key"]

            if key in config and server_name in config[key]:
                return True, f"mathematica server configured in {config_path}"
            return False, f"mathematica server not in {config_path}"
    except json.JSONDecodeError:
        return False, f"Invalid JSON in {config_path}"
    except Exception as e:
        return False, f"Error reading config: {e}"


def generate_mcp_config(use_uvx: bool = True, profile: str | None = None) -> dict[str, Any]:
    """Generate the MCP server configuration."""
    if use_uvx:
        config = {"command": resolve_launcher("uvx"), "args": ["mathematica-mcp-full"]}
    else:
        # Use absolute path for local development
        pkg_dir = get_package_dir()
        config = {
            "command": resolve_launcher("uv"),
            "args": ["--directory", str(pkg_dir), "run", "mathematica-mcp-full"],
        }

    if profile:
        config["env"] = {"MATHEMATICA_PROFILE": profile}
    return config


def install_addon(wolframscript: Path, addon_dir: Path) -> bool:
    """Install the Mathematica addon."""
    install_script = addon_dir / "install.wl"
    if not install_script.exists():
        error(f"Addon install script not found: {install_script}")
        return False

    info(f"Running: wolframscript -file {install_script}")
    try:
        result = subprocess.run(
            [str(wolframscript), "-file", str(install_script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(addon_dir),
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        error("Addon installation timed out")
        return False
    except Exception as e:
        error(f"Addon installation failed: {e}")
        return False


def generate_toml_config(server_name: str, use_uvx: bool = True, profile: str | None = None) -> str:
    """Generate TOML config for Codex CLI."""
    if use_uvx:
        launcher = resolve_launcher("uvx")
        config = f"""
[mcp_servers.{server_name}]
command = "{launcher}"
args = ["mathematica-mcp-full"]
"""
    else:
        launcher = resolve_launcher("uv")
        pkg_dir = get_package_dir()
        config = f'''
[mcp_servers.{server_name}]
command = "{launcher}"
args = ["--directory", "{pkg_dir}", "run", "mathematica-mcp-full"]
'''

    if profile:
        config += f'env = {{ MATHEMATICA_PROFILE = "{profile}" }}\n'
    return config


def update_client_config(client: str, use_uvx: bool = True, profile: str | None = None) -> bool:
    """Update the client configuration to add mathematica server."""
    config_path = get_config_path(client)
    if not config_path:
        error(f"Unknown client: {client}")
        return False

    client_info = CLIENT_CONFIGS[client]
    config_format = client_info.get("format", "json")
    server_name = client_info["server_name"]

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_format == "toml":
        # Handle TOML config (Codex CLI)
        return update_toml_config(config_path, server_name, use_uvx, profile)
    else:
        # Handle JSON config
        return update_json_config(config_path, client_info, use_uvx, profile)


def update_toml_config(
    config_path: Path,
    server_name: str,
    use_uvx: bool,
    profile: str | None,
) -> bool:
    """Update TOML config file for Codex CLI."""
    existing_content = ""
    if config_path.exists():
        existing_content = config_path.read_text()

    # Generate new TOML section
    new_section = generate_toml_config(server_name, use_uvx, profile)
    section_pattern = re.compile(rf"(?ms)^\[mcp_servers\.{re.escape(server_name)}\]\n.*?(?=^\[|\Z)")

    if section_pattern.search(existing_content):
        updated_content = section_pattern.sub(new_section.rstrip() + "\n\n", existing_content, count=1)
    else:
        updated_content = existing_content
        if updated_content and not updated_content.endswith("\n"):
            updated_content += "\n"
        updated_content += new_section

    try:
        config_path.write_text(updated_content)
        success(f"Updated {config_path}")
        return True
    except Exception as e:
        error(f"Failed to write config: {e}")
        return False


def update_json_config(
    config_path: Path,
    client_info: dict[str, Any],
    use_uvx: bool,
    profile: str | None,
) -> bool:
    """Update JSON config file."""
    key = client_info["key"]
    server_name = client_info["server_name"]

    # Read existing config or start fresh
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            warn(f"Invalid JSON in {config_path}, creating backup and starting fresh")
            backup = config_path.with_suffix(".json.bak")
            shutil.copy(config_path, backup)
            config = {}
    else:
        config = {}

    # Add server config
    if key not in config:
        config[key] = {}

    existing_server = config.get(key, {}).get(server_name, {})
    server_config = generate_mcp_config(use_uvx, profile)

    existing_env = existing_server.get("env", {}) if isinstance(existing_server, dict) else {}
    new_env = server_config.get("env", {})
    merged_env = {**existing_env, **new_env}
    if merged_env:
        server_config["env"] = merged_env

    # Add extra fields if needed (e.g., "type": "stdio" for VS Code)
    if "extra_fields" in client_info:
        server_config.update(client_info["extra_fields"])

    config[key][server_name] = server_config

    # Write config
    try:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        success(f"Updated {config_path}")
        return True
    except Exception as e:
        error(f"Failed to write config: {e}")
        return False


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def _upsert_marked_block(existing: str, block: str, *, start_marker: str, end_marker: str) -> str:
    marked = f"{start_marker}\n{block.rstrip()}\n{end_marker}\n"
    if start_marker in existing and end_marker in existing:
        before, _, rest = existing.partition(start_marker)
        _, _, after = rest.partition(end_marker)
        prefix = before.rstrip()
        suffix = after.lstrip("\n")
        parts = [part for part in [prefix, marked.rstrip(), suffix.rstrip()] if part]
        return "\n\n".join(parts) + "\n"

    base = existing.rstrip()
    if not base:
        return marked
    return f"{base}\n\n{marked}"


def install_claude_code_guidance(project_dir: Path, profile: str | None) -> list[Path]:
    features = FeatureFlags.from_env(profile_override=profile)
    written: list[Path] = []

    command_path = project_dir / ".claude" / "commands" / "mathematica.md"
    command_content = build_claude_command(features).rstrip() + "\n"
    if _write_if_changed(command_path, command_content):
        written.append(command_path)

    claude_md_path = project_dir / "CLAUDE.md"
    start_marker = "<!-- mathematica-mcp:start -->"
    end_marker = "<!-- mathematica-mcp:end -->"
    existing = claude_md_path.read_text() if claude_md_path.exists() else ""
    updated = _upsert_marked_block(
        existing,
        build_claude_hint(features).rstrip(),
        start_marker=start_marker,
        end_marker=end_marker,
    )
    if not claude_md_path.exists() or updated != existing:
        claude_md_path.write_text(updated)
        written.append(claude_md_path)

    return written


def install_codex_guidance(project_dir: Path, profile: str | None) -> list[Path]:
    """Install AGENTS.md for OpenAI Codex CLI."""
    features = FeatureFlags.from_env(profile_override=profile)
    written: list[Path] = []

    agents_md_path = project_dir / "AGENTS.md"
    start_marker = "<!-- mathematica-mcp:start -->"
    end_marker = "<!-- mathematica-mcp:end -->"
    existing = agents_md_path.read_text() if agents_md_path.exists() else ""
    updated = _upsert_marked_block(
        existing,
        build_codex_guidance(features).rstrip(),
        start_marker=start_marker,
        end_marker=end_marker,
    )
    if not agents_md_path.exists() or updated != existing:
        agents_md_path.write_text(updated)
        written.append(agents_md_path)

    return written


def cmd_setup(args: argparse.Namespace) -> int:
    """Run the setup command."""
    client = args.client.lower()

    # Normalize client names
    client_aliases = {
        "claude": "claude-desktop",
        "claudedesktop": "claude-desktop",
        "claude_desktop": "claude-desktop",
        "code": "claude-code",
        "claudecode": "claude-code",
        "claude_code": "claude-code",
        "vs-code": "vscode",
        "vs_code": "vscode",
        "openai-codex": "codex",
        "openai_codex": "codex",
        "codex-cli": "codex",
        "gemini-cli": "gemini",
        "google-gemini": "gemini",
    }
    client = client_aliases.get(client, client)

    if client not in CLIENT_CONFIGS:
        error(f"Unknown client: {args.client}")
        print(f"\nSupported clients: {', '.join(CLIENT_CONFIGS.keys())}")
        return 1

    client_name = CLIENT_CONFIGS[client]["name"]
    print(f"\n{color(f'Setting up mathematica-mcp-full for {client_name}', BOLD)}\n")

    # Step 1: Check wolframscript
    info("Checking wolframscript...")
    ws = find_wolframscript()
    if not ws:
        error("wolframscript not found!")
        print("\nPlease ensure Mathematica is installed and wolframscript is in your PATH.")
        print("See: https://github.com/AbhiRawat4841/mathematica-mcp#prerequisites")
        return 1
    success(f"Found wolframscript at {ws}")

    # Step 2: Install Mathematica addon
    if not args.skip_addon:
        info("Installing Mathematica addon...")
        addon_dir = get_addon_dir()
        if not addon_dir.exists():
            error(f"Addon directory not found: {addon_dir}")
            print("\nThis may happen if running via uvx before the package is published.")
            print("Please clone the repo and run: wolframscript -file addon/install.wl")
            return 1

        if install_addon(ws, addon_dir):
            success("Mathematica addon installed")
        else:
            error("Failed to install addon (you may need to run manually)")
            warn("Run: wolframscript -file addon/install.wl")
    else:
        info("Skipping addon installation (--skip-addon)")

    # Step 3: Update client config
    info(f"Configuring {client_name}...")
    use_uvx = not args.local
    if update_client_config(client, use_uvx=use_uvx, profile=args.profile):
        success(f"{client_name} configured")
    else:
        return 1

    if client == "claude-code" and args.project_dir:
        project_dir = expand_path(args.project_dir)
        changed_paths = install_claude_code_guidance(project_dir, args.profile)
        if changed_paths:
            for changed_path in changed_paths:
                success(f"Installed Claude Code guidance at {changed_path}")
        else:
            info(f"Claude Code guidance already up to date in {project_dir}")
    elif client == "claude-code":
        info("No project guidance installed. Re-run with --project-dir to add .claude/commands and CLAUDE.md hints.")

    if client == "codex" and args.project_dir:
        project_dir = expand_path(args.project_dir)
        changed_paths = install_codex_guidance(project_dir, args.profile)
        if changed_paths:
            for changed_path in changed_paths:
                success(f"Installed Codex guidance at {changed_path}")
        else:
            info(f"Codex guidance already up to date in {project_dir}")
    elif client == "codex":
        info("No project guidance installed. Re-run with --project-dir to add AGENTS.md hints.")

    # Done!
    print(f"\n{color('Setup complete!', GREEN + BOLD)}\n")
    print("Next steps:")
    print(f"  1. {color('Restart Mathematica', BOLD)} (for the addon to load)")
    print(f"  2. {color(f'Restart {client_name}', BOLD)} (to load the MCP server)")
    print(f"\nTo verify: {color('uvx mathematica-mcp-full doctor', BLUE)}")

    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run diagnostics to verify installation."""
    print(f"\n{color('mathematica-mcp-full doctor', BOLD)}\n")

    all_ok = True

    # Python version
    ok, msg = check_python_version()
    if ok:
        success(msg)
    else:
        error(msg)
        all_ok = False

    # wolframscript
    ok, msg = check_wolframscript()
    if ok:
        success(msg)
    else:
        error(msg)
        all_ok = False

    # Mathematica addon
    ok, msg = check_mathematica_addon()
    if ok:
        success(msg)
    else:
        warn(msg)
        print(f"    Run: {color('uvx mathematica-mcp-full setup <client>', BLUE)}")

    # MCP server port
    ok, msg = check_mcp_server_port()
    if ok:
        success(msg)
    else:
        warn(msg)
        print("    Start Mathematica to launch the MCP server")

    # Client configs
    print(f"\n{color('Client configurations:', BOLD)}")
    for client in CLIENT_CONFIGS:
        ok, msg = check_client_config(client)
        name = CLIENT_CONFIGS[client]["name"]
        if ok:
            success(f"{name}: {msg}")
        else:
            config_path = get_config_path(client)
            if config_path and config_path.exists():
                warn(f"{name}: {msg}")
            else:
                info(f"{name}: not configured")

    print()
    if all_ok:
        success("All core checks passed!")
    else:
        warn("Some checks failed. Run setup to fix.")

    return 0 if all_ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    """Print the MCP config for a client."""
    client = args.client.lower()

    # Normalize
    client_aliases = {
        "claude": "claude-desktop",
        "code": "claude-code",
        "openai-codex": "codex",
        "gemini-cli": "gemini",
    }
    client = client_aliases.get(client, client)

    if client not in CLIENT_CONFIGS:
        error(f"Unknown client: {args.client}")
        return 1

    client_info = CLIENT_CONFIGS[client]
    config_format = client_info.get("format", "json")
    use_uvx = not args.local
    config_path = get_config_path(client)

    print(f"# Add to: {config_path}\n")

    if config_format == "toml":
        # Output TOML format for Codex
        print(generate_toml_config(client_info["server_name"], use_uvx, args.profile).strip())
    else:
        # Output JSON format
        server_config = generate_mcp_config(use_uvx, args.profile)
        if "extra_fields" in client_info:
            server_config.update(client_info["extra_fields"])

        config = {client_info["key"]: {client_info["server_name"]: server_config}}
        print(json.dumps(config, indent=2))

    return 0


def main_cli() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mathematica-mcp-full",
        description="Mathematica MCP Server - Give your AI Agent the power of Wolfram Language",
    )
    parser.add_argument("--profile", choices=VALID_PROFILES, default=None, help="Runtime tool profile override")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Configure mathematica-mcp-full for an MCP client",
        description="Automatically configure mathematica-mcp-full for your editor/AI assistant",
    )
    setup_parser.add_argument(
        "client",
        choices=list(CLIENT_CONFIGS.keys()) + ["claude", "code", "openai-codex", "gemini-cli"],
        help="The MCP client to configure",
    )
    setup_parser.add_argument("--skip-addon", action="store_true", help="Skip Mathematica addon installation")
    setup_parser.add_argument("--local", action="store_true", help="Use local path instead of uvx (for development)")
    setup_parser.add_argument(
        "--profile", choices=VALID_PROFILES, default=None, help="Tool profile to configure in the client MCP config"
    )
    setup_parser.add_argument(
        "--project-dir", default=None, help="Project root for agent guidance installation (CLAUDE.md, AGENTS.md)"
    )
    setup_parser.set_defaults(func=cmd_setup)

    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Check installation and diagnose issues")
    doctor_parser.set_defaults(func=cmd_doctor)

    # config command
    config_parser = subparsers.add_parser("config", help="Print MCP config JSON for a client")
    config_parser.add_argument(
        "client",
        choices=list(CLIENT_CONFIGS.keys()) + ["claude", "code", "openai-codex", "gemini-cli"],
        help="The MCP client",
    )
    config_parser.add_argument("--local", action="store_true", help="Use local path instead of uvx")
    config_parser.add_argument(
        "--profile", choices=VALID_PROFILES, default=None, help="Tool profile to emit in the printed config"
    )
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if args.command is None:
        # No subcommand - run the MCP server (default behavior)
        if args.profile:
            os.environ["MATHEMATICA_PROFILE"] = args.profile
        try:
            import sys as _sys

            print("mathematica-mcp: importing server...", file=_sys.stderr)
            from .server import main as server_main

            print("mathematica-mcp: starting MCP server...", file=_sys.stderr)
            server_main()
        except Exception as _e:
            import sys as _sys
            import traceback

            print(f"mathematica-mcp: CRASH: {_e}", file=_sys.stderr)
            traceback.print_exc(file=_sys.stderr)
            return 1
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_cli())
