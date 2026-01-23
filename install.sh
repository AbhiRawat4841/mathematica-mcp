#!/bin/bash
# install.sh - Interactive installer for mathematica-mcp
#
# Usage:
#   ./install.sh                        # Interactive mode
#   ./install.sh claude-desktop         # Direct setup for Claude Desktop
#   ./install.sh --help                 # Show help
#
# One-liner (no clone needed):
#   curl -sSL https://raw.githubusercontent.com/AbhiRawat4841/mathematica-mcp/main/install.sh | bash
#   curl -sSL https://raw.githubusercontent.com/AbhiRawat4841/mathematica-mcp/main/install.sh | bash -s -- claude-desktop

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helpers
success() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }
info() { echo -e "${BLUE}→${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}\n"; }

# Supported clients
CLIENTS=("claude-desktop" "cursor" "vscode" "claude-code" "codex" "gemini")

show_help() {
    echo ""
    echo -e "${BOLD}mathematica-mcp installer${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC}"
    echo "  ./install.sh [client]        Install for a specific client"
    echo "  ./install.sh                 Interactive mode"
    echo "  ./install.sh --help          Show this help"
    echo ""
    echo -e "${BOLD}Supported clients:${NC}"
    echo "  claude-desktop   Claude Desktop app"
    echo "  cursor           Cursor editor"
    echo "  vscode           VS Code with MCP extension"
    echo "  claude-code      Claude Code CLI"
    echo "  codex            OpenAI Codex CLI"
    echo "  gemini           Google Gemini CLI"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  ./install.sh claude-desktop"
    echo "  curl -sSL https://raw.githubusercontent.com/.../install.sh | bash -s -- cursor"
    echo ""
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *)       echo "unknown" ;;
    esac
}

# Check if command exists
has_command() {
    command -v "$1" &> /dev/null
}

# Find wolframscript
find_wolframscript() {
    if has_command wolframscript; then
        which wolframscript
        return 0
    fi
    
    local os=$(detect_os)
    local candidates=()
    
    if [ "$os" = "macos" ]; then
        candidates=(
            "/Applications/Mathematica.app/Contents/MacOS/wolframscript"
            "/Applications/Wolfram Engine.app/Contents/MacOS/wolframscript"
        )
    elif [ "$os" = "linux" ]; then
        candidates=(
            "/usr/local/bin/wolframscript"
            "/usr/bin/wolframscript"
            "/opt/Wolfram/Mathematica/14.0/Executables/wolframscript"
            "/usr/local/Wolfram/Mathematica/14.0/Executables/wolframscript"
        )
    fi
    
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    
    return 1
}

# Get config file path for a client
get_config_path() {
    local client=$1
    local os=$(detect_os)
    
    case "$client" in
        claude-desktop)
            case "$os" in
                macos)   echo "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
                linux)   echo "$HOME/.config/Claude/claude_desktop_config.json" ;;
                windows) echo "$APPDATA/Claude/claude_desktop_config.json" ;;
            esac
            ;;
        cursor)
            echo "$HOME/.cursor/mcp.json"
            ;;
        vscode)
            echo "$HOME/.vscode/mcp.json"
            ;;
        claude-code)
            echo "$HOME/.claude.json"
            ;;
        codex)
            echo "$HOME/.codex/config.toml"
            ;;
        gemini)
            echo "$HOME/.gemini/settings.json"
            ;;
    esac
}

# Check if jq is available, if not use Python
json_tool() {
    if has_command jq; then
        echo "jq"
    elif has_command python3; then
        echo "python3"
    elif has_command python; then
        echo "python"
    else
        echo "none"
    fi
}

# Update TOML config (for Codex CLI)
update_toml_config() {
    local config_path=$1
    local repo_path=$2
    local use_uvx=$3
    
    mkdir -p "$(dirname "$config_path")"
    
    # Check if already configured
    if [ -f "$config_path" ] && grep -q "\[mcp_servers.mathematica\]" "$config_path"; then
        warn "mathematica already configured in $config_path"
        return 0
    fi
    
    # Append TOML config
    if [ "$use_uvx" = "true" ]; then
        cat >> "$config_path" << 'EOF'

[mcp_servers.mathematica]
command = "uvx"
args = ["mathematica-mcp"]
EOF
    else
        cat >> "$config_path" << EOF

[mcp_servers.mathematica]
command = "uv"
args = ["--directory", "$repo_path", "run", "mathematica-mcp"]
EOF
    fi
}

# Update JSON config file
update_config() {
    local config_path=$1
    local client=$2
    local repo_path=$3
    local use_uvx=$4
    
    # Handle TOML config for Codex
    if [ "$client" = "codex" ]; then
        update_toml_config "$config_path" "$repo_path" "$use_uvx"
        return $?
    fi
    
    local tool=$(json_tool)
    local key="mcpServers"
    local extra_type=""
    
    # VS Code uses "servers" instead of "mcpServers"
    if [ "$client" = "vscode" ]; then
        key="servers"
        extra_type='true'
    fi
    
    # Create directory if needed
    mkdir -p "$(dirname "$config_path")"
    
    # Generate server config
    if [ "$use_uvx" = "true" ]; then
        local server_config='{"command": "uvx", "args": ["mathematica-mcp"]}'
    else
        local server_config="{\"command\": \"uv\", \"args\": [\"--directory\", \"$repo_path\", \"run\", \"mathematica-mcp\"]}"
    fi
    
    # Add type: stdio for VS Code
    if [ -n "$extra_type" ]; then
        server_config=$(echo "$server_config" | sed 's/}$/, "type": "stdio"}/')
    fi
    
    if [ "$tool" = "jq" ]; then
        # Use jq
        if [ -f "$config_path" ]; then
            local tmp=$(mktemp)
            jq --arg key "$key" --argjson server "$server_config" '.[$key].mathematica = $server' "$config_path" > "$tmp" && mv "$tmp" "$config_path"
        else
            echo "{\"$key\": {\"mathematica\": $server_config}}" | jq '.' > "$config_path"
        fi
    elif [ "$tool" = "python3" ] || [ "$tool" = "python" ]; then
        # Use Python
        $tool << PYTHON
import json
import os

config_path = "$config_path"
key = "$key"
server_config = json.loads('$server_config')

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        try:
            config = json.load(f)
        except:
            config = {}
else:
    config = {}

if key not in config:
    config[key] = {}

config[key]["mathematica"] = server_config

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')
PYTHON
    else
        error "No JSON tool available (need jq or python)"
        return 1
    fi
}

# Interactive client selection
select_client() {
    # Check if running interactively (stdin is a terminal)
    if [ ! -t 0 ]; then
        echo "" >&2
        echo "Error: Interactive mode requires a terminal." >&2
        echo "When using curl | bash, specify the client:" >&2
        echo "  curl -sSL <url>/install.sh | bash -s -- claude-desktop" >&2
        echo "" >&2
        echo "Supported clients: ${CLIENTS[*]}" >&2
        echo ""
        return
    fi

    header "Select your MCP client:"

    local i=1
    for client in "${CLIENTS[@]}"; do
        case "$client" in
            claude-desktop) echo "  $i) Claude Desktop" ;;
            cursor)         echo "  $i) Cursor" ;;
            vscode)         echo "  $i) VS Code" ;;
            claude-code)    echo "  $i) Claude Code (CLI)" ;;
            codex)          echo "  $i) OpenAI Codex CLI" ;;
            gemini)         echo "  $i) Google Gemini CLI" ;;
        esac
        ((i++))
    done

    echo ""
    read -p "Enter number (1-${#CLIENTS[@]}): " choice

    if [[ "$choice" =~ ^[1-6]$ ]]; then
        echo "${CLIENTS[$((choice-1))]}"
    else
        echo ""
    fi
}

# Main installation
main() {
    # Parse arguments first
    local client=""
    local use_uvx="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --uvx)
                use_uvx="true"
                shift
                ;;
            *)
                client="$1"
                shift
                ;;
        esac
    done
    
    header "mathematica-mcp installer"
    
    # Detect OS
    local os=$(detect_os)
    info "Detected OS: $os"
    
    # Check Python
    if has_command python3; then
        success "Python 3 found: $(python3 --version 2>&1)"
    elif has_command python; then
        success "Python found: $(python --version 2>&1)"
    else
        error "Python not found. Please install Python 3.10+"
        exit 1
    fi
    
    # Check/find wolframscript
    info "Looking for wolframscript..."
    local ws=$(find_wolframscript)
    if [ -n "$ws" ]; then
        success "Found wolframscript: $ws"
    else
        error "wolframscript not found!"
        echo ""
        echo "Please install Mathematica and ensure wolframscript is in your PATH."
        echo ""
        if [ "$os" = "macos" ]; then
            echo "Add to ~/.zshrc:"
            echo '  export PATH="/Applications/Mathematica.app/Contents/MacOS:$PATH"'
        elif [ "$os" = "linux" ]; then
            echo "Add to ~/.bashrc:"
            echo '  export PATH="/usr/local/Wolfram/Mathematica/14.0/Executables:$PATH"'
        fi
        exit 1
    fi
    
    # Check/install uv
    if has_command uv; then
        success "uv found: $(uv --version 2>&1)"
    else
        warn "uv not found. Installing..."
        if [ "$os" = "macos" ] || [ "$os" = "linux" ]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            if has_command uv; then
                success "uv installed successfully"
            else
                error "Failed to install uv"
                exit 1
            fi
        else
            error "Please install uv manually: https://docs.astral.sh/uv/"
            exit 1
        fi
    fi
    
    # Determine repo path
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_path=""
    
    if [ -f "$script_dir/pyproject.toml" ]; then
        repo_path="$script_dir"
        info "Using local repository: $repo_path"
    else
        # Clone the repo
        info "Cloning mathematica-mcp repository..."
        local clone_dir="$HOME/.local/share/mathematica-mcp"
        
        if [ -d "$clone_dir" ]; then
            info "Updating existing clone..."
            cd "$clone_dir" && git pull --quiet
        else
            git clone --quiet https://github.com/AbhiRawat4841/mathematica-mcp.git "$clone_dir"
        fi
        
        repo_path="$clone_dir"
        success "Repository ready: $repo_path"
    fi
    
    # Install dependencies
    info "Installing dependencies..."
    cd "$repo_path"
    uv sync --quiet
    success "Dependencies installed"
    
    # Install Mathematica addon
    info "Installing Mathematica addon..."
    if [ -f "$repo_path/addon/install.wl" ]; then
        cd "$repo_path/addon"
        "$ws" -file install.wl
        success "Mathematica addon installed"
    else
        warn "Addon install script not found, skipping"
    fi
    
    # Select client
    if [ -z "$client" ]; then
        client=$(select_client)
        if [ -z "$client" ]; then
            error "Invalid selection"
            exit 1
        fi
    fi
    
    # Normalize client name
    case "$client" in
        claude|claudedesktop|claude_desktop) client="claude-desktop" ;;
        code|claudecode|claude_code) client="claude-code" ;;
        vs-code|vs_code) client="vscode" ;;
    esac
    
    # Validate client
    local valid=false
    for c in "${CLIENTS[@]}"; do
        if [ "$c" = "$client" ]; then
            valid=true
            break
        fi
    done
    
    if [ "$valid" = "false" ]; then
        error "Unknown client: $client"
        echo "Supported: ${CLIENTS[*]}"
        exit 1
    fi
    
    # Get config path
    local config_path=$(get_config_path "$client")
    info "Config file: $config_path"
    
    # Backup existing config
    if [ -f "$config_path" ]; then
        cp "$config_path" "$config_path.bak"
        info "Backed up existing config to $config_path.bak"
    fi
    
    # Update config
    info "Updating configuration..."
    if update_config "$config_path" "$client" "$repo_path" "$use_uvx"; then
        success "Configuration updated"
    else
        error "Failed to update configuration"
        exit 1
    fi
    
    # Done!
    header "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. ${BOLD}Restart Mathematica${NC} (for the addon to load)"
    
    case "$client" in
        claude-desktop) echo "  2. ${BOLD}Restart Claude Desktop${NC}" ;;
        cursor)         echo "  2. ${BOLD}Restart Cursor${NC}" ;;
        vscode)         echo "  2. ${BOLD}Restart VS Code${NC}" ;;
        claude-code)    echo "  2. ${BOLD}Restart Claude Code${NC}" ;;
    esac
    
    echo ""
    echo "To verify installation:"
    echo "  cd $repo_path && uv run mathematica-mcp doctor"
    echo ""
}

main "$@"
