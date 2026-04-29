#!/bin/bash
set -e

echo "Setting up environment for Vast.ai container..."

# Install system dependencies if needed (assuming Debian/Ubuntu based container)
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y curl git python3-venv
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
fi

# Make sure uv is available in path for the current session if installed to ~/.cargo/bin
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Sync Python dependencies using uv
echo "Installing Python dependencies..."
uv sync

echo "Setup complete!"
