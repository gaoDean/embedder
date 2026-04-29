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

if [ ! -d ".git" ]; then
    echo "Cloning repository..."
    # Clone into a temporary directory and move files, or clone into 'embedder'
    # To avoid "directory not empty" errors when cloning into '.'
    git clone https://github.com/gaoDean/embedder.git /workspace/embedder || git clone https://github.com/gaoDean/embedder.git
    cd *embedder
fi

# Sync Python dependencies using uv
echo "Installing Python dependencies..."
uv sync

echo "Setup complete!"
