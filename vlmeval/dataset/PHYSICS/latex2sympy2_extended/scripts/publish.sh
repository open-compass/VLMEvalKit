#!/bin/bash

# Exit on error
set -e

echo "=== Latex2Sympy2 Extended Package Publisher ==="

# Function to clean previous builds
clean_builds() {
    echo "Cleaning previous builds..."
    rm -rf dist/ build/ *.egg-info/
}

# Function to build package
build_package() {
    echo "Building package..."
    python -m build
}

# Function to upload to PyPI
upload_to_pypi() {
    echo "Uploading to PyPI..."
    python -m twine upload dist/*
}

# Main execution
echo "Installing publishing dependencies..."
pip install --upgrade pip build twine

# Ensure we're in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Execute the publishing process
clean_builds
build_package
upload_to_pypi

echo "Package successfully published to PyPI!"
echo "You can now install it with: pip install latex2sympy2-extended"
