#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Build the package
python3 -m build

# Publish the package to PyPI using the token from .pypirc
twine upload --config-file .pypirc dist/*

