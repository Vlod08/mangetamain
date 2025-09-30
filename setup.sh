#!/bin/bash

# This script sets up the Poetry environment for the project

set -e  # Exit on any error
PYTHON_PATH_MANGETAMAIN=`which python3.11`

echo "=== Kit Big Data Project Setup ==="
echo ""

# Check if Poetry is installed
echo "Checking Poetry installation..."
if ! command -v poetry &> /dev/null; then
    echo ""
    echo "Poetry is not installed on your system."
    echo "Poetry is required to manage this project's dependencies."
else
    # Get poetry version string
    VERSION=$(poetry --version 2>/dev/null | awk '{print $3}' | tr -d '()')
    if [ "$VERSION" = "2.2.0" ]; then
      echo "Poetry version 2.2.0 is installed !"
    else
      echo "Poetry version is $VERSION (not 2.2.0)"
    fi
fi

echo ""


echo "Checking if the specified Python installation exists..."
if [ ! -f "$PYTHON_PATH_MANGETAMAIN" ]; then
    echo "ERROR: Python 3.11 not found at: $PYTHON_PATH_MANGETAMAIN"
    exit 1
else
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

    if [[ "$PY_VERSION" == 3.11.* ]]; then
        echo "Python version 3.11.x is installed !"
    else
        echo "Python version is $PY_VERSION (not 3.11.x)"
    fi
fi

echo ""

# Set up Poetry virtual environment
echo "Setting up Poetry virtual environment..."
echo "Using Python: $PYTHON_PATH_MANGETAMAIN"

if poetry env use "$PYTHON_PATH_MANGETAMAIN"; then
    echo "Poetry virtual environment configured successfully"
else
    echo "ERROR: Failed to configure Poetry virtual environment"
    echo "Please check that Python 3.11 is properly installed"
    exit 1
fi

echo ""

# Install dependencies
echo "Installing project dependencies..."
if poetry install; then
    echo "Dependencies installed successfully"
else
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo ""

