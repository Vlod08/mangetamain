#!/bin/bash

# This script sets up the Poetry environment for the project

set -e  # Exit on any error

echo "=== Kit Big Data Project Setup ==="
echo ""

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "Environment variables loaded"
else
    echo "Warning: .env file not found"
    exit 1
fi

echo ""

# Check if Poetry is installed
echo "Checking Poetry installation..."
if ! command -v poetry &> /dev/null; then
    echo ""
    echo "ERROR: Poetry is not installed!"
    echo ""
    exit 1
else
    poetry_version=$(poetry --version 2>/dev/null || echo "unknown")
    echo "Poetry is installed: $poetry_version"
fi

echo ""

# Check if the specified Python path exists
if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: PYTHON_PATH not set in .env file"
    exit 1
fi

echo "Checking if the specified Python installation exists..."
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python 3.11 not found at: $PYTHON_PATH"
    echo ""
    echo "Then update the PYTHON_PATH in your .env file"
    exit 1
else
    python_version=$("$PYTHON_PATH" --version 2>/dev/null || echo "unknown")
    echo "Python version found: $python_version"
    echo "Path: $PYTHON_PATH"
fi

echo ""

# Set up Poetry virtual environment
echo "Setting up Poetry virtual environment..."
echo "Using Python: $PYTHON_PATH"

if poetry env use "$PYTHON_PATH"; then
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

