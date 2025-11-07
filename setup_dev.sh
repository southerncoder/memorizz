#!/bin/bash

# Memorizz Development Environment Setup Script

set -e

echo "======================================================================"
echo "  Memorizz Development Environment Setup"
echo "======================================================================"
echo ""

# Check if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "⚠ Warning: No conda environment detected"
    echo "  Consider creating one: conda create -n memorizz python=3.11"
fi
echo ""

# Install package in editable mode
echo "Step 1: Installing Memorizz in editable mode..."
pip install -e .
echo "✓ Package installed"
echo ""

# Install development dependencies
echo "Step 2: Installing development dependencies..."
pip install pre-commit black flake8 isort pytest ipython jupyter
echo "✓ Development dependencies installed"
echo ""

# Install pre-commit hooks
echo "Step 3: Setting up pre-commit hooks..."
pre-commit install
echo "✓ Pre-commit hooks installed"
echo ""

# Run initial format
echo "Step 4: Running initial code formatting..."
black src/memorizz --quiet || true
isort src/memorizz --profile black --quiet || true
echo "✓ Code formatted"
echo ""

# Check syntax
echo "Step 5: Checking Python syntax..."
find src/memorizz -name "*.py" -exec python -m py_compile {} \;
echo "✓ Syntax check passed"
echo ""

echo "======================================================================"
echo "  ✅ Development environment setup complete!"
echo "======================================================================"
echo ""
echo "Useful commands:"
echo "  make help       - Show all available commands"
echo "  make lint       - Check code quality"
echo "  make format     - Format code"
echo "  make test       - Run tests"
echo ""
echo "Git hooks are now active - code will be checked before each commit!"
echo ""
