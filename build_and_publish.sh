#!/bin/bash

# FLUX Scientific - Build and Publish Script
# This script builds and publishes the package to PyPI

echo "🚀 FLUX Scientific - Build and Publish Script"
echo "============================================"

# Clean previous builds
echo "📧 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Install build tools
echo "📦 Installing build tools..."
pip install --upgrade pip
pip install --upgrade build twine

# Build the package
echo "🔨 Building distribution packages..."
python -m build

# Check the build
echo "✅ Checking package with twine..."
twine check dist/*

# Display package info
echo ""
echo "📋 Package contents:"
ls -lh dist/

echo ""
echo "🎯 Ready to publish to PyPI!"
echo ""
echo "To publish to TEST PyPI (recommended first):"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "To publish to PRODUCTION PyPI:"
echo "  twine upload dist/*"
echo ""
echo "Note: You'll need your PyPI token configured in ~/.pypirc or use:"
echo "  twine upload -u __token__ -p YOUR_PYPI_TOKEN dist/*"