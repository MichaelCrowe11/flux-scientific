#!/bin/bash

# PyPI Publishing Script for FLUX Scientific Computing Language

echo "📦 Publishing FLUX to PyPI"
echo "=========================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install/upgrade build tools
echo "🔧 Installing build tools..."
pip install --upgrade pip build twine setuptools wheel

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/ flux_sci_lang.egg-info/ 2>/dev/null

# Build the package
echo "🏗️  Building package..."
python -m build

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Package built successfully!"
echo ""
echo "📋 Build artifacts:"
ls -la dist/
echo ""

# Check package with twine
echo "🔍 Checking package integrity..."
twine check dist/*

echo ""
echo "📤 Ready to upload to PyPI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: PyPI Upload Instructions"
echo ""
echo "You have two options:"
echo ""
echo "1. Upload to Test PyPI first (recommended for first-time):"
echo "   twine upload --repository testpypi dist/*"
echo "   Test install: pip install -i https://test.pypi.org/simple/ flux-sci-lang"
echo ""
echo "2. Upload directly to PyPI:"
echo "   twine upload dist/*"
echo ""
echo "When prompted for credentials:"
echo "   Username: __token__"
echo "   Password: [paste your PyPI token]"
echo ""
echo "To get a PyPI token:"
echo "1. Go to: https://pypi.org/manage/account/token/"
echo "2. Create a new API token"
echo "3. Copy the token (starts with 'pypi-')"
echo ""

read -p "Do you want to upload to PyPI now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Uploading to PyPI..."
    twine upload dist/*

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Successfully published to PyPI!"
        echo ""
        echo "📦 Package: https://pypi.org/project/flux-sci-lang/"
        echo ""
        echo "Install with:"
        echo "   pip install flux-sci-lang"
        echo ""
        echo "Import in Python:"
        echo "   import flux_sci"
        echo "   from flux_sci import FluxCompiler"
    else
        echo ""
        echo "❌ Upload failed. Please check your token and try again:"
        echo "   twine upload dist/*"
    fi
else
    echo ""
    echo "📝 Upload skipped. To upload later, run:"
    echo "   twine upload dist/*"
fi