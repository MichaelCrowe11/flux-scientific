#!/bin/bash

# GitHub Push Script for FLUX Scientific Computing Language

echo "🚀 Pushing FLUX to GitHub"
echo "========================="

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    git branch -M main
fi

# Add remote if not exists
if ! git remote | grep -q origin; then
    echo "🔗 Adding GitHub remote..."
    git remote add origin https://github.com/MichaelCrowe11/flux-sci-lang.git
fi

# Stage all files
echo "📦 Staging files..."
git add .

# Create comprehensive commit
echo "💾 Creating commit..."
git commit -m "🚀 Complete FLUX Scientific Computing Language v0.1.0

## 🎯 Production-Ready Features

### Core Functionality
- ✅ Finite difference PDE solvers with validated accuracy (<1e-6 error)
- ✅ Heat, Wave, Poisson, and Navier-Stokes equation solvers
- ✅ Multiple numerical methods (Explicit, Implicit, Crank-Nicolson, ADI)
- ✅ Stability analysis and CFL condition checking

### Advanced Features
- 🚀 GPU acceleration with CuPy
- 📝 Complete .flux compiler with lexer, parser, and AST
- 🎯 Multi-backend code generation (Python, C++, CUDA, Julia, Fortran)
- 🌐 Web application with interactive PDE solver
- 📚 Comprehensive documentation and examples

### Developer Tools
- 💻 VS Code extension with syntax highlighting and IntelliSense
- 📓 Interactive Jupyter notebooks
- 🧪 Validation suite with analytical solutions
- 📊 Performance benchmarking tools

### Deployment Ready
- 📦 PyPI package: flux-sci-lang
- 🚀 Fly.io deployment configuration
- 🐳 Docker containerization
- 📖 ReadTheDocs documentation

## Installation
pip install flux-sci-lang

## Web Demo
https://flux-sci-lang.fly.dev

## Documentation
https://flux-sci-lang.readthedocs.io" || echo "No changes to commit"

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
echo ""
echo "⚠️  IMPORTANT: You'll need to authenticate with GitHub"
echo ""
echo "Options for authentication:"
echo "1. Use GitHub CLI (recommended):"
echo "   gh auth login"
echo ""
echo "2. Use personal access token:"
echo "   - Go to: https://github.com/settings/tokens"
echo "   - Generate new token with 'repo' scope"
echo "   - Use token as password when prompted"
echo ""
echo "3. Use SSH key (if configured):"
echo "   git remote set-url origin git@github.com:MichaelCrowe11/flux-sci-lang.git"
echo ""

# Attempt push
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Repository: https://github.com/MichaelCrowe11/flux-sci-lang"
else
    echo ""
    echo "❌ Push failed. Please authenticate and try:"
    echo "   git push -u origin main"
fi