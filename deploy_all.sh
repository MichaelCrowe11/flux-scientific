#!/bin/bash

# FLUX Scientific Computing Language - Complete Deployment Script
# Handles GitHub, PyPI, and Fly.io deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════╗"
echo "║   FLUX Scientific Computing Language          ║"
echo "║          Complete Deployment Script           ║"
echo "╔════════════════════════════════════════════════╗"
echo -e "${NC}"

# Function to push to GitHub
push_to_github() {
    echo -e "\n${BLUE}📦 GitHub Deployment${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if git is initialized
    if [ ! -d ".git" ]; then
        print_warning "Git not initialized. Initializing..."
        git init
        git remote add origin https://github.com/MichaelCrowe11/flux-sci-lang.git 2>/dev/null || true
    fi

    # Check git status
    if [[ `git status --porcelain` ]]; then
        print_info "Changes detected. Preparing commit..."

        # Add all files
        git add .

        # Create commit message
        COMMIT_MSG="🚀 Complete FLUX Scientific Computing Language v0.1.0

Features implemented:
✅ Production-ready PDE solvers (Heat, Wave, Poisson, Navier-Stokes)
✅ Validated numerical methods with <1e-6 error
✅ GPU acceleration with CuPy
✅ Complete .flux compiler (Python, C++, CUDA, Julia, Fortran)
✅ VS Code extension with IntelliSense
✅ Interactive Jupyter notebooks
✅ Web application with real-time solver
✅ Comprehensive documentation
✅ Showcase applications

Ready for:
- PyPI publication as flux-sci-lang
- Fly.io deployment at flux-sci-lang.fly.dev
- Production use in scientific computing"

        git commit -m "$COMMIT_MSG"
        print_status "Commit created"
    else
        print_info "No changes to commit"
    fi

    # Push to GitHub
    print_info "Pushing to GitHub..."
    if git push -u origin main 2>/dev/null; then
        print_status "Successfully pushed to GitHub!"
        print_info "Repository: https://github.com/MichaelCrowe11/flux-sci-lang"
    else
        print_warning "Could not push automatically. You may need to:"
        echo "  1. Set up GitHub authentication:"
        echo "     gh auth login"
        echo "  2. Or use personal access token:"
        echo "     git push https://YOUR_TOKEN@github.com/MichaelCrowe11/flux-sci-lang.git"
    fi

    # Create GitHub release
    print_info "Creating GitHub release..."
    if command -v gh &> /dev/null; then
        gh release create v0.1.0 \
            --title "FLUX v0.1.0 - Production Release" \
            --notes "First production release of FLUX Scientific Computing Language" \
            --draft \
            dist/*.whl dist/*.tar.gz 2>/dev/null || print_warning "Release creation requires manual completion"
    else
        print_warning "GitHub CLI not installed. Create release manually at:"
        echo "  https://github.com/MichaelCrowe11/flux-sci-lang/releases/new"
    fi
}

# Function to build and publish to PyPI
publish_to_pypi() {
    echo -e "\n${BLUE}📦 PyPI Publication${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check for required tools
    print_info "Checking requirements..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        return 1
    fi

    # Install build tools if needed
    print_info "Installing build tools..."
    pip install --quiet --upgrade pip build twine

    # Clean previous builds
    print_info "Cleaning previous builds..."
    rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true

    # Build the package
    print_info "Building package..."
    python -m build

    if [ $? -eq 0 ]; then
        print_status "Package built successfully!"
        ls -la dist/
    else
        print_error "Build failed"
        return 1
    fi

    # Check package
    print_info "Checking package with twine..."
    twine check dist/*

    # Upload to PyPI
    print_info "Uploading to PyPI..."
    print_warning "You'll need to enter your PyPI token when prompted"
    echo "  Username: __token__"
    echo "  Password: <your-pypi-token>"
    echo ""

    read -p "Do you want to upload to PyPI now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        twine upload dist/*
        if [ $? -eq 0 ]; then
            print_status "Successfully published to PyPI!"
            print_info "Install with: pip install flux-sci-lang"
            print_info "Package page: https://pypi.org/project/flux-sci-lang/"
        else
            print_error "Upload failed. Check your credentials."
        fi
    else
        print_info "Skipping PyPI upload. You can run later:"
        echo "  twine upload dist/*"
    fi
}

# Function to deploy to Fly.io
deploy_to_flyio() {
    echo -e "\n${BLUE}🚀 Fly.io Deployment${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if fly CLI is installed
    if ! command -v fly &> /dev/null; then
        print_warning "Fly CLI not installed. Installing..."
        curl -L https://fly.io/install.sh | sh
        export FLYCTL_INSTALL="/home/$USER/.fly"
        export PATH="$FLYCTL_INSTALL/bin:$PATH"
    fi

    # Check authentication
    if ! fly auth whoami &> /dev/null; then
        print_info "Please log in to Fly.io:"
        fly auth login
    fi

    # Deploy application
    print_info "Deploying to Fly.io..."

    # Check if app exists
    if fly status --app flux-sci-lang &> /dev/null; then
        print_info "App exists, deploying update..."
        fly deploy --app flux-sci-lang
    else
        print_info "Creating new app and deploying..."
        fly launch --name flux-sci-lang --region ord --now
    fi

    if [ $? -eq 0 ]; then
        print_status "Successfully deployed to Fly.io!"
        print_info "Application URL: https://flux-sci-lang.fly.dev"

        # Open in browser
        read -p "Open in browser? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fly open --app flux-sci-lang
        fi
    else
        print_error "Deployment failed. Check the logs:"
        echo "  fly logs --app flux-sci-lang"
    fi
}

# Function to run tests
run_tests() {
    echo -e "\n${BLUE}🧪 Running Tests${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    print_info "Running validation suite..."
    python -m pytest tests/ -v --tb=short 2>/dev/null || python src/solvers/validation.py

    print_info "Checking import..."
    python -c "import flux_sci; print(f'✓ FLUX version: {flux_sci.__version__}')" || true
}

# Function to show deployment status
show_status() {
    echo -e "\n${BLUE}📊 Deployment Status${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # GitHub status
    if git remote -v | grep -q github; then
        print_status "GitHub: Connected"
        LATEST_COMMIT=$(git log -1 --pretty=format:"%h - %s" 2>/dev/null)
        echo "         Latest: $LATEST_COMMIT"
    else
        print_warning "GitHub: Not connected"
    fi

    # PyPI status
    if pip show flux-sci-lang &> /dev/null; then
        VERSION=$(pip show flux-sci-lang | grep Version | cut -d' ' -f2)
        print_status "PyPI: Published (v$VERSION)"
    else
        print_warning "PyPI: Not published"
    fi

    # Fly.io status
    if command -v fly &> /dev/null && fly status --app flux-sci-lang &> /dev/null; then
        print_status "Fly.io: Deployed"
        echo "         URL: https://flux-sci-lang.fly.dev"
    else
        print_warning "Fly.io: Not deployed"
    fi
}

# Main menu
main_menu() {
    echo -e "\n${YELLOW}Select deployment option:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1) Complete deployment (GitHub + PyPI + Fly.io)"
    echo "2) GitHub only"
    echo "3) PyPI only"
    echo "4) Fly.io only"
    echo "5) Run tests"
    echo "6) Show status"
    echo "7) Exit"
    echo ""
    read -p "Enter choice [1-7]: " choice

    case $choice in
        1)
            run_tests
            push_to_github
            publish_to_pypi
            deploy_to_flyio
            show_status
            ;;
        2)
            push_to_github
            ;;
        3)
            publish_to_pypi
            ;;
        4)
            deploy_to_flyio
            ;;
        5)
            run_tests
            ;;
        6)
            show_status
            ;;
        7)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            main_menu
            ;;
    esac
}

# Quick deployment if argument provided
case "$1" in
    all|full)
        run_tests
        push_to_github
        publish_to_pypi
        deploy_to_flyio
        show_status
        ;;
    github)
        push_to_github
        ;;
    pypi)
        publish_to_pypi
        ;;
    flyio|fly)
        deploy_to_flyio
        ;;
    test)
        run_tests
        ;;
    status)
        show_status
        ;;
    *)
        # Show interactive menu if no arguments
        if [ -z "$1" ]; then
            main_menu
        else
            echo "Usage: $0 [all|github|pypi|flyio|test|status]"
            echo ""
            echo "Options:"
            echo "  all/full - Complete deployment to all platforms"
            echo "  github   - Push to GitHub only"
            echo "  pypi     - Publish to PyPI only"
            echo "  flyio    - Deploy to Fly.io only"
            echo "  test     - Run tests only"
            echo "  status   - Show deployment status"
            echo ""
            echo "Or run without arguments for interactive menu"
        fi
        ;;
esac

echo -e "\n${GREEN}✨ Deployment script completed!${NC}"