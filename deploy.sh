#!/bin/bash

# FLUX Scientific Computing - Deployment Script for Fly.io

echo "🚀 FLUX Deployment to Fly.io"
echo "============================"

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "❌ Fly CLI not found. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Function to deploy
deploy() {
    echo "📦 Preparing deployment..."

    # Check if app exists
    if fly status --app flux-sci-lang 2>/dev/null; then
        echo "✓ App 'flux-sci-lang' exists"
    else
        echo "📱 Creating new Fly app..."
        fly launch --name flux-sci-lang --region ord --no-deploy
    fi

    # Deploy the app
    echo "🚀 Deploying to Fly.io..."
    fly deploy --app flux-sci-lang

    if [ $? -eq 0 ]; then
        echo "✅ Deployment successful!"
        echo ""
        echo "🌐 Your app is available at:"
        echo "   https://flux-sci-lang.fly.dev"
        echo ""
        echo "📊 View logs with:"
        echo "   fly logs --app flux-sci-lang"
        echo ""
        echo "📈 Check status with:"
        echo "   fly status --app flux-sci-lang"
    else
        echo "❌ Deployment failed. Check the error messages above."
        exit 1
    fi
}

# Function to scale app
scale() {
    echo "⚖️ Scaling app..."
    fly scale vm shared-cpu-1x --memory 512 --app flux-sci-lang
    echo "✓ Scaled to shared-cpu-1x with 512MB memory"
}

# Function to set secrets
set_secrets() {
    echo "🔐 Setting environment secrets..."
    # Add any API keys or secrets here
    # fly secrets set API_KEY=your_key --app flux-sci-lang
    echo "✓ Secrets configured"
}

# Function to setup volume (for persistent storage)
setup_volume() {
    echo "💾 Setting up persistent volume..."

    # Check if volume exists
    if fly volumes list --app flux-sci-lang | grep -q flux_data; then
        echo "✓ Volume 'flux_data' already exists"
    else
        echo "Creating new volume..."
        fly volumes create flux_data --size 1 --region ord --app flux-sci-lang
    fi
}

# Main menu
case "$1" in
    deploy)
        deploy
        ;;
    scale)
        scale
        ;;
    secrets)
        set_secrets
        ;;
    volume)
        setup_volume
        ;;
    full)
        setup_volume
        set_secrets
        deploy
        scale
        ;;
    logs)
        fly logs --app flux-sci-lang
        ;;
    status)
        fly status --app flux-sci-lang
        ;;
    open)
        fly open --app flux-sci-lang
        ;;
    *)
        echo "Usage: $0 {deploy|scale|secrets|volume|full|logs|status|open}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the app to Fly.io"
        echo "  scale   - Scale the app resources"
        echo "  secrets - Set environment secrets"
        echo "  volume  - Setup persistent storage"
        echo "  full    - Run full deployment (volume + secrets + deploy + scale)"
        echo "  logs    - View application logs"
        echo "  status  - Check app status"
        echo "  open    - Open app in browser"
        exit 1
        ;;
esac