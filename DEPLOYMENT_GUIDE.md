# 🚀 FLUX Scientific Computing Language - Complete Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying FLUX to GitHub, PyPI, and Fly.io.

---

## 📋 Pre-Deployment Checklist

- [x] All code implemented and tested
- [x] Documentation complete
- [x] VS Code extension created
- [x] Web application ready
- [x] Docker configuration complete
- [x] Deployment scripts prepared

---

## 1️⃣ GitHub Deployment

### Quick Method
```bash
./github_push.sh
```

### Manual Method

1. **Initialize Git** (if needed):
```bash
git init
git branch -M main
git remote add origin https://github.com/MichaelCrowe11/flux-sci-lang.git
```

2. **Commit Changes**:
```bash
git add .
git commit -m "Complete FLUX v0.1.0 - Production Release"
```

3. **Push to GitHub**:

**Option A: Using GitHub CLI (Recommended)**
```bash
# Install GitHub CLI if needed
brew install gh  # macOS
# or
winget install --id GitHub.cli  # Windows

# Authenticate
gh auth login

# Push
git push -u origin main
```

**Option B: Using Personal Access Token**
```bash
# Create token at: https://github.com/settings/tokens
# Select 'repo' scope

# Push with token
git push https://<YOUR_TOKEN>@github.com/MichaelCrowe11/flux-sci-lang.git main
```

**Option C: Using SSH**
```bash
# Setup SSH key if needed
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add key to GitHub: https://github.com/settings/keys

# Change remote to SSH
git remote set-url origin git@github.com:MichaelCrowe11/flux-sci-lang.git

# Push
git push -u origin main
```

4. **Create Release**:
```bash
gh release create v0.1.0 \
  --title "FLUX v0.1.0 - Production Release" \
  --notes "First production release" \
  dist/*.whl dist/*.tar.gz
```

---

## 2️⃣ PyPI Publication

### Quick Method
```bash
./pypi_publish.sh
```

### Manual Method

1. **Install Build Tools**:
```bash
pip install --upgrade pip build twine
```

2. **Build Package**:
```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info/

# Build
python -m build
```

3. **Check Package**:
```bash
twine check dist/*
```

4. **Get PyPI Token**:
   - Go to: https://pypi.org/manage/account/token/
   - Create new API token
   - Copy token (starts with `pypi-`)

5. **Upload to PyPI**:
```bash
# Test PyPI first (optional)
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: `<paste your token>`

6. **Verify Installation**:
```bash
pip install flux-sci-lang
python -c "import flux_sci; print(flux_sci.__version__)"
```

---

## 3️⃣ Fly.io Deployment

### Quick Method
```bash
./deploy.sh full
```

### Manual Method

1. **Install Fly CLI**:
```bash
# macOS
brew install flyctl

# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Linux
curl -L https://fly.io/install.sh | sh
```

2. **Login to Fly.io**:
```bash
fly auth login
```

3. **Launch App** (first time only):
```bash
fly launch --name flux-sci-lang --region ord
```

4. **Deploy**:
```bash
fly deploy
```

5. **Verify Deployment**:
```bash
fly status --app flux-sci-lang
fly open --app flux-sci-lang
```

6. **Monitor**:
```bash
fly logs --app flux-sci-lang
```

---

## 4️⃣ Complete Deployment (All Platforms)

### Automated Deployment
```bash
# Interactive menu
./deploy_all.sh

# Or direct deployment
./deploy_all.sh all
```

This will:
1. Run tests
2. Push to GitHub
3. Publish to PyPI
4. Deploy to Fly.io

---

## 📝 Post-Deployment Tasks

### Verify GitHub
- Check repository: https://github.com/MichaelCrowe11/flux-sci-lang
- Verify README displays correctly
- Check Actions/CI if configured

### Verify PyPI
- Check package: https://pypi.org/project/flux-sci-lang/
- Test installation: `pip install flux-sci-lang`
- Verify metadata and description

### Verify Fly.io
- Check app: https://flux-sci-lang.fly.dev
- Test PDE solver interface
- Monitor performance: `fly dashboard`

### Documentation
- ReadTheDocs will auto-build from `.readthedocs.yaml`
- Check: https://flux-sci-lang.readthedocs.io

### VS Code Extension
```bash
cd vscode-extension
npm install
vsce package

# Publish to marketplace
vsce publish
```

---

## 🔧 Troubleshooting

### GitHub Issues

**Authentication Failed**
```bash
# Use GitHub CLI
gh auth login

# Or create personal access token
# https://github.com/settings/tokens
```

**Push Rejected**
```bash
# Pull first if remote has changes
git pull origin main --rebase
git push origin main
```

### PyPI Issues

**Token Invalid**
- Ensure token starts with `pypi-`
- Check token scope (entire account or project-specific)
- Use `__token__` as username

**Package Name Taken**
- Package name must be unique
- Try `flux-scientific` or `fluxlang`

### Fly.io Issues

**Deployment Failed**
```bash
# Check logs
fly logs --app flux-sci-lang

# Check Docker build locally
docker build -f Dockerfile.web .
```

**Out of Memory**
```bash
# Scale up
fly scale vm dedicated-cpu-1x --memory 2048
```

---

## 🔐 Security Notes

### Tokens and Secrets
- Never commit tokens to git
- Use environment variables for secrets
- Store tokens securely (password manager)

### Fly.io Secrets
```bash
fly secrets set SECRET_KEY=your_secret --app flux-sci-lang
```

### PyPI 2FA
- Enable two-factor authentication on PyPI
- Use API tokens instead of password

---

## 📊 Monitoring

### GitHub
- Enable GitHub Insights
- Set up GitHub Actions for CI/CD
- Monitor issues and pull requests

### PyPI
- Check download statistics
- Monitor user issues
- Update regularly

### Fly.io
```bash
# Metrics
fly dashboard --app flux-sci-lang

# Logs
fly logs --app flux-sci-lang --tail

# SSH into container
fly ssh console --app flux-sci-lang
```

---

## 🔄 Updates and Maintenance

### Version Updates
1. Update version in `setup.py`
2. Update `__version__` in `__init__.py`
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. Build and publish new version

### Continuous Deployment
Create `.github/workflows/deploy.yml` for automated deployment on push to main.

---

## 📧 Support

- GitHub Issues: https://github.com/MichaelCrowe11/flux-sci-lang/issues
- Email: your-email@example.com
- Documentation: https://flux-sci-lang.readthedocs.io

---

## 🎉 Congratulations!

Your FLUX Scientific Computing Language is now:
- ✅ On GitHub for collaboration
- ✅ On PyPI for installation
- ✅ On Fly.io for web access
- ✅ Ready for production use!

**URLs:**
- GitHub: https://github.com/MichaelCrowe11/flux-sci-lang
- PyPI: https://pypi.org/project/flux-sci-lang/
- Web App: https://flux-sci-lang.fly.dev
- Docs: https://flux-sci-lang.readthedocs.io