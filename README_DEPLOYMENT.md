# FLUX Scientific Computing - Deployment Guide

## 🚀 Deployment to Fly.io

FLUX is ready to be deployed as a web application on Fly.io, providing an online PDE solver interface.

### Prerequisites

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Sign up/Login to Fly.io**:
   ```bash
   fly auth login
   ```

### Quick Deployment

1. **Full deployment** (recommended for first time):
   ```bash
   ./deploy.sh full
   ```

   This will:
   - Create a persistent volume for data
   - Set up environment secrets
   - Deploy the application
   - Scale appropriately

2. **Deploy updates**:
   ```bash
   ./deploy.sh deploy
   ```

### Manual Deployment Steps

1. **Launch the app** (first time only):
   ```bash
   fly launch --name flux-sci-lang --region ord
   ```

2. **Deploy**:
   ```bash
   fly deploy
   ```

3. **Open in browser**:
   ```bash
   fly open
   ```

### Available Commands

```bash
./deploy.sh deploy   # Deploy the app
./deploy.sh scale    # Scale resources
./deploy.sh logs     # View logs
./deploy.sh status   # Check status
./deploy.sh open     # Open in browser
```

### Configuration

The deployment is configured in:
- `fly.toml` - Fly.io configuration
- `Dockerfile.web` - Docker image for web app
- `web_app.py` - Flask application

### Features Deployed

The web application includes:

1. **Code Editor**
   - FLUX syntax highlighting
   - Example programs
   - Multi-backend compilation (Python, C++, CUDA, Julia, Fortran)

2. **Interactive PDE Solver**
   - Heat equation
   - Wave equation
   - Poisson equation
   - Real-time visualization

3. **API Endpoints**
   - `/api/compile` - Compile FLUX code
   - `/api/solve` - Solve PDEs
   - `/api/validate` - Validate syntax
   - `/api/examples` - Get example programs

### Monitoring

1. **View logs**:
   ```bash
   fly logs --app flux-sci-lang
   ```

2. **Check metrics**:
   ```bash
   fly dashboard --app flux-sci-lang
   ```

3. **SSH into container**:
   ```bash
   fly ssh console --app flux-sci-lang
   ```

### Scaling

Adjust resources as needed:

```bash
# Scale VM size
fly scale vm dedicated-cpu-1x --memory 2048

# Scale instance count
fly scale count 2 --app flux-sci-lang
```

### Custom Domain (Optional)

1. Add custom domain:
   ```bash
   fly certs add yourdomain.com --app flux-sci-lang
   ```

2. Configure DNS:
   - Add CNAME record pointing to `flux-sci-lang.fly.dev`

### Troubleshooting

1. **Deployment fails**:
   - Check logs: `fly logs --app flux-sci-lang`
   - Verify Docker build: `docker build -f Dockerfile.web .`

2. **App crashes**:
   - Check health endpoint: `https://flux-sci-lang.fly.dev/health`
   - Review memory usage: `fly scale show`

3. **Performance issues**:
   - Scale up: `fly scale vm dedicated-cpu-1x --memory 2048`
   - Add regions: `fly regions add sin ord`

### Environment Variables

Set any required secrets:

```bash
fly secrets set API_KEY=your_key --app flux-sci-lang
fly secrets set DATABASE_URL=your_db_url --app flux-sci-lang
```

### Backup and Recovery

1. **Export volume data**:
   ```bash
   fly ssh console -C "tar -czf /data/backup.tar.gz /data" --app flux-sci-lang
   fly ssh sftp get /data/backup.tar.gz --app flux-sci-lang
   ```

2. **Restore data**:
   ```bash
   fly ssh sftp put backup.tar.gz /data/backup.tar.gz --app flux-sci-lang
   fly ssh console -C "tar -xzf /data/backup.tar.gz -C /" --app flux-sci-lang
   ```

## 📦 Publishing to PyPI

1. **Build the package**:
   ```bash
   python -m build
   ```

2. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

   When prompted, use your PyPI token.

3. **Install from PyPI**:
   ```bash
   pip install flux-sci-lang
   ```

## 🐙 GitHub Deployment

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Complete FLUX implementation with web deployment"
   git push origin main
   ```

2. **Create Release**:
   - Go to GitHub releases
   - Create new release with tag `v0.1.0`
   - Attach built wheels from `dist/`

3. **GitHub Actions** (optional):
   - Set up CI/CD for automatic deployment
   - Add secrets for Fly.io and PyPI tokens

## 🌐 Live Demo

Once deployed, your FLUX application will be available at:

**https://flux-sci-lang.fly.dev**

Features available online:
- FLUX code editor with syntax highlighting
- Interactive PDE solver
- Real-time visualization
- Example programs
- Multi-backend compilation

## 📊 Performance Considerations

For production deployment:

1. **Caching**: Enable Redis for computation caching
2. **CDN**: Use Cloudflare for static assets
3. **Database**: Add PostgreSQL for result storage
4. **Monitoring**: Set up Datadog or New Relic

## 🔐 Security

1. **Rate limiting**: Implement to prevent abuse
2. **Input validation**: Sanitize all user inputs
3. **Resource limits**: Cap computation time and memory
4. **HTTPS**: Always enforced by Fly.io

## 📝 License

MIT License - See LICENSE file for details

---

**Need help?** Open an issue on [GitHub](https://github.com/MichaelCrowe11/flux-sci-lang/issues)