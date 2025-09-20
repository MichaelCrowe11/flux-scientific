#!/usr/bin/env python3
"""
FLUX Scientific Computing Language - Full-Featured Web Application
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
import json
import sys
import os
from datetime import datetime
import traceback

# Add src to path for FLUX modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import FLUX components (with fallbacks)
try:
    from src.pde_lexer import FluxPDELexer
    from src.pde_parser import FluxPDEParser
    from src.codegen import BackendManager
    COMPILER_AVAILABLE = True
except:
    COMPILER_AVAILABLE = False

# Simple finite difference solver for web demo
class SimplePDESolver:
    """Simplified PDE solver for web demonstrations"""

    def solve_heat(self, nx=50, ny=50, nt=100, alpha=0.1, initial='gaussian'):
        """Solve 2D heat equation"""
        dx = dy = 1.0 / (nx - 1)
        dt = 0.0001

        # Create grid
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Initial condition
        if initial == 'gaussian':
            u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        elif initial == 'multiple':
            u = (np.exp(-100 * ((X - 0.3)**2 + (Y - 0.3)**2)) +
                 np.exp(-100 * ((X - 0.7)**2 + (Y - 0.7)**2)))
        elif initial == 'sinusoidal':
            u = np.sin(np.pi * X) * np.sin(np.pi * Y)
        else:
            u = np.zeros((ny, nx))
            u[ny//4:3*ny//4, nx//4:3*nx//4] = 1.0

        u_initial = u.copy()

        # Time evolution
        for n in range(nt):
            un = u.copy()
            u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                             alpha * dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) +
                             alpha * dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))

            # Boundary conditions (Dirichlet = 0)
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0

        return X, Y, u_initial, u

    def solve_wave(self, nx=50, ny=50, nt=200, c=0.5, initial='gaussian'):
        """Solve 2D wave equation"""
        dx = dy = 1.0 / (nx - 1)
        dt = 0.001

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Initial conditions
        if initial == 'gaussian':
            u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        else:
            u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

        u_initial = u.copy()
        u_prev = u.copy()

        # Time evolution
        for n in range(nt):
            u_new = np.zeros_like(u)
            u_new[1:-1, 1:-1] = (2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                                 (c*dt/dx)**2 * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) +
                                 (c*dt/dy)**2 * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]))

            u_prev = u.copy()
            u = u_new

            # Boundary conditions
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0

        return X, Y, u_initial, u

    def solve_poisson(self, nx=50, ny=50, source='point'):
        """Solve 2D Poisson equation using Jacobi iteration"""
        dx = dy = 1.0 / (nx - 1)

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Source term
        if source == 'point':
            f = np.zeros((ny, nx))
            f[ny//2, nx//2] = 100.0 / (dx * dy)
        elif source == 'gaussian':
            f = 10 * np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        else:
            f = np.ones((ny, nx))

        # Initial guess
        u = np.zeros((ny, nx))

        # Jacobi iteration
        for iteration in range(1000):
            un = u.copy()
            u[1:-1, 1:-1] = 0.25 * (un[1:-1, 2:] + un[1:-1, :-2] +
                                    un[2:, 1:-1] + un[:-2, 1:-1] -
                                    dx**2 * f[1:-1, 1:-1])

        return X, Y, f, u

app = Flask(__name__)
solver = SimplePDESolver()

# Professional HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FLUX Scientific Computing - Interactive PDE Solver</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
        }

        /* Header */
        .header {
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo h1 {
            font-size: 28px;
            color: #1e3c72;
            font-weight: 700;
        }

        .logo span {
            font-size: 32px;
        }

        .nav-links {
            display: flex;
            gap: 30px;
        }

        .nav-links a {
            color: #2a5298;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }

        .nav-links a:hover {
            color: #1e3c72;
        }

        /* Main Content */
        .container {
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
        }

        /* Hero Section */
        .hero {
            background: white;
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        }

        .hero h2 {
            font-size: 48px;
            color: #1e3c72;
            margin-bottom: 20px;
            font-weight: 700;
        }

        .hero p {
            font-size: 20px;
            color: #666;
            margin-bottom: 40px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        .cta-buttons {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .cta-button {
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s;
            cursor: pointer;
            border: none;
        }

        .cta-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .cta-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        .cta-secondary {
            background: white;
            color: #1e3c72;
            border: 2px solid #1e3c72;
        }

        .cta-secondary:hover {
            background: #1e3c72;
            color: white;
        }

        /* PDE Solver Section */
        .solver-section {
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        }

        .solver-section h3 {
            font-size: 32px;
            color: #1e3c72;
            margin-bottom: 30px;
            text-align: center;
        }

        .solver-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
        }

        .control-group label {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .control-group select,
        .control-group input {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }

        .control-group select:focus,
        .control-group input:focus {
            outline: none;
            border-color: #667eea;
        }

        .slider-value {
            text-align: right;
            color: #667eea;
            font-weight: 600;
            margin-top: 5px;
        }

        .solve-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 50px;
            font-size: 20px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            margin: 30px auto;
            display: block;
        }

        .solve-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }

        .solve-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        /* Results Section */
        .results {
            margin-top: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
        }

        .results h4 {
            color: #1e3c72;
            margin-bottom: 20px;
            font-size: 24px;
        }

        .plot-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .plot-container img {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .stat-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }

        .stat-value {
            color: #1e3c72;
            font-size: 24px;
            font-weight: 700;
        }

        /* Features Grid */
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .feature-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }

        .feature-icon {
            font-size: 48px;
            margin-bottom: 20px;
        }

        .feature-title {
            font-size: 24px;
            color: #1e3c72;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .feature-description {
            color: #666;
            line-height: 1.6;
        }

        /* Code Example */
        .code-section {
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        }

        .code-section h3 {
            font-size: 32px;
            color: #1e3c72;
            margin-bottom: 30px;
        }

        .code-example {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 25px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            line-height: 1.6;
        }

        .code-example .keyword {
            color: #66d9ef;
        }

        .code-example .string {
            color: #a6e22e;
        }

        .code-example .number {
            color: #ae81ff;
        }

        .code-example .comment {
            color: #75715e;
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Footer */
        .footer {
            background: rgba(255, 255, 255, 0.98);
            padding: 40px;
            margin-top: 60px;
            text-align: center;
        }

        .footer-links {
            display: flex;
            gap: 30px;
            justify-content: center;
            margin-bottom: 20px;
        }

        .footer-links a {
            color: #1e3c72;
            text-decoration: none;
            font-weight: 500;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero {
                padding: 40px 20px;
            }

            .hero h2 {
                font-size: 32px;
            }

            .plot-container {
                grid-template-columns: 1fr;
            }

            .nav-links {
                display: none;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <div class="logo">
                <span>🔥</span>
                <h1>FLUX</h1>
            </div>
            <nav class="nav-links">
                <a href="#solver">Solver</a>
                <a href="#features">Features</a>
                <a href="#code">Code</a>
                <a href="https://github.com/MichaelCrowe11/flux-sci-lang">GitHub</a>
                <a href="https://pypi.org/project/flux-sci-lang/">PyPI</a>
            </nav>
        </div>
    </header>

    <div class="container">
        <!-- Hero Section -->
        <section class="hero">
            <h2>Scientific Computing Made Simple</h2>
            <p>
                FLUX is a domain-specific language designed for solving partial differential equations
                with validated numerical methods and GPU acceleration.
            </p>
            <div class="cta-buttons">
                <button class="cta-button cta-primary" onclick="scrollToSolver()">Try PDE Solver</button>
                <a href="https://pypi.org/project/flux-sci-lang/" class="cta-button cta-secondary">Install Package</a>
            </div>
        </section>

        <!-- Interactive PDE Solver -->
        <section id="solver" class="solver-section">
            <h3>Interactive PDE Solver</h3>

            <div class="solver-controls">
                <div class="control-group">
                    <label>PDE Type</label>
                    <select id="pde-type">
                        <option value="heat">Heat Equation</option>
                        <option value="wave">Wave Equation</option>
                        <option value="poisson">Poisson Equation</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>Initial Condition</label>
                    <select id="initial">
                        <option value="gaussian">Gaussian</option>
                        <option value="multiple">Multiple Sources</option>
                        <option value="sinusoidal">Sinusoidal</option>
                        <option value="square">Square</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>Grid Resolution</label>
                    <input type="range" id="resolution" min="20" max="100" value="50">
                    <div class="slider-value">50 × 50</div>
                </div>

                <div class="control-group">
                    <label>Time Steps</label>
                    <input type="range" id="timesteps" min="50" max="500" value="100">
                    <div class="slider-value">100</div>
                </div>

                <div class="control-group">
                    <label>Diffusivity/Speed</label>
                    <input type="range" id="parameter" min="0.05" max="1.0" step="0.05" value="0.1">
                    <div class="slider-value">0.1</div>
                </div>

                <div class="control-group">
                    <label>Colormap</label>
                    <select id="colormap">
                        <option value="viridis">Viridis</option>
                        <option value="hot">Hot</option>
                        <option value="coolwarm">Cool-Warm</option>
                        <option value="plasma">Plasma</option>
                    </select>
                </div>
            </div>

            <button class="solve-button" onclick="solvePDE()">Solve PDE</button>

            <div class="spinner" id="spinner"></div>

            <div id="results" class="results" style="display: none;">
                <h4>Solution Visualization</h4>
                <div class="plot-container">
                    <div>
                        <img id="plot-initial" src="" alt="Initial Condition">
                        <p style="text-align: center; margin-top: 10px; color: #666;">Initial Condition</p>
                    </div>
                    <div>
                        <img id="plot-final" src="" alt="Final Solution">
                        <p style="text-align: center; margin-top: 10px; color: #666;">Final Solution</p>
                    </div>
                </div>

                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-label">Max Value</div>
                        <div class="stat-value" id="max-value">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Min Value</div>
                        <div class="stat-value" id="min-value">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Grid Points</div>
                        <div class="stat-value" id="grid-points">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Time Steps</div>
                        <div class="stat-value" id="time-steps">-</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Features -->
        <section id="features" class="features">
            <div class="feature-card">
                <div class="feature-icon">🧮</div>
                <div class="feature-title">Validated Solvers</div>
                <div class="feature-description">
                    Production-ready PDE solvers with numerical validation against analytical solutions.
                    Accuracy verified to &lt;1e-6 error.
                </div>
            </div>

            <div class="feature-card">
                <div class="feature-icon">🚀</div>
                <div class="feature-title">GPU Acceleration</div>
                <div class="feature-description">
                    Leverage NVIDIA GPUs with CuPy for massive performance improvements on large-scale problems.
                </div>
            </div>

            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <div class="feature-title">Multi-Backend Compiler</div>
                <div class="feature-description">
                    Compile FLUX code to Python, C++, CUDA, Julia, or Fortran for maximum flexibility.
                </div>
            </div>
        </section>

        <!-- Code Example -->
        <section id="code" class="code-section">
            <h3>FLUX Language Example</h3>
            <pre class="code-example"><span class="comment">// Solve the heat equation with FLUX</span>
<span class="keyword">domain</span> heat_domain {
    rectangle(<span class="number">0</span>, <span class="number">1</span>, <span class="number">0</span>, <span class="number">1</span>)
    grid(<span class="number">100</span>, <span class="number">100</span>)
}

<span class="keyword">equation</span> heat_eq {
    dt(temperature) = <span class="number">0.1</span> * laplacian(temperature)
}

<span class="keyword">boundary</span> heat_bc {
    dirichlet(<span class="number">0.0</span>) on all
}

<span class="keyword">solver</span> heat_solver {
    method: crank_nicolson
    timestep: <span class="number">0.001</span>
    max_time: <span class="number">1.0</span>
}</pre>
        </section>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="footer-links">
            <a href="https://github.com/MichaelCrowe11/flux-sci-lang">GitHub</a>
            <a href="https://pypi.org/project/flux-sci-lang/">PyPI Package</a>
            <a href="https://flux-sci-lang.readthedocs.io">Documentation</a>
            <a href="/api/info">API</a>
        </div>
        <p style="color: #666;">© 2024 FLUX Scientific Computing Language</p>
    </footer>

    <script>
        // Update slider values
        document.getElementById('resolution').addEventListener('input', function(e) {
            e.target.nextElementSibling.textContent = e.target.value + ' × ' + e.target.value;
        });

        document.getElementById('timesteps').addEventListener('input', function(e) {
            e.target.nextElementSibling.textContent = e.target.value;
        });

        document.getElementById('parameter').addEventListener('input', function(e) {
            e.target.nextElementSibling.textContent = e.target.value;
        });

        function scrollToSolver() {
            document.getElementById('solver').scrollIntoView({ behavior: 'smooth' });
        }

        async function solvePDE() {
            const button = document.querySelector('.solve-button');
            const spinner = document.getElementById('spinner');
            const results = document.getElementById('results');

            // Get parameters
            const params = {
                pde_type: document.getElementById('pde-type').value,
                initial: document.getElementById('initial').value,
                resolution: parseInt(document.getElementById('resolution').value),
                timesteps: parseInt(document.getElementById('timesteps').value),
                parameter: parseFloat(document.getElementById('parameter').value),
                colormap: document.getElementById('colormap').value
            };

            // Show loading
            button.disabled = true;
            spinner.style.display = 'block';
            results.style.display = 'none';

            try {
                const response = await fetch('/api/solve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });

                const data = await response.json();

                if (data.success) {
                    // Display results
                    document.getElementById('plot-initial').src = 'data:image/png;base64,' + data.plot_initial;
                    document.getElementById('plot-final').src = 'data:image/png;base64,' + data.plot_final;
                    document.getElementById('max-value').textContent = data.stats.max_value.toFixed(4);
                    document.getElementById('min-value').textContent = data.stats.min_value.toFixed(4);
                    document.getElementById('grid-points').textContent = data.stats.grid_points.toLocaleString();
                    document.getElementById('time-steps').textContent = params.timesteps;

                    results.style.display = 'block';
                    results.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    alert('Error solving PDE: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Failed to connect to solver: ' + error);
            } finally {
                button.disabled = false;
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/solve', methods=['POST'])
def api_solve():
    """Solve PDE with given parameters"""
    try:
        data = request.json
        pde_type = data.get('pde_type', 'heat')
        initial = data.get('initial', 'gaussian')
        resolution = int(data.get('resolution', 50))
        timesteps = int(data.get('timesteps', 100))
        parameter = float(data.get('parameter', 0.1))
        colormap = data.get('colormap', 'viridis')

        # Solve PDE
        if pde_type == 'heat':
            X, Y, u_initial, u_final = solver.solve_heat(
                nx=resolution, ny=resolution, nt=timesteps,
                alpha=parameter, initial=initial
            )
        elif pde_type == 'wave':
            X, Y, u_initial, u_final = solver.solve_wave(
                nx=resolution, ny=resolution, nt=timesteps,
                c=parameter, initial=initial
            )
        else:  # poisson
            X, Y, u_initial, u_final = solver.solve_poisson(
                nx=resolution, ny=resolution, source=initial
            )

        # Create plots
        fig = Figure(figsize=(6, 5))

        # Initial condition plot
        ax1 = fig.add_subplot(111)
        im1 = ax1.contourf(X, Y, u_initial, levels=20, cmap=colormap)
        ax1.set_title('Initial Condition')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        fig.colorbar(im1, ax=ax1)

        # Save initial plot
        buf1 = io.BytesIO()
        fig.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0)
        plot_initial_b64 = base64.b64encode(buf1.getvalue()).decode()
        plt.close(fig)

        # Final solution plot
        fig = Figure(figsize=(6, 5))
        ax2 = fig.add_subplot(111)
        im2 = ax2.contourf(X, Y, u_final, levels=20, cmap=colormap)
        ax2.set_title(f'{pde_type.title()} Equation Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        fig.colorbar(im2, ax=ax2)

        # Save final plot
        buf2 = io.BytesIO()
        fig.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        plot_final_b64 = base64.b64encode(buf2.getvalue()).decode()
        plt.close(fig)

        # Statistics
        stats = {
            'max_value': float(np.max(u_final)),
            'min_value': float(np.min(u_final)),
            'mean_value': float(np.mean(u_final)),
            'grid_points': resolution * resolution
        }

        return jsonify({
            'success': True,
            'plot_initial': plot_initial_b64,
            'plot_final': plot_final_b64,
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/api/compile', methods=['POST'])
def api_compile():
    """Compile FLUX code"""
    if not COMPILER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Compiler not available in web version'
        }), 501

    try:
        data = request.json
        flux_code = data.get('code', '')
        backend = data.get('backend', 'python')

        lexer = FluxPDELexer(flux_code)
        tokens = lexer.tokenize()
        parser = FluxPDEParser(tokens)
        ast = parser.parse()
        backend_mgr = BackendManager()
        generated = backend_mgr.generate_code(backend, ast)

        return jsonify({
            'success': True,
            'generated_code': generated
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'FLUX Scientific Computing',
        'version': '0.1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/info')
def api_info():
    """API information"""
    return jsonify({
        'name': 'FLUX Scientific Computing Language',
        'version': '0.1.0',
        'features': [
            'Interactive PDE Solver',
            'Heat Equation',
            'Wave Equation',
            'Poisson Equation',
            'GPU Acceleration (in package)',
            'Multi-backend Compilation'
        ],
        'links': {
            'github': 'https://github.com/MichaelCrowe11/flux-sci-lang',
            'pypi': 'https://pypi.org/project/flux-sci-lang/',
            'documentation': 'https://flux-sci-lang.readthedocs.io'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)