#!/usr/bin/env python3
"""
FLUX Scientific Computing Language - Web Application
====================================================

Web interface for FLUX PDE solving, deployed on Fly.io
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import json
import sys
import os
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pde_lexer import FluxPDELexer
from src.pde_parser import FluxPDEParser
from src.codegen import BackendManager
from src.solvers.finite_difference import FiniteDifferenceSolver
from src.flux_compiler import FluxCompiler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize solvers
solver = FiniteDifferenceSolver()
compiler = FluxCompiler()

@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/api/compile', methods=['POST'])
def compile_flux():
    """Compile FLUX code to target backend"""
    try:
        data = request.json
        flux_code = data.get('code', '')
        backend = data.get('backend', 'python')

        # Tokenize
        lexer = FluxPDELexer(flux_code)
        tokens = lexer.tokenize()

        # Parse
        parser = FluxPDEParser(tokens)
        ast_nodes = parser.parse()

        # Generate code
        backend_mgr = BackendManager()
        generated_code = backend_mgr.generate_code(backend, ast_nodes)

        return jsonify({
            'success': True,
            'generated_code': generated_code,
            'token_count': len(tokens),
            'ast_count': len(ast_nodes),
            'backend': backend
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/api/solve', methods=['POST'])
def solve_pde():
    """Solve PDE with given parameters"""
    try:
        data = request.json
        pde_type = data.get('pde_type', 'heat')
        nx = int(data.get('nx', 50))
        ny = int(data.get('ny', 50))

        # Domain
        domain = {
            'x_min': 0.0, 'x_max': 1.0,
            'y_min': 0.0, 'y_max': 1.0
        }

        # Grid
        x = np.linspace(domain['x_min'], domain['x_max'], nx)
        y = np.linspace(domain['y_min'], domain['y_max'], ny)
        X, Y = np.meshgrid(x, y)

        # Initial condition
        initial_type = data.get('initial_condition', 'gaussian')
        if initial_type == 'gaussian':
            initial = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        elif initial_type == 'sinusoidal':
            initial = np.sin(np.pi * X) * np.sin(np.pi * Y)
        else:
            initial = np.zeros_like(X)

        # Boundary conditions
        bc_value = float(data.get('boundary_value', 0.0))
        boundary_conditions = {
            'left': ('dirichlet', bc_value),
            'right': ('dirichlet', bc_value),
            'top': ('dirichlet', bc_value),
            'bottom': ('dirichlet', bc_value)
        }

        # Solve based on type
        if pde_type == 'heat':
            alpha = float(data.get('alpha', 0.1))
            time_final = float(data.get('time_final', 1.0))
            dt = float(data.get('dt', 0.01))
            method = data.get('method', 'crank_nicolson')

            result = solver.solve_heat_equation(
                domain=domain,
                grid_points=(nx, ny),
                initial_condition=lambda x, y: initial,
                boundary_conditions=boundary_conditions,
                thermal_diffusivity=alpha,
                time_final=time_final,
                dt=dt,
                method=method
            )

        elif pde_type == 'wave':
            wave_speed = float(data.get('wave_speed', 0.5))
            time_final = float(data.get('time_final', 2.0))
            dt = float(data.get('dt', 0.01))

            result = solver.solve_wave_equation(
                domain=domain,
                grid_points=(nx, ny),
                initial_condition=lambda x, y: initial,
                initial_velocity=lambda x, y: np.zeros_like(initial),
                boundary_conditions=boundary_conditions,
                wave_speed=wave_speed,
                time_final=time_final,
                dt=dt
            )

        elif pde_type == 'poisson':
            # Source term
            source_strength = float(data.get('source_strength', 1.0))
            source = lambda x, y: source_strength * np.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))

            result = solver.solve_poisson_equation(
                domain=domain,
                grid_points=(nx, ny),
                source_term=source,
                boundary_conditions=boundary_conditions
            )
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Initial condition
        im1 = ax1.contourf(X, Y, initial, levels=20, cmap='viridis')
        ax1.set_title('Initial Condition')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)

        # Solution
        im2 = ax2.contourf(X, Y, result['u'], levels=20, cmap='hot')
        ax2.set_title(f'{pde_type.title()} Equation Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)

        # Convert plot to base64
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        # Statistics
        stats = {
            'max_value': float(np.max(result['u'])),
            'min_value': float(np.min(result['u'])),
            'mean_value': float(np.mean(result['u'])),
            'convergence': result.get('iterations', 0)
        }

        return jsonify({
            'success': True,
            'plot': img_base64,
            'stats': stats,
            'grid_size': f"{nx}×{ny}",
            'pde_type': pde_type
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/api/validate', methods=['POST'])
def validate_flux():
    """Validate FLUX syntax"""
    try:
        data = request.json
        flux_code = data.get('code', '')

        # Tokenize
        lexer = FluxPDELexer(flux_code)
        tokens = lexer.tokenize()

        # Parse
        parser = FluxPDEParser(tokens)
        ast = parser.parse()

        return jsonify({
            'success': True,
            'valid': True,
            'tokens': len(tokens),
            'ast_nodes': len(ast),
            'message': 'Syntax is valid!'
        })

    except Exception as e:
        return jsonify({
            'success': True,
            'valid': False,
            'error': str(e)
        })

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example FLUX programs"""
    examples = {
        'heat': '''// Heat equation example
domain heat_domain {
    rectangle(0, 1, 0, 1)
    grid(50, 50)
}

equation heat_eq {
    dt(u) = 0.1 * laplacian(u)
}

boundary heat_bc {
    dirichlet(0.0) on all
}

solver heat_solver {
    method: crank_nicolson
    timestep: 0.01
    max_time: 1.0
}''',

        'wave': '''// Wave equation example
domain wave_domain {
    rectangle(-1, 1, -1, 1)
    grid(100, 100)
}

equation wave_eq {
    d2dt(u) = 0.5^2 * laplacian(u)
}

boundary wave_bc {
    dirichlet(0.0) on all
}

solver wave_solver {
    method: explicit
    timestep: 0.001
    max_time: 2.0
}''',

        'poisson': '''// Poisson equation example
domain poisson_domain {
    rectangle(0, 1, 0, 1)
    grid(64, 64)
}

equation poisson_eq {
    laplacian(phi) = -rho
}

boundary poisson_bc {
    dirichlet(0.0) on all
}

solver poisson_solver {
    method: conjugate_gradient
    tolerance: 1e-6
}''',

        'navier_stokes': '''// Navier-Stokes equations
domain flow_domain {
    rectangle(0, 1, 0, 1)
    grid(64, 64)
}

equation momentum_x {
    dt(u) + u*dx(u) + v*dy(u) = -dx(p)/rho + nu*laplacian(u)
}

equation momentum_y {
    dt(v) + u*dx(v) + v*dy(v) = -dy(p)/rho + nu*laplacian(v)
}

equation continuity {
    dx(u) + dy(v) = 0
}

boundary flow_bc {
    dirichlet(1.0, 0.0) on top
    dirichlet(0.0, 0.0) on bottom, left, right
}

solver navier_solver {
    method: fractional_step
    reynolds: 100
    timestep: 0.001
}'''
    }

    return jsonify(examples)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'FLUX Scientific Computing',
        'version': '0.1.0',
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)