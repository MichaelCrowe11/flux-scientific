#!/usr/bin/env python3
"""
FLUX Scientific Computing Language - Simple Web Application
"""

from flask import Flask, jsonify, render_template_string
import os
from datetime import datetime

app = Flask(__name__)

# Simple HTML template embedded in code
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FLUX Scientific Computing</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 800px;
            text-align: center;
        }
        h1 {
            color: #2980B9;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #7F8C8D;
            margin-bottom: 30px;
        }
        .status {
            background: #D5F4E6;
            color: #27AE60;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
            margin: 20px 0;
        }
        .links {
            margin-top: 30px;
        }
        .links a {
            display: inline-block;
            margin: 10px;
            padding: 10px 20px;
            background: #3498DB;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.3s;
        }
        .links a:hover {
            background: #2980B9;
        }
        .features {
            text-align: left;
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        .features h3 {
            color: #2980B9;
            margin-bottom: 15px;
        }
        .features ul {
            margin: 0;
            padding-left: 20px;
        }
        .features li {
            margin: 8px 0;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 FLUX Scientific Computing Language</h1>
        <p class="subtitle">Domain-Specific Language for PDE Solving</p>

        <div class="status">✅ Service Online</div>

        <div class="features">
            <h3>Features</h3>
            <ul>
                <li>🧮 Validated PDE solvers (Heat, Wave, Poisson, Navier-Stokes)</li>
                <li>🚀 GPU acceleration with CuPy</li>
                <li>📝 Complete .flux compiler with multi-backend support</li>
                <li>💻 VS Code extension available</li>
                <li>📦 Install from PyPI: <code>pip install flux-sci-lang</code></li>
            </ul>
        </div>

        <div class="features">
            <h3>Quick Example</h3>
            <pre><code>// Heat equation in FLUX
domain heat_domain {
    rectangle(0, 1, 0, 1)
    grid(50, 50)
}

equation heat_eq {
    dt(u) = 0.1 * laplacian(u)
}

solver heat_solver {
    method: crank_nicolson
    timestep: 0.01
}</code></pre>
        </div>

        <div class="links">
            <a href="https://github.com/MichaelCrowe11/flux-sci-lang">GitHub</a>
            <a href="https://pypi.org/project/flux-sci-lang/">PyPI</a>
            <a href="/health">API Status</a>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint"""
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
        'description': 'Domain-Specific Language for PDE Solving',
        'features': [
            'Heat equation solver',
            'Wave equation solver',
            'Poisson equation solver',
            'Navier-Stokes solver',
            'GPU acceleration',
            'Multi-backend compilation'
        ],
        'links': {
            'github': 'https://github.com/MichaelCrowe11/flux-sci-lang',
            'pypi': 'https://pypi.org/project/flux-sci-lang/',
            'documentation': 'https://flux-sci-lang.readthedocs.io'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'status': 404}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)