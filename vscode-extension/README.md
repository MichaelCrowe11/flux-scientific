# FLUX Scientific Computing Language - VS Code Extension

This VS Code extension provides comprehensive language support for the FLUX Scientific Computing Language, a domain-specific language designed for solving partial differential equations (PDEs) and scientific computing tasks.

## Features

### 🔥 Syntax Highlighting
- Complete syntax highlighting for FLUX language constructs
- Support for domains, equations, boundaries, solvers, and field declarations
- Highlighting for mathematical operators, differential operators, and scientific functions

### 🚀 IntelliSense & Code Completion
- Auto-completion for FLUX keywords and functions
- Smart suggestions for PDE operators (laplacian, gradient, divergence)
- Context-aware completions for boundary conditions and solver methods

### 📝 Code Snippets
- Pre-built templates for common PDE problems:
  - Heat equation
  - Wave equation
  - Poisson equation
  - Navier-Stokes equations
- Domain, boundary, and solver configuration snippets
- Function and field declaration templates

### 🔧 Compilation & Execution
- One-click compilation to multiple backends (Python, C++, CUDA)
- Compile and run functionality with integrated output
- Configurable compilation settings
- Real-time syntax validation

### 📊 Debugging & Analysis
- Syntax validation on save
- AST (Abstract Syntax Tree) visualization
- Error highlighting and diagnostics
- Performance benchmarking tools

### ⚡ Commands
- `FLUX: Compile` (F5) - Compile current FLUX file
- `FLUX: Compile and Run` (Ctrl+F5) - Compile and execute immediately
- `FLUX: Validate Syntax` - Check syntax without compilation
- `FLUX: Show AST` - Display abstract syntax tree
- `FLUX: Run Benchmarks` - Execute performance benchmarks

## Installation

1. Install from VS Code Marketplace (coming soon)
2. Or install from VSIX:
   ```bash
   code --install-extension flux-sci-lang-0.1.0.vsix
   ```

## Requirements

- Python 3.7+ with FLUX Scientific Computing Language installed
- For GPU acceleration: NVIDIA CUDA toolkit and CuPy
- For C++ backend: CMake and a C++ compiler

## Extension Settings

- `flux.compiler.backend`: Default compilation backend (python, cpp, cuda, julia, fortran)
- `flux.compiler.optimization`: Optimization level (O0, O1, O2, O3)
- `flux.compiler.outputDirectory`: Default output directory for compiled code
- `flux.validation.onSave`: Enable syntax validation on file save
- `flux.python.path`: Path to Python interpreter for FLUX compiler

## Quick Start

1. Create a new file with `.flux` extension
2. Start typing `heat` and select the heat equation snippet
3. Customize the parameters for your problem
4. Press F5 to compile
5. View results in the integrated terminal

### Example FLUX Code

```flux
// Heat equation with Dirichlet boundaries
domain heat_domain {
    rectangle(0, 1, 0, 1)
    grid(50, 50)
}

equation heat_eq {
    dt(u) = alpha * laplacian(u)
}

boundary heat_bc {
    dirichlet(0.0) on left, right, top, bottom
}

solver heat_solver {
    method: crank_nicolson
    timestep: 0.001
    max_time: 1.0
}
```

## Supported PDE Types

- **Heat/Diffusion Equations**: Thermal diffusion, species transport
- **Wave Equations**: Acoustic waves, electromagnetic waves
- **Poisson Equations**: Electrostatics, steady-state diffusion
- **Navier-Stokes**: Incompressible fluid flow, CFD problems
- **Custom PDEs**: User-defined differential equations

## Compilation Backends

- **Python**: NumPy/SciPy implementation (default)
- **C++**: OpenMP-accelerated native code
- **CUDA**: GPU-accelerated for NVIDIA hardware
- **Julia**: High-performance Julia code (experimental)
- **Fortran**: Legacy Fortran 90 support (experimental)

## Contributing

This extension is part of the FLUX Scientific Computing Language project. Contributions welcome!

- GitHub: [https://github.com/MichaelCrowe11/flux-sci-lang](https://github.com/MichaelCrowe11/flux-sci-lang)
- Issues: [https://github.com/MichaelCrowe11/flux-sci-lang/issues](https://github.com/MichaelCrowe11/flux-sci-lang/issues)

## License

MIT License - see LICENSE file for details.

---

**Enjoy solving PDEs with FLUX! 🧮⚡**