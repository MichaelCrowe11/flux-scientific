# FLUX Scientific Computing Language

[![CI](https://github.com/MichaelCrowe11/flux-sci-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelCrowe11/flux-sci-lang/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MichaelCrowe11/flux-sci-lang/branch/main/graph/badge.svg)](https://codecov.io/gh/MichaelCrowe11/flux-sci-lang)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Domain-Specific Language for PDEs, CFD, and Computational Physics**

FLUX is a high-performance domain-specific language designed for scientific computing, specializing in partial differential equations, computational fluid dynamics, electromagnetic simulations, and finite element analysis.

## Why FLUX?

### 🧮 **Mathematical Notation**
Write equations naturally using Unicode operators:
```flux
∂u/∂t = α * ∇²u  // Heat equation
∇·v = 0           // Incompressible flow
```

### ⚡ **High Performance**
Multiple backend code generation:
- **Python**: NumPy/SciPy for rapid prototyping
- **C++**: Optimized for performance with OpenMP
- **CUDA**: GPU acceleration for massive parallelism

### 🔬 **Scientific Focus**
Built-in support for:
- Partial Differential Equations (PDEs)
- Computational Fluid Dynamics (CFD)
- Electromagnetic simulations
- Structural analysis
- Multi-physics coupling

### 🚀 **Modern Features**
- Adaptive mesh refinement
- GPU kernel generation
- Distributed computing
- Real-time visualization

## Quick Example

```flux
// Define computational domain
domain Ω = Rectangle(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

// Create mesh
mesh M = StructuredGrid(Ω, nx=100, ny=100)

// Heat equation PDE
pde heat_equation {
    variables: temperature u

    ∂u/∂t = α * ∇²u  in Ω

    boundary {
        u = 0.0  on walls
    }

    initial: u(x,y,0) = sin(π*x) * sin(π*y)
}

// Solve
solver = ImplicitEuler(dt=0.01)
solution = solve(heat_equation, mesh=M, solver=solver, t_end=1.0)
```

## Getting Started

### Installation
```bash
git clone https://github.com/MichaelCrowe11/flux-sci-lang.git
cd flux-sci-lang
pip install -e .
```

### First Program
```bash
# Create your first FLUX program
echo 'print("Hello, FLUX!")' > hello.flux

# Compile and run
python flux_scientific.py hello.flux
```

### Interactive Mode
```bash
python flux_scientific.py -i
```

## Core Features

### 📐 **Advanced Mesh Support**
- **Structured grids**: Cartesian, curvilinear
- **Unstructured meshes**: Triangular, tetrahedral
- **Adaptive refinement**: Error-driven mesh adaptation
- **Boundary layers**: High-quality meshes for CFD

### 🧪 **Comprehensive Solvers**
- **Time integration**: Explicit/implicit Euler, Runge-Kutta
- **Spatial discretization**: FEM, FVM, FDM, spectral methods
- **Linear solvers**: Direct sparse, iterative Krylov methods
- **Eigenvalue solvers**: Arnoldi, Lanczos algorithms

### 💻 **GPU Acceleration**
```flux
@gpu(backend="cuda")
kernel heat_kernel(u: Field, u_new: Field, α: float, dt: float) {
    idx = thread_index()
    u_new[idx] = u[idx] + α * dt * laplacian(u, idx)
}
```

### 🔗 **Multi-Physics Coupling**
```flux
pde fluid_structure_interaction {
    in fluid: navier_stokes_equations
    in solid: linear_elasticity_equations

    interface: {
        continuity_of_velocity
        equilibrium_of_traction
    }
}
```

## Example Applications

### 🌡️ **Heat Transfer**
- Transient heat conduction
- Convective heat transfer
- Radiative heat transfer
- Phase change problems

### 🌊 **Fluid Dynamics**
- Lid-driven cavity flow
- Flow around cylinders
- Turbulent channel flow
- Compressible gas dynamics

### ⚡ **Electromagnetics**
- Maxwell's equations
- Antenna radiation patterns
- Waveguide analysis
- Scattering problems

### 🏗️ **Structural Mechanics**
- Linear elasticity
- Nonlinear deformation
- Modal analysis
- Fatigue assessment

## Performance Benchmarks

| Problem Size | Python | C++ | CUDA |
|-------------|--------|-----|------|
| 100×100 grid | 0.5s | 0.1s | 0.02s |
| 500×500 grid | 12s | 2.4s | 0.15s |
| 1000×1000 grid | 95s | 19s | 0.6s |

*Heat equation benchmarks on Intel i7 + RTX 3080*

## Documentation

### 📚 **Comprehensive Guides**
- [Getting Started](getting-started.md) - Installation and first steps
- [Language Reference](language-reference.md) - Complete syntax guide
- [Tutorial](tutorial.md) - Step-by-step examples
- [API Documentation](api.md) - Function reference

### 🎯 **Specialized Topics**
- [GPU Programming](gpu.md) - CUDA kernel development
- [HPC Computing](hpc.md) - Distributed parallel computing
- [Advanced Examples](examples.md) - Complex multi-physics problems

## Community & Support

### 🤝 **Get Involved**
- **GitHub**: [flux-sci-lang](https://github.com/MichaelCrowe11/flux-sci-lang)
- **Issues**: [Bug reports & feature requests](https://github.com/MichaelCrowe11/flux-sci-lang/issues)
- **Discussions**: [Community forum](https://github.com/MichaelCrowe11/flux-sci-lang/discussions)

### 📧 **Contact**
- **Email**: michael@flux-sci.org
- **Website**: [flux-sci-lang.org](https://flux-sci-lang.org)

## License

FLUX is open-source software released under the MIT License. See [LICENSE](../LICENSE) for details.

## Acknowledgments

FLUX is inspired by:
- **Julia** (scientific computing)
- **FEniCS** (finite element methods)
- **OpenFOAM** (computational fluid dynamics)
- **COMSOL** (multi-physics simulation)
- **CUDA** (GPU computing)

---

**Ready to revolutionize your scientific computing workflow?**

[Get Started →](getting-started.md) | [View Examples →](examples.md) | [API Reference →](api.md)