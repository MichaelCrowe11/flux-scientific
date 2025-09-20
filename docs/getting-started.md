# Getting Started with FLUX Scientific Computing Language

FLUX is a domain-specific language designed for scientific computing, specializing in partial differential equations (PDEs), computational fluid dynamics (CFD), electromagnetic simulations, and finite element analysis.

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib

### Quick Install

```bash
# Clone the repository
git clone https://github.com/MichaelCrowe11/flux-sci-lang.git
cd flux-sci-lang

# Install dependencies
pip install numpy scipy matplotlib

# Install FLUX (development mode)
pip install -e .
```

### Optional Dependencies

```bash
# For GPU acceleration
pip install cupy pycuda

# For HPC features
pip install mpi4py h5py

# For development
pip install pytest black flake8 mypy
```

## Your First FLUX Program

Create a file called `my_first_pde.flux`:

```flux
// Heat Equation Example
domain Ω = Rectangle(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
mesh M = StructuredGrid(Ω, nx=50, ny=50)

pde heat_equation {
    variables: temperature u

    // Main equation: ∂u/∂t = α * ∇²u
    ∂u/∂t = α * ∇²u  in Ω

    boundary {
        u = 0.0      on left
        u = 0.0      on right
        u = 0.0      on top
        u = 0.0      on bottom
    }

    initial: u(x,y,0) = sin(π*x) * sin(π*y)
}

// Solver configuration
solver = ImplicitEuler(dt=0.01, tolerance=1e-6)

// Material properties
const α = 0.1  // Thermal diffusivity

// Solve the PDE
solution = solve(heat_equation, mesh=M, solver=solver, t_end=1.0)

// Output results
export(solution.u, format="vtk", filename="heat_solution.vtk")
```

## Compile and Run

```bash
# Compile to Python
python flux_scientific.py my_first_pde.flux -b python

# Compile to C++
python flux_scientific.py my_first_pde.flux -b cpp

# Compile to CUDA (GPU)
python flux_scientific.py my_first_pde.flux -b cuda
```

## Interactive Mode

FLUX provides an interactive environment for scientific computing:

```bash
python flux_scientific.py -i
```

Example session:
```
flux-sci> mesh StructuredGrid 50 50
Created StructuredGrid with 2601 nodes, 2500 cells

flux-sci> compile examples/heat_equation.flux python
Compiling examples/heat_equation.flux to python...
Generated code written to output/generated.py
Compilation successful!

flux-sci> help
Available backends: python, cpp, cuda
Available mesh types: StructuredGrid, UnstructuredMesh, AdaptiveMesh
```

## Core Concepts

### 1. Domains

Define computational domains for your problems:

```flux
// 2D rectangular domain
domain cavity = Rectangle(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

// 3D box domain
domain box = Box(x_min=0, x_max=2, y_min=0, y_max=1, z_min=0, z_max=0.5)

// Complex geometry (future feature)
domain airfoil = import("NACA0012.stl")
```

### 2. Meshes

Create meshes for numerical discretization:

```flux
// Structured grid
mesh uniform = StructuredGrid(cavity, nx=100, ny=100)

// Unstructured mesh
mesh flexible = UnstructuredMesh(cavity) {
    max_element_size: 0.01,
    boundary_layers: 5
}

// Adaptive mesh refinement
mesh adaptive = AdaptiveMesh(base_mesh=uniform) {
    refinement_criterion: gradient(u) > threshold,
    max_level: 4
}
```

### 3. PDE Definitions

Write equations in mathematical notation:

```flux
pde navier_stokes {
    variables: velocity v = [u, v], pressure p

    // Momentum equation
    ∂v/∂t + (v·∇)v = -∇p/ρ + ν*∇²v

    // Continuity equation
    ∇·v = 0

    boundary {
        v = [1.0, 0.0]    on inlet
        v = [0.0, 0.0]    on walls
        ∂v/∂n = 0        on outlet
    }
}
```

### 4. Solvers

Configure numerical methods:

```flux
// Time-stepping methods
solver = ImplicitEuler(dt=0.001, tolerance=1e-6)
solver = RungeKutta4(dt=0.001, cfl=0.8)

// Spatial discretization
solver = FiniteElement(basis=Lagrange(degree=2), quadrature=Gauss(order=3))
solver = FiniteVolume(flux_scheme=Roe, limiter=VanLeer)

// Linear solvers
solver = DirectSparse(factorization=Cholesky)
solver = IterativeKrylov(method=CG, preconditioner=AMG)
```

## Example Problems

### Heat Equation
Solve the classic heat diffusion equation with various boundary conditions.

### Navier-Stokes CFD
Simulate fluid flow in cavities, around objects, and through channels.

### Electromagnetic Scattering
Compute electromagnetic fields using Maxwell's equations.

### Structural Analysis
Analyze stress and deformation in solid mechanics problems.

### Multi-Physics Coupling
Combine multiple physical phenomena in coupled simulations.

## Performance Features

### GPU Acceleration
```flux
@gpu(backend="cuda")
kernel heat_kernel(u: Field, u_new: Field, α: float, dt: float) {
    idx = thread_index()
    u_new[idx] = u[idx] + α * dt * laplacian(u, idx)
}
```

### Parallel Computing
```flux
@parallel(backend="mpi")
solver = DistributedSolver(decomposition=CartesianDecomp(4, 4))
```

### High-Performance Backends
- **Python**: NumPy/SciPy for rapid prototyping
- **C++**: Optimized for performance with OpenMP
- **CUDA**: GPU acceleration for massive parallelism

## Next Steps

1. **Tutorial**: Work through the [FLUX Tutorial](tutorial.md)
2. **Examples**: Explore the `examples/` directory
3. **API Reference**: Browse the [API Documentation](api.md)
4. **Advanced Features**: Learn about [GPU Programming](gpu.md) and [HPC](hpc.md)

## Getting Help

- **Documentation**: [https://flux-sci-lang.readthedocs.io](https://flux-sci-lang.readthedocs.io)
- **Examples**: Check the `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/MichaelCrowe11/flux-sci-lang/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MichaelCrowe11/flux-sci-lang/discussions)

## Community

FLUX is open-source and welcomes contributions! Whether you're fixing bugs, adding features, writing documentation, or sharing examples, your help makes FLUX better for everyone.

**Happy Computing with FLUX! 🚀**