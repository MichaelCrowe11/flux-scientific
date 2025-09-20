# FLUX Language Reference

This document provides a comprehensive reference for the FLUX Scientific Computing Language syntax and features.

## Table of Contents

- [Basic Syntax](#basic-syntax)
- [Variables and Types](#variables-and-types)
- [Mathematical Operators](#mathematical-operators)
- [PDE Definitions](#pde-definitions)
- [Mesh and Domain Specification](#mesh-and-domain-specification)
- [Solver Configuration](#solver-configuration)
- [Boundary and Initial Conditions](#boundary-and-initial-conditions)
- [GPU Kernels](#gpu-kernels)
- [Built-in Functions](#built-in-functions)

## Basic Syntax

### Comments
```flux
// Single-line comment

/*
   Multi-line comment
   spans multiple lines
*/
```

### Statements
Statements are terminated by newlines or semicolons:
```flux
const pi = 3.14159
let x = 2.0; let y = 3.0
```

### Blocks
Code blocks use curly braces:
```flux
pde heat_equation {
    ∂u/∂t = α * ∇²u
}
```

## Variables and Types

### Variable Declarations
```flux
// Immutable variables
let temperature = 300.0
let name = "simulation"

// Mutable variables
var counter = 0
var position = [0.0, 0.0, 0.0]

// Constants
const pi = 3.14159
const MAX_ITERATIONS = 1000
```

### Type Annotations
```flux
let velocity: Vector = [1.0, 0.0, 0.0]
let pressure: Scalar = 101325.0
let temperature_field: Field = zeros(nx, ny)
```

### Built-in Types
- `Scalar`: Single floating-point value
- `Vector`: 3D vector [x, y, z]
- `Field`: Scalar field on mesh
- `VectorField`: Vector field on mesh
- `TensorField`: Tensor field on mesh
- `Matrix`: 2D matrix
- `Tensor`: Multi-dimensional tensor

## Mathematical Operators

### Arithmetic Operators
```flux
let result = a + b    // Addition
let result = a - b    // Subtraction
let result = a * b    // Multiplication
let result = a / b    // Division
let result = a ^ b    // Exponentiation
let result = a % b    // Modulo
```

### Vector and Tensor Operations
```flux
let dot = a · b       // Dot product
let cross = a × b     // Cross product
let tensor = a ⊗ b    // Tensor product
let norm = |v|        // Vector magnitude
```

### Differential Operators
```flux
// Partial derivatives
∂u/∂t                 // Time derivative
∂u/∂x                 // Spatial derivative
∂²u/∂x²              // Second derivative

// Vector calculus
∇u                    // Gradient
∇·v                   // Divergence
∇×v                   // Curl
∇²u                   // Laplacian
```

### Comparison Operators
```flux
a == b    // Equal to
a != b    // Not equal to
a < b     // Less than
a <= b    // Less than or equal
a > b     // Greater than
a >= b    // Greater than or equal
```

### Logic Operators
```flux
a && b    // Logical AND
a || b    // Logical OR
!a        // Logical NOT
```

## PDE Definitions

### Basic Structure
```flux
pde equation_name {
    variables: list_of_variables

    // Main equations
    equation1
    equation2
    ...

    // Boundary conditions
    boundary {
        condition1
        condition2
        ...
    }

    // Initial conditions
    initial: initial_condition
}
```

### Variable Declarations
```flux
pde navier_stokes {
    variables: velocity v = [u, v, w], pressure p, temperature T

    // Equations follow...
}
```

### Multi-Physics PDEs
```flux
pde fluid_heat_transfer {
    variables: velocity v, pressure p, temperature T

    // Navier-Stokes equations
    ∂v/∂t + (v·∇)v = -∇p/ρ + ν*∇²v
    ∇·v = 0

    // Energy equation
    ∂T/∂t + v·∇T = α*∇²T

    boundary {
        v = v_inlet     on inlet
        v = [0, 0, 0]   on walls
        T = T_wall      on walls
    }
}
```

### Regional PDEs
```flux
pde multi_region {
    regions: [fluid, solid]

    in fluid: {
        ∂v/∂t + (v·∇)v = -∇p/ρ + ν*∇²v
        ∇·v = 0
    }

    in solid: {
        ∂T/∂t = α*∇²T
    }

    interface: {
        T_fluid = T_solid
        k_fluid*∂T_fluid/∂n = k_solid*∂T_solid/∂n
    }
}
```

## Mesh and Domain Specification

### Domain Definition
```flux
// 2D rectangular domain
domain rectangle = Rectangle(
    x_min=0.0, x_max=1.0,
    y_min=0.0, y_max=1.0
)

// 3D box domain
domain box = Box(
    x_min=0, x_max=2,
    y_min=0, y_max=1,
    z_min=0, z_max=0.5
)

// Circular domain
domain circle = Circle(
    center=[0.0, 0.0],
    radius=1.0
)

// Custom polygon
domain polygon = Polygon(
    points=[[0,0], [1,0], [1,1], [0.5,1.5], [0,1]]
)
```

### Mesh Generation
```flux
// Structured grid
mesh uniform = StructuredGrid(rectangle, nx=100, ny=100)

// Unstructured triangular mesh
mesh triangular = UnstructuredMesh(rectangle) {
    max_element_size: 0.01,
    element_type: triangle,
    algorithm: delaunay
}

// Adaptive mesh refinement
mesh adaptive = AdaptiveMesh(base_mesh=uniform) {
    refinement_criterion: gradient(u) > 0.1,
    max_level: 4,
    min_level: 1
}

// Boundary layer mesh
mesh boundary_layer = UnstructuredMesh(airfoil) {
    boundary_layers: 20,
    first_layer_thickness: 1e-5,
    growth_rate: 1.2,
    far_field: Circle(radius=50)
}
```

## Solver Configuration

### Time Integration Methods
```flux
// Explicit methods
solver = ExplicitEuler(dt=0.001)
solver = RungeKutta4(dt=0.001, cfl=0.8)

// Implicit methods
solver = ImplicitEuler(dt=0.01, tolerance=1e-6)
solver = BackwardDifferentiation(order=2, dt=0.01)

// Semi-implicit methods
solver = CrankNicolson(dt=0.01, theta=0.5)
```

### Spatial Discretization
```flux
// Finite Element Method
solver = FEM(
    basis_functions = Lagrange(degree=2),
    quadrature = Gauss(order=3),
    linear_solver = DirectSparse()
)

// Finite Volume Method
solver = FVM(
    flux_scheme = Roe,
    limiter = VanLeer,
    reconstruction = MUSCL
)

// Finite Difference Method
solver = FDM(
    stencil = Central(order=2),
    boundary_scheme = Ghost
)

// Spectral Methods
solver = Spectral(
    basis = Chebyshev,
    transform = FFT
)
```

### Linear Solvers
```flux
// Direct solvers
linear_solver = DirectSparse(factorization=LU, reordering=METIS)
linear_solver = DirectDense(factorization=Cholesky)

// Iterative solvers
linear_solver = CG(tolerance=1e-8, max_iterations=1000)
linear_solver = GMRES(restart=30, tolerance=1e-8)
linear_solver = BiCGStab(tolerance=1e-8)

// Preconditioners
linear_solver = CG(
    tolerance=1e-8,
    preconditioner=AMG(levels=4, smoother=Jacobi)
)
```

## Boundary and Initial Conditions

### Boundary Condition Types
```flux
boundary {
    // Dirichlet (essential) boundary conditions
    u = 0.0           on walls
    v = [1.0, 0.0]    on inlet
    T = 300.0         on heated_wall

    // Neumann (natural) boundary conditions
    ∂u/∂n = 0         on symmetry
    ∂T/∂n = -q/k      on heat_flux

    // Robin (mixed) boundary conditions
    h*(T - T_ambient) + k*∂T/∂n = 0  on convection

    // Periodic boundary conditions
    periodic(u)       between left and right

    // Characteristic boundary conditions
    characteristic_inflow   on inlet
    characteristic_outflow  on outlet
}
```

### Initial Conditions
```flux
// Scalar fields
initial: u(x,y,0) = sin(π*x) * sin(π*y)
initial: T(x,y,z,0) = T0 * exp(-(x²+y²+z²)/σ²)

// Vector fields
initial: v(x,y,0) = [u0*y*(1-y), 0]

// Conditional initial conditions
initial: {
    if x < 0.5: u(x,y,0) = 1.0
    else: u(x,y,0) = 0.0
}

// File-based initial conditions
initial: u = import("initial_data.h5")
```

## GPU Kernels

### Basic GPU Kernel
```flux
@gpu(backend="cuda")
kernel heat_diffusion(
    u: Field,
    u_new: Field,
    α: Scalar,
    dt: Scalar
) {
    idx = blockIdx.x * blockDim.x + threadIdx.x

    if idx < u.size {
        u_new[idx] = u[idx] + α * dt * laplacian(u, idx)
    }
}
```

### Advanced GPU Features
```flux
@gpu(backend="cuda", shared_memory=true)
kernel stencil_computation(
    u: Field,
    result: Field,
    stencil: Array
) {
    // Shared memory for tile
    shared_u = shared_memory(TILE_SIZE)

    // Load data into shared memory
    load_tile(shared_u, u, threadIdx, blockIdx)
    syncthreads()

    // Compute stencil operation
    idx = get_global_index()
    result[idx] = apply_stencil(shared_u, stencil)
}
```

### Multi-GPU Support
```flux
@gpu(backend="cuda", devices=4)
distributed kernel multi_gpu_solve(
    u: DistributedField,
    forces: DistributedField
) {
    device_id = get_device_id()
    local_data = u.get_local_partition(device_id)

    // Local computation
    solve_local(local_data, forces)

    // Communication between devices
    exchange_boundaries(local_data)
}
```

## Built-in Functions

### Mathematical Functions
```flux
// Basic math
sin(x), cos(x), tan(x)
exp(x), log(x), sqrt(x)
abs(x), sign(x)
min(a, b), max(a, b)

// Vector operations
dot(a, b)           // Dot product
cross(a, b)         // Cross product
norm(v)             // Vector magnitude
normalize(v)        // Unit vector

// Matrix operations
transpose(A)        // Matrix transpose
inverse(A)          // Matrix inverse
determinant(A)      // Matrix determinant
eigenvalues(A)      // Eigenvalues
```

### Field Operations
```flux
// Differential operators
gradient(u)         // ∇u
divergence(v)       // ∇·v
curl(v)            // ∇×v
laplacian(u)       // ∇²u

// Field statistics
mean(u)            // Average value
variance(u)        // Variance
min(u), max(u)     // Extrema
integrate(u, mesh) // Numerical integration
```

### Solver Functions
```flux
// Main solver function
solution = solve(pde, mesh=M, solver=S, t_end=1.0)

// Eigenvalue problems
eigenvalues, eigenvectors = solve_eigenvalue(A, B, n_modes=10)

// Optimization
result = minimize(objective, constraints, variables)
result = maximize(objective, constraints, variables)
```

### I/O Functions
```flux
// File export
export(field, format="vtk", filename="solution.vtk")
export(field, format="hdf5", filename="data.h5")

// File import
data = import("input.csv", format="csv")
mesh = import("geometry.msh", format="gmsh")

// Plotting
plot(field, title="Solution")
contour(field, levels=20)
surface(field, colormap="hot")
```

### Utility Functions
```flux
// Mesh utilities
refine_mesh(mesh, criterion)
coarsen_mesh(mesh, criterion)
adapt_mesh(mesh, solution)

// Time stepping
advance_time(solution, dt)
check_convergence(residual, tolerance)

// Performance
profile_start("section_name")
profile_end("section_name")
get_memory_usage()
```

This reference covers the core language features. For more advanced topics and examples, see the [Tutorial](tutorial.md) and [Examples](examples.md).