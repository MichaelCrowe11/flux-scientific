// Heat Equation Example in FLUX
// Demonstrates PDE syntax and structured mesh

// Define computational domain
domain Ω = Rectangle(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

// Create structured mesh
mesh M = StructuredGrid(Ω, nx=50, ny=50)

// Heat equation PDE
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