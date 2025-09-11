// Lid-Driven Cavity Flow using Navier-Stokes equations
// Classic CFD benchmark problem

// Define unit square domain
domain cavity = Rectangle(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

// Create fine mesh for accurate boundary layer resolution
mesh M = StructuredGrid(cavity, nx=100, ny=100)

// Incompressible Navier-Stokes equations
pde navier_stokes {
    variables: velocity v, pressure p
    
    // Momentum equation: ∂v/∂t + (v·∇)v = -∇p/ρ + ν*∇²v
    ∂v/∂t + (v·∇)v = -∇p/ρ + ν*∇²v
    
    // Continuity equation: ∇·v = 0
    ∇·v = 0
    
    boundary {
        // No-slip walls
        v = [0.0, 0.0]    on bottom
        v = [0.0, 0.0]    on left
        v = [0.0, 0.0]    on right
        
        // Moving lid
        v = [1.0, 0.0]    on top
        
        // Pressure reference (set at one point)
        p = 0.0           at (0.0, 0.0)
    }
    
    initial: {
        v(x,y,0) = [0.0, 0.0]
        p(x,y,0) = 0.0
    }
}

// Fluid properties
const ρ = 1.0      // Density
const ν = 0.01     // Kinematic viscosity
const Re = 100.0   // Reynolds number = U*L/ν

// Solver for incompressible flow
solver = SIMPLE(
    dt = 0.001,
    pressure_correction = true,
    momentum_relaxation = 0.7,
    pressure_relaxation = 0.3,
    max_iterations = 1000
)

// Solve the equations
solution = solve(navier_stokes, mesh=M, solver=solver, t_end=10.0)

// Post-processing
vorticity = ∇ × solution.v
stream_function = solve_poisson(∇²ψ = -vorticity)

// Export results
export(solution.v, format="vtk", filename="velocity.vtk")
export(solution.p, format="vtk", filename="pressure.vtk")
export(vorticity, format="vtk", filename="vorticity.vtk")
export(stream_function, format="vtk", filename="streamlines.vtk")

// Extract velocity profiles for validation
u_centerline = extract_line(solution.v[0], start=[0.5, 0.0], end=[0.5, 1.0])
v_centerline = extract_line(solution.v[1], start=[0.0, 0.5], end=[1.0, 0.5])

print("Cavity flow simulation completed")
print("Reynolds number: " + str(Re))
print("Final time: " + str(solution.time))
print("Peak u-velocity at centerline: " + str(max(u_centerline)))