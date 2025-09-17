// FLUX Heat Equation Example - Full Working Demo
// Solves 2D heat equation with validated numerical methods

domain omega = Rectangle(xmin=0, xmax=1, ymin=0, ymax=1)

mesh grid = StructuredGrid(omega, nx=50, ny=50)

pde HeatEquation {
    variables: u, T

    // Heat equation: ∂u/∂t = α∇²u
    ∂u/∂t = 0.1 * ∇²u

    boundary {
        u = 0 on left
        u = 0 on right
        u = 0 on top
        u = 0 on bottom
    }

    initial: u(x, y, 0) = exp(-20*((x-0.5)² + (y-0.5)²))
}

solver heat_solver = CrankNicolson(
    time_step = 0.001,
    tolerance = 1e-6,
    max_iterations = 1000
)

// Solve and visualize
solve(HeatEquation, heat_solver, t_final=0.5)
plot(u, title="Temperature Distribution")
animate(u, fps=30, filename="heat_evolution.mp4")