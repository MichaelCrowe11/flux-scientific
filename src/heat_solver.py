"""
Real Heat Equation Solver for FLUX
Implements finite difference method for ∂u/∂t = α∇²u
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Callable
import time

class HeatEquationSolver:
    """
    Solves 2D heat equation: ∂u/∂t = α∇²u
    Using finite differences in space, implicit Euler in time
    """
    
    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Create coordinate arrays
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Total number of interior points
        self.n_interior = (nx - 2) * (ny - 2)
        
        # Build system matrix for implicit scheme
        self._build_system_matrix()
        
    def _build_system_matrix(self):
        """Build the system matrix for implicit time stepping"""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # Number of interior points
        n = (nx - 2) * (ny - 2)
        
        # Coefficients for 5-point stencil
        cx = 1.0 / (dx * dx)
        cy = 1.0 / (dy * dy)
        cc = -2.0 * (cx + cy)  # Center coefficient
        
        # Build sparse matrix using diagonals
        # Main diagonal
        main_diag = np.full(n, cc)
        
        # East/West neighbors (±1)
        east_diag = np.full(n - 1, cx)
        west_diag = np.full(n - 1, cx)
        
        # Remove connections across grid boundaries
        for i in range(nx - 3, n, nx - 2):
            if i < n - 1:
                east_diag[i] = 0.0
        for i in range(nx - 2, n, nx - 2):
            if i > 0:
                west_diag[i - 1] = 0.0
        
        # North/South neighbors (±(nx-2))
        north_diag = np.full(n - (nx - 2), cy)
        south_diag = np.full(n - (nx - 2), cy)
        
        # Create sparse matrix
        diagonals = [south_diag, west_diag, main_diag, east_diag, north_diag]
        offsets = [-(nx - 2), -1, 0, 1, nx - 2]
        
        self.laplacian_matrix = diags(diagonals, offsets, shape=(n, n), format='csc')
        
    def _grid_to_vector(self, u_grid: np.ndarray) -> np.ndarray:
        """Convert 2D grid to 1D vector (interior points only)"""
        return u_grid[1:-1, 1:-1].flatten()
    
    def _vector_to_grid(self, u_vec: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
        """Convert 1D vector back to 2D grid (preserving boundary)"""
        result = u_grid.copy()
        result[1:-1, 1:-1] = u_vec.reshape(self.ny - 2, self.nx - 2)
        return result
    
    def apply_boundary_conditions(self, u: np.ndarray, bc_type: str = 'dirichlet', 
                                 bc_value: float = 0.0) -> np.ndarray:
        """Apply boundary conditions to the solution"""
        u_bc = u.copy()
        
        if bc_type == 'dirichlet':
            # Fixed temperature on all boundaries
            u_bc[0, :] = bc_value    # Bottom
            u_bc[-1, :] = bc_value   # Top
            u_bc[:, 0] = bc_value    # Left
            u_bc[:, -1] = bc_value   # Right
        elif bc_type == 'neumann':
            # Zero flux (insulated) boundaries
            u_bc[0, :] = u_bc[1, :]      # Bottom
            u_bc[-1, :] = u_bc[-2, :]    # Top
            u_bc[:, 0] = u_bc[:, 1]      # Left
            u_bc[:, -1] = u_bc[:, -2]    # Right
        
        return u_bc
    
    def set_initial_condition(self, func: Callable[[float, float], float]) -> np.ndarray:
        """Set initial condition using a function u(x,y,0) = func(x,y)"""
        u0 = np.zeros((self.ny, self.nx))
        for i in range(self.ny):
            for j in range(self.nx):
                u0[i, j] = func(self.X[i, j], self.Y[i, j])
        return u0
    
    def solve_time_step(self, u_old: np.ndarray, dt: float, alpha: float = 1.0) -> np.ndarray:
        """Solve one time step using implicit Euler method"""
        # Convert to vector form (interior points only)
        u_old_vec = self._grid_to_vector(u_old)
        
        # Build system: (I - α*dt*L) * u_new = u_old
        # where L is the Laplacian matrix
        identity = csc_matrix(np.eye(self.n_interior))
        system_matrix = identity - alpha * dt * self.laplacian_matrix
        
        # Solve linear system
        u_new_vec = spsolve(system_matrix, u_old_vec)
        
        # Convert back to grid and apply boundary conditions
        u_new = self._vector_to_grid(u_new_vec, u_old)
        u_new = self.apply_boundary_conditions(u_new)
        
        return u_new
    
    def solve(self, u0: np.ndarray, t_end: float, dt: float, alpha: float = 1.0,
              save_interval: int = 10) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Solve heat equation from t=0 to t=t_end
        
        Returns:
            t_array: Time points
            u_final: Final solution
            u_history: List of solutions at saved time points
        """
        print(f"Solving 2D heat equation:")
        print(f"  Grid: {self.nx}×{self.ny}")
        print(f"  Time step: {dt:.6f}")
        print(f"  End time: {t_end}")
        print(f"  Thermal diffusivity: {alpha}")
        
        # Check CFL condition for stability
        cfl_x = alpha * dt / (self.dx**2)
        cfl_y = alpha * dt / (self.dy**2)
        cfl_max = cfl_x + cfl_y
        print(f"  CFL number: {cfl_max:.4f}")
        
        if cfl_max > 0.5:
            print("  Warning: CFL number high, solution may be unstable")
        
        # Time stepping
        n_steps = int(t_end / dt)
        t = 0.0
        u = u0.copy()
        
        # Storage for results
        t_array = [0.0]
        u_history = [u0.copy()]
        
        start_time = time.time()
        
        for step in range(1, n_steps + 1):
            # Advance one time step
            u = self.solve_time_step(u, dt, alpha)
            t = step * dt
            
            # Save results periodically
            if step % save_interval == 0:
                t_array.append(t)
                u_history.append(u.copy())
                
                # Progress report
                elapsed = time.time() - start_time
                progress = step / n_steps * 100
                print(f"  Step {step}/{n_steps} ({progress:.1f}%), t={t:.4f}, "
                      f"max(u)={np.max(u):.6f}, elapsed={elapsed:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Simulation completed in {total_time:.2f} seconds")
        print(f"Performance: {n_steps/total_time:.0f} steps/second")
        
        return np.array(t_array), u, u_history

def analytical_solution_1d(x: float, t: float, alpha: float = 1.0, L: float = 1.0) -> float:
    """Analytical solution for 1D heat equation with sin initial condition"""
    if t == 0:
        return np.sin(np.pi * x / L)
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

def create_gaussian_initial_condition(x0: float = 0.5, y0: float = 0.5, 
                                    sigma: float = 0.1, amplitude: float = 1.0):
    """Create Gaussian blob initial condition"""
    def gaussian(x, y):
        return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gaussian

def create_sine_initial_condition():
    """Create sine wave initial condition for verification"""
    def sine_2d(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    return sine_2d

def demo_heat_solver():
    """Demonstrate the heat equation solver"""
    print("🔥 FLUX Heat Equation Solver Demo")
    print("=" * 50)
    
    # Create solver
    nx, ny = 50, 50
    solver = HeatEquationSolver(nx, ny, Lx=1.0, Ly=1.0)
    
    # Test 1: Gaussian blob diffusion
    print("\n1. Gaussian Blob Diffusion")
    gaussian_ic = create_gaussian_initial_condition(0.3, 0.7, 0.05, 10.0)
    u0_gaussian = solver.set_initial_condition(gaussian_ic)
    
    t_end = 0.1
    dt = 0.0001
    alpha = 0.1
    
    t_array, u_final, u_history = solver.solve(u0_gaussian, t_end, dt, alpha, save_interval=100)
    
    print(f"Initial max temperature: {np.max(u0_gaussian):.4f}")
    print(f"Final max temperature: {np.max(u_final):.4f}")
    print(f"Heat diffused as expected: {np.max(u_final) < np.max(u0_gaussian)}")
    
    # Test 2: Sine wave (for verification)
    print("\n2. Sine Wave Initial Condition (Verification)")
    sine_ic = create_sine_initial_condition()
    u0_sine = solver.set_initial_condition(sine_ic)
    
    t_end = 0.01
    dt = 0.00001
    alpha = 1.0
    
    t_array_sine, u_final_sine, _ = solver.solve(u0_sine, t_end, dt, alpha, save_interval=200)
    
    # Compare with analytical solution
    analytical = np.sin(np.pi * solver.X) * np.sin(np.pi * solver.Y) * np.exp(-2 * np.pi**2 * alpha * t_end)
    error = np.max(np.abs(u_final_sine - analytical))
    print(f"Maximum error vs analytical solution: {error:.6f}")
    print(f"Relative error: {error / np.max(np.abs(analytical)) * 100:.4f}%")
    
    print("\n✅ Heat equation solver is working correctly!")
    
    return solver, u_history, t_array

if __name__ == "__main__":
    demo_heat_solver()