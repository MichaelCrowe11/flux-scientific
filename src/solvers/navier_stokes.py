"""
Navier-Stokes Solver for FLUX Scientific Computing
Solves incompressible fluid flow equations with projection methods
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
import warnings


class NavierStokesSolver:
    """
    Solver for 2D incompressible Navier-Stokes equations:

    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    ∇·u = 0

    Uses fractional step (projection) method for pressure-velocity coupling.
    """

    def __init__(self, nx: int = 64, ny: int = 64,
                 Lx: float = 1.0, Ly: float = 1.0,
                 viscosity: float = 0.01, density: float = 1.0,
                 verbose: bool = False):
        """
        Initialize Navier-Stokes solver

        Parameters:
        -----------
        nx, ny : int
            Grid points in x and y directions
        Lx, Ly : float
            Domain size
        viscosity : float
            Kinematic viscosity ν
        density : float
            Fluid density ρ
        """
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.nu = viscosity  # Kinematic viscosity
        self.rho = density
        self.verbose = verbose

        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        # Coordinate arrays
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Staggered grid arrays
        self._setup_staggered_grid()

        # Velocity and pressure fields
        self.u = np.zeros((ny, nx))      # x-velocity
        self.v = np.zeros((ny, nx))      # y-velocity
        self.p = np.zeros((ny, nx))      # pressure

        # Temporary arrays
        self.u_star = np.zeros((ny, nx))
        self.v_star = np.zeros((ny, nx))

        # Build matrices
        self._build_laplacian_matrix()
        self._build_gradient_matrices()

        if self.verbose:
            print(f"Navier-Stokes solver initialized:")
            print(f"  Grid: {nx}×{ny}")
            print(f"  Domain: {Lx}×{Ly}")
            print(f"  Viscosity: {viscosity}")
            print(f"  Reynolds number (est): {1.0/(viscosity+1e-12):.1f}")

    def _setup_staggered_grid(self):
        """Setup staggered grid coordinates for MAC method"""
        # u-velocity points (face centers in x-direction)
        self.x_u = np.linspace(-self.dx/2, self.Lx + self.dx/2, self.nx + 1)
        self.y_u = self.y

        # v-velocity points (face centers in y-direction)
        self.x_v = self.x
        self.y_v = np.linspace(-self.dy/2, self.Ly + self.dy/2, self.ny + 1)

    def _build_laplacian_matrix(self):
        """Build discrete Laplacian matrix for pressure Poisson equation"""
        n = self.nx * self.ny

        # Coefficients
        cx = 1.0 / (self.dx**2)
        cy = 1.0 / (self.dy**2)
        cc = -2.0 * (cx + cy)

        # Main diagonal
        main_diag = np.full(n, cc)

        # Off-diagonals for x-direction
        east_diag = np.full(n - 1, cx)
        west_diag = np.full(n - 1, cx)

        # Remove connections across boundaries
        for i in range(self.nx - 1, n, self.nx):
            if i < n - 1:
                east_diag[i] = 0.0
        for i in range(self.nx, n, self.nx):
            if i > 0:
                west_diag[i - 1] = 0.0

        # Off-diagonals for y-direction
        north_diag = np.full(n - self.nx, cy)
        south_diag = np.full(n - self.nx, cy)

        # Apply boundary conditions (Neumann for pressure)
        # Bottom boundary
        for i in range(self.nx):
            main_diag[i] -= cy  # Remove south connection

        # Top boundary
        for i in range(n - self.nx, n):
            main_diag[i] -= cy  # Remove north connection

        # Left boundary
        for i in range(0, n, self.nx):
            main_diag[i] -= cx  # Remove west connection

        # Right boundary
        for i in range(self.nx - 1, n, self.nx):
            main_diag[i] -= cx  # Remove east connection

        # Create sparse matrix
        diagonals = [south_diag, west_diag, main_diag, east_diag, north_diag]
        offsets = [-self.nx, -1, 0, 1, self.nx]

        self.L = sp.diags(diagonals, offsets, shape=(n, n), format='csr')

    def _build_gradient_matrices(self):
        """Build gradient matrices for pressure correction"""
        n = self.nx * self.ny

        # x-gradient matrix
        main_x = np.zeros(n)
        east_x = np.full(n - 1, 1.0/self.dx)
        west_x = np.full(n - 1, -1.0/self.dx)

        # Handle boundaries
        for i in range(self.nx - 1, n, self.nx):
            if i < n - 1:
                east_x[i] = 0.0
        for i in range(self.nx, n, self.nx):
            if i > 0:
                west_x[i - 1] = 0.0

        self.Gx = sp.diags([west_x, main_x, east_x], [-1, 0, 1], shape=(n, n), format='csr')

        # y-gradient matrix
        main_y = np.zeros(n)
        north_y = np.full(n - self.nx, 1.0/self.dy)
        south_y = np.full(n - self.nx, -1.0/self.dy)

        self.Gy = sp.diags([south_y, main_y, north_y], [-self.nx, 0, self.nx], shape=(n, n), format='csr')

    def apply_boundary_conditions(self):
        """Apply boundary conditions for lid-driven cavity"""
        # No-slip walls (u = v = 0)
        self.u[0, :] = 0      # Bottom
        self.u[-1, :] = 1.0   # Top (lid velocity)
        self.u[:, 0] = 0      # Left
        self.u[:, -1] = 0     # Right

        self.v[0, :] = 0      # Bottom
        self.v[-1, :] = 0     # Top
        self.v[:, 0] = 0      # Left
        self.v[:, -1] = 0     # Right

    def compute_convection_term(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute convection terms (u·∇)u using upwind differencing
        Returns: (conv_u, conv_v)
        """
        conv_u = np.zeros_like(u)
        conv_v = np.zeros_like(v)

        # u-momentum: u∂u/∂x + v∂u/∂y
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # u∂u/∂x term
                if u[j, i] > 0:
                    dudx = (u[j, i] - u[j, i-1]) / self.dx
                else:
                    dudx = (u[j, i+1] - u[j, i]) / self.dx

                # v∂u/∂y term
                v_avg = 0.25 * (v[j, i] + v[j+1, i] + v[j, i-1] + v[j+1, i-1])
                if v_avg > 0:
                    dudy = (u[j, i] - u[j-1, i]) / self.dy
                else:
                    dudy = (u[j+1, i] - u[j, i]) / self.dy

                conv_u[j, i] = u[j, i] * dudx + v_avg * dudy

        # v-momentum: u∂v/∂x + v∂v/∂y
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # u∂v/∂x term
                u_avg = 0.25 * (u[j, i] + u[j, i+1] + u[j-1, i] + u[j-1, i+1])
                if u_avg > 0:
                    dvdx = (v[j, i] - v[j, i-1]) / self.dx
                else:
                    dvdx = (v[j, i+1] - v[j, i]) / self.dx

                # v∂v/∂y term
                if v[j, i] > 0:
                    dvdy = (v[j, i] - v[j-1, i]) / self.dy
                else:
                    dvdy = (v[j+1, i] - v[j, i]) / self.dy

                conv_v[j, i] = u_avg * dvdx + v[j, i] * dvdy

        return conv_u, conv_v

    def compute_diffusion_term(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute viscous diffusion terms ν∇²u
        Returns: (diff_u, diff_v)
        """
        diff_u = np.zeros_like(u)
        diff_v = np.zeros_like(v)

        # Interior points using 5-point stencil
        diff_u[1:-1, 1:-1] = self.nu * (
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (self.dx**2) +
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (self.dy**2)
        )

        diff_v[1:-1, 1:-1] = self.nu * (
            (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / (self.dx**2) +
            (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / (self.dy**2)
        )

        return diff_u, diff_v

    def solve_pressure_poisson(self, div_u_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve pressure Poisson equation:
        ∇²p = ρ/dt * ∇·u*
        """
        # Right-hand side
        rhs = (self.rho / dt) * div_u_star.flatten()

        # Solve linear system
        try:
            p_flat = spsolve(self.L, rhs)
            return p_flat.reshape((self.ny, self.nx))
        except:
            warnings.warn("Pressure solve failed, using zeros")
            return np.zeros((self.ny, self.nx))

    def compute_divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute velocity divergence ∇·u"""
        div = np.zeros((self.ny, self.nx))

        div[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2*self.dx) +
            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*self.dy)
        )

        return div

    def time_step(self, dt: float) -> Dict[str, float]:
        """
        Perform one time step using fractional step method

        1. Predictor step: solve momentum without pressure
        2. Pressure correction: solve Poisson equation
        3. Velocity correction: project to divergence-free space
        """

        # Step 1: Compute convection and diffusion terms
        conv_u, conv_v = self.compute_convection_term(self.u, self.v)
        diff_u, diff_v = self.compute_diffusion_term(self.u, self.v)

        # Step 2: Predictor step (momentum without pressure gradient)
        self.u_star = self.u + dt * (-conv_u + diff_u)
        self.v_star = self.v + dt * (-conv_v + diff_v)

        # Apply boundary conditions to u*, v*
        self.u_star[0, :] = 0     # Bottom
        self.u_star[-1, :] = 1.0  # Top (lid)
        self.u_star[:, 0] = 0     # Left
        self.u_star[:, -1] = 0    # Right

        self.v_star[0, :] = 0     # Bottom
        self.v_star[-1, :] = 0    # Top
        self.v_star[:, 0] = 0     # Left
        self.v_star[:, -1] = 0    # Right

        # Step 3: Solve pressure Poisson equation
        div_u_star = self.compute_divergence(self.u_star, self.v_star)
        dp = self.solve_pressure_poisson(div_u_star, dt)

        # Step 4: Velocity correction
        # Compute pressure gradients
        dpdx = np.zeros_like(dp)
        dpdy = np.zeros_like(dp)

        dpdx[1:-1, 1:-1] = (dp[1:-1, 2:] - dp[1:-1, :-2]) / (2*self.dx)
        dpdy[1:-1, 1:-1] = (dp[2:, 1:-1] - dp[:-2, 1:-1]) / (2*self.dy)

        # Correct velocities
        self.u = self.u_star - (dt/self.rho) * dpdx
        self.v = self.v_star - (dt/self.rho) * dpdy

        # Update pressure
        self.p += dp

        # Final boundary conditions
        self.apply_boundary_conditions()

        # Compute diagnostics
        div_final = self.compute_divergence(self.u, self.v)
        max_div = np.max(np.abs(div_final))
        max_vel = np.sqrt(np.max(self.u**2 + self.v**2))

        return {
            'max_divergence': max_div,
            'max_velocity': max_vel,
            'max_pressure': np.max(np.abs(self.p))
        }

    def solve(self, time_final: float, dt: Optional[float] = None,
              save_interval: int = 10, cfl_target: float = 0.5) -> Dict[str, Any]:
        """
        Solve Navier-Stokes equations until time_final

        Parameters:
        -----------
        time_final : float
            Final simulation time
        dt : float, optional
            Time step (computed automatically if None)
        save_interval : int
            Save solutions every N steps
        cfl_target : float
            Target CFL number for automatic time step
        """

        # Auto-compute time step based on CFL condition
        if dt is None:
            max_vel = max(1.0, np.sqrt(np.max(self.u**2 + self.v**2)))  # Include lid velocity
            dt_cfl = cfl_target * min(self.dx, self.dy) / max_vel
            dt_visc = cfl_target * min(self.dx, self.dy)**2 / (4 * self.nu) if self.nu > 0 else dt_cfl
            dt = min(dt_cfl, dt_visc)

        if self.verbose:
            print(f"Auto-computed dt = {dt:.6f}")
            print(f"CFL number: {dt * 1.0 / min(self.dx, self.dy):.3f}")
            print(f"Viscous number: {self.nu * dt / min(self.dx, self.dy)**2:.3f}")

        n_steps = int(np.ceil(time_final / dt))
        actual_dt = time_final / n_steps

        if self.verbose:
            print(f"Solving for {n_steps} steps, dt = {actual_dt:.6f}")

        # Storage
        time_history = [0.0]
        u_history = [self.u.copy()]
        v_history = [self.v.copy()]
        p_history = [self.p.copy()]
        diagnostics = []

        # Apply initial boundary conditions
        self.apply_boundary_conditions()

        # Time stepping
        t = 0.0
        for step in range(n_steps):
            # Advance one time step
            diag = self.time_step(actual_dt)
            t += actual_dt

            # Save results
            if step % save_interval == 0:
                time_history.append(t)
                u_history.append(self.u.copy())
                v_history.append(self.v.copy())
                p_history.append(self.p.copy())
                diagnostics.append(diag)

                if self.verbose:
                    print(f"Step {step:4d}/{n_steps}, t={t:.4f}, "
                          f"max|∇·u|={diag['max_divergence']:.2e}, "
                          f"max|u|={diag['max_velocity']:.4f}")

        if self.verbose:
            print(f"Simulation complete! Final time: {t:.4f}")

        return {
            'time': np.array(time_history),
            'u_history': u_history,
            'v_history': v_history,
            'p_history': p_history,
            'diagnostics': diagnostics,
            'final_u': self.u,
            'final_v': self.v,
            'final_p': self.p,
            'mesh': (self.X, self.Y),
            'dt_used': actual_dt
        }

    def plot_solution(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot velocity field and pressure"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Velocity magnitude
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        im1 = axes[0].contourf(self.X, self.Y, vel_mag, levels=20, cmap='viridis')
        axes[0].streamplot(self.X, self.Y, self.u, self.v, density=1, color='white',
                          linewidth=0.5, alpha=0.7)
        axes[0].set_title('Velocity Magnitude |u|')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])

        # Pressure
        im2 = axes[1].contourf(self.X, self.Y, self.p, levels=20, cmap='RdBu_r')
        axes[1].set_title('Pressure p')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])

        # Vorticity
        vorticity = np.zeros_like(self.u)
        vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2*self.dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2*self.dy)
        )
        im3 = axes[2].contourf(self.X, self.Y, vorticity, levels=20, cmap='RdBu_r')
        axes[2].set_title('Vorticity ∇×u')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()


def lid_driven_cavity_demo():
    """Demonstrate lid-driven cavity flow"""
    print("🌊 Navier-Stokes Solver Demo: Lid-Driven Cavity")
    print("=" * 50)

    # Create solver
    solver = NavierStokesSolver(nx=64, ny=64, viscosity=0.01, verbose=True)

    # Solve
    result = solver.solve(time_final=10.0, save_interval=20)

    # Plot results
    solver.plot_solution()

    # Show convergence
    if result['diagnostics']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        times = result['time'][1:]  # Skip initial
        max_divs = [d['max_divergence'] for d in result['diagnostics']]
        max_vels = [d['max_velocity'] for d in result['diagnostics']]

        axes[0].semilogy(times, max_divs, 'b-', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Max Divergence |∇·u|')
        axes[0].set_title('Mass Conservation')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, max_vels, 'r-', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Max Velocity |u|')
        axes[1].set_title('Flow Development')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("✅ Lid-driven cavity simulation complete!")
    return result


if __name__ == "__main__":
    lid_driven_cavity_demo()