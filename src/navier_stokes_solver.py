"""
Navier-Stokes Solver for FLUX
Implements incompressible fluid flow solver using projection method
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib import animation
import time


@dataclass
class NavierStokesSolver:
    """
    Solver for 2D incompressible Navier-Stokes equations:
    ∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v
    ∇·v = 0
    """

    nx: int = 50
    ny: int = 50
    Lx: float = 1.0
    Ly: float = 1.0
    dt: float = 0.001
    nu: float = 0.01  # Kinematic viscosity
    rho: float = 1.0  # Density

    def __post_init__(self):
        """Initialize grid and variables"""
        # Grid spacing
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny

        # Staggered grid for velocity components
        # u is defined at (i+1/2, j) - horizontal faces
        # v is defined at (i, j+1/2) - vertical faces
        # p is defined at (i, j) - cell centers

        self.u = np.zeros((self.ny, self.nx + 1))  # x-velocity
        self.v = np.zeros((self.ny + 1, self.nx))  # y-velocity
        self.p = np.zeros((self.ny, self.nx))      # pressure

        # Intermediate velocity fields
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)

        # Grid coordinates
        self.x_u = np.linspace(0, self.Lx, self.nx + 1)
        self.y_u = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)

        self.x_v = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        self.y_v = np.linspace(0, self.Ly, self.ny + 1)

        self.x_p = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        self.y_p = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)

        # Create mesh grids
        self.X_u, self.Y_u = np.meshgrid(self.x_u, self.y_u)
        self.X_v, self.Y_v = np.meshgrid(self.x_v, self.y_v)
        self.X_p, self.Y_p = np.meshgrid(self.x_p, self.y_p)

        # Setup Poisson solver for pressure
        self._setup_pressure_solver()

        # Reynolds number
        self.Re = self.Lx * 1.0 / self.nu
        print(f"Reynolds number: Re = {self.Re:.1f}")

    def _setup_pressure_solver(self):
        """Setup sparse matrix for pressure Poisson equation"""
        n = self.nx * self.ny

        # Laplacian matrix for pressure (Neumann boundary conditions)
        main_diag = -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(n)
        x_off_diag = (1/self.dx**2) * np.ones(n - 1)
        y_off_diag = (1/self.dy**2) * np.ones(n - self.nx)

        # Handle periodic boundaries in x-direction (optional)
        for i in range(self.nx - 1, n, self.nx):
            if i < n - 1:
                x_off_diag[i] = 0

        # Build sparse matrix
        self.L_p = sp.diags(
            [y_off_diag, x_off_diag, main_diag, x_off_diag, y_off_diag],
            [-self.nx, -1, 0, 1, self.nx],
            shape=(n, n),
            format='csr'
        )

        # For Neumann BC, fix pressure at one point to remove singularity
        self.L_p = self.L_p.tolil()
        self.L_p[0, :] = 0
        self.L_p[0, 0] = 1
        self.L_p = self.L_p.tocsr()

    def set_lid_driven_cavity_bc(self, u_lid: float = 1.0):
        """Set boundary conditions for lid-driven cavity"""
        # Top lid moves with velocity u_lid
        self.u[-1, 1:-1] = u_lid

        # No-slip on other walls
        self.u[0, :] = 0    # Bottom
        self.u[:, 0] = 0    # Left
        self.u[:, -1] = 0  # Right

        self.v[0, :] = 0    # Bottom
        self.v[-1, :] = 0   # Top
        self.v[:, 0] = 0    # Left walls
        self.v[:, -1] = 0   # Right

    def compute_divergence(self) -> np.ndarray:
        """Compute divergence of velocity field"""
        div = np.zeros((self.ny, self.nx))

        # ∇·v = ∂u/∂x + ∂v/∂y
        div = ((self.u[:, 1:] - self.u[:, :-1]) / self.dx +
               (self.v[1:, :] - self.v[:-1, :]) / self.dy)

        return div

    def advection_term_u(self) -> np.ndarray:
        """Compute advection term for u-momentum: (v·∇)u"""
        adv_u = np.zeros_like(self.u)

        # Interior points
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx):
                # Interpolate velocities to u-grid point
                u_e = self.u[j, i]
                u_w = self.u[j, i-1]
                u_ee = self.u[j, i+1] if i < self.nx - 1 else u_e

                v_n = 0.25 * (self.v[j+1, i-1] + self.v[j+1, i] +
                             self.v[j, i-1] + self.v[j, i])

                # Compute derivatives using upwind scheme
                dudx = (u_e - u_w) / self.dx if u_e > 0 else (u_ee - u_e) / self.dx
                dudy = (self.u[j+1, i] - self.u[j, i]) / self.dy if v_n > 0 else \
                       (self.u[j, i] - self.u[j-1, i]) / self.dy

                adv_u[j, i] = u_e * dudx + v_n * dudy

        return adv_u

    def advection_term_v(self) -> np.ndarray:
        """Compute advection term for v-momentum: (v·∇)v"""
        adv_v = np.zeros_like(self.v)

        # Interior points
        for j in range(1, self.ny):
            for i in range(1, self.nx - 1):
                # Interpolate velocities to v-grid point
                v_n = self.v[j, i]
                v_s = self.v[j-1, i]
                v_nn = self.v[j+1, i] if j < self.ny - 1 else v_n

                u_e = 0.25 * (self.u[j-1, i+1] + self.u[j, i+1] +
                             self.u[j-1, i] + self.u[j, i])

                # Compute derivatives using upwind scheme
                dvdx = (self.v[j, i+1] - self.v[j, i]) / self.dx if u_e > 0 else \
                       (self.v[j, i] - self.v[j, i-1]) / self.dx
                dvdy = (v_n - v_s) / self.dy if v_n > 0 else (v_nn - v_n) / self.dy

                adv_v[j, i] = u_e * dvdx + v_n * dvdy

        return adv_v

    def diffusion_term_u(self) -> np.ndarray:
        """Compute diffusion term for u-momentum: ν∇²u"""
        diff_u = np.zeros_like(self.u)

        # Interior points
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx):
                d2udx2 = (self.u[j, i-1] - 2*self.u[j, i] +
                         (self.u[j, i+1] if i < self.nx - 1 else 0)) / self.dx**2
                d2udy2 = (self.u[j-1, i] - 2*self.u[j, i] + self.u[j+1, i]) / self.dy**2

                diff_u[j, i] = self.nu * (d2udx2 + d2udy2)

        return diff_u

    def diffusion_term_v(self) -> np.ndarray:
        """Compute diffusion term for v-momentum: ν∇²v"""
        diff_v = np.zeros_like(self.v)

        # Interior points
        for j in range(1, self.ny):
            for i in range(1, self.nx - 1):
                d2vdx2 = (self.v[j, i-1] - 2*self.v[j, i] + self.v[j, i+1]) / self.dx**2
                d2vdy2 = (self.v[j-1, i] - 2*self.v[j, i] +
                         (self.v[j+1, i] if j < self.ny - 1 else 0)) / self.dy**2

                diff_v[j, i] = self.nu * (d2vdx2 + d2vdy2)

        return diff_v

    def projection_step(self):
        """
        Projection method (Chorin's splitting):
        1. Compute intermediate velocity without pressure
        2. Solve pressure Poisson equation
        3. Correct velocity with pressure gradient
        """

        # Step 1: Compute intermediate velocity (u*, v*)
        # u* = u^n + dt * (-advection + diffusion)
        adv_u = self.advection_term_u()
        diff_u = self.diffusion_term_u()
        self.u_star = self.u + self.dt * (-adv_u + diff_u)

        adv_v = self.advection_term_v()
        diff_v = self.diffusion_term_v()
        self.v_star = self.v + self.dt * (-adv_v + diff_v)

        # Apply boundary conditions to intermediate velocity
        self.set_lid_driven_cavity_bc()
        self.u_star[0, :] = 0
        self.u_star[:, 0] = 0
        self.u_star[:, -1] = 0

        self.v_star[0, :] = 0
        self.v_star[-1, :] = 0
        self.v_star[:, 0] = 0
        self.v_star[:, -1] = 0

        # Step 2: Solve pressure Poisson equation
        # ∇²p = (ρ/dt) * ∇·v*
        div_star = ((self.u_star[:, 1:] - self.u_star[:, :-1]) / self.dx +
                    (self.v_star[1:, :] - self.v_star[:-1, :]) / self.dy)

        rhs = (self.rho / self.dt) * div_star.flatten()
        rhs[0] = 0  # Fix pressure at one point

        p_flat = spla.spsolve(self.L_p, rhs)
        self.p = p_flat.reshape((self.ny, self.nx))

        # Step 3: Pressure correction
        # u^(n+1) = u* - (dt/ρ) * ∂p/∂x
        # v^(n+1) = v* - (dt/ρ) * ∂p/∂y

        # Pressure gradient at u-points
        for j in range(self.ny):
            for i in range(1, self.nx):
                dpdx = (self.p[j, i] - self.p[j, i-1]) / self.dx
                self.u[j, i] = self.u_star[j, i] - (self.dt / self.rho) * dpdx

        # Pressure gradient at v-points
        for j in range(1, self.ny):
            for i in range(self.nx):
                dpdy = (self.p[j, i] - self.p[j-1, i]) / self.dy
                self.v[j, i] = self.v_star[j, i] - (self.dt / self.rho) * dpdy

        # Reapply boundary conditions
        self.set_lid_driven_cavity_bc()

    def solve(self, t_end: float, plot_interval: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Navier-Stokes equations"""
        n_steps = int(t_end / self.dt)

        print(f"Solving 2D Navier-Stokes equations:")
        print(f"  Grid: {self.nx}×{self.ny}")
        print(f"  Time steps: {n_steps}")
        print(f"  Final time: {t_end}")
        print(f"  Reynolds number: {self.Re:.1f}")

        # Check CFL condition
        cfl_u = self.dt / self.dx
        cfl_v = self.dt / self.dy
        cfl_diffusion = self.nu * self.dt / min(self.dx**2, self.dy**2)
        print(f"  CFL numbers: u={cfl_u:.3f}, v={cfl_v:.3f}, diffusion={cfl_diffusion:.3f}")

        start_time = time.time()

        for step in range(n_steps):
            # Time integration using projection method
            self.projection_step()

            # Check divergence (should be near zero)
            if step % max(1, n_steps // 10) == 0:
                div = self.compute_divergence()
                max_div = np.max(np.abs(div))
                max_u = np.max(np.abs(self.u))
                max_v = np.max(np.abs(self.v))

                t = step * self.dt
                print(f"  Step {step:5d}/{n_steps}, t={t:.3f}, "
                      f"max|div|={max_div:.2e}, max|u|={max_u:.3f}, max|v|={max_v:.3f}")

                if plot_interval > 0 and step % plot_interval == 0:
                    self.plot_flow(title=f"Flow at t={t:.3f}")

        elapsed = time.time() - start_time
        print(f"Solution completed in {elapsed:.2f} seconds")
        print(f"Performance: {n_steps / elapsed:.1f} steps/second")

        return self.u, self.v

    def compute_vorticity(self) -> np.ndarray:
        """Compute vorticity ω = ∂v/∂x - ∂u/∂y"""
        omega = np.zeros((self.ny - 1, self.nx - 1))

        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                dvdx = (self.v[j+1, i+1] + self.v[j, i+1] -
                       self.v[j+1, i] - self.v[j, i]) / (2 * self.dx)
                dudy = (self.u[j+1, i+1] + self.u[j+1, i] -
                       self.u[j, i+1] - self.u[j, i]) / (2 * self.dy)
                omega[j, i] = dvdx - dudy

        return omega

    def compute_stream_function(self) -> np.ndarray:
        """Compute stream function from vorticity"""
        omega = self.compute_vorticity()

        # Solve Poisson equation: ∇²ψ = -ω
        n = (self.nx - 1) * (self.ny - 1)

        # Setup Laplacian for stream function
        main_diag = -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(n)
        x_off_diag = (1/self.dx**2) * np.ones(n - 1)
        y_off_diag = (1/self.dy**2) * np.ones(n - (self.nx - 1))

        L_psi = sp.diags(
            [y_off_diag, x_off_diag, main_diag, x_off_diag, y_off_diag],
            [-(self.nx - 1), -1, 0, 1, (self.nx - 1)],
            shape=(n, n),
            format='csr'
        )

        # Fix stream function at boundaries (ψ = 0)
        rhs = -omega.flatten()

        psi_flat = spla.spsolve(L_psi, rhs)
        psi = psi_flat.reshape((self.ny - 1, self.nx - 1))

        return psi

    def plot_flow(self, title: str = "Fluid Flow"):
        """Visualize flow field"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Interpolate velocities to cell centers for plotting
        u_center = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        v_center = 0.5 * (self.v[:-1, :] + self.v[1:, :])

        # 1. Velocity magnitude
        vel_mag = np.sqrt(u_center**2 + v_center**2)
        im1 = axes[0, 0].contourf(self.X_p, self.Y_p, vel_mag, levels=20, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Streamlines
        axes[0, 1].streamplot(self.x_p, self.y_p, u_center, v_center,
                             density=1.5, color=vel_mag, cmap='viridis')
        axes[0, 1].set_title('Streamlines')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_aspect('equal')

        # 3. Vorticity
        omega = self.compute_vorticity()
        X_omega = self.x_p[:-1]
        Y_omega = self.y_p[:-1]
        im3 = axes[1, 0].contourf(X_omega, Y_omega, omega, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('Vorticity')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])

        # 4. Pressure
        im4 = axes[1, 1].contourf(self.X_p, self.Y_p, self.p, levels=20, cmap='coolwarm')
        axes[1, 1].set_title('Pressure')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    """Demonstrate Navier-Stokes solver with lid-driven cavity"""
    print("=" * 60)
    print("FLUX CFD - Lid-Driven Cavity Flow")
    print("=" * 60)

    # Create solver
    solver = NavierStokesSolver(
        nx=50, ny=50,
        Lx=1.0, Ly=1.0,
        dt=0.001,
        nu=0.01  # Re = 100
    )

    # Set boundary conditions
    solver.set_lid_driven_cavity_bc(u_lid=1.0)

    # Solve
    t_final = 10.0
    u_final, v_final = solver.solve(t_end=t_final, plot_interval=0)

    # Final visualization
    solver.plot_flow(title=f"Lid-Driven Cavity at Re={solver.Re:.0f}, t={t_final}")

    # Compute and report flow characteristics
    omega = solver.compute_vorticity()
    psi = solver.compute_stream_function()

    print(f"\nFlow characteristics:")
    print(f"  Max vorticity: {np.max(np.abs(omega)):.3f}")
    print(f"  Stream function range: [{np.min(psi):.3e}, {np.max(psi):.3e}]")

    # Check mass conservation
    div = solver.compute_divergence()
    print(f"  Max divergence (mass conservation): {np.max(np.abs(div)):.2e}")

    # Save results
    np.savez("cavity_flow_solution.npz",
             u=u_final, v=v_final, p=solver.p,
             omega=omega, psi=psi)
    print("\nResults saved to cavity_flow_solution.npz")


if __name__ == "__main__":
    main()