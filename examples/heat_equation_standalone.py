#!/usr/bin/env python3
"""
Standalone Heat Equation Solver
This demonstrates what FLUX should generate for the heat equation example
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


@dataclass
class HeatEquationSolver:
    """Solver for 2D heat equation: ∂u/∂t = α∇²u"""

    nx: int = 100
    ny: int = 100
    dx: float = 0.01
    dy: float = 0.01
    dt: float = 0.001
    alpha: float = 0.1  # Thermal diffusivity

    def __post_init__(self):
        # Initialize solution arrays
        self.u = np.zeros((self.ny, self.nx))
        self.u_new = np.zeros_like(self.u)

        # Setup finite difference matrices
        self._setup_matrices()

        # Create coordinate arrays
        self.x = np.linspace(0, 1, self.nx)
        self.y = np.linspace(0, 1, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def _setup_matrices(self):
        """Setup sparse finite difference matrices for efficient solving"""
        n = self.nx * self.ny

        # Main diagonal: -4
        main_diag = -4 * np.ones(n)

        # Off-diagonals for x-direction: +1
        x_off_diag = np.ones(n - 1)
        # Zero out connections across y-boundaries
        for i in range(self.nx - 1, n, self.nx):
            if i < n - 1:
                x_off_diag[i] = 0

        # Off-diagonals for y-direction: +1
        y_off_diag = np.ones(n - self.nx)

        # Create sparse Laplacian matrix
        self.L = sp.diags(
            [y_off_diag, x_off_diag, main_diag, x_off_diag, y_off_diag],
            [-self.nx, -1, 0, 1, self.nx],
            shape=(n, n),
            format='csr'
        )

        # Scale by grid spacing
        self.L = self.L / (self.dx ** 2)

        print(f"Created sparse Laplacian matrix: {n}×{n} with {self.L.nnz} non-zeros")

    def apply_initial_condition(self, func):
        """Apply initial condition u(x,y,0) = func(x,y)"""
        self.u = func(self.X, self.Y)
        print(f"Applied initial condition. Max value: {np.max(self.u):.6f}")

    def apply_boundary_conditions(self):
        """Apply homogeneous Dirichlet boundary conditions"""
        self.u[0, :] = 0    # Bottom boundary
        self.u[-1, :] = 0   # Top boundary
        self.u[:, 0] = 0    # Left boundary
        self.u[:, -1] = 0   # Right boundary

    def time_step_explicit(self):
        """Explicit Euler time step (unstable for large dt)"""
        u_flat = self.u.flatten()
        dudt_flat = self.alpha * self.L.dot(u_flat)

        u_new_flat = u_flat + self.dt * dudt_flat
        self.u = u_new_flat.reshape((self.ny, self.nx))

        self.apply_boundary_conditions()

    def time_step_implicit(self):
        """Implicit Euler time step (unconditionally stable)"""
        u_flat = self.u.flatten()

        # Solve: (I - dt*α*L) u_new = u_old
        A = sp.eye(len(u_flat)) - self.dt * self.alpha * self.L
        u_new_flat = spla.spsolve(A, u_flat)

        self.u = u_new_flat.reshape((self.ny, self.nx))
        self.apply_boundary_conditions()

    def solve(self, t_end: float, method: str = 'implicit') -> np.ndarray:
        """Solve heat equation until t_end"""
        n_steps = int(t_end / self.dt)

        print(f"Solving 2D heat equation:")
        print(f"  Grid: {self.nx}×{self.ny}")
        print(f"  Time steps: {n_steps}")
        print(f"  Final time: {t_end}")
        print(f"  Method: {method}")
        print(f"  α = {self.alpha}, dt = {self.dt}")

        # Check stability for explicit method
        if method == 'explicit':
            r = self.alpha * self.dt / (self.dx ** 2)
            print(f"  Stability parameter r = {r:.4f} (stable if r ≤ 0.25)")
            if r > 0.25:
                print("  WARNING: Explicit method may be unstable!")

        start_time = time.time()

        for step in range(n_steps):
            if method == 'implicit':
                self.time_step_implicit()
            else:
                self.time_step_explicit()

            # Progress reporting
            if step % max(1, n_steps // 20) == 0:
                t = step * self.dt
                max_val = np.max(self.u)
                energy = np.sum(self.u ** 2) * self.dx * self.dy
                print(f"  Step {step:6d}/{n_steps}, t = {t:.3f}, max = {max_val:.6f}, energy = {energy:.6f}")

        elapsed = time.time() - start_time
        print(f"Solution completed in {elapsed:.2f} seconds")
        print(f"Performance: {n_steps / elapsed:.1f} steps/second")

        return self.u

    def compute_energy(self) -> float:
        """Compute total thermal energy"""
        return np.sum(self.u ** 2) * self.dx * self.dy

    def compute_analytical_solution(self, t: float) -> np.ndarray:
        """Analytical solution for comparison"""
        # For initial condition u(x,y,0) = sin(πx)sin(πy)
        # Analytical solution: u(x,y,t) = exp(-2π²αt) * sin(πx)sin(πy)
        decay_factor = np.exp(-2 * np.pi**2 * self.alpha * t)
        return decay_factor * np.sin(np.pi * self.X) * np.sin(np.pi * self.Y)

    def plot_solution(self, title: str = "Heat Equation Solution"):
        """Plot current solution"""
        plt.figure(figsize=(12, 5))

        # Contour plot
        plt.subplot(1, 2, 1)
        levels = np.linspace(np.min(self.u), np.max(self.u), 20)
        cs = plt.contourf(self.X, self.Y, self.u, levels=levels, cmap='hot')
        plt.colorbar(cs, label='Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{title} - Contour Plot')
        plt.axis('equal')

        # 3D surface plot
        ax = plt.subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(self.X, self.Y, self.u, cmap='hot', alpha=0.8)
        plt.colorbar(surf, label='Temperature', shrink=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Temperature')
        ax.set_title(f'{title} - 3D Surface')

        plt.tight_layout()
        plt.show()


def main():
    """Demonstrate heat equation solver"""
    print("=" * 60)
    print("FLUX Scientific Computing - Heat Equation Demo")
    print("=" * 60)

    # Create solver
    solver = HeatEquationSolver(nx=100, ny=100, dx=0.01, dy=0.01, dt=0.0001, alpha=0.1)

    # Set initial condition: sine wave
    def initial_condition(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    solver.apply_initial_condition(initial_condition)

    # Solve for t=0.1
    t_final = 0.1
    solution = solver.solve(t_end=t_final, method='implicit')

    # Compare with analytical solution
    analytical = solver.compute_analytical_solution(t_final)
    error = np.abs(solution - analytical)

    print(f"\nValidation against analytical solution:")
    print(f"  Max error: {np.max(error):.2e}")
    print(f"  RMS error: {np.sqrt(np.mean(error**2)):.2e}")
    print(f"  Relative error: {np.max(error) / np.max(analytical):.2e}")

    # Plot results
    solver.plot_solution(f"Heat Equation Solution at t = {t_final}")

    # Energy conservation check
    initial_energy = solver.compute_energy()
    print(f"\nEnergy analysis:")
    print(f"  Initial energy: {initial_energy:.6f}")
    print(f"  Final energy: {solver.compute_energy():.6f}")
    print(f"  Energy decay factor: {solver.compute_energy() / initial_energy:.6f}")

    # Save results
    np.save("heat_solution.npy", solution)
    np.save("heat_analytical.npy", analytical)
    print(f"\nResults saved to heat_solution.npy and heat_analytical.npy")


if __name__ == "__main__":
    main()