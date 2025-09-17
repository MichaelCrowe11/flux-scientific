"""
Finite Difference Solver for FLUX PDEs
Production-ready implementation with validated numerical methods
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Dict, Optional, Callable, Tuple, Any
import warnings


class FiniteDifferenceSolver:
    """
    Finite difference solver for partial differential equations.
    Supports 1D, 2D, and 3D problems with various boundary conditions.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.convergence_history = []
        self.stability_check = True

    def solve_heat_equation(
        self,
        domain: Tuple[float, ...],
        grid_points: Tuple[int, ...],
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        thermal_diffusivity: float = 1.0,
        time_final: float = 1.0,
        dt: Optional[float] = None,
        method: str = 'crank-nicolson',
        return_history: bool = False
    ) -> Dict[str, Any]:
        """
        Solve the heat equation: ∂u/∂t = α∇²u

        Parameters:
        -----------
        domain : tuple
            Domain boundaries ((x_min, x_max), (y_min, y_max), ...)
        grid_points : tuple
            Number of grid points in each dimension (nx, ny, ...)
        initial_condition : np.ndarray
            Initial temperature distribution
        boundary_conditions : dict
            BC specification: {'left': value/function, 'right': value/function, ...}
        thermal_diffusivity : float
            Thermal diffusivity coefficient α
        time_final : float
            Final simulation time
        dt : float, optional
            Time step (auto-calculated if None)
        method : str
            Numerical method: 'explicit', 'implicit', 'crank-nicolson'
        return_history : bool
            Whether to return full time history

        Returns:
        --------
        dict containing:
            - 'solution': final solution
            - 'mesh': coordinate arrays
            - 'time': time points
            - 'history': solution at all time steps (if requested)
            - 'convergence': convergence metrics
            - 'stability': stability analysis
        """

        dim = len(grid_points)

        if dim == 1:
            return self._solve_heat_1d(
                domain[0], grid_points[0], initial_condition,
                boundary_conditions, thermal_diffusivity,
                time_final, dt, method, return_history
            )
        elif dim == 2:
            return self._solve_heat_2d(
                domain, grid_points, initial_condition,
                boundary_conditions, thermal_diffusivity,
                time_final, dt, method, return_history
            )
        elif dim == 3:
            return self._solve_heat_3d(
                domain, grid_points, initial_condition,
                boundary_conditions, thermal_diffusivity,
                time_final, dt, method, return_history
            )
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def _solve_heat_1d(
        self,
        domain: Tuple[float, float],
        nx: int,
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        alpha: float,
        t_final: float,
        dt: Optional[float],
        method: str,
        return_history: bool
    ) -> Dict[str, Any]:
        """Solve 1D heat equation"""

        x_min, x_max = domain
        dx = (x_max - x_min) / (nx - 1)
        x = np.linspace(x_min, x_max, nx)

        # Auto-calculate time step for stability
        if dt is None:
            if method == 'explicit':
                # CFL condition for explicit method
                dt = 0.4 * dx**2 / alpha
                if self.verbose:
                    print(f"Auto-calculated dt={dt:.6f} for stability (CFL)")
            else:
                # Implicit methods are unconditionally stable
                dt = 0.01 * t_final

        # Check stability for explicit method
        if method == 'explicit':
            stability_number = alpha * dt / dx**2
            if stability_number > 0.5:
                warnings.warn(f"Unstable! Stability number {stability_number:.3f} > 0.5")
                if self.stability_check:
                    dt = 0.4 * dx**2 / alpha
                    stability_number = alpha * dt / dx**2
                    if self.verbose:
                        print(f"Adjusted dt to {dt:.6f} for stability")

        nt = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, nt + 1)

        # Initialize solution
        u = initial_condition.copy()
        if return_history:
            history = [u.copy()]
        else:
            history = None

        # Select method
        if method == 'explicit':
            solver_func = self._heat_1d_explicit
        elif method == 'implicit':
            solver_func = self._heat_1d_implicit
        elif method == 'crank-nicolson':
            solver_func = self._heat_1d_crank_nicolson
        else:
            raise ValueError(f"Unknown method: {method}")

        # Time stepping
        for n in range(nt):
            u = solver_func(u, dx, dt, alpha, boundary_conditions)

            if return_history:
                history.append(u.copy())

            # Check convergence
            if n > 0 and n % 100 == 0:
                change = np.max(np.abs(u - history[-2] if history else u))
                self.convergence_history.append(change)
                if self.verbose and n % 500 == 0:
                    print(f"Step {n}/{nt}, t={t[n]:.3f}, max change={change:.6e}")

        return {
            'solution': u,
            'mesh': x,
            'time': t,
            'history': np.array(history) if history else None,
            'convergence': {
                'final_residual': self.convergence_history[-1] if self.convergence_history else None,
                'history': self.convergence_history
            },
            'stability': {
                'method': method,
                'stability_number': alpha * dt / dx**2,
                'dt_used': dt,
                'dx': dx
            }
        }

    def _heat_1d_explicit(
        self,
        u: np.ndarray,
        dx: float,
        dt: float,
        alpha: float,
        bc: Dict[str, Any]
    ) -> np.ndarray:
        """Explicit (Forward Euler) method for 1D heat equation"""
        r = alpha * dt / dx**2
        u_new = u.copy()

        # Interior points
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

        # Boundary conditions
        u_new[0] = self._apply_bc(bc.get('left', 0), u[0], 0, dt)
        u_new[-1] = self._apply_bc(bc.get('right', 0), u[-1], 0, dt)

        return u_new

    def _heat_1d_implicit(
        self,
        u: np.ndarray,
        dx: float,
        dt: float,
        alpha: float,
        bc: Dict[str, Any]
    ) -> np.ndarray:
        """Implicit (Backward Euler) method for 1D heat equation"""
        nx = len(u)
        r = alpha * dt / dx**2

        # Build tridiagonal matrix
        main_diag = np.ones(nx) * (1 + 2*r)
        off_diag = np.ones(nx - 1) * (-r)

        # Boundary conditions
        main_diag[0] = 1
        main_diag[-1] = 1
        off_diag[0] = 0
        off_diag[-2] = 0

        # Build RHS
        b = u.copy()
        b[0] = self._apply_bc(bc.get('left', 0), u[0], 0, dt)
        b[-1] = self._apply_bc(bc.get('right', 0), u[-1], 0, dt)

        # Solve tridiagonal system
        A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(nx, nx))
        u_new = spsolve(A.tocsr(), b)

        return u_new

    def _heat_1d_crank_nicolson(
        self,
        u: np.ndarray,
        dx: float,
        dt: float,
        alpha: float,
        bc: Dict[str, Any]
    ) -> np.ndarray:
        """Crank-Nicolson method for 1D heat equation (2nd order accurate)"""
        nx = len(u)
        r = alpha * dt / (2 * dx**2)

        # Build implicit matrix (LHS)
        main_diag_lhs = np.ones(nx) * (1 + 2*r)
        off_diag_lhs = np.ones(nx - 1) * (-r)

        # Build explicit matrix (RHS)
        main_diag_rhs = np.ones(nx) * (1 - 2*r)
        off_diag_rhs = np.ones(nx - 1) * r

        # Apply explicit part
        b = np.zeros(nx)
        b[1:-1] = main_diag_rhs[1:-1] * u[1:-1] + off_diag_rhs[:-1] * u[:-2] + off_diag_rhs[1:] * u[2:]

        # Boundary conditions
        main_diag_lhs[0] = 1
        main_diag_lhs[-1] = 1
        off_diag_lhs[0] = 0
        off_diag_lhs[-2] = 0
        b[0] = self._apply_bc(bc.get('left', 0), u[0], 0, dt)
        b[-1] = self._apply_bc(bc.get('right', 0), u[-1], 0, dt)

        # Solve system
        A = sp.diags([off_diag_lhs, main_diag_lhs, off_diag_lhs], [-1, 0, 1], shape=(nx, nx))
        u_new = spsolve(A.tocsr(), b)

        return u_new

    def _solve_heat_2d(
        self,
        domain: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_points: Tuple[int, int],
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        alpha: float,
        t_final: float,
        dt: Optional[float],
        method: str,
        return_history: bool
    ) -> Dict[str, Any]:
        """Solve 2D heat equation using ADI (Alternating Direction Implicit) method"""

        (x_min, x_max), (y_min, y_max) = domain
        nx, ny = grid_points

        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        # Auto-calculate time step
        if dt is None:
            if method == 'explicit':
                dt = 0.2 * min(dx**2, dy**2) / alpha
            else:
                dt = 0.01 * t_final

        nt = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, nt + 1)

        # Initialize solution
        u = initial_condition.reshape(ny, nx)
        if return_history:
            history = [u.copy()]
        else:
            history = None

        # ADI parameters
        rx = alpha * dt / (2 * dx**2)
        ry = alpha * dt / (2 * dy**2)

        # Time stepping with ADI
        for n in range(nt):
            # Step 1: Implicit in x, explicit in y
            u_half = np.zeros_like(u)

            for j in range(ny):
                if j == 0 or j == ny - 1:
                    # Boundary rows
                    u_half[j, :] = self._apply_2d_bc_row(
                        u[j, :], boundary_conditions, 'y', j, ny
                    )
                else:
                    # Interior rows
                    # Build tridiagonal system
                    main_diag = np.ones(nx) * (1 + 2*rx)
                    off_diag = np.ones(nx - 1) * (-rx)

                    # RHS with explicit y-direction
                    b = u[j, :] + ry * (u[j+1, :] - 2*u[j, :] + u[j-1, :])

                    # Apply x-boundary conditions
                    main_diag[0] = 1
                    main_diag[-1] = 1
                    off_diag[0] = 0
                    off_diag[-2] = 0
                    b[0] = self._apply_bc(boundary_conditions.get('left', 0), u[j, 0], 0, dt)
                    b[-1] = self._apply_bc(boundary_conditions.get('right', 0), u[j, -1], 0, dt)

                    # Solve
                    A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(nx, nx))
                    u_half[j, :] = spsolve(A.tocsr(), b)

            # Step 2: Implicit in y, explicit in x
            u_new = np.zeros_like(u)

            for i in range(nx):
                if i == 0 or i == nx - 1:
                    # Boundary columns
                    u_new[:, i] = self._apply_2d_bc_col(
                        u_half[:, i], boundary_conditions, 'x', i, nx
                    )
                else:
                    # Interior columns
                    # Build tridiagonal system
                    main_diag = np.ones(ny) * (1 + 2*ry)
                    off_diag = np.ones(ny - 1) * (-ry)

                    # RHS with explicit x-direction
                    b = u_half[:, i] + rx * (u_half[:, i+1] - 2*u_half[:, i] + u_half[:, i-1])

                    # Apply y-boundary conditions
                    main_diag[0] = 1
                    main_diag[-1] = 1
                    off_diag[0] = 0
                    off_diag[-2] = 0
                    b[0] = self._apply_bc(boundary_conditions.get('bottom', 0), u_half[0, i], 0, dt)
                    b[-1] = self._apply_bc(boundary_conditions.get('top', 0), u_half[-1, i], 0, dt)

                    # Solve
                    A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(ny, ny))
                    u_new[:, i] = spsolve(A.tocsr(), b)

            u = u_new

            if return_history:
                history.append(u.copy())

            # Convergence check
            if n > 0 and n % 50 == 0:
                if history:
                    change = np.max(np.abs(u - history[-2]))
                else:
                    change = np.max(np.abs(u - u_new))
                self.convergence_history.append(change)
                if self.verbose and n % 200 == 0:
                    print(f"Step {n}/{nt}, t={t[n]:.3f}, max change={change:.6e}")

        return {
            'solution': u,
            'mesh': (X, Y),
            'time': t,
            'history': np.array(history) if history else None,
            'convergence': {
                'final_residual': self.convergence_history[-1] if self.convergence_history else None,
                'history': self.convergence_history
            },
            'stability': {
                'method': 'ADI',
                'rx': rx,
                'ry': ry,
                'dt_used': dt,
                'dx': dx,
                'dy': dy
            }
        }

    def _solve_heat_3d(
        self,
        domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        grid_points: Tuple[int, int, int],
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        alpha: float,
        t_final: float,
        dt: Optional[float],
        method: str,
        return_history: bool
    ) -> Dict[str, Any]:
        """Solve 3D heat equation using explicit method (for simplicity)"""

        (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain
        nx, ny, nz = grid_points

        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        dz = (z_max - z_min) / (nz - 1)

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Auto-calculate time step for stability
        if dt is None:
            dt = 0.1 * min(dx**2, dy**2, dz**2) / alpha

        # Stability check
        stability_number = alpha * dt * (1/dx**2 + 1/dy**2 + 1/dz**2)
        if stability_number > 0.5:
            warnings.warn(f"3D stability number {stability_number:.3f} > 0.5, may be unstable")

        nt = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, nt + 1)

        # Initialize solution
        u = initial_condition.reshape(nx, ny, nz)
        if return_history:
            history = [u.copy()]
        else:
            history = None

        # Coefficients
        rx = alpha * dt / dx**2
        ry = alpha * dt / dy**2
        rz = alpha * dt / dz**2

        # Time stepping (explicit method)
        for n in range(nt):
            u_new = u.copy()

            # Interior points
            u_new[1:-1, 1:-1, 1:-1] = u[1:-1, 1:-1, 1:-1] + \
                rx * (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) + \
                ry * (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) + \
                rz * (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2])

            # Apply boundary conditions (simplified - zero flux)
            u_new[0, :, :] = u_new[1, :, :]
            u_new[-1, :, :] = u_new[-2, :, :]
            u_new[:, 0, :] = u_new[:, 1, :]
            u_new[:, -1, :] = u_new[:, -2, :]
            u_new[:, :, 0] = u_new[:, :, 1]
            u_new[:, :, -1] = u_new[:, :, -2]

            u = u_new

            if return_history:
                history.append(u.copy())

            # Convergence check
            if n > 0 and n % 20 == 0:
                if history:
                    change = np.max(np.abs(u - history[-2]))
                else:
                    change = 0
                self.convergence_history.append(change)
                if self.verbose and n % 100 == 0:
                    print(f"Step {n}/{nt}, t={t[n]:.3f}, max change={change:.6e}")

        return {
            'solution': u,
            'mesh': (X, Y, Z),
            'time': t,
            'history': np.array(history) if history else None,
            'convergence': {
                'final_residual': self.convergence_history[-1] if self.convergence_history else None,
                'history': self.convergence_history
            },
            'stability': {
                'method': 'explicit',
                'stability_number': stability_number,
                'dt_used': dt,
                'dx': dx,
                'dy': dy,
                'dz': dz
            }
        }

    def solve_wave_equation(
        self,
        domain: Tuple[float, ...],
        grid_points: Tuple[int, ...],
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        boundary_conditions: Dict[str, Any],
        wave_speed: float = 1.0,
        time_final: float = 1.0,
        dt: Optional[float] = None,
        return_history: bool = False
    ) -> Dict[str, Any]:
        """
        Solve the wave equation: ∂²u/∂t² = c²∇²u

        Uses leapfrog method for time integration.
        """

        dim = len(grid_points)

        if dim == 1:
            return self._solve_wave_1d(
                domain[0], grid_points[0], initial_position, initial_velocity,
                boundary_conditions, wave_speed, time_final, dt, return_history
            )
        elif dim == 2:
            return self._solve_wave_2d(
                domain, grid_points, initial_position, initial_velocity,
                boundary_conditions, wave_speed, time_final, dt, return_history
            )
        else:
            raise NotImplementedError(f"Wave equation in {dim}D not yet implemented")

    def _solve_wave_1d(
        self,
        domain: Tuple[float, float],
        nx: int,
        u0: np.ndarray,
        v0: np.ndarray,
        bc: Dict[str, Any],
        c: float,
        t_final: float,
        dt: Optional[float],
        return_history: bool
    ) -> Dict[str, Any]:
        """Solve 1D wave equation using leapfrog method"""

        x_min, x_max = domain
        dx = (x_max - x_min) / (nx - 1)
        x = np.linspace(x_min, x_max, nx)

        # CFL condition for stability
        if dt is None:
            dt = 0.9 * dx / c  # CFL number < 1 for stability

        CFL = c * dt / dx
        if CFL > 1:
            warnings.warn(f"CFL number {CFL:.3f} > 1, solution may be unstable!")

        nt = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, nt + 1)

        # Initialize
        u_prev = u0.copy()
        u_curr = u0 + dt * v0  # First step using initial velocity

        if return_history:
            history = [u_prev.copy(), u_curr.copy()]
        else:
            history = None

        # Time stepping
        for n in range(2, nt + 1):
            u_next = np.zeros(nx)

            # Interior points (leapfrog scheme)
            u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] +
                           CFL**2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))

            # Boundary conditions
            u_next[0] = self._apply_bc(bc.get('left', 0), u_curr[0], t[n], dt)
            u_next[-1] = self._apply_bc(bc.get('right', 0), u_curr[-1], t[n], dt)

            # Update
            u_prev = u_curr
            u_curr = u_next

            if return_history:
                history.append(u_curr.copy())

        return {
            'solution': u_curr,
            'mesh': x,
            'time': t,
            'history': np.array(history) if history else None,
            'stability': {
                'CFL': CFL,
                'dt_used': dt,
                'dx': dx
            }
        }

    def _solve_wave_2d(
        self,
        domain: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_points: Tuple[int, int],
        u0: np.ndarray,
        v0: np.ndarray,
        bc: Dict[str, Any],
        c: float,
        t_final: float,
        dt: Optional[float],
        return_history: bool
    ) -> Dict[str, Any]:
        """Solve 2D wave equation"""

        (x_min, x_max), (y_min, y_max) = domain
        nx, ny = grid_points

        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        # CFL condition
        if dt is None:
            dt = 0.5 * min(dx, dy) / (c * np.sqrt(2))

        CFL = c * dt * np.sqrt(1/dx**2 + 1/dy**2)
        if CFL > 1:
            warnings.warn(f"2D CFL number {CFL:.3f} > 1, may be unstable!")

        nt = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, nt + 1)

        # Initialize
        u_prev = u0.reshape(ny, nx)
        u_curr = u_prev + dt * v0.reshape(ny, nx)

        if return_history:
            history = [u_prev.copy(), u_curr.copy()]
        else:
            history = None

        # Coefficients
        cx = (c * dt / dx) ** 2
        cy = (c * dt / dy) ** 2

        # Time stepping
        for n in range(2, nt + 1):
            u_next = np.zeros((ny, nx))

            # Interior points
            u_next[1:-1, 1:-1] = (2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                                  cx * (u_curr[1:-1, 2:] - 2*u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) +
                                  cy * (u_curr[2:, 1:-1] - 2*u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]))

            # Boundary conditions (simplified - fixed boundaries)
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0

            # Update
            u_prev = u_curr
            u_curr = u_next

            if return_history:
                history.append(u_curr.copy())

        return {
            'solution': u_curr,
            'mesh': (X, Y),
            'time': t,
            'history': np.array(history) if history else None,
            'stability': {
                'CFL': CFL,
                'dt_used': dt,
                'dx': dx,
                'dy': dy
            }
        }

    def solve_poisson_equation(
        self,
        domain: Tuple[float, ...],
        grid_points: Tuple[int, ...],
        source_term: np.ndarray,
        boundary_conditions: Dict[str, Any],
        tol: float = 1e-6,
        max_iter: int = 10000,
        method: str = 'jacobi'
    ) -> Dict[str, Any]:
        """
        Solve Poisson equation: ∇²u = f

        Uses iterative methods (Jacobi, Gauss-Seidel, SOR).
        """

        dim = len(grid_points)

        if dim == 1:
            return self._solve_poisson_1d(
                domain[0], grid_points[0], source_term,
                boundary_conditions, tol, max_iter
            )
        elif dim == 2:
            return self._solve_poisson_2d(
                domain, grid_points, source_term,
                boundary_conditions, tol, max_iter, method
            )
        else:
            raise NotImplementedError(f"Poisson equation in {dim}D not yet implemented")

    def _solve_poisson_1d(
        self,
        domain: Tuple[float, float],
        nx: int,
        f: np.ndarray,
        bc: Dict[str, Any],
        tol: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Solve 1D Poisson equation using direct method"""

        x_min, x_max = domain
        dx = (x_max - x_min) / (nx - 1)
        x = np.linspace(x_min, x_max, nx)

        # Build linear system
        A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx)) / dx**2
        b = f.copy()

        # Apply boundary conditions
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        A[-1, :] = 0
        A[-1, -1] = 1
        b[0] = self._apply_bc(bc.get('left', 0), 0, 0, 0)
        b[-1] = self._apply_bc(bc.get('right', 0), 0, 0, 0)

        # Solve
        u = spsolve(A.tocsr(), b)

        return {
            'solution': u,
            'mesh': x,
            'iterations': 1,  # Direct method
            'residual': np.linalg.norm(A @ u - b)
        }

    def _solve_poisson_2d(
        self,
        domain: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_points: Tuple[int, int],
        f: np.ndarray,
        bc: Dict[str, Any],
        tol: float,
        max_iter: int,
        method: str
    ) -> Dict[str, Any]:
        """Solve 2D Poisson equation using iterative methods"""

        (x_min, x_max), (y_min, y_max) = domain
        nx, ny = grid_points

        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        # Initialize solution
        u = np.zeros((ny, nx))
        f_2d = f.reshape(ny, nx)

        # Apply initial boundary conditions
        u[0, :] = self._apply_bc(bc.get('bottom', 0), 0, 0, 0)
        u[-1, :] = self._apply_bc(bc.get('top', 0), 0, 0, 0)
        u[:, 0] = self._apply_bc(bc.get('left', 0), 0, 0, 0)
        u[:, -1] = self._apply_bc(bc.get('right', 0), 0, 0, 0)

        # Select solver
        if method == 'jacobi':
            solver_func = self._jacobi_2d
        elif method == 'gauss-seidel':
            solver_func = self._gauss_seidel_2d
        elif method == 'sor':
            solver_func = self._sor_2d
        else:
            raise ValueError(f"Unknown method: {method}")

        # Iterate
        residuals = []
        for iter in range(max_iter):
            u_old = u.copy()

            u = solver_func(u, f_2d, dx, dy, bc)

            # Check convergence
            residual = np.max(np.abs(u - u_old))
            residuals.append(residual)

            if residual < tol:
                if self.verbose:
                    print(f"Converged in {iter + 1} iterations, residual = {residual:.6e}")
                break

            if self.verbose and (iter + 1) % 100 == 0:
                print(f"Iteration {iter + 1}, residual = {residual:.6e}")

        return {
            'solution': u,
            'mesh': (X, Y),
            'iterations': iter + 1,
            'residual': residual,
            'residual_history': residuals
        }

    def _jacobi_2d(
        self,
        u: np.ndarray,
        f: np.ndarray,
        dx: float,
        dy: float,
        bc: Dict[str, Any]
    ) -> np.ndarray:
        """Jacobi iteration for 2D Poisson"""
        ny, nx = u.shape
        u_new = u.copy()

        # Interior points
        u_new[1:-1, 1:-1] = 0.25 * (
            u[1:-1, 2:] + u[1:-1, :-2] +
            u[2:, 1:-1] + u[:-2, 1:-1] -
            dx**2 * f[1:-1, 1:-1]
        )

        return u_new

    def _gauss_seidel_2d(
        self,
        u: np.ndarray,
        f: np.ndarray,
        dx: float,
        dy: float,
        bc: Dict[str, Any]
    ) -> np.ndarray:
        """Gauss-Seidel iteration for 2D Poisson"""
        ny, nx = u.shape

        # Interior points (in-place update)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u[j, i] = 0.25 * (
                    u[j, i+1] + u[j, i-1] +
                    u[j+1, i] + u[j-1, i] -
                    dx**2 * f[j, i]
                )

        return u

    def _sor_2d(
        self,
        u: np.ndarray,
        f: np.ndarray,
        dx: float,
        dy: float,
        bc: Dict[str, Any],
        omega: float = 1.5
    ) -> np.ndarray:
        """Successive Over-Relaxation for 2D Poisson"""
        ny, nx = u.shape

        # Interior points with SOR
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_jacobi = 0.25 * (
                    u[j, i+1] + u[j, i-1] +
                    u[j+1, i] + u[j-1, i] -
                    dx**2 * f[j, i]
                )
                u[j, i] = (1 - omega) * u[j, i] + omega * u_jacobi

        return u

    def _apply_bc(self, bc_value: Any, current_value: float, position: float, time: float) -> float:
        """Apply boundary condition (can be constant or function)"""
        if callable(bc_value):
            return bc_value(position, time)
        else:
            return bc_value

    def _apply_2d_bc_row(
        self,
        row: np.ndarray,
        bc: Dict[str, Any],
        direction: str,
        index: int,
        total: int
    ) -> np.ndarray:
        """Apply boundary conditions to a row"""
        if index == 0:
            return np.full_like(row, self._apply_bc(bc.get('bottom', 0), 0, 0, 0))
        elif index == total - 1:
            return np.full_like(row, self._apply_bc(bc.get('top', 0), 0, 0, 0))
        else:
            return row

    def _apply_2d_bc_col(
        self,
        col: np.ndarray,
        bc: Dict[str, Any],
        direction: str,
        index: int,
        total: int
    ) -> np.ndarray:
        """Apply boundary conditions to a column"""
        if index == 0:
            return np.full_like(col, self._apply_bc(bc.get('left', 0), 0, 0, 0))
        elif index == total - 1:
            return np.full_like(col, self._apply_bc(bc.get('right', 0), 0, 0, 0))
        else:
            return col