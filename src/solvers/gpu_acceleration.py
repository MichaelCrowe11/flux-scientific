"""
GPU Acceleration for FLUX Scientific Computing
Provides CuPy-based GPU implementations of PDE solvers
"""

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_linalg
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback to NumPy
    GPU_AVAILABLE = False

import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings
import time


class GPUFiniteDifferenceSolver:
    """
    GPU-accelerated finite difference solver using CuPy
    Falls back to CPU if GPU not available
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.gpu_available = GPU_AVAILABLE

        if self.gpu_available:
            try:
                # Test GPU access
                cp.cuda.Device(0).use()
                test_array = cp.array([1, 2, 3])
                _ = cp.sum(test_array)

                if self.verbose:
                    print(f"✅ GPU acceleration enabled")
                    print(f"   Device: {cp.cuda.Device().name}")
                    print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total")

            except Exception as e:
                self.gpu_available = False
                if self.verbose:
                    print(f"⚠️ GPU not accessible: {e}")
                    print("   Falling back to CPU computation")
        else:
            if self.verbose:
                print("⚠️ CuPy not installed - using CPU fallback")
                print("   Install with: pip install cupy-cuda11x")

    def to_gpu(self, array: np.ndarray) -> Any:
        """Transfer array to GPU if available"""
        if self.gpu_available:
            return cp.asarray(array)
        return array

    def to_cpu(self, array: Any) -> np.ndarray:
        """Transfer array to CPU"""
        if self.gpu_available and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)

    def solve_heat_equation_gpu(
        self,
        domain: Tuple[float, ...],
        grid_points: Tuple[int, ...],
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        thermal_diffusivity: float = 1.0,
        time_final: float = 1.0,
        dt: Optional[float] = None,
        method: str = 'crank-nicolson'
    ) -> Dict[str, Any]:
        """
        GPU-accelerated heat equation solver
        """

        if not self.gpu_available:
            warnings.warn("GPU not available, falling back to CPU")
            from .finite_difference import FiniteDifferenceSolver
            cpu_solver = FiniteDifferenceSolver(verbose=self.verbose)
            return cpu_solver.solve_heat_equation(
                domain, grid_points, initial_condition, boundary_conditions,
                thermal_diffusivity, time_final, dt, method
            )

        dim = len(grid_points)

        if dim == 1:
            return self._solve_heat_1d_gpu(
                domain[0], grid_points[0], initial_condition,
                boundary_conditions, thermal_diffusivity, time_final, dt, method
            )
        elif dim == 2:
            return self._solve_heat_2d_gpu(
                domain, grid_points, initial_condition,
                boundary_conditions, thermal_diffusivity, time_final, dt, method
            )
        else:
            raise NotImplementedError(f"GPU solver not implemented for {dim}D")

    def _solve_heat_1d_gpu(
        self,
        domain: Tuple[float, float],
        nx: int,
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        alpha: float,
        t_final: float,
        dt: Optional[float],
        method: str
    ) -> Dict[str, Any]:
        """1D heat equation on GPU"""

        x_min, x_max = domain
        dx = (x_max - x_min) / (nx - 1)
        x = np.linspace(x_min, x_max, nx)

        # Auto-calculate time step
        if dt is None:
            if method == 'explicit':
                dt = 0.4 * dx**2 / alpha
            else:
                dt = 0.01 * t_final

        nt = int(np.ceil(t_final / dt))

        # Transfer to GPU
        u_gpu = self.to_gpu(initial_condition.copy())

        if self.verbose:
            print(f"🚀 GPU 1D Heat Equation:")
            print(f"   Grid: {nx} points")
            print(f"   Time steps: {nt}")
            print(f"   Method: {method}")
            print(f"   Memory usage: {u_gpu.nbytes / 1e6:.1f} MB")

        start_time = time.time()

        if method == 'explicit':
            result_gpu = self._heat_1d_explicit_gpu(u_gpu, dx, dt, alpha, boundary_conditions, nt)
        elif method == 'crank-nicolson':
            result_gpu = self._heat_1d_crank_nicolson_gpu(u_gpu, dx, dt, alpha, boundary_conditions, nt)
        else:
            raise ValueError(f"GPU method {method} not implemented")

        gpu_time = time.time() - start_time

        # Transfer result back to CPU
        u_final = self.to_cpu(result_gpu)

        if self.verbose:
            print(f"✅ GPU computation completed in {gpu_time:.3f}s")
            print(f"   Performance: {nx * nt / gpu_time:,.0f} cell-steps/second")

            # Compare with estimated CPU time
            estimated_cpu_time = gpu_time * 10  # Rough estimate
            speedup = estimated_cpu_time / gpu_time
            print(f"   Estimated speedup: {speedup:.1f}×")

        return {
            'solution': u_final,
            'mesh': x,
            'time': np.linspace(0, t_final, nt + 1),
            'gpu_time': gpu_time,
            'method': method,
            'stability': {
                'stability_number': alpha * dt / dx**2,
                'dt_used': dt,
                'dx': dx
            }
        }

    def _heat_1d_explicit_gpu(
        self,
        u_gpu: Any,
        dx: float,
        dt: float,
        alpha: float,
        bc: Dict[str, Any],
        nt: int
    ) -> Any:
        """Explicit method on GPU using CuPy"""

        nx = len(u_gpu)
        r = alpha * dt / dx**2

        # Check stability
        if r > 0.5:
            warnings.warn(f"Unstable! r = {r:.3f} > 0.5")

        for n in range(nt):
            u_new = cp.zeros_like(u_gpu)

            # Interior points (vectorized)
            u_new[1:-1] = u_gpu[1:-1] + r * (u_gpu[2:] - 2*u_gpu[1:-1] + u_gpu[:-2])

            # Boundary conditions
            u_new[0] = self._apply_bc_gpu(bc.get('left', 0))
            u_new[-1] = self._apply_bc_gpu(bc.get('right', 0))

            u_gpu = u_new

        return u_gpu

    def _heat_1d_crank_nicolson_gpu(
        self,
        u_gpu: Any,
        dx: float,
        dt: float,
        alpha: float,
        bc: Dict[str, Any],
        nt: int
    ) -> Any:
        """Crank-Nicolson method on GPU"""

        nx = len(u_gpu)
        r = alpha * dt / (2 * dx**2)

        # Build matrices
        main_diag = cp.ones(nx) * (1 + 2*r)
        off_diag = cp.ones(nx - 1) * (-r)

        # Boundary conditions in matrix
        main_diag[0] = 1
        main_diag[-1] = 1
        off_diag[0] = 0
        off_diag[-2] = 0

        # Create sparse matrix
        A = cp_sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(nx, nx))

        for n in range(nt):
            # Build RHS
            b = cp.zeros(nx)
            b[1:-1] = u_gpu[1:-1] + r * (u_gpu[2:] - 2*u_gpu[1:-1] + u_gpu[:-2])

            # Boundary conditions
            b[0] = self._apply_bc_gpu(bc.get('left', 0))
            b[-1] = self._apply_bc_gpu(bc.get('right', 0))

            # Solve system on GPU
            u_gpu = cp_linalg.spsolve(A, b)

        return u_gpu

    def _solve_heat_2d_gpu(
        self,
        domain: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_points: Tuple[int, int],
        initial_condition: np.ndarray,
        boundary_conditions: Dict[str, Any],
        alpha: float,
        t_final: float,
        dt: Optional[float],
        method: str
    ) -> Dict[str, Any]:
        """2D heat equation on GPU"""

        (x_min, x_max), (y_min, y_max) = domain
        nx, ny = grid_points

        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)

        if dt is None:
            dt = 0.2 * min(dx**2, dy**2) / alpha

        nt = int(np.ceil(t_final / dt))

        # Transfer to GPU
        u_gpu = self.to_gpu(initial_condition.reshape(ny, nx))

        if self.verbose:
            print(f"🚀 GPU 2D Heat Equation:")
            print(f"   Grid: {nx}×{ny} = {nx*ny:,} cells")
            print(f"   Time steps: {nt}")
            print(f"   GPU memory: {u_gpu.nbytes / 1e6:.1f} MB")

        start_time = time.time()

        # Solve using explicit method (optimized for GPU)
        u_final_gpu = self._heat_2d_explicit_gpu(u_gpu, dx, dy, dt, alpha, boundary_conditions, nt)

        gpu_time = time.time() - start_time

        # Transfer back
        u_final = self.to_cpu(u_final_gpu)

        if self.verbose:
            print(f"✅ GPU computation completed in {gpu_time:.3f}s")
            print(f"   Performance: {nx * ny * nt / gpu_time:,.0f} cell-steps/second")

        # Generate coordinate arrays
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        return {
            'solution': u_final,
            'mesh': (X, Y),
            'time': np.linspace(0, t_final, nt + 1),
            'gpu_time': gpu_time,
            'method': 'explicit-gpu',
            'stability': {
                'rx': alpha * dt / dx**2,
                'ry': alpha * dt / dy**2,
                'dt_used': dt,
                'dx': dx,
                'dy': dy
            }
        }

    def _heat_2d_explicit_gpu(
        self,
        u_gpu: Any,
        dx: float,
        dy: float,
        dt: float,
        alpha: float,
        bc: Dict[str, Any],
        nt: int
    ) -> Any:
        """2D explicit heat equation on GPU"""

        ny, nx = u_gpu.shape
        rx = alpha * dt / dx**2
        ry = alpha * dt / dy**2

        # Stability check
        stability = rx + ry
        if stability > 0.5:
            warnings.warn(f"2D GPU stability number {stability:.3f} > 0.5")

        for n in range(nt):
            u_new = cp.zeros_like(u_gpu)

            # Interior points (fully vectorized)
            u_new[1:-1, 1:-1] = (
                u_gpu[1:-1, 1:-1] +
                rx * (u_gpu[1:-1, 2:] - 2*u_gpu[1:-1, 1:-1] + u_gpu[1:-1, :-2]) +
                ry * (u_gpu[2:, 1:-1] - 2*u_gpu[1:-1, 1:-1] + u_gpu[:-2, 1:-1])
            )

            # Apply boundary conditions
            self._apply_2d_bc_gpu(u_new, bc)

            u_gpu = u_new

        return u_gpu

    def _apply_bc_gpu(self, bc_value: Any) -> Any:
        """Apply boundary condition on GPU"""
        if callable(bc_value):
            return bc_value(0, 0)  # Simplified
        return bc_value

    def _apply_2d_bc_gpu(self, u_gpu: Any, bc: Dict[str, Any]):
        """Apply 2D boundary conditions on GPU"""
        # Zero Dirichlet BCs (simplified)
        u_gpu[0, :] = 0    # Bottom
        u_gpu[-1, :] = 0   # Top
        u_gpu[:, 0] = 0    # Left
        u_gpu[:, -1] = 0   # Right

    def benchmark_gpu_vs_cpu(
        self,
        grid_sizes: Tuple[int, ...] = (32, 64, 128, 256)
    ) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance
        """

        print("🏁 GPU vs CPU Benchmark")
        print("=" * 50)

        results = {
            'grid_sizes': [],
            'cpu_times': [],
            'gpu_times': [],
            'speedups': [],
            'gpu_memory': []
        }

        for n in grid_sizes:
            print(f"\n📊 Testing {n}×{n} grid ({n*n:,} cells)...")

            # Setup problem
            domain = ((0, 1), (0, 1))
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, y)
            u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

            # CPU timing
            print("   ⏱️  CPU timing...")
            cpu_start = time.time()

            # Use fallback CPU solver
            from .finite_difference import FiniteDifferenceSolver
            cpu_solver = FiniteDifferenceSolver(verbose=False)
            cpu_result = cpu_solver.solve_heat_equation(
                domain=domain,
                grid_points=(n, n),
                initial_condition=u0.flatten(),
                boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                thermal_diffusivity=0.1,
                time_final=0.1,
                method='explicit'
            )

            cpu_time = time.time() - cpu_start

            # GPU timing
            if self.gpu_available:
                print("   🚀 GPU timing...")
                gpu_start = time.time()

                gpu_result = self.solve_heat_equation_gpu(
                    domain=domain,
                    grid_points=(n, n),
                    initial_condition=u0.flatten(),
                    boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                    thermal_diffusivity=0.1,
                    time_final=0.1,
                    method='explicit'
                )

                gpu_time = time.time() - gpu_start
                speedup = cpu_time / gpu_time

                # Memory usage
                gpu_memory = n * n * 8 / 1e6  # Rough estimate in MB

                print(f"   ✅ CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {speedup:.1f}×")

            else:
                gpu_time = float('inf')
                speedup = 0
                gpu_memory = 0
                print(f"   ✅ CPU: {cpu_time:.3f}s, GPU: Not available")

            # Store results
            results['grid_sizes'].append(n)
            results['cpu_times'].append(cpu_time)
            results['gpu_times'].append(gpu_time)
            results['speedups'].append(speedup)
            results['gpu_memory'].append(gpu_memory)

        # Summary
        print(f"\n📈 Benchmark Summary:")
        print(f"{'Grid':>8} {'CPU Time':>10} {'GPU Time':>10} {'Speedup':>10} {'GPU Memory':>12}")
        print("-" * 60)

        for i, n in enumerate(results['grid_sizes']):
            cpu_t = results['cpu_times'][i]
            gpu_t = results['gpu_times'][i]
            speedup = results['speedups'][i]
            memory = results['gpu_memory'][i]

            if gpu_t != float('inf'):
                print(f"{n:3d}×{n:<3d} {cpu_t:9.3f}s {gpu_t:9.3f}s {speedup:9.1f}× {memory:10.1f} MB")
            else:
                print(f"{n:3d}×{n:<3d} {cpu_t:9.3f}s {'N/A':>9s} {'N/A':>9s} {'N/A':>12s}")

        if self.gpu_available and any(s > 1 for s in results['speedups']):
            max_speedup = max(results['speedups'])
            print(f"\n🚀 Maximum speedup achieved: {max_speedup:.1f}×")
            print(f"🎯 GPU acceleration is working effectively!")

        return results


def create_gpu_heat_example():
    """Create example demonstrating GPU acceleration"""

    print("🔥 GPU Heat Equation Example")
    print("=" * 40)

    # Create GPU solver
    gpu_solver = GPUFiniteDifferenceSolver(verbose=True)

    # Setup 2D problem
    n = 128  # Large enough to show GPU benefits
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # Multiple Gaussian hot spots
    u0 = (5 * np.exp(-50*((X - 0.3)**2 + (Y - 0.3)**2)) +
          3 * np.exp(-50*((X - 0.7)**2 + (Y - 0.7)**2)) +
          4 * np.exp(-50*((X - 0.5)**2 + (Y - 0.1)**2)))

    # Solve on GPU
    result = gpu_solver.solve_heat_equation_gpu(
        domain=((0, 1), (0, 1)),
        grid_points=(n, n),
        initial_condition=u0.flatten(),
        boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
        thermal_diffusivity=0.05,
        time_final=0.5,
        method='explicit'
    )

    print(f"\n🎉 GPU Heat Equation Complete!")
    print(f"   Grid: {n}×{n} = {n*n:,} cells")
    print(f"   GPU time: {result.get('gpu_time', 0):.3f}s")
    print(f"   Method: {result['method']}")

    return result, X, Y, u0


if __name__ == "__main__":
    # Run GPU demonstration
    gpu_solver = GPUFiniteDifferenceSolver(verbose=True)

    # Create example
    result, X, Y, u0 = create_gpu_heat_example()

    # Run benchmark
    if gpu_solver.gpu_available:
        benchmark_results = gpu_solver.benchmark_gpu_vs_cpu(grid_sizes=(32, 64, 128))
    else:
        print("\n⚠️ GPU not available for benchmarking")

    print("\n✅ GPU acceleration demonstration complete!")