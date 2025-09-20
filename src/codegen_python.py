"""
Python Scientific Code Generator for FLUX
Generates optimized Python/NumPy/SciPy code from FLUX PDE definitions
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .pde_parser import *


class PythonScientificGenerator:
    """Generate scientific Python code from FLUX definitions"""

    def __init__(self):
        self.indent_level = 0
        self.code_lines = []

    def generate(self, ast_nodes: List[ASTNode]) -> str:
        """Generate complete Python scientific computing code"""
        self.code_lines = []

        # Add imports
        self._generate_imports()

        # Process each AST node
        for node in ast_nodes:
            if isinstance(node, PDEDefinition):
                self._generate_pde_solver(node)
            elif isinstance(node, MeshDefinition):
                self._generate_mesh_creation(node)
            elif isinstance(node, SolverDefinition):
                self._generate_solver_config(node)
            elif isinstance(node, KernelDefinition):
                self._generate_python_kernel(node)

        # Add main execution block
        self._generate_main()

        return '\n'.join(self.code_lines)

    def _generate_imports(self):
        """Generate import statements"""
        imports = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated Python code from FLUX Scientific Computing Language",
            "This code solves PDEs using NumPy/SciPy numerical methods",
            '"""',
            "",
            "import numpy as np",
            "import scipy.sparse as sp",
            "import scipy.sparse.linalg as spla",
            "from scipy.integrate import solve_ivp",
            "from scipy.ndimage import laplace",
            "import matplotlib.pyplot as plt",
            "from dataclasses import dataclass",
            "from typing import Tuple, Callable, Optional",
            "import time",
            ""
        ]
        self.code_lines.extend(imports)

    def _generate_pde_solver(self, pde: PDEDefinition):
        """Generate PDE solver class"""
        class_def = f"""
@dataclass
class {pde.name.capitalize()}Solver:
    \"\"\"Solver for {pde.name} PDE\"\"\"

    nx: int = 100
    ny: int = 100
    dx: float = 0.01
    dy: float = 0.01
    dt: float = 0.001
    alpha: float = 0.1  # Diffusion coefficient

    def __post_init__(self):
        self.u = np.zeros((self.ny, self.nx))
        self.u_new = np.zeros_like(self.u)
        self._setup_matrices()

    def _setup_matrices(self):
        \"\"\"Setup finite difference matrices\"\"\"
        # Create Laplacian operator matrix (sparse)
        n = self.nx * self.ny
        diag = -4 * np.ones(n)
        off_diag = np.ones(n - 1)
        off_diag[self.nx-1::self.nx] = 0  # Handle boundary

        # Build sparse Laplacian
        self.L = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(n, n))
        self.L += sp.diags([np.ones(n - self.nx), np.ones(n - self.nx)],
                          [-self.nx, self.nx], shape=(n, n))
        self.L = self.L / (self.dx**2)

    def apply_initial_condition(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        \"\"\"Apply initial condition u(x,y,0) = func(x,y)\"\"\"
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y)
        self.u = func(X, Y)

    def apply_boundary_conditions(self):
        \"\"\"Apply Dirichlet boundary conditions\"\"\"
        self.u[0, :] = 0    # Bottom
        self.u[-1, :] = 0   # Top
        self.u[:, 0] = 0    # Left
        self.u[:, -1] = 0   # Right

    def time_step(self):
        \"\"\"Perform one time step using implicit Euler method\"\"\"
        # Flatten for matrix operations
        u_flat = self.u.flatten()

        # Implicit Euler: (I - dt*alpha*L) u_new = u_old
        A = sp.eye(len(u_flat)) - self.dt * self.alpha * self.L
        u_new_flat = spla.spsolve(A, u_flat)

        # Reshape back to 2D
        self.u = u_new_flat.reshape((self.ny, self.nx))
        self.apply_boundary_conditions()

    def solve(self, t_end: float) -> np.ndarray:
        \"\"\"Solve PDE until t_end\"\"\"
        n_steps = int(t_end / self.dt)

        print(f"Solving {pde.name} for {{n_steps}} time steps...")
        start_time = time.time()

        for step in range(n_steps):
            self.time_step()

            if step % 100 == 0:
                print(f"  Step {{step}}/{{n_steps}}, t = {{step * self.dt:.3f}}")

        elapsed = time.time() - start_time
        print(f"Solution complete in {{elapsed:.2f}} seconds")

        return self.u

    def plot_solution(self, title: str = "Solution"):
        \"\"\"Plot the current solution\"\"\"
        plt.figure(figsize=(10, 8))
        plt.imshow(self.u, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(label='u(x,y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.show()
"""
        self.code_lines.append(class_def)

    def _generate_mesh_creation(self, mesh: MeshDefinition):
        """Generate mesh creation code"""
        mesh_code = f"""
class {mesh.type}:
    \"\"\"Mesh generation for {mesh.name}\"\"\"

    def __init__(self, nx: int, ny: int, domain: Tuple[float, float, float, float] = (0, 1, 0, 1)):
        self.nx = nx
        self.ny = ny
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        self.dx = (self.x_max - self.x_min) / (nx - 1)
        self.dy = (self.y_max - self.y_min) / (ny - 1)

        # Create mesh points
        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.y = np.linspace(self.y_min, self.y_max, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        print(f"Created {{mesh.type}} with {{nx*ny}} nodes, {{(nx-1)*(ny-1)}} cells")

    def get_node_count(self) -> int:
        return self.nx * self.ny

    def get_cell_count(self) -> int:
        return (self.nx - 1) * (self.ny - 1)
"""
        self.code_lines.append(mesh_code)

    def _generate_solver_config(self, solver: SolverDefinition):
        """Generate solver configuration"""
        solver_code = f"""
class SolverConfig:
    \"\"\"Configuration for {solver.name} solver\"\"\"

    def __init__(self):
        self.method = "{solver.method}"
        self.dt = {solver.dt if hasattr(solver, 'dt') else 0.001}
        self.tolerance = {solver.tolerance if hasattr(solver, 'tolerance') else 1e-6}
        self.max_iterations = {solver.max_iterations if hasattr(solver, 'max_iterations') else 1000}
"""
        self.code_lines.append(solver_code)

    def _generate_python_kernel(self, kernel: KernelDefinition):
        """Generate Python version of GPU kernel"""
        kernel_code = f"""
def {kernel.name}_kernel(U: np.ndarray, **params) -> np.ndarray:
    \"\"\"Python implementation of {kernel.name} kernel\"\"\"
    # This is a Python/NumPy version of the GPU kernel
    # For actual GPU execution, use CuPy or PyCUDA

    result = np.zeros_like(U)

    # Kernel computation (simplified)
    # In production, this would use vectorized NumPy operations
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            # Kernel body implementation
            result[i, j] = U[i, j]  # Placeholder

    return result
"""
        self.code_lines.append(kernel_code)

    def _generate_main(self):
        """Generate main execution block"""
        main_code = """

def main():
    \"\"\"Main execution function\"\"\"
    print("=" * 60)
    print("FLUX Scientific Computing - Python Execution")
    print("=" * 60)

    # Create mesh
    mesh = StructuredGrid(nx=100, ny=100)

    # Initialize solver
    solver = Heat_equationSolver(nx=100, ny=100, dx=0.01, dy=0.01)

    # Set initial condition
    def initial_condition(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    solver.apply_initial_condition(initial_condition)

    # Solve PDE
    solution = solver.solve(t_end=1.0)

    # Visualize results
    solver.plot_solution("Final Solution at t=1.0")

    # Export results
    np.save("solution.npy", solution)
    print("Solution saved to solution.npy")

    # Compute statistics
    print(f"\\nSolution statistics:")
    print(f"  Min value: {np.min(solution):.6f}")
    print(f"  Max value: {np.max(solution):.6f}")
    print(f"  Mean value: {np.mean(solution):.6f}")
    print(f"  L2 norm: {np.linalg.norm(solution):.6f}")


if __name__ == "__main__":
    main()
"""
        self.code_lines.append(main_code)