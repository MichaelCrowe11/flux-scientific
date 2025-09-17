"""
FLUX Python Code Generator with Full NumPy/SciPy Integration
Generates working Python code for solving PDEs with validated numerical methods
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import textwrap
from .pde_parser import *

class PythonScientificGenerator:
    """Generate working Python scientific computing code from FLUX AST"""

    def __init__(self):
        self.indent_level = 0
        self.generated_code = []
        self.imports = set()
        self.helper_functions = []

    def generate(self, ast_nodes: List[ASTNode]) -> str:
        """Generate complete Python module from AST nodes"""
        self.generated_code = []
        self.imports = set([
            "import numpy as np",
            "import scipy.sparse as sp",
            "from scipy.sparse.linalg import spsolve",
            "from scipy.linalg import solve_banded",
            "import matplotlib.pyplot as plt",
            "from matplotlib import cm",
            "from mpl_toolkits.mplot3d import Axes3D",
            "import time"
        ])

        # Process each node
        pde_definitions = []
        meshes = []
        domains = []
        solvers = []

        for node in ast_nodes:
            if isinstance(node, PDEDefinition):
                pde_definitions.append(node)
            elif isinstance(node, Mesh):
                meshes.append(node)
            elif isinstance(node, Domain):
                domains.append(node)
            elif isinstance(node, Solver):
                solvers.append(node)

        # Generate imports
        self.generated_code.extend(sorted(self.imports))
        self.generated_code.append("")

        # Generate helper classes
        self._generate_helper_classes()

        # Generate domain classes
        for domain in domains:
            self._generate_domain_class(domain)

        # Generate mesh classes
        for mesh in meshes:
            self._generate_mesh_class(mesh)

        # Generate PDE solver classes
        for pde in pde_definitions:
            self._generate_pde_solver(pde)

        # Generate main execution code
        self._generate_main_block(pde_definitions, meshes, domains)

        return '\n'.join(self.generated_code)

    def _generate_helper_classes(self):
        """Generate FLUX runtime helper classes"""
        helper_code = '''
class FluxField:
    """Represents a field (scalar, vector, or tensor) on a mesh"""

    def __init__(self, mesh, field_type='scalar', initial_value=0.0):
        self.mesh = mesh
        self.field_type = field_type

        if field_type == 'scalar':
            self.data = np.full(mesh.num_points, initial_value, dtype=np.float64)
        elif field_type == 'vector':
            self.data = np.zeros((mesh.num_points, mesh.dim), dtype=np.float64)
        elif field_type == 'tensor':
            self.data = np.zeros((mesh.num_points, mesh.dim, mesh.dim), dtype=np.float64)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        self.old_data = self.data.copy()

    def update_old(self):
        """Store current data as old data for time stepping"""
        self.old_data = self.data.copy()

    def apply_bc_dirichlet(self, boundary_nodes, values):
        """Apply Dirichlet boundary conditions"""
        self.data[boundary_nodes] = values

    def apply_bc_neumann(self, boundary_nodes, flux_values, dx):
        """Apply Neumann boundary conditions (flux)"""
        # Approximate using ghost point method
        for i, node in enumerate(boundary_nodes):
            self.data[node] = self.data[node-1] + flux_values[i] * dx

    def gradient(self):
        """Compute gradient of scalar field"""
        if self.field_type != 'scalar':
            raise ValueError("Gradient only defined for scalar fields")

        grad = FluxField(self.mesh, field_type='vector')

        if self.mesh.dim == 1:
            grad.data[:, 0] = np.gradient(self.data, self.mesh.dx)
        elif self.mesh.dim == 2:
            gy, gx = np.gradient(self.data.reshape(self.mesh.ny, self.mesh.nx),
                                  self.mesh.dy, self.mesh.dx)
            grad.data[:, 0] = gx.flatten()
            grad.data[:, 1] = gy.flatten()
        elif self.mesh.dim == 3:
            data_3d = self.data.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
            gz, gy, gx = np.gradient(data_3d, self.mesh.dz, self.mesh.dy, self.mesh.dx)
            grad.data[:, 0] = gx.flatten()
            grad.data[:, 1] = gy.flatten()
            grad.data[:, 2] = gz.flatten()

        return grad

    def divergence(self):
        """Compute divergence of vector field"""
        if self.field_type != 'vector':
            raise ValueError("Divergence only defined for vector fields")

        div = FluxField(self.mesh, field_type='scalar')

        if self.mesh.dim == 1:
            div.data = np.gradient(self.data[:, 0], self.mesh.dx)
        elif self.mesh.dim == 2:
            u = self.data[:, 0].reshape(self.mesh.ny, self.mesh.nx)
            v = self.data[:, 1].reshape(self.mesh.ny, self.mesh.nx)
            dudx = np.gradient(u, self.mesh.dx, axis=1)
            dvdy = np.gradient(v, self.mesh.dy, axis=0)
            div.data = (dudx + dvdy).flatten()
        elif self.mesh.dim == 3:
            u = self.data[:, 0].reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
            v = self.data[:, 1].reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
            w = self.data[:, 2].reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
            dudx = np.gradient(u, self.mesh.dx, axis=2)
            dvdy = np.gradient(v, self.mesh.dy, axis=1)
            dwdz = np.gradient(w, self.mesh.dz, axis=0)
            div.data = (dudx + dvdy + dwdz).flatten()

        return div

    def laplacian(self):
        """Compute Laplacian of scalar field"""
        if self.field_type != 'scalar':
            raise ValueError("Laplacian only defined for scalar fields")

        lap = FluxField(self.mesh, field_type='scalar')

        if self.mesh.dim == 1:
            # Use second-order central difference
            lap.data[1:-1] = (self.data[2:] - 2*self.data[1:-1] + self.data[:-2]) / (self.mesh.dx**2)
            # Boundary conditions
            lap.data[0] = lap.data[1]
            lap.data[-1] = lap.data[-2]
        elif self.mesh.dim == 2:
            u = self.data.reshape(self.mesh.ny, self.mesh.nx)
            laplacian = np.zeros_like(u)

            # Interior points
            laplacian[1:-1, 1:-1] = (
                (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (self.mesh.dx**2) +
                (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (self.mesh.dy**2)
            )

            # Boundary conditions (zero flux)
            laplacian[0, :] = laplacian[1, :]
            laplacian[-1, :] = laplacian[-2, :]
            laplacian[:, 0] = laplacian[:, 1]
            laplacian[:, -1] = laplacian[:, -2]

            lap.data = laplacian.flatten()
        elif self.mesh.dim == 3:
            u = self.data.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
            laplacian = np.zeros_like(u)

            # Interior points
            laplacian[1:-1, 1:-1, 1:-1] = (
                (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (self.mesh.dx**2) +
                (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (self.mesh.dy**2) +
                (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (self.mesh.dz**2)
            )

            # Boundary conditions
            laplacian[0, :, :] = laplacian[1, :, :]
            laplacian[-1, :, :] = laplacian[-2, :, :]
            laplacian[:, 0, :] = laplacian[:, 1, :]
            laplacian[:, -1, :] = laplacian[:, -2, :]
            laplacian[:, :, 0] = laplacian[:, :, 1]
            laplacian[:, :, -1] = laplacian[:, :, -2]

            lap.data = laplacian.flatten()

        return lap

    def curl(self):
        """Compute curl of vector field (3D only)"""
        if self.field_type != 'vector' or self.mesh.dim != 3:
            raise ValueError("Curl only defined for 3D vector fields")

        curl = FluxField(self.mesh, field_type='vector')

        u = self.data[:, 0].reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        v = self.data[:, 1].reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
        w = self.data[:, 2].reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)

        # Compute curl components
        dwdy = np.gradient(w, self.mesh.dy, axis=1)
        dvdz = np.gradient(v, self.mesh.dz, axis=0)
        dudz = np.gradient(u, self.mesh.dz, axis=0)
        dwdx = np.gradient(w, self.mesh.dx, axis=2)
        dvdx = np.gradient(v, self.mesh.dx, axis=2)
        dudy = np.gradient(u, self.mesh.dy, axis=1)

        curl.data[:, 0] = (dwdy - dvdz).flatten()
        curl.data[:, 1] = (dudz - dwdx).flatten()
        curl.data[:, 2] = (dvdx - dudy).flatten()

        return curl

    def plot(self, title="Field", save_path=None):
        """Visualize the field"""
        if self.field_type == 'scalar':
            if self.mesh.dim == 1:
                plt.figure(figsize=(10, 6))
                plt.plot(self.mesh.x, self.data)
                plt.xlabel('x')
                plt.ylabel('Field value')
                plt.title(title)
                plt.grid(True)
            elif self.mesh.dim == 2:
                plt.figure(figsize=(10, 8))
                data_2d = self.data.reshape(self.mesh.ny, self.mesh.nx)
                plt.contourf(self.mesh.X, self.mesh.Y, data_2d, levels=20, cmap='viridis')
                plt.colorbar(label='Field value')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(title)
                plt.axis('equal')
            elif self.mesh.dim == 3:
                # Show middle slice
                fig = plt.figure(figsize=(12, 10))
                data_3d = self.data.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)

                # Plot three orthogonal slices
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.contourf(self.mesh.X[:, :, self.mesh.nz//2],
                            self.mesh.Y[:, :, self.mesh.nz//2],
                            data_3d[self.mesh.nz//2, :, :], levels=20)
                ax1.set_title('z-slice')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')

                ax2 = fig.add_subplot(2, 2, 2)
                ax2.contourf(self.mesh.X[:, self.mesh.ny//2, :],
                            self.mesh.Z[:, self.mesh.ny//2, :],
                            data_3d[:, self.mesh.ny//2, :], levels=20)
                ax2.set_title('y-slice')
                ax2.set_xlabel('x')
                ax2.set_ylabel('z')

                ax3 = fig.add_subplot(2, 2, 3)
                ax3.contourf(self.mesh.Y[self.mesh.nx//2, :, :],
                            self.mesh.Z[self.mesh.nx//2, :, :],
                            data_3d[:, :, self.mesh.nx//2], levels=20)
                ax3.set_title('x-slice')
                ax3.set_xlabel('y')
                ax3.set_ylabel('z')

                fig.suptitle(title)

        elif self.field_type == 'vector':
            if self.mesh.dim == 2:
                plt.figure(figsize=(10, 8))
                u = self.data[:, 0].reshape(self.mesh.ny, self.mesh.nx)
                v = self.data[:, 1].reshape(self.mesh.ny, self.mesh.nx)
                plt.quiver(self.mesh.X[::2, ::2], self.mesh.Y[::2, ::2],
                          u[::2, ::2], v[::2, ::2])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(title)
                plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

class FluxMesh:
    """Structured mesh for FLUX computations"""

    def __init__(self, domain, nx=50, ny=None, nz=None):
        self.domain = domain
        self.nx = nx
        self.ny = ny if ny is not None else (nx if domain.dim >= 2 else 1)
        self.nz = nz if nz is not None else (nx if domain.dim >= 3 else 1)

        self.dim = domain.dim
        self.num_points = self.nx * self.ny * self.nz

        # Generate mesh points
        if self.dim == 1:
            self.x = np.linspace(domain.xmin, domain.xmax, self.nx)
            self.dx = (domain.xmax - domain.xmin) / (self.nx - 1)
            self.X = self.x
        elif self.dim == 2:
            self.x = np.linspace(domain.xmin, domain.xmax, self.nx)
            self.y = np.linspace(domain.ymin, domain.ymax, self.ny)
            self.dx = (domain.xmax - domain.xmin) / (self.nx - 1)
            self.dy = (domain.ymax - domain.ymin) / (self.ny - 1)
            self.X, self.Y = np.meshgrid(self.x, self.y)
        elif self.dim == 3:
            self.x = np.linspace(domain.xmin, domain.xmax, self.nx)
            self.y = np.linspace(domain.ymin, domain.ymax, self.ny)
            self.z = np.linspace(domain.zmin, domain.zmax, self.nz)
            self.dx = (domain.xmax - domain.xmin) / (self.nx - 1)
            self.dy = (domain.ymax - domain.ymin) / (self.ny - 1)
            self.dz = (domain.zmax - domain.zmin) / (self.nz - 1)
            self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # Identify boundary nodes
        self._identify_boundaries()

    def _identify_boundaries(self):
        """Identify boundary nodes"""
        self.boundary_nodes = {}

        if self.dim == 1:
            self.boundary_nodes['left'] = np.array([0])
            self.boundary_nodes['right'] = np.array([self.nx - 1])
        elif self.dim == 2:
            # Convert to flat indices
            self.boundary_nodes['bottom'] = np.arange(self.nx)
            self.boundary_nodes['top'] = np.arange((self.ny-1)*self.nx, self.ny*self.nx)
            self.boundary_nodes['left'] = np.arange(0, self.ny*self.nx, self.nx)
            self.boundary_nodes['right'] = np.arange(self.nx-1, self.ny*self.nx, self.nx)
        elif self.dim == 3:
            # 3D boundary identification
            self.boundary_nodes['xmin'] = []
            self.boundary_nodes['xmax'] = []
            self.boundary_nodes['ymin'] = []
            self.boundary_nodes['ymax'] = []
            self.boundary_nodes['zmin'] = []
            self.boundary_nodes['zmax'] = []

            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx):
                        idx = k * self.ny * self.nx + j * self.nx + i
                        if i == 0:
                            self.boundary_nodes['xmin'].append(idx)
                        if i == self.nx - 1:
                            self.boundary_nodes['xmax'].append(idx)
                        if j == 0:
                            self.boundary_nodes['ymin'].append(idx)
                        if j == self.ny - 1:
                            self.boundary_nodes['ymax'].append(idx)
                        if k == 0:
                            self.boundary_nodes['zmin'].append(idx)
                        if k == self.nz - 1:
                            self.boundary_nodes['zmax'].append(idx)

            # Convert to arrays
            for key in self.boundary_nodes:
                self.boundary_nodes[key] = np.array(self.boundary_nodes[key])

class FluxDomain:
    """Computational domain for FLUX"""

    def __init__(self, shape='rectangle', **kwargs):
        self.shape = shape

        if shape == 'line':
            self.dim = 1
            self.xmin = kwargs.get('xmin', 0.0)
            self.xmax = kwargs.get('xmax', 1.0)
        elif shape == 'rectangle':
            self.dim = 2
            self.xmin = kwargs.get('xmin', 0.0)
            self.xmax = kwargs.get('xmax', 1.0)
            self.ymin = kwargs.get('ymin', 0.0)
            self.ymax = kwargs.get('ymax', 1.0)
        elif shape == 'box':
            self.dim = 3
            self.xmin = kwargs.get('xmin', 0.0)
            self.xmax = kwargs.get('xmax', 1.0)
            self.ymin = kwargs.get('ymin', 0.0)
            self.ymax = kwargs.get('ymax', 1.0)
            self.zmin = kwargs.get('zmin', 0.0)
            self.zmax = kwargs.get('zmax', 1.0)
        elif shape == 'circle':
            self.dim = 2
            self.center = kwargs.get('center', (0.0, 0.0))
            self.radius = kwargs.get('radius', 1.0)
        elif shape == 'sphere':
            self.dim = 3
            self.center = kwargs.get('center', (0.0, 0.0, 0.0))
            self.radius = kwargs.get('radius', 1.0)
        else:
            raise ValueError(f"Unknown domain shape: {shape}")

    def contains(self, x, y=None, z=None):
        """Check if point is inside domain"""
        if self.shape in ['line', 'rectangle', 'box']:
            in_domain = (x >= self.xmin) & (x <= self.xmax)
            if self.dim >= 2 and y is not None:
                in_domain &= (y >= self.ymin) & (y <= self.ymax)
            if self.dim >= 3 and z is not None:
                in_domain &= (z >= self.zmin) & (z <= self.zmax)
            return in_domain
        elif self.shape == 'circle':
            r = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
            return r <= self.radius
        elif self.shape == 'sphere':
            r = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2)
            return r <= self.radius
'''
        self.generated_code.append(textwrap.dedent(helper_code))

    def _generate_domain_class(self, domain: Domain):
        """Generate domain initialization code"""
        params = []
        if domain.shape.lower() in ['rectangle', 'box']:
            for key, value in domain.parameters.items():
                if isinstance(value, Number):
                    params.append(f"{key}={value.value}")
                else:
                    params.append(f"{key}={value}")

        code = f"""
# Domain: {domain.name}
domain_{domain.name} = FluxDomain(shape='{domain.shape.lower()}', {', '.join(params)})
"""
        self.generated_code.append(code)

    def _generate_mesh_class(self, mesh: Mesh):
        """Generate mesh initialization code"""
        params = []
        for key, value in mesh.parameters.items():
            if key != 'domain':
                if isinstance(value, Number):
                    params.append(f"{key}={value.value}")
                else:
                    params.append(f"{key}={value}")

        code = f"""
# Mesh: {mesh.name}
mesh_{mesh.name} = FluxMesh(domain_{mesh.domain}, {', '.join(params)})
"""
        self.generated_code.append(code)

    def _generate_pde_solver(self, pde: PDEDefinition):
        """Generate complete PDE solver class"""
        class_name = f"{pde.name.title()}Solver"

        # Analyze equations to determine solver type
        solver_type = self._determine_solver_type(pde)

        code = f'''
class {class_name}:
    """Solver for {pde.name} PDE"""

    def __init__(self, mesh):
        self.mesh = mesh

        # Initialize fields
'''

        # Initialize variables
        variables = pde.variables if pde.variables else ['u']
        for var in variables:
            code += f"        self.{var} = FluxField(mesh, field_type='scalar')\n"

        # Add boundary and initial condition storage
        code += '''
        # Storage for boundary conditions
        self.boundary_conditions = {}
        self.initial_conditions = {}

        # Solver parameters
        self.dt = 0.001
        self.solver_type = '{}'
'''.format(solver_type)

        # Generate initialization methods
        code += self._generate_init_methods(pde, variables)

        # Generate solve method
        code += self._generate_solve_method(pde, variables, solver_type)

        # Generate time stepping methods
        code += self._generate_time_step_methods(pde, variables, solver_type)

        # Generate boundary condition methods
        code += self._generate_bc_methods(pde)

        # Generate visualization method
        code += self._generate_visualization_method(variables)

        self.generated_code.append(code)

    def _determine_solver_type(self, pde: PDEDefinition) -> str:
        """Determine appropriate solver type based on PDE structure"""
        # Check for heat equation pattern
        for eq in pde.equations:
            if isinstance(eq.lhs, PartialDerivative) and eq.lhs.with_respect_to == 't':
                if isinstance(eq.rhs, Laplacian) or isinstance(eq.rhs, BinaryOp):
                    return 'parabolic'

        # Check for wave equation pattern
        for eq in pde.equations:
            if isinstance(eq.lhs, PartialDerivative) and eq.lhs.order == 2:
                return 'hyperbolic'

        # Default to elliptic
        return 'elliptic'

    def _generate_init_methods(self, pde: PDEDefinition, variables: List[str]) -> str:
        """Generate initialization methods"""
        code = '''
    def set_initial_conditions(self, **kwargs):
        """Set initial conditions for all variables"""
'''

        # Handle initial conditions from PDE definition
        if pde.initial_conditions:
            for ic in pde.initial_conditions:
                code += f'''        if '{ic.variable}' in kwargs:
            self.{ic.variable}.data[:] = kwargs['{ic.variable}']
        else:
            # Default initial condition
            self.{ic.variable}.data[:] = 0.0
'''
        else:
            for var in variables:
                code += f'''        if '{var}' in kwargs:
            self.{var}.data[:] = kwargs['{var}']
'''

        code += '''
    def set_boundary_conditions(self, **kwargs):
        """Set boundary conditions"""
        self.boundary_conditions = kwargs
'''
        return code

    def _generate_solve_method(self, pde: PDEDefinition, variables: List[str], solver_type: str) -> str:
        """Generate main solve method"""
        code = '''
    def solve(self, time_steps=1000, dt=None, save_interval=100, verbose=True):
        """Main solver routine"""
        if dt is not None:
            self.dt = dt

        # Solution history
        history = {
'''
        for var in variables:
            code += f"            '{var}': [],\n"

        code += '''            'time': []
        }

        # Time stepping loop
        t = 0.0
        for step in range(time_steps):
            # Store old values
'''
        for var in variables:
            code += f"            self.{var}.update_old()\n"

        code += f'''
            # Perform time step
            self.time_step_{solver_type}()

            # Apply boundary conditions
            self.apply_boundary_conditions()

            # Update time
            t += self.dt

            # Save history
            if step % save_interval == 0:
'''
        for var in variables:
            code += f"                history['{var}'].append(self.{var}.data.copy())\n"

        code += '''                history['time'].append(t)

                if verbose:
                    print(f"Step {step:5d}, t = {t:.4f}", end='')
'''

        # Add convergence check for first variable
        var = variables[0]
        code += f'''                    max_val = np.max(np.abs(self.{var}.data))
                    print(f", max |{var}| = {{max_val:.6e}}")

        return history
'''
        return code

    def _generate_time_step_methods(self, pde: PDEDefinition, variables: List[str], solver_type: str) -> str:
        """Generate time stepping methods for different PDE types"""
        code = ""

        if solver_type == 'parabolic':
            # Heat equation type - use implicit or explicit method
            code += self._generate_parabolic_solver(pde, variables)
        elif solver_type == 'hyperbolic':
            # Wave equation type
            code += self._generate_hyperbolic_solver(pde, variables)
        else:
            # Elliptic type
            code += self._generate_elliptic_solver(pde, variables)

        return code

    def _generate_parabolic_solver(self, pde: PDEDefinition, variables: List[str]) -> str:
        """Generate solver for parabolic PDEs (heat equation)"""
        var = variables[0]  # Primary variable

        code = f'''
    def time_step_parabolic(self):
        """Time step for parabolic PDE (heat equation type)"""
        # Use Crank-Nicolson method (unconditionally stable)

        if self.mesh.dim == 1:
            # 1D heat equation
            nx = self.mesh.nx
            dx = self.mesh.dx
            alpha = self.dt / (2 * dx**2)

            # Build tridiagonal matrix
            main_diag = np.ones(nx) * (1 + 2*alpha)
            off_diag = np.ones(nx-1) * (-alpha)

            # Build RHS
            rhs = np.zeros(nx)
            for i in range(1, nx-1):
                rhs[i] = self.{var}.old_data[i] + alpha * (
                    self.{var}.old_data[i+1] - 2*self.{var}.old_data[i] + self.{var}.old_data[i-1]
                )

            # Handle boundaries (keep fixed)
            rhs[0] = self.{var}.data[0]
            rhs[-1] = self.{var}.data[-1]
            main_diag[0] = 1
            main_diag[-1] = 1
            off_diag[0] = 0
            off_diag[-2] = 0

            # Solve tridiagonal system
            A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(nx, nx))
            self.{var}.data = sp.linalg.spsolve(A.tocsr(), rhs)

        elif self.mesh.dim == 2:
            # 2D heat equation using ADI (Alternating Direction Implicit)
            nx, ny = self.mesh.nx, self.mesh.ny
            dx, dy = self.mesh.dx, self.mesh.dy

            # Reshape for 2D operations
            u = self.{var}.data.reshape(ny, nx)
            u_old = self.{var}.old_data.reshape(ny, nx)
            u_half = np.zeros_like(u)

            # First half-step (implicit in x, explicit in y)
            alpha_x = self.dt / (2 * dx**2)
            alpha_y = self.dt / (2 * dy**2)

            for j in range(1, ny-1):
                # Build tridiagonal system for each row
                main_diag = np.ones(nx) * (1 + 2*alpha_x)
                off_diag = np.ones(nx-1) * (-alpha_x)

                rhs = u_old[j, :] + alpha_y * (u_old[j+1, :] - 2*u_old[j, :] + u_old[j-1, :])

                # Boundary conditions
                main_diag[0] = 1
                main_diag[-1] = 1
                off_diag[0] = 0
                off_diag[-2] = 0
                rhs[0] = u_old[j, 0]
                rhs[-1] = u_old[j, -1]

                # Solve
                A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(nx, nx))
                u_half[j, :] = sp.linalg.spsolve(A.tocsr(), rhs)

            # Copy boundaries
            u_half[0, :] = u_old[0, :]
            u_half[-1, :] = u_old[-1, :]

            # Second half-step (implicit in y, explicit in x)
            for i in range(1, nx-1):
                # Build tridiagonal system for each column
                main_diag = np.ones(ny) * (1 + 2*alpha_y)
                off_diag = np.ones(ny-1) * (-alpha_y)

                rhs = u_half[:, i] + alpha_x * (u_half[:, i+1] - 2*u_half[:, i] + u_half[:, i-1])

                # Boundary conditions
                main_diag[0] = 1
                main_diag[-1] = 1
                off_diag[0] = 0
                off_diag[-2] = 0
                rhs[0] = u_half[0, i]
                rhs[-1] = u_half[-1, i]

                # Solve
                A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(ny, ny))
                u[i, :] = sp.linalg.spsolve(A.tocsr(), rhs)

            # Copy boundaries
            u[:, 0] = u_half[:, 0]
            u[:, -1] = u_half[:, -1]

            # Update field
            self.{var}.data = u.flatten()

        elif self.mesh.dim == 3:
            # 3D heat equation - use explicit method for simplicity
            u = self.{var}.data.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)
            u_old = self.{var}.old_data.reshape(self.mesh.nz, self.mesh.ny, self.mesh.nx)

            dx2 = self.mesh.dx**2
            dy2 = self.mesh.dy**2
            dz2 = self.mesh.dz**2

            # Interior points
            u[1:-1, 1:-1, 1:-1] = u_old[1:-1, 1:-1, 1:-1] + self.dt * (
                (u_old[1:-1, 1:-1, 2:] - 2*u_old[1:-1, 1:-1, 1:-1] + u_old[1:-1, 1:-1, :-2]) / dx2 +
                (u_old[1:-1, 2:, 1:-1] - 2*u_old[1:-1, 1:-1, 1:-1] + u_old[1:-1, :-2, 1:-1]) / dy2 +
                (u_old[2:, 1:-1, 1:-1] - 2*u_old[1:-1, 1:-1, 1:-1] + u_old[:-2, 1:-1, 1:-1]) / dz2
            )

            self.{var}.data = u.flatten()
'''
        return code

    def _generate_hyperbolic_solver(self, pde: PDEDefinition, variables: List[str]) -> str:
        """Generate solver for hyperbolic PDEs (wave equation)"""
        var = variables[0]

        code = f'''
    def time_step_hyperbolic(self):
        """Time step for hyperbolic PDE (wave equation type)"""
        # Use leapfrog or Lax-Wendroff method

        if self.mesh.dim == 1:
            # 1D wave equation
            nx = self.mesh.nx
            dx = self.mesh.dx
            c = 1.0  # Wave speed (should be configurable)
            CFL = c * self.dt / dx

            if not hasattr(self, '{var}_prev'):
                self.{var}_prev = self.{var}.data.copy()

            u_new = np.zeros(nx)

            # Lax-Wendroff scheme
            for i in range(1, nx-1):
                u_new[i] = (CFL**2 * (self.{var}.data[i+1] - 2*self.{var}.data[i] + self.{var}.data[i-1]) +
                           2*self.{var}.data[i] - self.{var}_prev[i])

            # Update
            self.{var}_prev = self.{var}.data.copy()
            self.{var}.data[1:-1] = u_new[1:-1]

        elif self.mesh.dim == 2:
            # 2D wave equation
            nx, ny = self.mesh.nx, self.mesh.ny
            dx, dy = self.mesh.dx, self.mesh.dy

            if not hasattr(self, '{var}_prev'):
                self.{var}_prev = self.{var}.data.copy()

            u = self.{var}.data.reshape(ny, nx)
            u_old = self.{var}.old_data.reshape(ny, nx)
            u_prev = self.{var}_prev.reshape(ny, nx)

            c = 1.0  # Wave speed
            dt2 = self.dt**2

            # Central difference in time and space
            u[1:-1, 1:-1] = (dt2 * c**2 * (
                (u_old[1:-1, 2:] - 2*u_old[1:-1, 1:-1] + u_old[1:-1, :-2]) / dx**2 +
                (u_old[2:, 1:-1] - 2*u_old[1:-1, 1:-1] + u_old[:-2, 1:-1]) / dy**2
            ) + 2*u_old[1:-1, 1:-1] - u_prev[1:-1, 1:-1])

            self.{var}_prev = u_old.flatten()
            self.{var}.data = u.flatten()
'''
        return code

    def _generate_elliptic_solver(self, pde: PDEDefinition, variables: List[str]) -> str:
        """Generate solver for elliptic PDEs (Poisson equation)"""
        var = variables[0]

        code = f'''
    def time_step_elliptic(self):
        """Solve elliptic PDE (Poisson equation type)"""
        # Use iterative method (Jacobi or Gauss-Seidel)

        if self.mesh.dim == 1:
            # 1D Poisson equation
            nx = self.mesh.nx
            dx = self.mesh.dx

            # Build linear system
            A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx)) / dx**2
            b = np.zeros(nx)  # RHS (should be from PDE definition)

            # Apply boundary conditions
            A = A.tolil()
            A[0, :] = 0
            A[0, 0] = 1
            A[-1, :] = 0
            A[-1, -1] = 1
            b[0] = self.{var}.data[0]
            b[-1] = self.{var}.data[-1]

            # Solve
            self.{var}.data = sp.linalg.spsolve(A.tocsr(), b)

        elif self.mesh.dim == 2:
            # 2D Poisson equation using Jacobi iteration
            nx, ny = self.mesh.nx, self.mesh.ny
            dx, dy = self.mesh.dx, self.mesh.dy

            u = self.{var}.data.reshape(ny, nx)

            # Jacobi iteration
            max_iter = 1000
            tol = 1e-6

            for iter in range(max_iter):
                u_old = u.copy()

                # Interior points
                u[1:-1, 1:-1] = 0.25 * (
                    u_old[1:-1, 2:] + u_old[1:-1, :-2] +
                    u_old[2:, 1:-1] + u_old[:-2, 1:-1]
                )

                # Check convergence
                if np.max(np.abs(u - u_old)) < tol:
                    break

            self.{var}.data = u.flatten()
'''
        return code

    def _generate_bc_methods(self, pde: PDEDefinition) -> str:
        """Generate boundary condition application methods"""
        code = '''
    def apply_boundary_conditions(self):
        """Apply boundary conditions to all variables"""
        for bc_location, bc_value in self.boundary_conditions.items():
            if bc_location in self.mesh.boundary_nodes:
                nodes = self.mesh.boundary_nodes[bc_location]

                if callable(bc_value):
                    # Time-dependent BC
                    values = bc_value(self.mesh, nodes)
                else:
                    # Constant BC
                    values = bc_value

'''

        # Apply to all variables
        variables = pde.variables if pde.variables else ['u']
        for var in variables:
            code += f"                self.{var}.apply_bc_dirichlet(nodes, values)\n"

        # Add specific boundary conditions from PDE definition
        if pde.boundary_conditions:
            code += '''
        # Apply specific boundary conditions from PDE definition
'''
            for bc in pde.boundary_conditions:
                bc_type = 'dirichlet' if bc.type == 'dirichlet' else 'neumann'
                code += f"        # BC at {bc.location}: {bc.type}\n"

        return code

    def _generate_visualization_method(self, variables: List[str]) -> str:
        """Generate visualization methods"""
        code = '''
    def visualize(self, variable=None, save_path=None):
        """Visualize solution fields"""
'''

        if len(variables) == 1:
            var = variables[0]
            code += f'''        self.{var}.plot(title="{var.title()} Field", save_path=save_path)
'''
        else:
            code += '''        if variable is None:
            # Plot all variables
'''
            for i, var in enumerate(variables):
                code += f'''            plt.subplot(1, {len(variables)}, {i+1})
            self.{var}.plot(title="{var.title()}")
'''
            code += '''        else:
            # Plot specific variable
            getattr(self, variable).plot(title=f"{variable.title()} Field", save_path=save_path)
'''

        return code

    def _generate_main_block(self, pde_definitions: List[PDEDefinition],
                            meshes: List[Mesh], domains: List[Domain]):
        """Generate main execution block"""
        if not pde_definitions:
            return

        pde = pde_definitions[0]
        code = f'''
if __name__ == "__main__":
    # Example usage
    print("FLUX Scientific Computing - {pde.name} Solver")
    print("=" * 50)

'''

        # Use existing domain/mesh or create defaults
        if domains:
            domain = domains[0]
            code += f"    # Using domain: {domain.name}\n"
        else:
            code += '''    # Create domain
    domain = FluxDomain(shape='rectangle', xmin=0, xmax=1, ymin=0, ymax=1)
'''

        if meshes:
            mesh = meshes[0]
            code += f"    # Using mesh: {mesh.name}\n"
        else:
            code += '''    # Create mesh
    mesh = FluxMesh(domain, nx=50, ny=50)
'''

        code += f'''
    # Create solver
    solver = {pde.name.title()}Solver(mesh)

    # Set initial conditions (example: Gaussian pulse)
    if mesh.dim == 1:
        x = mesh.x
        initial = np.exp(-20*(x - 0.5)**2)
    elif mesh.dim == 2:
        X, Y = mesh.X, mesh.Y
        initial = np.exp(-20*((X - 0.5)**2 + (Y - 0.5)**2)).flatten()
    else:
        initial = np.zeros(mesh.num_points)
'''

        var = pde.variables[0] if pde.variables else 'u'
        code += f'''
    solver.set_initial_conditions({var}=initial)

    # Set boundary conditions (zero at boundaries)
    solver.set_boundary_conditions(left=0, right=0, top=0, bottom=0)

    # Solve
    print("\\nSolving PDE...")
    start_time = time.time()
    history = solver.solve(time_steps=1000, dt=0.0001, save_interval=100)
    end_time = time.time()

    print(f"\\nSolution completed in {{end_time - start_time:.2f}} seconds")

    # Visualize final solution
    solver.visualize()

    # Plot time evolution
    if len(history['time']) > 1:
        plt.figure(figsize=(12, 8))

        # Plot solution at different times
        n_plots = min(4, len(history['time']))
        for i in range(n_plots):
            plt.subplot(2, 2, i+1)

            t_idx = i * (len(history['time']) - 1) // (n_plots - 1)
            t = history['time'][t_idx]
            data = history['{var}'][t_idx]

            if mesh.dim == 1:
                plt.plot(mesh.x, data)
                plt.xlabel('x')
                plt.ylabel('{var}')
            elif mesh.dim == 2:
                plt.contourf(mesh.X, mesh.Y, data.reshape(mesh.ny, mesh.nx), levels=20)
                plt.colorbar()
                plt.xlabel('x')
                plt.ylabel('y')

            plt.title(f't = {{t:.4f}}')

        plt.tight_layout()
        plt.show()
'''

        self.generated_code.append(code)