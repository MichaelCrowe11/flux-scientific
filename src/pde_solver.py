"""
FLUX PDE Solver Integration
Connects FLUX syntax to real numerical solvers
"""

import numpy as np
from typing import Dict, Any, List, Optional
from heat_solver import HeatEquationSolver, create_gaussian_initial_condition, create_sine_initial_condition
from mesh import Mesh, StructuredGrid, Field

class FLUXSolver:
    """Main solver class that interprets FLUX PDE definitions"""
    
    def __init__(self):
        self.solvers = {}
        self.solutions = {}
        
    def solve_heat_equation(self, pde_def: Dict[str, Any], mesh: Mesh, 
                           solver_params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve heat equation PDE"""
        
        # Extract parameters
        alpha = solver_params.get('alpha', 1.0)
        dt = solver_params.get('dt', 0.001)
        t_end = solver_params.get('t_end', 1.0)
        
        # Get mesh parameters
        if isinstance(mesh, StructuredGrid):
            nx = mesh.nx + 1
            ny = mesh.ny + 1
        else:
            # For unstructured meshes, approximate with structured grid
            nx = ny = int(np.sqrt(mesh.get_node_count()))
        
        # Create heat solver
        heat_solver = HeatEquationSolver(nx, ny)
        
        # Set initial condition based on PDE definition
        initial_condition = pde_def.get('initial_condition', 'gaussian')
        
        if initial_condition == 'gaussian':
            ic_func = create_gaussian_initial_condition(0.5, 0.5, 0.1, 1.0)
        elif initial_condition == 'sine':
            ic_func = create_sine_initial_condition()
        elif callable(initial_condition):
            ic_func = initial_condition
        else:
            # Default to Gaussian
            ic_func = create_gaussian_initial_condition(0.5, 0.5, 0.1, 1.0)
        
        u0 = heat_solver.set_initial_condition(ic_func)
        
        # Solve the PDE
        print(f"🔥 Solving heat equation with FLUX:")
        print(f"   α (thermal diffusivity) = {alpha}")
        print(f"   dt (time step) = {dt}")
        print(f"   t_end (final time) = {t_end}")
        print(f"   Grid: {nx}×{ny}")
        
        t_array, u_final, u_history = heat_solver.solve(u0, t_end, dt, alpha, save_interval=10)
        
        # Package results
        solution = {
            'u': u_final,                    # Final temperature field
            'time_history': u_history,       # Temperature at all saved times
            'time_array': t_array,           # Time points
            'mesh_x': heat_solver.X,         # X coordinates
            'mesh_y': heat_solver.Y,         # Y coordinates
            'parameters': {
                'alpha': alpha,
                'dt': dt,
                't_end': t_end,
                'nx': nx,
                'ny': ny
            },
            'solver_info': {
                'method': 'finite_difference',
                'time_scheme': 'implicit_euler',
                'boundary_conditions': 'dirichlet_zero'
            }
        }
        
        return solution

def solve_flux_pde(pde_name: str, mesh_name: str = "default", **solver_params) -> Dict[str, Any]:
    """
    Main entry point for solving FLUX PDEs
    This is what gets called from FLUX programs
    """
    
    # Create default mesh if not provided
    if mesh_name == "default":
        from mesh import create_mesh
        mesh = create_mesh("StructuredGrid", "default_mesh", nx=50, ny=50)
    else:
        # In a full implementation, look up mesh by name
        mesh = create_mesh("StructuredGrid", mesh_name, nx=50, ny=50)
    
    # Create solver
    flux_solver = FLUXSolver()
    
    # Route to appropriate solver based on PDE type
    if pde_name.lower() in ['heat', 'heat_equation', 'diffusion']:
        pde_def = {
            'type': 'heat_equation',
            'initial_condition': solver_params.get('initial_condition', 'gaussian')
        }
        solution = flux_solver.solve_heat_equation(pde_def, mesh, solver_params)
    else:
        raise NotImplementedError(f"PDE type '{pde_name}' not yet implemented")
    
    print("✅ FLUX PDE solved successfully!")
    return solution

def demonstrate_flux_solver():
    """Demonstrate the FLUX solver integration"""
    print("🚀 FLUX PDE Solver Demo")
    print("=" * 50)
    
    # Example 1: Basic heat equation
    print("\n📖 Example 1: Heat equation with Gaussian initial condition")
    solution1 = solve_flux_pde(
        pde_name="heat_equation",
        alpha=0.1,
        dt=0.001,
        t_end=0.5,
        initial_condition='gaussian'
    )
    
    print(f"✅ Solved! Final temperature range: [{np.min(solution1['u']):.4f}, {np.max(solution1['u']):.4f}]")
    
    # Example 2: Heat equation with sine initial condition
    print("\n📖 Example 2: Heat equation with sine initial condition")
    solution2 = solve_flux_pde(
        pde_name="heat_equation", 
        alpha=1.0,
        dt=0.0001,
        t_end=0.01,
        initial_condition='sine'
    )
    
    print(f"✅ Solved! Final temperature range: [{np.min(solution2['u']):.4f}, {np.max(solution2['u']):.4f}]")
    
    return solution1, solution2

if __name__ == "__main__":
    demonstrate_flux_solver()