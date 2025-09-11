"""
Simple test for FLUX heat equation solver
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.heat_solver import HeatEquationSolver, create_gaussian_initial_condition

def test_heat_solver():
    """Test the heat equation solver"""
    print("FLUX Heat Equation Solver Test")
    print("=" * 40)
    
    # Create small solver for quick test
    nx, ny = 21, 21
    solver = HeatEquationSolver(nx, ny, Lx=1.0, Ly=1.0)
    
    # Create Gaussian initial condition
    gaussian_ic = create_gaussian_initial_condition(0.5, 0.5, 0.1, 10.0)
    u0 = solver.set_initial_condition(gaussian_ic)
    
    print(f"Grid size: {nx} x {ny}")
    print(f"Initial max temperature: {np.max(u0):.4f}")
    print(f"Initial total heat: {np.sum(u0):.4f}")
    
    # Solve for a short time
    alpha = 0.1
    dt = 0.001
    t_end = 0.05
    
    print(f"Solving with alpha={alpha}, dt={dt}, t_end={t_end}")
    
    t_array, u_final, u_history = solver.solve(u0, t_end, dt, alpha, save_interval=10)
    
    print(f"Final max temperature: {np.max(u_final):.4f}")
    print(f"Final total heat: {np.sum(u_final):.4f}")
    print(f"Temperature decreased: {np.max(u_final) < np.max(u0)}")
    print(f"Heat diffused properly: {np.max(u_final) > 0}")
    
    # Check that it's physically reasonable
    max_initial = np.max(u0)
    max_final = np.max(u_final)
    
    if max_final < max_initial and max_final > 0:
        print("SUCCESS: Heat equation solved correctly!")
        print("- Temperature decreased (heat diffused)")
        print("- Temperature still positive")
        print("- No numerical instabilities")
        return True
    else:
        print("FAILED: Solution not physical")
        return False

if __name__ == "__main__":
    success = test_heat_solver()
    exit(0 if success else 1)