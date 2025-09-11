"""
Test suite for FLUX heat equation solver
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.heat_solver import HeatEquationSolver, create_gaussian_initial_condition, create_sine_initial_condition


class TestHeatEquationSolver:
    
    def test_solver_creation(self):
        """Test that solver can be created"""
        solver = HeatEquationSolver(10, 10)
        assert solver.nx == 10
        assert solver.ny == 10
        assert solver.dx > 0
        assert solver.dy > 0
    
    def test_initial_conditions(self):
        """Test initial condition setting"""
        solver = HeatEquationSolver(21, 21)
        
        # Gaussian initial condition
        gaussian_ic = create_gaussian_initial_condition()
        u0 = solver.set_initial_condition(gaussian_ic)
        
        assert u0.shape == (21, 21)
        assert np.max(u0) > 0
        assert np.min(u0) >= 0
        assert np.max(u0) <= 1.1  # Should be close to amplitude
    
    def test_boundary_conditions(self):
        """Test boundary condition application"""
        solver = HeatEquationSolver(10, 10)
        
        u = np.ones((10, 10))  # All ones
        u_bc = solver.apply_boundary_conditions(u, bc_type='dirichlet', bc_value=0.0)
        
        # Boundaries should be zero
        assert np.all(u_bc[0, :] == 0.0)   # Bottom
        assert np.all(u_bc[-1, :] == 0.0)  # Top  
        assert np.all(u_bc[:, 0] == 0.0)   # Left
        assert np.all(u_bc[:, -1] == 0.0)  # Right
        
        # Interior should still be one
        assert u_bc[5, 5] == 1.0
    
    def test_time_stepping_stability(self):
        """Test that time stepping is stable"""
        solver = HeatEquationSolver(11, 11)  # Small grid for speed
        
        # Simple initial condition
        u0 = np.zeros((11, 11))
        u0[5, 5] = 1.0  # Hot spot in center
        
        # Solve for short time
        alpha = 0.1
        dt = 0.001
        t_end = 0.01
        
        t_array, u_final, u_history = solver.solve(u0, t_end, dt, alpha, save_interval=5)
        
        # Check stability
        assert not np.any(np.isnan(u_final)), "Solution should not contain NaN"
        assert not np.any(np.isinf(u_final)), "Solution should not contain inf"
        assert np.all(u_final >= -1e-10), "Solution should be non-negative (within numerical precision)"
        assert np.max(u_final) <= np.max(u0), "Maximum temperature should not increase (heat diffusion)"
    
    def test_heat_diffusion_physics(self):
        """Test that heat diffuses correctly"""
        solver = HeatEquationSolver(21, 21)
        
        # Hot spot in center
        u0 = np.zeros((21, 21))
        u0[10, 10] = 10.0
        
        # Apply boundary conditions
        u0 = solver.apply_boundary_conditions(u0)
        
        alpha = 0.1
        dt = 0.001
        t_end = 0.1
        
        t_array, u_final, u_history = solver.solve(u0, t_end, dt, alpha, save_interval=20)
        
        # Physics checks
        initial_max = np.max(u0)
        final_max = np.max(u_final)
        
        assert final_max < initial_max, "Peak temperature should decrease (heat diffusion)"
        assert final_max > 0, "Some heat should remain"
        
        # Heat should spread from center
        center_final = u_final[10, 10]
        neighbor_final = u_final[10, 11]  # Adjacent cell
        assert neighbor_final > 0, "Heat should spread to neighbors"
    
    def test_conservation_property(self):
        """Test heat conservation (with appropriate boundary conditions)"""
        solver = HeatEquationSolver(15, 15)
        
        # Use Neumann (insulated) boundaries for conservation
        gaussian_ic = create_gaussian_initial_condition(0.5, 0.5, 0.2, 1.0)
        u0 = solver.set_initial_condition(gaussian_ic)
        
        # Apply Neumann boundary conditions
        u0 = solver.apply_boundary_conditions(u0, bc_type='neumann')
        
        alpha = 0.05
        dt = 0.0005
        
        # Single time step
        u1 = solver.solve_time_step(u0, dt, alpha)
        u1 = solver.apply_boundary_conditions(u1, bc_type='neumann')
        
        initial_heat = np.sum(u0)
        final_heat = np.sum(u1)
        
        # With Neumann BCs, heat should be conserved (within numerical precision)
        relative_error = abs(final_heat - initial_heat) / initial_heat
        assert relative_error < 0.01, f"Heat conservation error: {relative_error:.4f}"
    
    def test_analytical_comparison(self):
        """Compare with analytical solution"""
        # For sin(πx)sin(πy) initial condition, analytical solution is:
        # u(x,y,t) = sin(πx)sin(πy)exp(-2π²αt)
        
        solver = HeatEquationSolver(31, 31, Lx=1.0, Ly=1.0)  # Odd number for exact center
        
        sine_ic = create_sine_initial_condition()
        u0 = solver.set_initial_condition(sine_ic)
        
        alpha = 1.0
        dt = 0.00005  # Small time step for accuracy
        t_end = 0.01
        
        t_array, u_numerical, _ = solver.solve(u0, t_end, dt, alpha, save_interval=200)
        
        # Analytical solution
        X, Y = solver.X, solver.Y
        u_analytical = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi**2 * alpha * t_end)
        
        # Compare at center point (should be most accurate)
        center_idx = solver.nx // 2, solver.ny // 2
        numerical_center = u_numerical[center_idx]
        analytical_center = u_analytical[center_idx]
        
        relative_error = abs(numerical_center - analytical_center) / abs(analytical_center)
        assert relative_error < 0.05, f"Large error vs analytical solution: {relative_error:.4f}"
    
    @pytest.mark.performance
    def test_performance_benchmark(self):
        """Test solver performance"""
        solver = HeatEquationSolver(50, 50)
        
        gaussian_ic = create_gaussian_initial_condition()
        u0 = solver.set_initial_condition(gaussian_ic)
        
        alpha = 0.1
        dt = 0.001
        t_end = 0.1
        n_steps = int(t_end / dt)
        
        import time
        start = time.time()
        
        t_array, u_final, _ = solver.solve(u0, t_end, dt, alpha, save_interval=100)
        
        elapsed = time.time() - start
        
        # Performance metrics
        total_operations = 50 * 50 * n_steps  # grid_size * time_steps
        ops_per_second = total_operations / elapsed
        
        print(f"Performance: {ops_per_second:.0f} cell-steps/second")
        
        # Should be reasonably fast (at least 10k cell-steps/second) 
        assert ops_per_second > 10000, f"Performance too slow: {ops_per_second:.0f} ops/sec"


class TestInitialConditions:
    
    def test_gaussian_initial_condition(self):
        """Test Gaussian initial condition creator"""
        gaussian_ic = create_gaussian_initial_condition(0.3, 0.7, 0.1, 5.0)
        
        # Test at specified center
        center_value = gaussian_ic(0.3, 0.7)
        assert abs(center_value - 5.0) < 0.01, "Peak should be at specified amplitude"
        
        # Test decay away from center
        edge_value = gaussian_ic(0.0, 0.0)
        assert edge_value < center_value, "Should decay away from center"
    
    def test_sine_initial_condition(self):
        """Test sine initial condition creator"""
        sine_ic = create_sine_initial_condition()
        
        # Test boundary values
        assert abs(sine_ic(0.0, 0.0)) < 1e-10, "Should be zero at origin"
        assert abs(sine_ic(1.0, 1.0)) < 1e-10, "Should be zero at corner"
        
        # Test maximum at center
        center_value = sine_ic(0.5, 0.5)
        assert abs(center_value - 1.0) < 1e-10, "Should be 1 at center"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])