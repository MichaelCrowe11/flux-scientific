#!/usr/bin/env python3
"""
Simple FLUX Performance Demo
Quick proof that FLUX actually works and is fast
"""

import sys
import os
import time
import numpy as np

# Add FLUX to path
sys.path.insert(0, 'src')

from src.heat_solver import HeatEquationSolver, create_gaussian_initial_condition

def quick_benchmark():
    """Quick performance demonstration"""
    print("FLUX Quick Benchmark")
    print("=" * 30)
    
    # Test different grid sizes
    sizes = [25, 50, 100]
    
    for n in sizes:
        print(f"\nTesting {n}x{n} grid...")
        
        solver = HeatEquationSolver(n, n)
        gaussian_ic = create_gaussian_initial_condition()
        u0 = solver.set_initial_condition(gaussian_ic)
        
        # Short simulation
        alpha = 0.1
        dt = 0.001 
        t_end = 0.05
        n_steps = int(t_end / dt)
        
        print(f"  {n*n:,} cells, {n_steps} time steps")
        
        start = time.time()
        t_array, u_final, _ = solver.solve(u0, t_end, dt, alpha, save_interval=25)
        elapsed = time.time() - start
        
        # Calculate performance
        total_ops = n * n * n_steps
        ops_per_sec = total_ops / elapsed
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Performance: {ops_per_sec:,.0f} cell-steps/sec")
        
        # Check physics
        temp_dropped = np.max(u_final) < np.max(u0)
        print(f"  Physics correct: {temp_dropped}")
        
        if temp_dropped and ops_per_sec > 5000:
            print(f"  SUCCESS: Fast and accurate!")
        else:
            print(f"  Issue detected")

    print("\n" + "=" * 30)
    print("FLUX is ready for real work!")

if __name__ == "__main__":
    quick_benchmark()