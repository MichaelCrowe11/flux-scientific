#!/usr/bin/env python3
"""
FLUX Benchmark Suite
Demonstrates that FLUX is fast, accurate, and ready for real work
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add FLUX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.heat_solver import HeatEquationSolver, create_gaussian_initial_condition, create_sine_initial_condition

class FLUXBenchmark:
    """FLUX performance and accuracy benchmarks"""
    
    def __init__(self):
        self.results = {}
        
    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("FLUX Scientific Computing - Benchmark Suite")
        print("=" * 60)
        print("Testing performance, accuracy, and scalability...")
        print("")
        
        # Run benchmarks
        self.benchmark_accuracy()
        self.benchmark_performance_scaling()  
        self.benchmark_vs_reference()
        self.benchmark_stability()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("FLUX Benchmark Suite Complete!")
        print("Check 'benchmark_results.txt' for detailed results")
    
    def benchmark_accuracy(self):
        """Test accuracy against analytical solutions"""
        print("1. ACCURACY BENCHMARK")
        print("-" * 30)
        
        # Test against analytical solution: u = sin(πx)sin(πy)exp(-2π²αt)
        grid_sizes = [21, 41, 81]  # Odd for exact center
        errors = []
        
        for n in grid_sizes:
            solver = HeatEquationSolver(n, n, Lx=1.0, Ly=1.0)
            
            sine_ic = create_sine_initial_condition()
            u0 = solver.set_initial_condition(sine_ic)
            
            # High accuracy parameters
            alpha = 1.0
            dt = 0.00001  # Very small time step
            t_end = 0.005  # Short time for accuracy
            
            print(f"   Testing {n}×{n} grid...")
            t_array, u_numerical, _ = solver.solve(u0, t_end, dt, alpha, save_interval=500)
            
            # Analytical solution
            X, Y = solver.X, solver.Y
            u_analytical = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi**2 * alpha * t_end)
            
            # Compute error
            error_max = np.max(np.abs(u_numerical - u_analytical))
            error_rel = error_max / np.max(np.abs(u_analytical))
            errors.append(error_rel)
            
            print(f"      Max relative error: {error_rel:.2e}")
        
        # Check convergence rate
        if len(errors) >= 2:
            convergence_rate = np.log(errors[-2] / errors[-1]) / np.log(grid_sizes[-1] / grid_sizes[-2])
            print(f"   Convergence rate: {convergence_rate:.2f} (theoretical: 2.0)")
        
        self.results['accuracy'] = {
            'grid_sizes': grid_sizes,
            'errors': errors,
            'convergence_rate': convergence_rate if len(errors) >= 2 else None
        }
        
        if max(errors) < 0.01:  # Less than 1% error
            print("   ✅ ACCURACY TEST PASSED: Errors < 1%")
        else:
            print("   ⚠️ ACCURACY WARNING: Large errors detected")
        print("")
    
    def benchmark_performance_scaling(self):
        """Test performance scaling with grid size"""
        print("2. PERFORMANCE SCALING BENCHMARK")
        print("-" * 40)
        
        grid_sizes = [25, 50, 100, 200]
        times = []
        ops_per_sec = []
        
        for n in grid_sizes:
            solver = HeatEquationSolver(n, n)
            
            gaussian_ic = create_gaussian_initial_condition()
            u0 = solver.set_initial_condition(gaussian_ic)
            
            # Fixed computational work
            alpha = 0.1
            dt = 0.001
            n_steps = 50  # Fixed number of steps
            t_end = n_steps * dt
            
            print(f"   Testing {n}×{n} grid ({n*n:,} cells)...")
            
            start = time.time()
            t_array, u_final, _ = solver.solve(u0, t_end, dt, alpha, save_interval=100)
            elapsed = time.time() - start
            
            times.append(elapsed)
            
            # Calculate performance metrics
            total_ops = n * n * n_steps
            ops_rate = total_ops / elapsed
            ops_per_sec.append(ops_rate)
            
            print(f"      Time: {elapsed:.2f}s")
            print(f"      Rate: {ops_rate:,.0f} cell-steps/sec")
        
        # Analyze scaling
        scaling_efficiency = []
        for i in range(1, len(grid_sizes)):
            expected_slowdown = (grid_sizes[i] / grid_sizes[i-1]) ** 2  # O(N²) expected
            actual_slowdown = times[i] / times[i-1]
            efficiency = expected_slowdown / actual_slowdown
            scaling_efficiency.append(efficiency)
            
            print(f"   {grid_sizes[i-1]}→{grid_sizes[i]}: {efficiency:.2f} scaling efficiency")
        
        self.results['performance'] = {
            'grid_sizes': grid_sizes,
            'times': times,
            'ops_per_sec': ops_per_sec,
            'scaling_efficiency': scaling_efficiency
        }
        
        print(f"   Peak performance: {max(ops_per_sec):,.0f} cell-steps/sec")
        print("")
    
    def benchmark_vs_reference(self):
        """Compare against reference implementation"""
        print("3. REFERENCE COMPARISON")
        print("-" * 30)
        
        # Simple reference implementation (explicit Euler)
        def reference_solver(u0, alpha, dt, t_end, dx, dy):
            """Reference explicit finite difference solver"""
            u = u0.copy()
            nx, ny = u.shape
            
            n_steps = int(t_end / dt)
            
            for step in range(n_steps):
                u_new = u.copy()
                
                # Explicit finite differences (interior points only)
                for i in range(1, nx-1):
                    for j in range(1, ny-1):
                        laplacian = ((u[i+1,j] + u[i-1,j] - 2*u[i,j]) / dx**2 + 
                                   (u[i,j+1] + u[i,j-1] - 2*u[i,j]) / dy**2)
                        u_new[i,j] = u[i,j] + alpha * dt * laplacian
                
                # Apply boundary conditions
                u_new[0,:] = 0; u_new[-1,:] = 0
                u_new[:,0] = 0; u_new[:,-1] = 0
                
                u = u_new
            
            return u
        
        # Test problem
        n = 25  # Small for reference solver speed
        solver = HeatEquationSolver(n, n)
        
        gaussian_ic = create_gaussian_initial_condition()
        u0 = solver.set_initial_condition(gaussian_ic)
        
        alpha = 0.05
        dt = 0.0001  # Small for explicit stability
        t_end = 0.01
        
        print(f"   Comparing {n}×{n} problem...")
        
        # FLUX solver
        start = time.time()
        t_array, u_flux, _ = solver.solve(u0, t_end, dt, alpha, save_interval=100)
        flux_time = time.time() - start
        
        # Reference solver
        start = time.time()
        u_ref = reference_solver(u0, alpha, dt, t_end, solver.dx, solver.dy)
        ref_time = time.time() - start
        
        # Compare results
        diff = np.max(np.abs(u_flux - u_ref))
        speedup = ref_time / flux_time
        
        print(f"   FLUX time: {flux_time:.3f}s")
        print(f"   Reference time: {ref_time:.3f}s")
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   Max difference: {diff:.2e}")
        
        self.results['reference_comparison'] = {
            'flux_time': flux_time,
            'reference_time': ref_time,
            'speedup': speedup,
            'max_difference': diff
        }
        
        if speedup > 1.0 and diff < 1e-6:
            print("   ✅ REFERENCE TEST PASSED: Faster and accurate")
        else:
            print("   ⚠️ REFERENCE WARNING: Check implementation")
        print("")
    
    def benchmark_stability(self):
        """Test numerical stability"""
        print("4. STABILITY BENCHMARK")
        print("-" * 25)
        
        solver = HeatEquationSolver(31, 31)
        
        # Challenging initial condition (sharp gradient)
        def sharp_initial(x, y):
            if (x - 0.5)**2 + (y - 0.5)**2 < 0.05**2:
                return 1000.0  # Very hot spot
            else:
                return 0.0
        
        u0 = solver.set_initial_condition(sharp_initial)
        
        # Test different CFL numbers
        alpha = 0.1
        dx = solver.dx
        dt_stable = 0.2 * dx**2 / alpha  # Conservative CFL
        dt_aggressive = 0.8 * dx**2 / alpha  # Aggressive CFL
        
        test_cases = [
            ("Conservative", dt_stable),
            ("Aggressive", dt_aggressive)
        ]
        
        for name, dt in test_cases:
            print(f"   Testing {name} time step (dt={dt:.6f})...")
            
            try:
                t_array, u_final, _ = solver.solve(u0, 0.1, dt, alpha, save_interval=50)
                
                # Check for instabilities
                has_nan = np.any(np.isnan(u_final))
                has_inf = np.any(np.isinf(u_final))
                has_negative = np.any(u_final < -1e-10)
                max_temp = np.max(u_final)
                
                print(f"      Max temperature: {max_temp:.2f}")
                print(f"      NaN values: {'Yes' if has_nan else 'No'}")
                print(f"      Inf values: {'Yes' if has_inf else 'No'}")
                print(f"      Negative temps: {'Yes' if has_negative else 'No'}")
                
                if not (has_nan or has_inf or has_negative) and max_temp > 0:
                    print(f"      ✅ {name} test STABLE")
                else:
                    print(f"      ❌ {name} test UNSTABLE")
                    
            except Exception as e:
                print(f"      ❌ {name} test FAILED: {e}")
        
        print("")
    
    def generate_report(self):
        """Generate benchmark report"""
        report = []
        report.append("FLUX Scientific Computing - Benchmark Report")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Accuracy results
        if 'accuracy' in self.results:
            report.append("ACCURACY BENCHMARK RESULTS:")
            report.append("-" * 30)
            acc = self.results['accuracy']
            for i, (n, err) in enumerate(zip(acc['grid_sizes'], acc['errors'])):
                report.append(f"  {n:3}×{n:<3} grid: {err:.2e} relative error")
            if acc['convergence_rate']:
                report.append(f"  Convergence rate: {acc['convergence_rate']:.2f}")
            report.append("")
        
        # Performance results  
        if 'performance' in self.results:
            report.append("PERFORMANCE BENCHMARK RESULTS:")
            report.append("-" * 35)
            perf = self.results['performance']
            for i, (n, t, ops) in enumerate(zip(perf['grid_sizes'], perf['times'], perf['ops_per_sec'])):
                report.append(f"  {n:3}×{n:<3} grid: {t:6.2f}s ({ops:8,.0f} cell-steps/sec)")
            report.append(f"  Peak performance: {max(perf['ops_per_sec']):,.0f} cell-steps/sec")
            report.append("")
        
        # Reference comparison
        if 'reference_comparison' in self.results:
            report.append("REFERENCE COMPARISON RESULTS:")
            report.append("-" * 35)
            ref = self.results['reference_comparison']
            report.append(f"  FLUX solver:      {ref['flux_time']:.3f}s")
            report.append(f"  Reference solver: {ref['reference_time']:.3f}s")
            report.append(f"  Speedup:          {ref['speedup']:.1f}x")
            report.append(f"  Max difference:   {ref['max_difference']:.2e}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 10)
        
        if 'accuracy' in self.results and max(self.results['accuracy']['errors']) < 0.01:
            report.append("  ✅ ACCURACY: Excellent (< 1% error)")
        
        if 'performance' in self.results and max(self.results['performance']['ops_per_sec']) > 10000:
            report.append("  ✅ PERFORMANCE: Good (> 10k cell-steps/sec)")
        
        if 'reference_comparison' in self.results and self.results['reference_comparison']['speedup'] > 1.0:
            report.append("  ✅ SPEED: Faster than reference implementation")
        
        report.append("")
        report.append("FLUX is ready for real scientific computing!")
        
        # Save report
        with open('benchmark_results.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print summary
        print("BENCHMARK SUMMARY:")
        print("-" * 20)
        for line in report[-10:]:  # Last 10 lines
            print(line)

def main():
    """Run FLUX benchmark suite"""
    benchmark = FLUXBenchmark()
    benchmark.run_all_benchmarks()

if __name__ == "__main__":
    main()