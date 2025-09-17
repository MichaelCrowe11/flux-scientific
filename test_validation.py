#!/usr/bin/env python3
"""
Test the FLUX validation suite
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.solvers.finite_difference import FiniteDifferenceSolver
from src.solvers.validation import ValidationSuite, AnalyticalSolutions


def test_heat_equation_accuracy():
    """Test that heat equation solver meets accuracy requirements"""
    print("Testing Heat Equation Solver Accuracy...")
    print("-" * 40)

    solver = FiniteDifferenceSolver(verbose=False)

    # 1D test with known solution
    nx = 101
    x = np.linspace(0, 1, nx)
    u0 = np.sin(np.pi * x)

    result = solver.solve_heat_equation(
        domain=((0, 1),),
        grid_points=(nx,),
        initial_condition=u0,
        boundary_conditions={'left': 0, 'right': 0},
        thermal_diffusivity=1.0,
        time_final=0.1,
        method='crank-nicolson'
    )

    # Compare with analytical
    analytical = AnalyticalSolutions()
    u_exact = analytical.heat_1d_dirichlet(x, 0.1, alpha=1.0)
    error = np.max(np.abs(result['solution'] - u_exact))

    print(f"  Grid: {nx} points")
    print(f"  Max error: {error:.6e}")
    print(f"  Relative error: {error/np.max(u_exact)*100:.4f}%")

    assert error < 1e-3, f"Error {error:.6e} exceeds tolerance"
    print("  ✅ PASSED: Heat equation accuracy test\n")

    return True


def test_convergence_rates():
    """Test that numerical methods achieve expected convergence rates"""
    print("Testing Convergence Rates...")
    print("-" * 40)

    solver = FiniteDifferenceSolver(verbose=False)
    validator = ValidationSuite(solver, verbose=False)

    # Test heat equation convergence
    result = validator.validate_heat_equation_1d(nx_values=[21, 41, 81])

    rate = result['convergence_rate']
    print(f"  Heat equation convergence rate: {rate:.2f}")
    print(f"  Expected: 2.0 (second-order)")

    assert abs(rate - 2.0) < 0.3, f"Convergence rate {rate:.2f} not second-order"
    print("  ✅ PASSED: Convergence rate test\n")

    return True


def test_stability():
    """Test numerical stability limits"""
    print("Testing Numerical Stability...")
    print("-" * 40)

    solver = FiniteDifferenceSolver(verbose=False)

    # Test with stable CFL
    nx = 51
    x = np.linspace(0, 1, nx)
    u0 = np.sin(np.pi * x)
    dx = 1.0 / (nx - 1)
    alpha = 1.0

    # Stable case
    dt_stable = 0.4 * dx**2 / alpha
    result_stable = solver.solve_heat_equation(
        domain=((0, 1),),
        grid_points=(nx,),
        initial_condition=u0,
        boundary_conditions={'left': 0, 'right': 0},
        thermal_diffusivity=alpha,
        time_final=0.01,
        dt=dt_stable,
        method='explicit'
    )

    max_stable = np.max(np.abs(result_stable['solution']))
    print(f"  Stable case (CFL=0.4): max|u| = {max_stable:.3f}")

    assert max_stable < 2.0, "Solution appears unstable"
    print("  ✅ PASSED: Stability test\n")

    return True


def test_2d_heat():
    """Test 2D heat equation solver"""
    print("Testing 2D Heat Equation...")
    print("-" * 40)

    solver = FiniteDifferenceSolver(verbose=False)

    # 2D problem
    n = 31
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

    result = solver.solve_heat_equation(
        domain=((0, 1), (0, 1)),
        grid_points=(n, n),
        initial_condition=u0.flatten(),
        boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
        thermal_diffusivity=1.0,
        time_final=0.01,
        method='crank-nicolson'
    )

    # Check solution properties
    u_final = result['solution'].reshape(n, n)
    print(f"  Grid: {n}×{n}")
    print(f"  Initial max: {np.max(u0):.4f}")
    print(f"  Final max: {np.max(u_final):.4f}")

    # Heat should dissipate
    assert np.max(u_final) < np.max(u0), "Heat not dissipating"
    print("  ✅ PASSED: 2D heat equation test\n")

    return True


def test_wave_equation():
    """Test wave equation solver"""
    print("Testing Wave Equation...")
    print("-" * 40)

    solver = FiniteDifferenceSolver(verbose=False)

    # 1D wave
    nx = 101
    x = np.linspace(0, 1, nx)
    u0 = np.sin(2 * np.pi * x)
    v0 = np.zeros(nx)

    result = solver.solve_wave_equation(
        domain=((0, 1),),
        grid_points=(nx,),
        initial_position=u0,
        initial_velocity=v0,
        boundary_conditions={'left': 0, 'right': 0},
        wave_speed=1.0,
        time_final=0.5
    )

    # Check energy conservation (approximately)
    energy_initial = np.sum(u0**2)
    energy_final = np.sum(result['solution']**2)
    energy_ratio = energy_final / energy_initial

    print(f"  Grid: {nx} points")
    print(f"  CFL number: {result['stability']['CFL']:.3f}")
    print(f"  Energy ratio: {energy_ratio:.4f}")

    assert 0.8 < energy_ratio < 1.2, "Energy not approximately conserved"
    print("  ✅ PASSED: Wave equation test\n")

    return True


def test_poisson():
    """Test Poisson equation solver"""
    print("Testing Poisson Equation...")
    print("-" * 40)

    solver = FiniteDifferenceSolver(verbose=False)

    # 2D Poisson
    n = 21
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # Source term
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    result = solver.solve_poisson_equation(
        domain=((0, 1), (0, 1)),
        grid_points=(n, n),
        source_term=f.flatten(),
        boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
        tol=1e-6,
        method='gauss-seidel'
    )

    print(f"  Grid: {n}×{n}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final residual: {result['residual']:.6e}")

    assert result['iterations'] < 1000, "Too many iterations"
    assert result['residual'] < 1e-5, "Residual too large"
    print("  ✅ PASSED: Poisson equation test\n")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FLUX FINITE DIFFERENCE SOLVER - VALIDATION TESTS")
    print("="*60 + "\n")

    tests = [
        test_heat_equation_accuracy,
        test_convergence_rates,
        test_stability,
        test_2d_heat,
        test_wave_equation,
        test_poisson
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}\n")
            failed += 1

    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All validation tests passed!")
        print("The FLUX finite difference solver is working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)