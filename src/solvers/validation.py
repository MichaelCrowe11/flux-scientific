"""
Validation Suite for FLUX Numerical Solvers
Provides analytical solutions and convergence testing
"""

import numpy as np
from typing import Dict, Tuple, Callable, List
import matplotlib.pyplot as plt
from scipy.special import erf


class AnalyticalSolutions:
    """Collection of analytical solutions for PDE validation"""

    @staticmethod
    def heat_1d_dirichlet(x: np.ndarray, t: float, alpha: float = 1.0,
                          L: float = 1.0, n_terms: int = 100) -> np.ndarray:
        """
        1D heat equation with Dirichlet BCs: u(0,t) = u(L,t) = 0
        Initial condition: u(x,0) = sin(πx/L)
        """
        return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

    @staticmethod
    def heat_2d_dirichlet(x: np.ndarray, y: np.ndarray, t: float,
                          alpha: float = 1.0, Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
        """
        2D heat equation with Dirichlet BCs on rectangular domain
        Initial condition: u(x,y,0) = sin(πx/Lx) * sin(πy/Ly)
        """
        X, Y = np.meshgrid(x, y)
        return (np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) *
                np.exp(-alpha * np.pi**2 * (1/Lx**2 + 1/Ly**2) * t))

    @staticmethod
    def heat_gaussian_infinite(x: np.ndarray, t: float, alpha: float = 1.0,
                               x0: float = 0.5, sigma0: float = 0.01) -> np.ndarray:
        """
        Heat equation with Gaussian initial condition on infinite domain
        u(x,0) = exp(-(x-x0)²/(2σ₀²))
        """
        if t == 0:
            return np.exp(-(x - x0)**2 / (2 * sigma0**2))

        # Variance grows with time
        sigma_t = np.sqrt(sigma0**2 + 2 * alpha * t)
        # Amplitude decreases to conserve total heat
        amplitude = sigma0 / sigma_t

        return amplitude * np.exp(-(x - x0)**2 / (2 * sigma_t**2))

    @staticmethod
    def wave_1d_dalembert(x: np.ndarray, t: float, c: float = 1.0,
                          f: Callable = None, g: Callable = None) -> np.ndarray:
        """
        1D wave equation solution via d'Alembert's formula
        ∂²u/∂t² = c²∂²u/∂x²
        u(x,0) = f(x), ∂u/∂t(x,0) = g(x)
        """
        if f is None:
            f = lambda x: np.sin(np.pi * x)
        if g is None:
            g = lambda x: 0 * x

        # d'Alembert's solution
        u = 0.5 * (f(x - c*t) + f(x + c*t))

        # Add integral term for initial velocity
        if t > 0:
            # Simplified for g=0 case
            pass

        return u

    @staticmethod
    def wave_2d_standing(x: np.ndarray, y: np.ndarray, t: float,
                         c: float = 1.0, Lx: float = 1.0, Ly: float = 1.0,
                         m: int = 1, n: int = 1) -> np.ndarray:
        """
        2D standing wave solution on rectangular domain
        Mode (m,n) oscillation
        """
        X, Y = np.meshgrid(x, y)
        omega = c * np.pi * np.sqrt((m/Lx)**2 + (n/Ly)**2)

        return (np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly) *
                np.cos(omega * t))

    @staticmethod
    def poisson_2d_rectangle(x: np.ndarray, y: np.ndarray,
                            Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
        """
        2D Poisson equation ∇²u = -2π² sin(πx/Lx)sin(πy/Ly)
        with homogeneous Dirichlet BCs
        """
        X, Y = np.meshgrid(x, y)
        return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

    @staticmethod
    def advection_1d(x: np.ndarray, t: float, c: float = 1.0,
                    initial: Callable = None) -> np.ndarray:
        """
        1D advection equation: ∂u/∂t + c∂u/∂x = 0
        Solution: u(x,t) = u₀(x - ct)
        """
        if initial is None:
            # Gaussian pulse
            initial = lambda x: np.exp(-100*(x - 0.5)**2)

        return initial(x - c*t)

    @staticmethod
    def diffusion_advection_1d(x: np.ndarray, t: float,
                              D: float = 0.01, c: float = 1.0,
                              x0: float = 0.2) -> np.ndarray:
        """
        1D advection-diffusion equation
        ∂u/∂t + c∂u/∂x = D∂²u/∂x²
        """
        if t == 0:
            return np.exp(-100*(x - x0)**2)

        # Solution for Gaussian initial condition
        sigma = np.sqrt(2 * D * t + 0.01)
        x_center = x0 + c * t

        return (0.1/sigma) * np.exp(-(x - x_center)**2 / (2*sigma**2))


class ValidationSuite:
    """Comprehensive validation for numerical PDE solvers"""

    def __init__(self, solver, verbose: bool = True):
        self.solver = solver
        self.verbose = verbose
        self.analytical = AnalyticalSolutions()
        self.results = {}

    def validate_heat_equation_1d(self, nx_values: List[int] = [21, 41, 81, 161]) -> Dict:
        """
        Validate 1D heat equation solver with grid refinement study
        """
        if self.verbose:
            print("\n" + "="*60)
            print("1D Heat Equation Validation")
            print("="*60)

        errors = []
        dx_values = []

        for nx in nx_values:
            # Setup problem
            domain = (0, 1)
            dx = 1.0 / (nx - 1)
            dx_values.append(dx)

            # Initial condition: sin(πx)
            x = np.linspace(0, 1, nx)
            u0 = np.sin(np.pi * x)

            # Solve
            alpha = 1.0
            t_final = 0.1
            dt = 0.5 * dx**2 / alpha  # CFL condition

            result = self.solver.solve_heat_equation(
                domain=(domain,),
                grid_points=(nx,),
                initial_condition=u0,
                boundary_conditions={'left': 0, 'right': 0},
                thermal_diffusivity=alpha,
                time_final=t_final,
                dt=dt,
                method='crank-nicolson'
            )

            # Analytical solution
            u_exact = self.analytical.heat_1d_dirichlet(x, t_final, alpha)

            # Compute error
            error = np.max(np.abs(result['solution'] - u_exact))
            errors.append(error)

            if self.verbose:
                print(f"  nx={nx:4d}, dx={dx:.4f}: L∞ error = {error:.6e}")

        # Compute convergence rate
        errors = np.array(errors)
        dx_values = np.array(dx_values)

        if len(errors) > 1:
            # Linear regression in log-log space
            p = np.polyfit(np.log(dx_values), np.log(errors), 1)
            convergence_rate = p[0]

            if self.verbose:
                print(f"\nConvergence rate: {convergence_rate:.2f}")
                print(f"Expected: 2.0 (second-order accurate)")

                if abs(convergence_rate - 2.0) < 0.2:
                    print("✅ Convergence rate validated!")
                else:
                    print("⚠️ Convergence rate differs from expected")

        self.results['heat_1d'] = {
            'errors': errors.tolist(),
            'dx_values': dx_values.tolist(),
            'convergence_rate': convergence_rate if len(errors) > 1 else None
        }

        return self.results['heat_1d']

    def validate_heat_equation_2d(self, n_values: List[int] = [11, 21, 41]) -> Dict:
        """
        Validate 2D heat equation solver
        """
        if self.verbose:
            print("\n" + "="*60)
            print("2D Heat Equation Validation")
            print("="*60)

        errors = []
        dx_values = []

        for n in n_values:
            # Setup problem
            domain = ((0, 1), (0, 1))
            dx = 1.0 / (n - 1)
            dx_values.append(dx)

            # Initial condition: sin(πx)sin(πy)
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, y)
            u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

            # Solve
            alpha = 1.0
            t_final = 0.01
            dt = 0.2 * dx**2 / alpha

            result = self.solver.solve_heat_equation(
                domain=domain,
                grid_points=(n, n),
                initial_condition=u0.flatten(),
                boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                thermal_diffusivity=alpha,
                time_final=t_final,
                dt=dt,
                method='crank-nicolson'
            )

            # Analytical solution
            u_exact = self.analytical.heat_2d_dirichlet(x, y, t_final, alpha)

            # Compute error
            u_numerical = result['solution']
            error = np.max(np.abs(u_numerical - u_exact))
            errors.append(error)

            if self.verbose:
                print(f"  n={n:3d}×{n:<3d}, dx={dx:.4f}: L∞ error = {error:.6e}")

        # Convergence rate
        errors = np.array(errors)
        dx_values = np.array(dx_values)

        if len(errors) > 1:
            p = np.polyfit(np.log(dx_values), np.log(errors), 1)
            convergence_rate = p[0]

            if self.verbose:
                print(f"\nConvergence rate: {convergence_rate:.2f}")
                print(f"Expected: 2.0 (second-order accurate)")

        self.results['heat_2d'] = {
            'errors': errors.tolist(),
            'dx_values': dx_values.tolist(),
            'convergence_rate': convergence_rate if len(errors) > 1 else None
        }

        return self.results['heat_2d']

    def validate_wave_equation(self, nx_values: List[int] = [51, 101, 201]) -> Dict:
        """
        Validate wave equation solver with CFL condition
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Wave Equation Validation")
            print("="*60)

        errors = []
        dx_values = []

        for nx in nx_values:
            # Setup
            domain = (0, 1)
            dx = 1.0 / (nx - 1)
            dx_values.append(dx)

            # Initial conditions
            x = np.linspace(0, 1, nx)
            u0 = np.sin(np.pi * x)  # Initial position
            v0 = np.zeros(nx)       # Initial velocity

            # Solve
            c = 1.0  # Wave speed
            t_final = 0.5
            dt = 0.8 * dx / c  # CFL < 1

            result = self.solver.solve_wave_equation(
                domain=(domain,),
                grid_points=(nx,),
                initial_position=u0,
                initial_velocity=v0,
                boundary_conditions={'left': 0, 'right': 0},
                wave_speed=c,
                time_final=t_final,
                dt=dt
            )

            # Analytical: standing wave
            u_exact = np.sin(np.pi * x) * np.cos(np.pi * c * t_final)

            # Error
            error = np.max(np.abs(result['solution'] - u_exact))
            errors.append(error)

            if self.verbose:
                CFL = c * dt / dx
                print(f"  nx={nx:4d}, CFL={CFL:.3f}: L∞ error = {error:.6e}")

        # Convergence
        errors = np.array(errors)
        dx_values = np.array(dx_values)

        if len(errors) > 1:
            p = np.polyfit(np.log(dx_values), np.log(errors), 1)
            convergence_rate = p[0]

            if self.verbose:
                print(f"\nConvergence rate: {convergence_rate:.2f}")
                print(f"Expected: 2.0 (second-order accurate)")

        self.results['wave_1d'] = {
            'errors': errors.tolist(),
            'dx_values': dx_values.tolist(),
            'convergence_rate': convergence_rate if len(errors) > 1 else None
        }

        return self.results['wave_1d']

    def validate_poisson_equation(self, n_values: List[int] = [11, 21, 41, 81]) -> Dict:
        """
        Validate Poisson equation solver
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Poisson Equation Validation")
            print("="*60)

        errors = []
        dx_values = []
        iterations = []

        for n in n_values:
            # Setup
            domain = ((0, 1), (0, 1))
            dx = 1.0 / (n - 1)
            dx_values.append(dx)

            # Source term for ∇²u = f
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, y)
            f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

            # Solve
            result = self.solver.solve_poisson_equation(
                domain=domain,
                grid_points=(n, n),
                source_term=f.flatten(),
                boundary_conditions={'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
                tol=1e-8,
                method='gauss-seidel'
            )

            # Analytical
            u_exact = self.analytical.poisson_2d_rectangle(x, y)

            # Error
            u_numerical = result['solution']
            error = np.max(np.abs(u_numerical - u_exact))
            errors.append(error)
            iterations.append(result['iterations'])

            if self.verbose:
                print(f"  n={n:3d}×{n:<3d}, iterations={result['iterations']:5d}: "
                      f"L∞ error = {error:.6e}")

        # Convergence
        errors = np.array(errors)
        dx_values = np.array(dx_values)

        if len(errors) > 1:
            p = np.polyfit(np.log(dx_values), np.log(errors), 1)
            convergence_rate = p[0]

            if self.verbose:
                print(f"\nConvergence rate: {convergence_rate:.2f}")
                print(f"Expected: 2.0 (second-order accurate)")

        self.results['poisson_2d'] = {
            'errors': errors.tolist(),
            'dx_values': dx_values.tolist(),
            'iterations': iterations,
            'convergence_rate': convergence_rate if len(errors) > 1 else None
        }

        return self.results['poisson_2d']

    def test_stability(self) -> Dict:
        """
        Test numerical stability with various CFL numbers
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Stability Analysis")
            print("="*60)

        # Test explicit method stability
        nx = 51
        domain = (0, 1)
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        dx = 1.0 / (nx - 1)
        alpha = 1.0

        CFL_values = [0.4, 0.5, 0.6]  # Below, at, and above stability limit
        stability_results = {}

        for CFL in CFL_values:
            dt = CFL * dx**2 / alpha

            try:
                result = self.solver.solve_heat_equation(
                    domain=(domain,),
                    grid_points=(nx,),
                    initial_condition=u0,
                    boundary_conditions={'left': 0, 'right': 0},
                    thermal_diffusivity=alpha,
                    time_final=0.01,
                    dt=dt,
                    method='explicit'
                )

                # Check if solution is bounded
                max_val = np.max(np.abs(result['solution']))
                is_stable = max_val < 10  # Arbitrary threshold for explosion

                stability_results[CFL] = {
                    'stable': is_stable,
                    'max_value': max_val
                }

                if self.verbose:
                    status = "✅ Stable" if is_stable else "❌ Unstable"
                    print(f"  CFL={CFL:.2f}: {status} (max|u|={max_val:.3e})")

            except Exception as e:
                stability_results[CFL] = {
                    'stable': False,
                    'error': str(e)
                }
                if self.verbose:
                    print(f"  CFL={CFL:.2f}: ❌ Failed ({str(e)[:30]}...)")

        self.results['stability'] = stability_results
        return stability_results

    def generate_report(self, save_plots: bool = True) -> str:
        """
        Generate comprehensive validation report
        """
        report = "\n" + "="*70 + "\n"
        report += "FLUX SOLVER VALIDATION REPORT\n"
        report += "="*70 + "\n\n"

        # Summary
        report += "SUMMARY\n"
        report += "-"*30 + "\n"

        all_tests_passed = True

        for test_name, result in self.results.items():
            if 'convergence_rate' in result and result['convergence_rate']:
                rate = result['convergence_rate']
                expected = 2.0
                passed = abs(rate - expected) < 0.3
                all_tests_passed &= passed

                status = "✅ PASS" if passed else "❌ FAIL"
                report += f"{test_name:15s}: {status} (rate={rate:.2f}, expected={expected:.1f})\n"

        report += "\n"
        report += "Overall: " + ("✅ ALL TESTS PASSED" if all_tests_passed else "⚠️ SOME TESTS FAILED")
        report += "\n\n"

        # Detailed results
        report += "DETAILED RESULTS\n"
        report += "-"*30 + "\n"

        for test_name, result in self.results.items():
            report += f"\n{test_name}:\n"
            if 'errors' in result:
                report += f"  Errors: {result['errors']}\n"
            if 'convergence_rate' in result:
                report += f"  Convergence rate: {result['convergence_rate']:.3f}\n"

        # Generate plots
        if save_plots and len(self.results) > 0:
            self._generate_validation_plots()
            report += "\nPlots saved to validation_plots.png\n"

        return report

    def _generate_validation_plots(self):
        """Generate validation plots"""
        n_tests = len([k for k in self.results.keys() if 'errors' in self.results[k]])
        if n_tests == 0:
            return

        fig, axes = plt.subplots(1, n_tests, figsize=(5*n_tests, 4))
        if n_tests == 1:
            axes = [axes]

        plot_idx = 0
        for test_name, result in self.results.items():
            if 'errors' not in result:
                continue

            ax = axes[plot_idx]
            dx = np.array(result['dx_values'])
            errors = np.array(result['errors'])

            # Log-log plot
            ax.loglog(dx, errors, 'o-', label='Numerical')

            # Reference slope
            if result.get('convergence_rate'):
                rate = result['convergence_rate']
                ref_line = errors[0] * (dx / dx[0])**2
                ax.loglog(dx, ref_line, 'k--', alpha=0.5, label='O(dx²)')

            ax.set_xlabel('Grid spacing (dx)')
            ax.set_ylabel('L∞ Error')
            ax.set_title(test_name.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        plt.tight_layout()
        plt.savefig('validation_plots.png', dpi=150, bbox_inches='tight')
        plt.close()


def run_full_validation():
    """Run complete validation suite"""
    print("🔬 FLUX Numerical Solver Validation Suite")
    print("="*60)

    # Import the finite difference solver
    from .finite_difference import FiniteDifferenceSolver

    # Create solver and validator
    solver = FiniteDifferenceSolver(verbose=False)
    validator = ValidationSuite(solver, verbose=True)

    # Run all validations
    print("\nRunning validation tests...")

    validator.validate_heat_equation_1d()
    validator.validate_heat_equation_2d()
    validator.validate_wave_equation()
    validator.validate_poisson_equation()
    validator.test_stability()

    # Generate report
    report = validator.generate_report(save_plots=True)
    print(report)

    # Save report
    with open('validation_report.txt', 'w') as f:
        f.write(report)

    print("\nValidation complete! Report saved to validation_report.txt")

    return validator.results


if __name__ == "__main__":
    run_full_validation()