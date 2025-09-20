#!/usr/bin/env python3
"""
FLUX Showcase: Lid-Driven Cavity Flow
=====================================

This showcase demonstrates advanced CFD capabilities of FLUX by solving
the classic lid-driven cavity flow problem using the Navier-Stokes equations.

Features demonstrated:
- Navier-Stokes equations for incompressible flow
- Fractional step method for pressure-velocity coupling
- Advanced visualization with streamlines and vorticity
- Performance benchmarking
- Reynolds number studies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add FLUX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from solvers.navier_stokes import NavierStokesSolver
from mesh import StructuredGrid
from visualization import FluxVisualizer

class CavityFlowShowcase:
    """Showcase application for lid-driven cavity flow"""

    def __init__(self, nx=64, ny=64, reynolds=100):
        """
        Initialize cavity flow simulation

        Parameters:
        -----------
        nx, ny : int
            Grid resolution
        reynolds : float
            Reynolds number
        """
        self.nx = nx
        self.ny = ny
        self.reynolds = reynolds

        # Create domain
        self.domain = {
            'x_min': 0.0, 'x_max': 1.0,
            'y_min': 0.0, 'y_max': 1.0
        }

        # Create mesh
        self.mesh = StructuredGrid("cavity_mesh")
        self.mesh.create_rectangle(
            self.domain['x_min'], self.domain['x_max'],
            self.domain['y_min'], self.domain['y_max'],
            nx, ny
        )

        # Boundary conditions (lid velocity = 1.0)
        self.boundary_conditions = {
            'u': {
                'top': ('dirichlet', 1.0),
                'bottom': ('dirichlet', 0.0),
                'left': ('dirichlet', 0.0),
                'right': ('dirichlet', 0.0)
            },
            'v': {
                'top': ('dirichlet', 0.0),
                'bottom': ('dirichlet', 0.0),
                'left': ('dirichlet', 0.0),
                'right': ('dirichlet', 0.0)
            }
        }

        # Initialize solver
        self.solver = NavierStokesSolver(
            self.mesh,
            reynolds=reynolds,
            boundary_conditions=self.boundary_conditions
        )

        # Visualization
        self.visualizer = FluxVisualizer()

    def run_simulation(self, max_time=10.0, dt=0.01, save_frequency=10):
        """Run the cavity flow simulation"""

        print(f"🌊 FLUX Cavity Flow Simulation")
        print(f"   Grid: {self.nx}×{self.ny}")
        print(f"   Reynolds number: {self.reynolds}")
        print(f"   Time step: {dt}")
        print(f"   Max time: {max_time}")
        print("=" * 50)

        # Time stepping
        t = 0.0
        step = 0
        start_time = time.time()

        # Storage for animation
        self.time_history = []
        self.velocity_history = []
        self.vorticity_history = []

        while t < max_time:
            # Time step
            residuals = self.solver.time_step(dt)

            # Progress output
            if step % save_frequency == 0:
                # Calculate vorticity
                u, v = self.solver.get_velocity()
                vorticity = self.calculate_vorticity(u, v)
                max_vorticity = np.max(np.abs(vorticity))

                print(f"Step {step:5d}, t={t:6.3f}, "
                      f"max|ω|={max_vorticity:8.4f}, "
                      f"res_u={residuals['u_residual']:.2e}, "
                      f"res_v={residuals['v_residual']:.2e}")

                # Store for visualization
                self.time_history.append(t)
                self.velocity_history.append((u.copy(), v.copy()))
                self.vorticity_history.append(vorticity.copy())

            t += dt
            step += 1

            # Check convergence
            if (residuals['u_residual'] < 1e-8 and
                residuals['v_residual'] < 1e-8):
                print("✓ Converged to steady state!")
                break

        end_time = time.time()
        print(f"\n⏱️  Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"   Steps: {step}")
        print(f"   Performance: {step/(end_time - start_time):.1f} steps/sec")

        return self.solver.get_velocity()

    def calculate_vorticity(self, u, v):
        """Calculate vorticity field"""
        # Get grid spacing
        x = self.mesh.x
        y = self.mesh.y
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Calculate vorticity: ω = ∂v/∂x - ∂u/∂y
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        vorticity = dvdx - dudy

        return vorticity

    def create_advanced_visualization(self):
        """Create comprehensive visualization of results"""

        if not hasattr(self, 'velocity_history'):
            print("❌ No simulation data available. Run simulation first.")
            return

        # Get final state
        u_final, v_final = self.velocity_history[-1]
        vorticity_final = self.vorticity_history[-1]

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Velocity magnitude with streamlines
        ax1 = plt.subplot(2, 3, 1)
        x, y = np.meshgrid(self.mesh.x, self.mesh.y)
        velocity_mag = np.sqrt(u_final**2 + v_final**2)

        im1 = ax1.contourf(x, y, velocity_mag, levels=20, cmap='viridis')
        ax1.streamplot(x, y, u_final, v_final, density=2, color='white', linewidth=0.5)
        plt.colorbar(im1, ax=ax1, label='Velocity Magnitude')
        ax1.set_title(f'Velocity Field (Re={self.reynolds})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')

        # 2. Vorticity contours
        ax2 = plt.subplot(2, 3, 2)
        vort_levels = np.linspace(-10, 10, 21)
        im2 = ax2.contourf(x, y, vorticity_final, levels=vort_levels, cmap='RdBu_r')
        ax2.contour(x, y, vorticity_final, levels=vort_levels[::2], colors='black', linewidths=0.3)
        plt.colorbar(im2, ax=ax2, label='Vorticity')
        ax2.set_title('Vorticity Field')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')

        # 3. Pressure field
        ax3 = plt.subplot(2, 3, 3)
        pressure = self.solver.get_pressure()
        im3 = ax3.contourf(x, y, pressure, levels=20, cmap='coolwarm')
        plt.colorbar(im3, ax=ax3, label='Pressure')
        ax3.set_title('Pressure Field')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')

        # 4. Velocity profiles along centerlines
        ax4 = plt.subplot(2, 3, 4)
        mid_i = self.nx // 2
        mid_j = self.ny // 2

        # Vertical centerline (u-velocity)
        ax4.plot(u_final[:, mid_i], y[:, 0], 'b-', label='u at x=0.5', linewidth=2)
        # Horizontal centerline (v-velocity)
        ax4.plot(x[0, :], v_final[mid_j, :], 'r-', label='v at y=0.5', linewidth=2)

        # Benchmark data (Ghia et al., 1982)
        self.plot_benchmark_data(ax4)

        ax4.set_xlabel('Velocity / Position')
        ax4.set_ylabel('Position / Velocity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Centerline Velocity Profiles')

        # 5. Convergence history
        ax5 = plt.subplot(2, 3, 5)
        if len(self.time_history) > 1:
            max_vorticity = [np.max(np.abs(vort)) for vort in self.vorticity_history]
            ax5.semilogy(self.time_history, max_vorticity, 'g-', linewidth=2)
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Max |Vorticity|')
            ax5.set_title('Convergence History')
            ax5.grid(True, alpha=0.3)

        # 6. Performance metrics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # Display simulation parameters and results
        info_text = f"""
        FLUX Cavity Flow Results
        ========================

        Grid Resolution: {self.nx} × {self.ny}
        Reynolds Number: {self.reynolds}

        Final Statistics:
        • Max Velocity: {np.max(velocity_mag):.4f}
        • Max Vorticity: {np.max(np.abs(vorticity_final)):.4f}
        • Simulation Steps: {len(self.time_history)}

        Computational Performance:
        • Grid Points: {self.nx * self.ny:,}
        • Time Steps: {len(self.time_history)}

        FLUX Features Used:
        ✓ Navier-Stokes solver
        ✓ Fractional step method
        ✓ Advanced visualization
        ✓ Performance monitoring
        """

        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig('flux_cavity_flow_showcase.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_benchmark_data(self, ax):
        """Plot benchmark data from Ghia et al. (1982)"""
        # Benchmark data for Re=100 (representative points)
        if abs(self.reynolds - 100) < 1:
            # u-velocity along vertical centerline
            y_ghia = [0.0, 0.0547, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5,
                     0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0]
            u_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                     -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                     0.68717, 0.73722, 1.0]

            ax.scatter(u_ghia, y_ghia, c='blue', marker='o', s=30,
                      label='Ghia et al. (1982)', zorder=10)

    def create_animation(self, save_path='cavity_flow_animation.gif'):
        """Create animation of the flow evolution"""

        if not hasattr(self, 'velocity_history'):
            print("❌ No simulation data available. Run simulation first.")
            return

        print("🎬 Creating animation...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x, y = np.meshgrid(self.mesh.x, self.mesh.y)

        def animate(frame):
            ax1.clear()
            ax2.clear()

            u, v = self.velocity_history[frame]
            vorticity = self.vorticity_history[frame]
            time = self.time_history[frame]

            # Velocity streamlines
            velocity_mag = np.sqrt(u**2 + v**2)
            im1 = ax1.contourf(x, y, velocity_mag, levels=20, cmap='viridis')
            ax1.streamplot(x, y, u, v, density=2, color='white', linewidth=0.5)
            ax1.set_title(f'Velocity Field (t={time:.2f})')
            ax1.set_aspect('equal')

            # Vorticity
            vort_levels = np.linspace(-10, 10, 21)
            im2 = ax2.contourf(x, y, vorticity, levels=vort_levels, cmap='RdBu_r')
            ax2.set_title(f'Vorticity Field (t={time:.2f})')
            ax2.set_aspect('equal')

            return [im1, im2]

        ani = FuncAnimation(fig, animate, frames=len(self.velocity_history),
                          interval=200, blit=False, repeat=True)

        # Save animation
        ani.save(save_path, writer='pillow', fps=5)
        print(f"✓ Animation saved to {save_path}")

        plt.show()
        return ani

def reynolds_study():
    """Demonstrate Reynolds number effects"""

    print("🔬 FLUX Reynolds Number Study")
    print("=" * 40)

    reynolds_numbers = [50, 100, 400, 1000]
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, re in enumerate(reynolds_numbers):
        print(f"\n🌊 Running Re = {re}...")

        # Create and run simulation
        showcase = CavityFlowShowcase(nx=64, ny=64, reynolds=re)
        u, v = showcase.run_simulation(max_time=5.0, dt=0.005, save_frequency=50)

        # Calculate vorticity
        vorticity = showcase.calculate_vorticity(u, v)

        # Plot results
        x, y = np.meshgrid(showcase.mesh.x, showcase.mesh.y)
        im = axes[i].contourf(x, y, vorticity, levels=20, cmap='RdBu_r')
        axes[i].streamplot(x, y, u, v, density=1.5, color='black', linewidth=0.5)
        axes[i].set_title(f'Re = {re}')
        axes[i].set_aspect('equal')

        # Store results
        results[re] = {
            'u': u, 'v': v, 'vorticity': vorticity,
            'max_vorticity': np.max(np.abs(vorticity))
        }

    plt.tight_layout()
    plt.savefig('flux_reynolds_study.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Summary
    print("\n📊 Reynolds Number Study Results:")
    print("-" * 40)
    for re in reynolds_numbers:
        max_vort = results[re]['max_vorticity']
        print(f"Re = {re:4d}: Max |ω| = {max_vort:.4f}")

    return results

def performance_benchmark():
    """Benchmark FLUX performance"""

    print("⚡ FLUX Performance Benchmark")
    print("=" * 40)

    grid_sizes = [32, 64, 96, 128]
    results = []

    for nx in grid_sizes:
        print(f"\n🔧 Testing {nx}×{nx} grid...")

        showcase = CavityFlowShowcase(nx=nx, ny=nx, reynolds=100)

        start_time = time.time()
        showcase.run_simulation(max_time=1.0, dt=0.01, save_frequency=100)
        end_time = time.time()

        total_time = end_time - start_time
        grid_points = nx * nx
        performance = grid_points / total_time

        results.append({
            'grid_size': nx,
            'grid_points': grid_points,
            'time': total_time,
            'performance': performance
        })

        print(f"   Time: {total_time:.2f} s")
        print(f"   Performance: {performance:.0f} grid points/second")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sizes = [r['grid_size'] for r in results]
    times = [r['time'] for r in results]
    perfs = [r['performance'] for r in results]

    ax1.loglog(sizes, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Grid Size (N×N)')
    ax1.set_ylabel('Computation Time (s)')
    ax1.set_title('FLUX Scaling Performance')
    ax1.grid(True, alpha=0.3)

    ax2.semilogx([r['grid_points'] for r in results], perfs, 'ro-',
                linewidth=2, markersize=8)
    ax2.set_xlabel('Total Grid Points')
    ax2.set_ylabel('Performance (grid points/s)')
    ax2.set_title('FLUX Throughput')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('flux_performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

def main():
    """Main showcase demonstration"""

    print("🚀 FLUX Scientific Computing Language")
    print("     Advanced CFD Showcase")
    print("=" * 50)

    # 1. Basic cavity flow simulation
    print("\n1️⃣  Basic Cavity Flow Simulation")
    showcase = CavityFlowShowcase(nx=64, ny=64, reynolds=100)
    showcase.run_simulation(max_time=10.0, dt=0.01, save_frequency=20)
    showcase.create_advanced_visualization()

    # 2. Create animation
    print("\n2️⃣  Creating Flow Animation")
    showcase.create_animation()

    # 3. Reynolds number study
    print("\n3️⃣  Reynolds Number Study")
    reynolds_study()

    # 4. Performance benchmark
    print("\n4️⃣  Performance Benchmark")
    performance_benchmark()

    print("\n🎉 FLUX Showcase Complete!")
    print("   Generated files:")
    print("   • flux_cavity_flow_showcase.png")
    print("   • cavity_flow_animation.gif")
    print("   • flux_reynolds_study.png")
    print("   • flux_performance_benchmark.png")

if __name__ == "__main__":
    main()