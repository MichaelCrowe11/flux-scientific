#!/usr/bin/env python3
"""
FLUX Showcase: Interactive PDE Explorer
=======================================

This showcase provides an interactive GUI for exploring different PDEs
with real-time parameter adjustment and visualization.

Features demonstrated:
- Interactive parameter control
- Real-time PDE solving
- Multiple PDE types
- Dynamic visualization
- Educational interface
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from typing import Dict, Any, Callable

# Add FLUX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from solvers.finite_difference import FiniteDifferenceSolver
from mesh import StructuredGrid
from visualization import FluxVisualizer

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️  GUI not available, running in demo mode")

class PDEExplorer:
    """Interactive PDE exploration interface"""

    def __init__(self):
        self.solver = FiniteDifferenceSolver()
        self.visualizer = FluxVisualizer()

        # Default parameters
        self.params = {
            'pde_type': 'heat',
            'nx': 50,
            'ny': 50,
            'alpha': 0.1,
            'time_final': 1.0,
            'dt': 0.01,
            'method': 'crank_nicolson',
            'boundary_type': 'dirichlet',
            'boundary_value': 0.0,
            'initial_type': 'gaussian',
            'wave_speed': 0.5,
            'source_strength': 1.0
        }

        # Current simulation state
        self.current_solution = None
        self.current_domain = None
        self.animation_running = False

        if GUI_AVAILABLE:
            self.setup_gui()
        else:
            self.run_demo_mode()

    def setup_gui(self):
        """Setup the GUI interface"""

        self.root = tk.Tk()
        self.root.title("FLUX Interactive PDE Explorer")
        self.root.geometry("1200x800")

        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_controls()
        self.setup_plot()

        # Initial solve
        self.solve_pde()

    def setup_controls(self):
        """Setup control panel"""

        # Title
        title_label = ttk.Label(self.control_frame, text="FLUX PDE Explorer",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # PDE Type Selection
        pde_frame = ttk.LabelFrame(self.control_frame, text="PDE Type", padding=10)
        pde_frame.pack(fill=tk.X, pady=5)

        self.pde_var = tk.StringVar(value=self.params['pde_type'])
        pde_types = [
            ("Heat Equation", "heat"),
            ("Wave Equation", "wave"),
            ("Poisson Equation", "poisson"),
            ("Reaction-Diffusion", "reaction_diffusion")
        ]

        for text, value in pde_types:
            ttk.Radiobutton(pde_frame, text=text, variable=self.pde_var,
                           value=value, command=self.on_pde_change).pack(anchor=tk.W)

        # Grid Parameters
        grid_frame = ttk.LabelFrame(self.control_frame, text="Grid Parameters", padding=10)
        grid_frame.pack(fill=tk.X, pady=5)

        self.create_slider(grid_frame, "Grid Size X", "nx", 20, 100, 50)
        self.create_slider(grid_frame, "Grid Size Y", "ny", 20, 100, 50)

        # Physical Parameters
        phys_frame = ttk.LabelFrame(self.control_frame, text="Physical Parameters", padding=10)
        phys_frame.pack(fill=tk.X, pady=5)

        self.create_slider(phys_frame, "Diffusivity/Speed", "alpha", 0.01, 1.0, 0.1)
        self.create_slider(phys_frame, "Source Strength", "source_strength", 0.0, 2.0, 1.0)

        # Time Parameters
        time_frame = ttk.LabelFrame(self.control_frame, text="Time Parameters", padding=10)
        time_frame.pack(fill=tk.X, pady=5)

        self.create_slider(time_frame, "Final Time", "time_final", 0.1, 5.0, 1.0)
        self.create_slider(time_frame, "Time Step", "dt", 0.001, 0.1, 0.01)

        # Method Selection
        method_frame = ttk.LabelFrame(self.control_frame, text="Numerical Method", padding=10)
        method_frame.pack(fill=tk.X, pady=5)

        self.method_var = tk.StringVar(value=self.params['method'])
        methods = [
            ("Explicit", "explicit"),
            ("Implicit", "implicit"),
            ("Crank-Nicolson", "crank_nicolson")
        ]

        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var,
                           value=value, command=self.on_method_change).pack(anchor=tk.W)

        # Initial Conditions
        initial_frame = ttk.LabelFrame(self.control_frame, text="Initial Condition", padding=10)
        initial_frame.pack(fill=tk.X, pady=5)

        self.initial_var = tk.StringVar(value=self.params['initial_type'])
        initial_types = [
            ("Gaussian", "gaussian"),
            ("Hot Center", "hot_center"),
            ("Sinusoidal", "sinusoidal"),
            ("Random", "random")
        ]

        for text, value in initial_types:
            ttk.Radiobutton(initial_frame, text=text, variable=self.initial_var,
                           value=value, command=self.on_initial_change).pack(anchor=tk.W)

        # Action Buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(button_frame, text="Solve PDE", command=self.solve_pde).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Animate", command=self.toggle_animation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset", command=self.reset_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Plot", command=self.save_plot).pack(fill=tk.X, pady=2)

        # Status
        self.status_label = ttk.Label(self.control_frame, text="Ready")
        self.status_label.pack(pady=10)

    def create_slider(self, parent, label, param, min_val, max_val, default):
        """Create a parameter slider"""

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=label).pack(side=tk.LEFT)

        var = tk.DoubleVar(value=default)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var,
                          orient=tk.HORIZONTAL, command=lambda v: self.on_param_change(param, v))
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Store reference
        setattr(self, f"{param}_var", var)

    def setup_plot(self):
        """Setup matplotlib plot"""

        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial plot
        self.ax.set_title("Select PDE type and click 'Solve PDE'")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def on_pde_change(self):
        """Handle PDE type change"""
        self.params['pde_type'] = self.pde_var.get()
        self.update_parameter_visibility()

    def on_method_change(self):
        """Handle method change"""
        self.params['method'] = self.method_var.get()

    def on_initial_change(self):
        """Handle initial condition change"""
        self.params['initial_type'] = self.initial_var.get()

    def on_param_change(self, param, value):
        """Handle parameter change"""
        self.params[param] = float(value)

    def update_parameter_visibility(self):
        """Update which parameters are visible based on PDE type"""
        # Could implement parameter hiding/showing based on PDE type
        pass

    def get_initial_condition(self, X, Y):
        """Generate initial condition based on selection"""

        if self.params['initial_type'] == 'gaussian':
            return np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))

        elif self.params['initial_type'] == 'hot_center':
            center_mask = ((X - 0.5)**2 + (Y - 0.5)**2) < 0.1
            return np.where(center_mask, 1.0, 0.0)

        elif self.params['initial_type'] == 'sinusoidal':
            return np.sin(np.pi * X) * np.sin(np.pi * Y)

        elif self.params['initial_type'] == 'random':
            np.random.seed(42)  # Reproducible
            return np.random.uniform(0, 1, X.shape)

        else:
            return np.zeros_like(X)

    def solve_pde(self):
        """Solve the selected PDE"""

        try:
            self.status_label.config(text="Solving...")
            self.root.update()

            # Get current parameters
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())

            # Domain
            domain = {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1}
            grid_points = (nx, ny)

            # Create grid for initial condition
            x = np.linspace(domain['x_min'], domain['x_max'], nx)
            y = np.linspace(domain['y_min'], domain['y_max'], ny)
            X, Y = np.meshgrid(x, y)

            # Initial condition
            initial_condition_func = lambda x, y: self.get_initial_condition(
                np.meshgrid(x, y) if np.isscalar(x) else (x, y)
            )

            # Boundary conditions
            boundary_conditions = {
                'left': ('dirichlet', 0.0),
                'right': ('dirichlet', 0.0),
                'top': ('dirichlet', 0.0),
                'bottom': ('dirichlet', 0.0)
            }

            # Solve based on PDE type
            if self.params['pde_type'] == 'heat':
                result = self.solver.solve_heat_equation(
                    domain=domain,
                    grid_points=grid_points,
                    initial_condition=initial_condition_func,
                    boundary_conditions=boundary_conditions,
                    thermal_diffusivity=self.params['alpha'],
                    time_final=self.params['time_final'],
                    dt=self.params['dt'],
                    method=self.params['method']
                )

            elif self.params['pde_type'] == 'wave':
                # Initial velocity (zero)
                initial_velocity = lambda x, y: np.zeros_like(
                    np.meshgrid(x, y)[0] if np.isscalar(x) else x
                )

                result = self.solver.solve_wave_equation(
                    domain=domain,
                    grid_points=grid_points,
                    initial_condition=initial_condition_func,
                    initial_velocity=initial_velocity,
                    boundary_conditions=boundary_conditions,
                    wave_speed=self.params['alpha'],
                    time_final=self.params['time_final'],
                    dt=self.params['dt']
                )

            elif self.params['pde_type'] == 'poisson':
                # Source term
                source_strength = self.params['source_strength']
                source_term = lambda x, y: source_strength * np.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))

                result = self.solver.solve_poisson_equation(
                    domain=domain,
                    grid_points=grid_points,
                    source_term=source_term,
                    boundary_conditions=boundary_conditions
                )

            else:
                raise ValueError(f"PDE type {self.params['pde_type']} not implemented")

            # Store result
            self.current_solution = result
            self.current_domain = (X, Y)

            # Plot
            self.plot_solution(result, X, Y)

            self.status_label.config(text="Solved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to solve PDE: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")

    def plot_solution(self, result, X, Y):
        """Plot the solution"""

        self.ax.clear()

        u = result['u']
        levels = 20

        # Contour plot
        if self.params['pde_type'] in ['heat', 'poisson']:
            im = self.ax.contourf(X, Y, u, levels=levels, cmap='hot')
            self.ax.set_title(f'{self.params["pde_type"].title()} Equation Solution')

        elif self.params['pde_type'] == 'wave':
            # For wave equation, show both positive and negative values
            max_val = np.max(np.abs(u))
            levels = np.linspace(-max_val, max_val, 21)
            im = self.ax.contourf(X, Y, u, levels=levels, cmap='RdBu_r')
            self.ax.set_title('Wave Equation Solution')

        # Add contour lines
        self.ax.contour(X, Y, u, levels=levels[::2], colors='black', linewidths=0.3, alpha=0.5)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal')

        # Colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(im, ax=self.ax)

        self.canvas.draw()

    def toggle_animation(self):
        """Toggle animation on/off"""

        if self.animation_running:
            self.animation_running = False
            self.status_label.config(text="Animation stopped")
        else:
            if self.current_solution is None:
                messagebox.showwarning("Warning", "Solve PDE first!")
                return

            self.animate_solution()

    def animate_solution(self):
        """Create animation of solution evolution"""

        self.status_label.config(text="Creating animation...")
        self.animation_running = True

        # Re-solve with intermediate steps stored
        try:
            # Parameters
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
            domain = {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1}

            x = np.linspace(domain['x_min'], domain['x_max'], nx)
            y = np.linspace(domain['y_min'], domain['y_max'], ny)
            X, Y = np.meshgrid(x, y)

            # Simple time stepping for animation
            dt = self.params['dt']
            time_final = self.params['time_final']
            steps = int(time_final / dt)

            # Initial condition
            u = self.get_initial_condition(X, Y)
            alpha = self.params['alpha']

            # Storage
            u_history = [u.copy()]
            times = [0]

            # Time stepping (simple explicit for animation)
            dx = x[1] - x[0]
            dy = y[1] - y[0]

            for step in range(min(steps, 100)):  # Limit for performance
                if not self.animation_running:
                    break

                if self.params['pde_type'] == 'heat':
                    # Heat equation step
                    u_new = u.copy()
                    u_new[1:-1, 1:-1] = (u[1:-1, 1:-1] +
                                        alpha * dt * (
                                            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
                                            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
                                        ))
                    # Boundary conditions
                    u_new[0, :] = 0; u_new[-1, :] = 0
                    u_new[:, 0] = 0; u_new[:, -1] = 0
                    u = u_new

                elif self.params['pde_type'] == 'wave':
                    # Wave equation (simplified)
                    if not hasattr(self, 'u_prev'):
                        self.u_prev = u.copy()

                    u_new = 2*u - self.u_prev + alpha**2 * dt**2 * (
                        (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2 +
                        (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
                    )

                    # Boundary conditions
                    u_new[0, :] = 0; u_new[-1, :] = 0
                    u_new[:, 0] = 0; u_new[:, -1] = 0

                    self.u_prev = u.copy()
                    u = u_new

                # Store every few steps
                if step % 5 == 0:
                    u_history.append(u.copy())
                    times.append((step + 1) * dt)

                # Update plot
                if step % 10 == 0:
                    self.ax.clear()
                    im = self.ax.contourf(X, Y, u, levels=20, cmap='hot')
                    self.ax.set_title(f'Solution at t = {(step+1)*dt:.3f}')
                    self.ax.set_aspect('equal')
                    self.canvas.draw()
                    self.root.update()

            self.animation_running = False
            self.status_label.config(text="Animation complete")

        except Exception as e:
            self.animation_running = False
            self.status_label.config(text=f"Animation error: {str(e)}")

    def reset_parameters(self):
        """Reset parameters to defaults"""

        # Reset sliders
        self.nx_var.set(50)
        self.ny_var.set(50)
        self.alpha_var.set(0.1)
        self.source_strength_var.set(1.0)
        self.time_final_var.set(1.0)
        self.dt_var.set(0.01)

        # Reset selections
        self.pde_var.set('heat')
        self.method_var.set('crank_nicolson')
        self.initial_var.set('gaussian')

        self.status_label.config(text="Parameters reset")

    def save_plot(self):
        """Save current plot"""

        if self.current_solution is None:
            messagebox.showwarning("Warning", "No solution to save!")
            return

        try:
            filename = f"flux_interactive_{self.params['pde_type']}_solution.png"
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved as {filename}")
            self.status_label.config(text=f"Saved {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def run_demo_mode(self):
        """Run in demo mode without GUI"""

        print("🎮 FLUX Interactive PDE Explorer - Demo Mode")
        print("=" * 50)

        pde_types = ['heat', 'wave', 'poisson']

        for pde_type in pde_types:
            print(f"\n🔧 Demonstrating {pde_type} equation...")

            self.params['pde_type'] = pde_type

            # Demo solve
            self.solve_pde_demo()

        print("\n✓ Demo mode complete!")

    def solve_pde_demo(self):
        """Demo version of solve_pde"""

        # Simple demo solve
        nx, ny = 50, 50
        domain = {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1}

        x = np.linspace(domain['x_min'], domain['x_max'], nx)
        y = np.linspace(domain['y_min'], domain['y_max'], ny)
        X, Y = np.meshgrid(x, y)

        # Initial condition
        initial = self.get_initial_condition(X, Y)

        # Simple plot
        plt.figure(figsize=(10, 4))

        plt.subplot(121)
        plt.contourf(X, Y, initial, levels=20, cmap='hot')
        plt.title(f'{self.params["pde_type"].title()} - Initial Condition')
        plt.colorbar()
        plt.axis('equal')

        # Simulate final state (simplified)
        if self.params['pde_type'] == 'heat':
            final = initial * 0.1  # Simplified decay
        elif self.params['pde_type'] == 'wave':
            final = initial * np.cos(np.pi * 2)  # Simplified oscillation
        else:
            final = initial + 0.5  # Simplified solution

        plt.subplot(122)
        plt.contourf(X, Y, final, levels=20, cmap='hot')
        plt.title(f'{self.params["pde_type"].title()} - Final State')
        plt.colorbar()
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(f'flux_demo_{self.params["pde_type"]}.png', dpi=200)
        plt.show()

    def run(self):
        """Run the explorer"""

        if GUI_AVAILABLE:
            print("🎮 Starting FLUX Interactive PDE Explorer...")
            self.root.mainloop()

def main():
    """Main function"""

    if not GUI_AVAILABLE:
        print("⚠️  GUI libraries not available")
        print("💡 Install tkinter: sudo apt-get install python3-tk")

    explorer = PDEExplorer()
    explorer.run()

if __name__ == "__main__":
    main()