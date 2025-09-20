"""
Visualization utilities for FLUX Scientific Computing
Provides plotting and animation capabilities for PDE solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Callable, Any
import os


class FluxVisualizer:
    """Main visualization class for FLUX solutions"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.figures = []

    def plot_2d_field(self,
                      field: np.ndarray,
                      x: Optional[np.ndarray] = None,
                      y: Optional[np.ndarray] = None,
                      title: str = "Field Plot",
                      xlabel: str = "x",
                      ylabel: str = "y",
                      cmap: str = "viridis",
                      levels: int = 20,
                      show_colorbar: bool = True,
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot 2D scalar field"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if x is None:
            x = np.arange(field.shape[1])
        if y is None:
            y = np.arange(field.shape[0])

        X, Y = np.meshgrid(x, y) if x.ndim == 1 else (x, y)

        # Create contour plot
        cs = ax.contourf(X, Y, field, levels=levels, cmap=cmap)

        if show_colorbar:
            cbar = plt.colorbar(cs, ax=ax, label='Value')
            cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def plot_3d_surface(self,
                        field: np.ndarray,
                        x: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None,
                        title: str = "3D Surface",
                        xlabel: str = "x",
                        ylabel: str = "y",
                        zlabel: str = "z",
                        cmap: str = "viridis",
                        alpha: float = 0.9,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot 3D surface"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        if x is None:
            x = np.arange(field.shape[1])
        if y is None:
            y = np.arange(field.shape[0])

        X, Y = np.meshgrid(x, y) if x.ndim == 1 else (x, y)

        # Create surface plot
        surf = ax.plot_surface(X, Y, field, cmap=cmap, alpha=alpha,
                              linewidth=0, antialiased=True)

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set viewing angle
        ax.view_init(elev=30, azim=45)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def plot_vector_field(self,
                         u: np.ndarray,
                         v: np.ndarray,
                         x: Optional[np.ndarray] = None,
                         y: Optional[np.ndarray] = None,
                         title: str = "Vector Field",
                         xlabel: str = "x",
                         ylabel: str = "y",
                         arrow_scale: int = 20,
                         background: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot 2D vector field with optional background"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if x is None:
            x = np.arange(u.shape[1])
        if y is None:
            y = np.arange(u.shape[0])

        X, Y = np.meshgrid(x, y) if x.ndim == 1 else (x, y)

        # Plot background if provided
        if background is not None:
            cs = ax.contourf(X, Y, background, levels=20, cmap='viridis', alpha=0.6)
            plt.colorbar(cs, ax=ax, label='Magnitude')

        # Plot vector field
        skip = max(1, min(u.shape[0], u.shape[1]) // arrow_scale)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 u[::skip, ::skip], v[::skip, ::skip],
                 angles='xy', scale_units='xy', scale=1)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def plot_streamlines(self,
                        u: np.ndarray,
                        v: np.ndarray,
                        x: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None,
                        title: str = "Streamlines",
                        density: float = 1.5,
                        color_field: Optional[np.ndarray] = None,
                        cmap: str = "viridis",
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot streamlines of vector field"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if x is None:
            x = np.linspace(0, 1, u.shape[1])
        if y is None:
            y = np.linspace(0, 1, u.shape[0])

        # Color streamlines by magnitude or custom field
        if color_field is None:
            color_field = np.sqrt(u**2 + v**2)

        stream = ax.streamplot(x, y, u, v, color=color_field,
                              cmap=cmap, density=density, linewidth=1.5)

        # Add colorbar
        cbar = fig.colorbar(stream.lines, ax=ax, label='Speed')

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def plot_convergence(self,
                        residuals: List[float],
                        iterations: Optional[List[int]] = None,
                        title: str = "Convergence History",
                        xlabel: str = "Iteration",
                        ylabel: str = "Residual",
                        log_scale: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot convergence history"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        if iterations is None:
            iterations = list(range(len(residuals)))

        ax.plot(iterations, residuals, 'b-', linewidth=2, label='Residual')

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def create_animation(self,
                        time_series: List[np.ndarray],
                        times: Optional[List[float]] = None,
                        title: str = "Solution Evolution",
                        cmap: str = "viridis",
                        save_path: Optional[str] = None,
                        fps: int = 10) -> animation.FuncAnimation:
        """Create animation of time-evolving field"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if times is None:
            times = list(range(len(time_series)))

        # Initial plot
        vmin = min(np.min(field) for field in time_series)
        vmax = max(np.max(field) for field in time_series)

        im = ax.imshow(time_series[0], cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect='auto', origin='lower')

        cbar = plt.colorbar(im, ax=ax)
        ax.set_title(f"{title} (t = {times[0]:.3f})")

        def update(frame):
            im.set_array(time_series[frame])
            ax.set_title(f"{title} (t = {times[frame]:.3f})")
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=len(time_series),
                                     interval=1000/fps, blit=True)

        if save_path:
            if save_path.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            else:
                writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")

        return anim

    def plot_comparison(self,
                       fields: List[np.ndarray],
                       titles: List[str],
                       main_title: str = "Comparison",
                       cmap: str = "viridis",
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot multiple fields for comparison"""
        n_fields = len(fields)
        n_cols = min(3, n_fields)
        n_rows = (n_fields + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(5*n_cols, 4*n_rows),
                                dpi=self.dpi)

        if n_fields == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Find global min/max for consistent coloring
        vmin = min(np.min(field) for field in fields)
        vmax = max(np.max(field) for field in fields)

        for i, (field, title) in enumerate(zip(fields, titles)):
            if i < n_fields:
                im = axes[i].imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                                   aspect='auto', origin='lower')
                axes[i].set_title(title, fontsize=12)
                plt.colorbar(im, ax=axes[i])
            else:
                axes[i].axis('off')

        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def plot_mesh(self,
                  mesh,
                  show_nodes: bool = True,
                  show_cells: bool = True,
                  show_boundaries: bool = True,
                  title: str = "Mesh",
                  save_path: Optional[str] = None) -> plt.Figure:
        """Visualize computational mesh"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot cells
        if show_cells and hasattr(mesh, 'cells'):
            for cell in mesh.cells:
                if cell.type == "triangle":
                    nodes = [mesh.nodes[i] for i in cell.nodes]
                    triangle = plt.Polygon([(n.x, n.y) for n in nodes],
                                          fill=False, edgecolor='blue', linewidth=0.5)
                    ax.add_patch(triangle)
                elif cell.type == "quad":
                    nodes = [mesh.nodes[i] for i in cell.nodes]
                    quad = plt.Polygon([(n.x, n.y) for n in nodes],
                                      fill=False, edgecolor='blue', linewidth=0.5)
                    ax.add_patch(quad)

        # Plot nodes
        if show_nodes and hasattr(mesh, 'nodes'):
            x_coords = [node.x for node in mesh.nodes]
            y_coords = [node.y for node in mesh.nodes]
            ax.plot(x_coords, y_coords, 'ko', markersize=2)

        # Plot boundaries
        if show_boundaries and hasattr(mesh, 'boundaries'):
            boundary_colors = {'wall': 'red', 'inlet': 'green',
                             'outlet': 'orange', 'symmetry': 'purple'}

            for boundary in mesh.boundaries:
                color = boundary_colors.get(boundary.type, 'black')
                for face in boundary.faces:
                    if len(face) == 2:  # Edge
                        n1, n2 = mesh.nodes[face[0]], mesh.nodes[face[1]]
                        ax.plot([n1.x, n2.x], [n1.y, n2.y],
                               color=color, linewidth=2, label=boundary.name)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        self.figures.append(fig)
        return fig

    def close_all(self):
        """Close all figures"""
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()


class SolutionMonitor:
    """Real-time monitoring of solution convergence"""

    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.fig = None
        self.axes = None
        self.data = {
            'iterations': [],
            'residuals': [],
            'max_values': [],
            'energies': []
        }

    def initialize_plot(self):
        """Initialize monitoring plots"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.ion()  # Interactive mode

        # Setup axes
        self.axes[0, 0].set_title('Residual')
        self.axes[0, 0].set_xlabel('Iteration')
        self.axes[0, 0].set_ylabel('Residual')
        self.axes[0, 0].set_yscale('log')

        self.axes[0, 1].set_title('Maximum Value')
        self.axes[0, 1].set_xlabel('Iteration')
        self.axes[0, 1].set_ylabel('Max Value')

        self.axes[1, 0].set_title('Energy')
        self.axes[1, 0].set_xlabel('Iteration')
        self.axes[1, 0].set_ylabel('Energy')

        self.axes[1, 1].set_title('Solution Snapshot')
        self.axes[1, 1].set_xlabel('x')
        self.axes[1, 1].set_ylabel('y')

    def update(self,
               iteration: int,
               residual: float,
               max_value: float,
               energy: float,
               solution: Optional[np.ndarray] = None):
        """Update monitoring plots"""
        self.data['iterations'].append(iteration)
        self.data['residuals'].append(residual)
        self.data['max_values'].append(max_value)
        self.data['energies'].append(energy)

        if iteration % self.update_interval == 0:
            # Update plots
            self.axes[0, 0].clear()
            self.axes[0, 0].semilogy(self.data['iterations'],
                                    self.data['residuals'], 'b-')
            self.axes[0, 0].set_xlabel('Iteration')
            self.axes[0, 0].set_ylabel('Residual')
            self.axes[0, 0].set_title('Residual')

            self.axes[0, 1].clear()
            self.axes[0, 1].plot(self.data['iterations'],
                                self.data['max_values'], 'r-')
            self.axes[0, 1].set_xlabel('Iteration')
            self.axes[0, 1].set_ylabel('Max Value')
            self.axes[0, 1].set_title('Maximum Value')

            self.axes[1, 0].clear()
            self.axes[1, 0].plot(self.data['iterations'],
                                self.data['energies'], 'g-')
            self.axes[1, 0].set_xlabel('Iteration')
            self.axes[1, 0].set_ylabel('Energy')
            self.axes[1, 0].set_title('Energy')

            if solution is not None:
                self.axes[1, 1].clear()
                im = self.axes[1, 1].imshow(solution, cmap='viridis',
                                           aspect='auto', origin='lower')
                self.axes[1, 1].set_title(f'Solution at iter {iteration}')

            plt.tight_layout()
            plt.pause(0.01)

    def finalize(self, save_path: Optional[str] = None):
        """Finalize and save monitoring results"""
        plt.ioff()
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Monitoring results saved to {save_path}")


def create_report(solution_data: dict,
                  output_dir: str = "flux_results",
                  report_name: str = "solution_report.html"):
    """Generate HTML report with solution visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FLUX Solution Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .section { margin: 20px 0; }
            img { max-width: 100%; height: auto; margin: 10px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>FLUX Scientific Computing - Solution Report</h1>
    """

    # Add problem description
    if 'problem' in solution_data:
        html_content += f"""
        <div class="section">
            <h2>Problem Description</h2>
            <p>{solution_data['problem']}</p>
        </div>
        """

    # Add solution statistics
    if 'statistics' in solution_data:
        html_content += """
        <div class="section">
            <h2>Solution Statistics</h2>
            <table>
        """
        for key, value in solution_data['statistics'].items():
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html_content += """
            </table>
        </div>
        """

    # Add plots
    if 'plots' in solution_data:
        html_content += """
        <div class="section">
            <h2>Visualizations</h2>
        """
        for plot in solution_data['plots']:
            html_content += f'<img src="{plot}" alt="Solution plot"><br>'
        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    # Write report
    report_path = os.path.join(output_dir, report_name)
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Report generated: {report_path}")
    return report_path