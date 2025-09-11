#!/usr/bin/env python3
"""
My First PDE with FLUX - Heat Equation
Follow along with the 5-minute tutorial!
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add FLUX to path
sys.path.insert(0, 'src')

from src.heat_solver import HeatEquationSolver, create_gaussian_initial_condition

def main():
    print("FLUX: My First PDE")
    print("="*40)
    
    # Step 1: Create a solver
    print("Creating FLUX heat equation solver...")
    solver = HeatEquationSolver(nx=50, ny=50, Lx=1.0, Ly=1.0)
    
    # Step 2: Set initial condition (hot spot in center)
    print("Setting up initial condition (hot pancake)...")
    gaussian_ic = create_gaussian_initial_condition(x0=0.5, y0=0.5, sigma=0.1, amplitude=100.0)
    u0 = solver.set_initial_condition(gaussian_ic)
    
    print(f"Initial temperature: {np.max(u0):.1f}°C")
    
    # Step 3: Solve the PDE
    print("Solving heat equation...")
    alpha = 0.1      # Thermal diffusivity (material property)
    dt = 0.001       # Time step
    t_end = 1.0      # Final time
    
    t_array, u_final, u_history = solver.solve(u0, t_end, dt, alpha, save_interval=100)
    
    # Step 4: Show results
    print(f"Final temperature: {np.max(u_final):.1f}°C")
    print(f"Heat diffused correctly: {np.max(u_final) < np.max(u0)}")
    print("")
    print("SUCCESS: You just solved your first PDE with FLUX!")
    print("")
    print("The pancake cooled from {:.1f}°C to {:.1f}°C".format(np.max(u0), np.max(u_final)))
    
    # Step 5: Create awesome visualization
    print("Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial state
    im1 = axes[0].imshow(u_history[0], cmap='hot', extent=[0,1,0,1])
    axes[0].set_title('t = 0: Hot Pancake')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Middle state  
    mid_idx = len(u_history) // 2
    im2 = axes[1].imshow(u_history[mid_idx], cmap='hot', extent=[0,1,0,1])
    axes[1].set_title(f't = {t_array[mid_idx]:.2f}: Cooling Down')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Final state
    im3 = axes[2].imshow(u_final, cmap='hot', extent=[0,1,0,1])
    axes[2].set_title(f't = {t_end}: Cool Pancake')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y') 
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle('FLUX Heat Equation: ∂u/∂t = α∇²u', fontsize=16)
    plt.tight_layout()
    plt.savefig('my_first_flux_pde.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'my_first_flux_pde.png'")
    
    # Show some stats
    print("")
    print("Solution Statistics:")
    print(f"   Grid size: {solver.nx} x {solver.ny}")
    print(f"   Time steps: {len(t_array)}")
    print(f"   Final time: {t_end}")
    print(f"   Initial heat: {np.sum(u0):.2f}")
    print(f"   Final heat: {np.sum(u_final):.2f}")
    print(f"   Heat conservation: {abs(np.sum(u_final) - np.sum(u0)) < 1.0}")
    
    plt.show()
    
    print("")
    print("SUCCESS! You've solved your first PDE with FLUX!")
    print("   Try modifying the initial conditions or material properties!")

if __name__ == "__main__":
    main()