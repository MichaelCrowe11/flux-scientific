// Linear Elastic Structural Analysis
// Demonstrates FEM for solid mechanics

// Define L-shaped bracket geometry
domain bracket = LShapedDomain(
    width = 0.1,    // 10 cm
    height = 0.1,   // 10 cm  
    thickness = 0.01 // 1 cm thick
)

// Create mesh with refinement near stress concentrations
mesh M = UnstructuredMesh(bracket) {
    element_type: quadrilateral,
    max_element_size: 0.005,
    refinement_regions: [
        {region: corners, size: 0.001},
        {region: re_entrant_corner, size: 0.0005}
    ]
}

// Linear elasticity PDE
pde linear_elasticity {
    variables: displacement u = [u_x, u_y]
    
    // Equilibrium equation: ∇·σ + f = 0
    // Constitutive: σ = C : ε
    // Kinematic: ε = (∇u + ∇uᵀ)/2
    
    ∇·(C : (∇u + ∇uᵀ)/2) + f = 0  in bracket
    
    boundary {
        // Fixed boundary (clamped at left edge)
        u = [0.0, 0.0]  on left_edge
        
        // Applied load (distributed force on top)
        σ·n = [0.0, -P]  on top_edge
        
        // Free boundaries (traction-free)
        σ·n = [0.0, 0.0]  on free_surfaces
    }
    
    // Body forces (self-weight)
    f = [0.0, -ρ*g]
}

// Material properties (Aluminum 6061)
const E = 69e9        // Young's modulus [Pa]
const ν = 0.33        // Poisson's ratio
const ρ = 2700        // Density [kg/m³]
const g = 9.81        // Gravity [m/s²]

// Applied load
const P = 1000.0      // Pressure [Pa]

// Compute elasticity tensor
function compute_elasticity_tensor(E, ν) -> Tensor {
    // Plane stress elasticity matrix
    factor = E / (1 - ν²)
    C = factor * [
        [1,  ν,  0],
        [ν,  1,  0], 
        [0,  0,  (1-ν)/2]
    ]
    return C
}

C = compute_elasticity_tensor(E, ν)

// FEM solver with direct factorization
solver = FEM(
    basis_functions = Lagrange(degree=2),
    quadrature = Gauss(order=3),
    linear_solver = DirectSparse(
        factorization = Cholesky,
        reordering = METIS
    ),
    integration_scheme = full
)

// Solve structural problem
solution = solve(linear_elasticity, mesh=M, solver=solver)

// Post-processing: compute stress and strain
strain = (∇solution.u + ∇solution.uᵀ) / 2
stress = C : strain

// Derived quantities
von_mises_stress = √(3/2 * |stress_dev|²)  // von Mises equivalent stress
principal_stresses = eigenvalues(stress)
displacement_magnitude = |solution.u|

// Safety factor analysis
yield_strength = 276e6  // Yield strength of Al 6061 [Pa]
safety_factor = yield_strength / von_mises_stress

// Find critical locations
max_stress_location = argmax(von_mises_stress)
max_displacement_location = argmax(displacement_magnitude)

// Modal analysis for natural frequencies
pde modal_analysis {
    // Eigenvalue problem: K·φ = ω²·M·φ
    // where K is stiffness matrix, M is mass matrix
    
    ∇·(C : (∇φ + ∇φᵀ)/2) = ω²·ρ·φ  in bracket
    
    boundary {
        φ = [0.0, 0.0]  on left_edge  // Fixed boundary
    }
}

// Solve eigenvalue problem for first 10 modes
modes = solve_eigenvalue(modal_analysis, num_modes=10, solver=ARPACK)

natural_frequencies = √(modes.eigenvalues) / (2*π)  // Convert to Hz

// Export results
export(displacement_magnitude, format="vtk", filename="displacement.vtk")
export(von_mises_stress, format="vtk", filename="stress.vtk")
export(safety_factor, format="vtk", filename="safety_factor.vtk")
export(modes.eigenvectors, format="vtk", filename="mode_shapes.vtk")

// Generate report
print("=== Structural Analysis Results ===")
print("Material: Aluminum 6061")
print("Applied pressure: " + str(P) + " Pa")
print("")
print("Maximum displacement: " + str(max(displacement_magnitude)*1000) + " mm")
print("Maximum von Mises stress: " + str(max(von_mises_stress)/1e6) + " MPa")
print("Minimum safety factor: " + str(min(safety_factor)))
print("")
print("Natural frequencies [Hz]:")
for i, freq in enumerate(natural_frequencies) {
    print("Mode " + str(i+1) + ": " + str(freq))
}

// Fatigue analysis (simplified)
function estimate_fatigue_life(stress_amplitude, material="Al6061") -> int {
    // Basquin's law: N = (Sf'/σa)^(1/b)
    if material == "Al6061" {
        Sf_prime = 138e6  // Fatigue strength coefficient [Pa]
        b = -0.085        // Fatigue strength exponent
    }
    
    N = (Sf_prime / stress_amplitude)^(1/b)
    return int(N)
}

stress_amplitude = max(von_mises_stress) / 2  // Assume fully reversed loading
fatigue_life = estimate_fatigue_life(stress_amplitude)

print("")
print("Estimated fatigue life: " + str(fatigue_life) + " cycles")

// Design optimization setup (conceptual)
optimization design_optimization {
    objective: minimize(mass)
    
    constraints: [
        max(von_mises_stress) <= yield_strength / safety_factor_target,
        min(natural_frequencies) >= min_frequency_target,
        max(displacement_magnitude) <= max_displacement_allowed
    ]
    
    design_variables: [
        thickness ∈ [0.005, 0.02],
        fillet_radius ∈ [0.001, 0.01]
    ]
    
    method: TopologyOptimization(
        density_filter = true,
        volume_fraction = 0.5
    )
}

print("")
print("Optimization setup ready for topology optimization")
print("Target: Minimize mass while meeting structural constraints")