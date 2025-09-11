// Electromagnetic Scattering from Cylinder
// Solves Maxwell's equations using Finite Element Method

// Define computational domain with PML
domain scattering_domain = Circle(radius=2.0, center=[0.0, 0.0])
domain pml_region = Annulus(inner_radius=2.0, outer_radius=3.0, center=[0.0, 0.0])
domain obstacle = Circle(radius=0.1, center=[0.0, 0.0])

// Create unstructured mesh
mesh M = UnstructuredMesh(scattering_domain) {
    obstacle: obstacle,
    max_element_size: 0.02,
    boundary_layers: 5,
    growth_rate: 1.1
}

// Maxwell's equations in frequency domain
pde maxwell_2d {
    variables: electric_field E
    
    // Helmholtz equation: ∇²E + k²E = 0
    ∇²E + k²E = 0  in scattering_domain
    
    boundary {
        // Perfect Electric Conductor (PEC) - metallic cylinder
        E = 0.0  on obstacle
        
        // Perfectly Matched Layer (PML) boundary
        ∇E·n + ik*E = ∇E_inc·n + ik*E_inc  on pml_region
        
        // Far-field radiation condition
        ∇E·n - ik*E = 0  on outer_boundary
    }
    
    // Incident plane wave
    incident_wave: E_inc = exp(ik*x)  // Propagating in +x direction
}

// EM parameters
const c = 299792458.0      // Speed of light [m/s]
const f = 3.0e9           // Frequency [Hz] (X-band)
const λ = c / f           // Wavelength [m]
const k = 2*π / λ         // Wave number [1/m]
const Z₀ = 376.73         // Free space impedance [Ω]

// FEM solver with complex arithmetic
solver = FEM(
    basis_functions = Lagrange(degree=2),
    quadrature = Gauss(order=4),
    linear_solver = BiCGSTAB(
        preconditioner = ILU,
        tolerance = 1e-8
    ),
    complex_arithmetic = true
)

// Solve Maxwell's equations
solution = solve(maxwell_2d, mesh=M, solver=solver)

// Post-processing: Radar Cross Section (RCS)
function compute_rcs(E_field, incident_field, frequency) -> float {
    // Compute scattered field
    E_scattered = E_field - incident_field
    
    // Integrate over far-field circle
    far_field_radius = 10.0 * λ
    rcs_integral = integrate_circle(|E_scattered|², far_field_radius)
    
    return 4*π * rcs_integral / |incident_field|²
}

rcs = compute_rcs(solution.E, maxwell_2d.incident_wave, f)

// Field intensity and phase
intensity = |solution.E|²
phase = arg(solution.E)

// Near-field to far-field transformation
far_field_pattern = nf2ff_transform(solution.E, mesh=M, frequency=f)

// Export results
export(intensity, format="vtk", filename="em_intensity.vtk")
export(phase, format="vtk", filename="em_phase.vtk") 
export(far_field_pattern, format="csv", filename="radiation_pattern.csv")

// Frequency sweep for broadband analysis
frequencies = linspace(2.0e9, 4.0e9, 50)
rcs_spectrum = []

for freq in frequencies {
    k_sweep = 2*π*freq / c
    maxwell_sweep = maxwell_2d.substitute(k = k_sweep)
    sol_sweep = solve(maxwell_sweep, mesh=M, solver=solver)
    rcs_value = compute_rcs(sol_sweep.E, maxwell_sweep.incident_wave, freq)
    rcs_spectrum.append(rcs_value)
}

export(rcs_spectrum, frequencies, format="csv", filename="rcs_spectrum.csv")

print("Electromagnetic scattering simulation completed")
print("Frequency: " + str(f/1e9) + " GHz")
print("Wavelength: " + str(λ*1000) + " mm")
print("RCS at " + str(f/1e9) + " GHz: " + str(rcs) + " m²")
print("Peak RCS in band: " + str(max(rcs_spectrum)) + " m²")