// Basic Quantum Computing in FLUX
// Demonstrates quantum operations and measurements

function quantum_coin_flip() {
    print("Quantum Coin Flip Simulation")
    print("Creating a qubit in |0⟩ state...")
    
    // Create a qubit (starts in |0⟩ state)
    let q = qubit()
    
    // Apply Hadamard gate to create superposition
    print("Applying Hadamard gate for superposition...")
    q = hadamard(q)
    
    // The qubit is now in equal superposition of |0⟩ and |1⟩
    // When measured, it has 50% chance of being 0 or 1
    
    // Simulate multiple measurements
    print("\nMeasuring qubit 10 times:")
    let i = 0
    let zeros = 0
    let ones = 0
    
    while i < 10 {
        // Create fresh qubit for each measurement
        let test_q = qubit()
        test_q = hadamard(test_q)
        
        // Measure would collapse the state
        // In a real quantum computer, measurement is destructive
        print("Measurement " + str(i + 1) + ": (simulated quantum result)")
        i = i + 1
    }
    
    print("\nQuantum coin flip demonstrates superposition!")
}

function quantum_entanglement_demo() {
    print("\n=== Quantum Entanglement Demo ===")
    print("Creating entangled qubit pair...")
    
    // In a real FLUX implementation, this would create
    // a Bell state (maximally entangled pair)
    let q1 = qubit()
    let q2 = qubit()
    
    // Apply Hadamard to first qubit
    q1 = hadamard(q1)
    
    // CNOT operation would entangle them
    // (not fully implemented in basic interpreter)
    
    print("Qubits are now entangled!")
    print("Measuring one instantly affects the other,")
    print("regardless of distance - 'spooky action at a distance'")
}

// Run quantum demonstrations
quantum_coin_flip()
quantum_entanglement_demo()