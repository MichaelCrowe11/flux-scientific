# FLUX Scientific Computing Language

[![CI](https://github.com/MichaelCrowe11/flux-sci-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelCrowe11/flux-sci-lang/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MichaelCrowe11/flux-sci-lang/branch/main/graph/badge.svg)](https://codecov.io/gh/MichaelCrowe11/flux-sci-lang)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Domain-Specific Language for PDEs, CFD, and Computational Physics**

FLUX is a high-performance DSL designed for scientific computing, specializing in partial differential equations, computational fluid dynamics, electromagnetic simulations, and finite element analysis.

## Features

### Core Scientific Computing Features
- **Native PDE Syntax**: Write equations in mathematical notation (∂u/∂t = ∇²u)
- **Advanced Mesh Support**: Structured, unstructured, and adaptive mesh refinement
- **Multi-Physics**: CFD, electromagnetics, structural analysis, heat transfer
- **GPU Acceleration**: CUDA kernel generation and parallel execution
- **Modern Solvers**: Finite element, finite volume, and spectral methods
- **Code Generation**: Compile to C++, CUDA, Python for maximum performance

### Implemented Features (v0.1.0)
- ✅ PDE-specific lexer with Unicode math operators (∇, ∂, ×, ⊗)
- ✅ Parser for PDE definitions and boundary conditions
- ✅ Mesh generation (structured grids, unstructured, AMR)
- ✅ Code generation backends (Python, C++, CUDA)
- ✅ Scientific templates (heat equation, Navier-Stokes, Maxwell)
- ✅ GPU kernel generation for parallel computing
- ✅ Field operations and mathematical operators
- ✅ Boundary condition specification

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flux-lang.git
cd flux-lang

# Install dependencies
pip install numpy scipy matplotlib

# Optional: CUDA toolkit for GPU acceleration
# Download from https://developer.nvidia.com/cuda-downloads
```

## Quick Start

### Compile a FLUX Program

Compile FLUX scientific code to your target backend:

```bash
# Compile to Python (default)
python flux_scientific.py examples/heat_equation.flux

# Compile to CUDA for GPU acceleration  
python flux_scientific.py examples/gpu_accelerated_cfd.flux -b cuda

# Compile to C++ for performance
python flux_scientific.py examples/navier_stokes_cavity.flux -b cpp
```

### Interactive Scientific Computing

Start the interactive scientific environment:

```bash
python flux_scientific.py -i
```

Example session:
```
flux-sci> mesh StructuredGrid 50 50
Created StructuredGrid with 2601 nodes, 2500 cells

flux-sci> compile examples/heat_equation.flux python
Compiling examples/heat_equation.flux to python...
Generated code written to output/generated.py
Compilation successful!
```

### Run Benchmarks

Test FLUX with included scientific benchmarks:

```bash
python flux_scientific.py --benchmark
```

## Language Examples

### Hello World
```flux
function main() {
    print("Hello, FLUX World!")
}

main()
```

### Variables and Types
```flux
// Immutable variable
let pi = 3.14159

// Mutable variable
var counter = 0
counter = counter + 1

// Constant
const MAX_SIZE = 100

// Type annotations (optional)
let name: string = "FLUX"
let age: int = 1
```

### Functions
```flux
function add(a: int, b: int) -> int {
    return a + b
}

// Async function (syntax supported, async not yet implemented)
async function fetch_data(url: string) -> string {
    // Implementation
}
```

### Vectors and AI Operations
```flux
// Create vectors
let embedding1 = embed("artificial intelligence")
let embedding2 = embed("machine learning")

// Semantic similarity using ~~ operator
let similarity = embedding1 ~~ embedding2
print("Similarity: " + str(similarity))

// Vector operations
let v1 = vector(1.0, 2.0, 3.0)
let v2 = vector(4.0, 5.0, 6.0)
let dot_product = v1 @ v2  // Matrix multiplication operator
```

### Tensor Operations
```flux
// Create tensors
let matrix = tensor([[1, 2], [3, 4]])
let weights = tensor([[0.1, 0.2], [0.3, 0.4]])

// Matrix multiplication
let result = matrix @ weights

// Special tensors
let zeros_3x3 = zeros(3, 3)
let ones_2x4 = ones(2, 4)
let random_5x5 = random(5, 5)
```

### Quantum Computing (Basic)
```flux
// Create a qubit
let q = qubit()

// Apply quantum gates
q = hadamard(q)  // Put in superposition

// Quantum circuits (syntax supported)
quantum circuit bell_pair() {
    classical {
        // Classical preprocessing
    }
    quantum {
        // Quantum operations
    }
    classical {
        // Classical postprocessing
    }
}
```

### Control Flow
```flux
// If statement
if x > 0 {
    print("Positive")
} else if x < 0 {
    print("Negative")
} else {
    print("Zero")
}

// While loop
while condition {
    // Loop body
}

// For loop (syntax planned)
for item in collection {
    process(item)
}

// Match expression (syntax supported)
match value {
    1 => print("One"),
    2 => print("Two"),
    _ => print("Other")
}
```

## Examples Directory

The `examples/` directory contains several demonstration programs:

- `hello_world.flux` - Basic hello world and string operations
- `fibonacci.flux` - Recursive and iterative Fibonacci implementations
- `vectors_ai.flux` - Vector operations and semantic similarity
- `tensors_ml.flux` - Tensor operations and ML concepts
- `quantum_basic.flux` - Basic quantum computing demonstrations

## Architecture

```
flux-lang/
├── src/
│   ├── __init__.py      # Package initialization
│   ├── lexer.py         # Tokenization
│   ├── parser.py        # AST generation
│   └── interpreter.py   # Execution engine
├── examples/            # Example FLUX programs
├── flux.py             # CLI and REPL
└── README.md           # This file
```

## Development Status

FLUX is currently in early development (v0.1.0). The following features are planned:

### Near-term Goals
- [ ] Complete type system implementation
- [ ] Async/await execution
- [ ] Import/module system
- [ ] Standard library expansion
- [ ] Error handling improvements
- [ ] Debugger support

### Long-term Goals
- [ ] MLIR backend for compilation
- [ ] WebAssembly target
- [ ] GPU acceleration
- [ ] Distributed computing primitives
- [ ] Full quantum circuit simulation
- [ ] Package manager
- [ ] IDE plugins (VSCode, etc.)

## Contributing

Contributions are welcome! Areas where help is needed:

1. **Standard Library**: Implementing built-in functions and types
2. **Quantum Operations**: Expanding quantum computing support
3. **AI Integration**: Connecting to real LLM APIs
4. **Optimization**: Performance improvements
5. **Documentation**: Tutorials and examples
6. **Testing**: Unit tests and integration tests

## License

FLUX is open-source software. License details to be determined.

## Acknowledgments

FLUX is inspired by:
- Shopify Liquid (templating philosophy)
- Python (syntax and simplicity)
- Rust (memory safety concepts)
- Julia (scientific computing)
- Q# (quantum computing)
- Mojo (AI compilation)

---

**FLUX: Write once, run everywhere, understand everything.**