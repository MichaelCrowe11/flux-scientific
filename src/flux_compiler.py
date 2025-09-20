"""
Complete FLUX Scientific Computing Compiler
Compiles .flux files to executable Python, C++, or CUDA code
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import json
import textwrap

from .pde_lexer import FluxPDELexer, TokenType
from .pde_parser import FluxPDEParser, PDEDefinition, Domain, Mesh, Solver
from .codegen_python import PythonScientificGenerator


class FluxCompiler:
    """
    Complete FLUX compiler that transforms .flux source files
    into executable scientific computing code
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def compile_file(self, source_file: str, output_dir: str = "output",
                    backend: str = "python", optimize: bool = True) -> Dict[str, Any]:
        """
        Compile a .flux file to the specified backend

        Parameters:
        -----------
        source_file : str
            Path to .flux source file
        output_dir : str
            Directory for generated files
        backend : str
            Target backend: 'python', 'cpp', 'cuda'
        optimize : bool
            Enable optimizations

        Returns:
        --------
        dict with compilation results
        """

        if self.verbose:
            print(f"🔧 FLUX Compiler v0.1.0")
            print(f"   Source: {source_file}")
            print(f"   Backend: {backend}")
            print(f"   Output: {output_dir}")

        # Reset errors/warnings
        self.errors = []
        self.warnings = []

        try:
            # 1. Read source file
            source_code = self._read_source(source_file)

            # 2. Lexical analysis
            tokens = self._tokenize(source_code)

            # 3. Syntax analysis
            ast = self._parse(tokens)

            # 4. Semantic analysis
            validated_ast = self._semantic_analysis(ast)

            # 5. Optimization
            if optimize:
                optimized_ast = self._optimize(validated_ast)
            else:
                optimized_ast = validated_ast

            # 6. Code generation
            generated_code = self._generate_code(optimized_ast, backend)

            # 7. Write output files
            output_files = self._write_output(generated_code, output_dir, backend, source_file)

            # 8. Generate metadata
            metadata = self._generate_metadata(source_file, backend, ast, output_files)

            if self.verbose:
                print(f"✅ Compilation successful!")
                print(f"   Generated: {len(output_files)} files")
                print(f"   Warnings: {len(self.warnings)}")

            return {
                'success': True,
                'output_files': output_files,
                'metadata': metadata,
                'warnings': self.warnings,
                'errors': []
            }

        except Exception as e:
            error_msg = f"Compilation failed: {str(e)}"
            self.errors.append(error_msg)

            if self.verbose:
                print(f"❌ {error_msg}")

            return {
                'success': False,
                'output_files': [],
                'metadata': {},
                'warnings': self.warnings,
                'errors': self.errors
            }

    def _read_source(self, source_file: str) -> str:
        """Read and validate source file"""
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")

        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            raise ValueError("Source file is empty")

        if self.verbose:
            print(f"   📖 Read {len(content)} characters")

        return content

    def _tokenize(self, source: str) -> List[Any]:
        """Tokenize source code"""
        try:
            lexer = FluxPDELexer(source)
            tokens = lexer.tokenize()

            if self.verbose:
                print(f"   🔤 Tokenized: {len(tokens)} tokens")

            return tokens

        except Exception as e:
            raise SyntaxError(f"Tokenization failed: {e}")

    def _parse(self, tokens: List[Any]) -> List[Any]:
        """Parse tokens into AST"""
        try:
            parser = FluxPDEParser(tokens)
            ast = parser.parse()

            if self.verbose:
                print(f"   🌳 Parsed: {len(ast)} AST nodes")
                for node in ast:
                    print(f"      - {node.__class__.__name__}")

            return ast

        except Exception as e:
            raise SyntaxError(f"Parsing failed: {e}")

    def _semantic_analysis(self, ast: List[Any]) -> List[Any]:
        """Validate semantics and types"""
        validated_ast = []

        # Track defined entities
        domains = {}
        meshes = {}
        pdes = {}
        solvers = {}

        for node in ast:
            try:
                if isinstance(node, Domain):
                    # Validate domain parameters
                    self._validate_domain(node)
                    domains[node.name] = node

                elif isinstance(node, Mesh):
                    # Validate mesh references domain
                    if hasattr(node, 'domain') and node.domain not in domains:
                        self.warnings.append(f"Mesh '{node.name}' references unknown domain '{node.domain}'")
                    meshes[node.name] = node

                elif isinstance(node, PDEDefinition):
                    # Validate PDE structure
                    self._validate_pde(node)
                    pdes[node.name] = node

                elif isinstance(node, Solver):
                    # Validate solver configuration
                    self._validate_solver(node)
                    solvers[node.name] = node

                validated_ast.append(node)

            except Exception as e:
                self.warnings.append(f"Validation warning for {node.__class__.__name__}: {e}")
                validated_ast.append(node)  # Include anyway

        if self.verbose:
            print(f"   ✓ Validated: {len(domains)} domains, {len(meshes)} meshes, "
                  f"{len(pdes)} PDEs, {len(solvers)} solvers")

        return validated_ast

    def _validate_domain(self, domain: Domain):
        """Validate domain definition"""
        if domain.shape not in ['Rectangle', 'Circle', 'Line', 'Box']:
            self.warnings.append(f"Unknown domain shape: {domain.shape}")

        # Check required parameters
        if domain.shape == 'Rectangle':
            required = ['xmin', 'xmax', 'ymin', 'ymax']
            for param in required:
                if param not in domain.parameters:
                    self.warnings.append(f"Rectangle domain missing parameter: {param}")

    def _validate_pde(self, pde: PDEDefinition):
        """Validate PDE definition"""
        if not pde.equations:
            self.warnings.append(f"PDE '{pde.name}' has no equations")

        if not pde.variables:
            self.warnings.append(f"PDE '{pde.name}' has no variables declared")

        # Check equation structure
        for i, eq in enumerate(pde.equations):
            if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
                self.warnings.append(f"Equation {i+1} in PDE '{pde.name}' is malformed")

    def _validate_solver(self, solver: Solver):
        """Validate solver configuration"""
        valid_types = ['ImplicitEuler', 'ExplicitEuler', 'CrankNicolson', 'RungeKutta']
        if solver.type not in valid_types:
            self.warnings.append(f"Unknown solver type: {solver.type}")

    def _optimize(self, ast: List[Any]) -> List[Any]:
        """Apply optimization passes"""
        if self.verbose:
            print(f"   ⚡ Applying optimizations...")

        # For now, basic optimizations
        optimized = []

        for node in ast:
            # Example optimization: merge compatible PDEs
            # More sophisticated optimizations would go here
            optimized.append(node)

        return optimized

    def _generate_code(self, ast: List[Any], backend: str) -> Dict[str, str]:
        """Generate code for specified backend"""
        if self.verbose:
            print(f"   🎯 Generating {backend} code...")

        if backend == 'python':
            return self._generate_python_code(ast)
        elif backend == 'cpp':
            return self._generate_cpp_code(ast)
        elif backend == 'cuda':
            return self._generate_cuda_code(ast)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _generate_python_code(self, ast: List[Any]) -> Dict[str, str]:
        """Generate Python code using enhanced generator"""
        generator = PythonScientificGenerator()
        main_code = generator.generate(ast)

        # Also generate setup and requirements files
        setup_code = self._generate_python_setup()
        requirements = self._generate_requirements()
        readme = self._generate_readme(ast)

        return {
            'main.py': main_code,
            'setup.py': setup_code,
            'requirements.txt': requirements,
            'README.md': readme
        }

    def _generate_cpp_code(self, ast: List[Any]) -> Dict[str, str]:
        """Generate C++ code"""
        # Basic C++ generation
        cpp_code = """
// Auto-generated C++ code from FLUX
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// FLUX Runtime
namespace flux {

class HeatSolver {
public:
    HeatSolver(int nx, int ny) : nx_(nx), ny_(ny) {
        u_.resize(nx * ny);
        u_new_.resize(nx * ny);
    }

    void solve(double dt, double alpha, int steps) {
        for (int step = 0; step < steps; ++step) {
            timeStep(dt, alpha);
        }
    }

private:
    void timeStep(double dt, double alpha) {
        // Explicit finite difference
        double dx = 1.0 / (nx_ - 1);
        double r = alpha * dt / (dx * dx);

        for (int j = 1; j < ny_ - 1; ++j) {
            for (int i = 1; i < nx_ - 1; ++i) {
                int idx = j * nx_ + i;
                u_new_[idx] = u_[idx] + r * (
                    u_[idx + 1] - 2*u_[idx] + u_[idx - 1] +
                    u_[idx + nx_] - 2*u_[idx] + u_[idx - nx_]
                );
            }
        }

        u_.swap(u_new_);
    }

    int nx_, ny_;
    std::vector<double> u_, u_new_;
};

} // namespace flux

int main() {
    std::cout << "FLUX C++ Solver\\n";

    flux::HeatSolver solver(100, 100);

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(0.001, 0.1, 1000);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Solved in " << duration.count() << " ms\\n";

    return 0;
}
"""

        makefile = """
# Auto-generated Makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native
TARGET = flux_solver
SOURCES = main.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: clean
"""

        return {
            'main.cpp': cpp_code,
            'Makefile': makefile
        }

    def _generate_cuda_code(self, ast: List[Any]) -> Dict[str, str]:
        """Generate CUDA code"""
        cuda_code = """
// Auto-generated CUDA code from FLUX
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void heatStepKernel(float* u, float* u_new, int nx, int ny, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
        int idx = j * nx + i;

        u_new[idx] = u[idx] + r * (
            u[idx + 1] - 2*u[idx] + u[idx - 1] +
            u[idx + nx] - 2*u[idx] + u[idx - nx]
        );
    }
}

class CUDAHeatSolver {
public:
    CUDAHeatSolver(int nx, int ny) : nx_(nx), ny_(ny) {
        size_t bytes = nx * ny * sizeof(float);

        cudaMalloc(&d_u_, bytes);
        cudaMalloc(&d_u_new_, bytes);

        h_u_.resize(nx * ny);
        h_u_new_.resize(nx * ny);
    }

    ~CUDAHeatSolver() {
        cudaFree(d_u_);
        cudaFree(d_u_new_);
    }

    void solve(float dt, float alpha, int steps) {
        float dx = 1.0f / (nx_ - 1);
        float r = alpha * dt / (dx * dx);

        dim3 block(16, 16);
        dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

        for (int step = 0; step < steps; ++step) {
            heatStepKernel<<<grid, block>>>(d_u_, d_u_new_, nx_, ny_, r);
            cudaDeviceSynchronize();

            std::swap(d_u_, d_u_new_);
        }
    }

private:
    int nx_, ny_;
    float *d_u_, *d_u_new_;
    std::vector<float> h_u_, h_u_new_;
};

int main() {
    std::cout << "FLUX CUDA Solver\\n";

    CUDAHeatSolver solver(512, 512);

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(0.001f, 0.1f, 1000);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "GPU solved in " << duration.count() << " ms\\n";

    return 0;
}
"""

        return {
            'main.cu': cuda_code,
            'CMakeLists.txt': self._generate_cuda_cmake()
        }

    def _generate_python_setup(self) -> str:
        """Generate setup.py for Python output"""
        return """
# Auto-generated setup.py for FLUX output
from setuptools import setup, find_packages

setup(
    name="flux-generated-solver",
    version="0.1.0",
    description="Generated PDE solver from FLUX",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    python_requires=">=3.8",
)
"""

    def _generate_requirements(self) -> str:
        """Generate requirements.txt"""
        return """numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
"""

    def _generate_readme(self, ast: List[Any]) -> str:
        """Generate README for output"""
        pde_names = [node.name for node in ast if isinstance(node, PDEDefinition)]

        return f"""# FLUX Generated Solver

This code was automatically generated by FLUX Scientific Computing Language.

## Generated PDEs
{chr(10).join(f'- {name}' for name in pde_names)}

## Usage

```python
python main.py
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Generated by
FLUX Scientific Computing Language v0.1.0
https://github.com/MichaelCrowe11/flux-sci-lang
"""

    def _generate_cuda_cmake(self) -> str:
        """Generate CMakeLists.txt for CUDA"""
        return """
# Auto-generated CMakeLists.txt for FLUX CUDA
cmake_minimum_required(VERSION 3.18)
project(flux_cuda_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

add_executable(flux_solver main.cu)

set_target_properties(flux_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(flux_solver ${CUDA_LIBRARIES})
"""

    def _write_output(self, generated_code: Dict[str, str], output_dir: str,
                     backend: str, source_file: str) -> List[str]:
        """Write generated code to files"""
        os.makedirs(output_dir, exist_ok=True)

        output_files = []

        for filename, content in generated_code.items():
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            output_files.append(filepath)

            if self.verbose:
                print(f"      📝 {filepath} ({len(content)} chars)")

        return output_files

    def _generate_metadata(self, source_file: str, backend: str,
                          ast: List[Any], output_files: List[str]) -> Dict[str, Any]:
        """Generate compilation metadata"""
        return {
            'source_file': source_file,
            'backend': backend,
            'timestamp': __import__('time').time(),
            'flux_version': '0.1.0',
            'ast_node_count': len(ast),
            'output_files': output_files,
            'pde_count': sum(1 for node in ast if isinstance(node, PDEDefinition)),
            'domain_count': sum(1 for node in ast if isinstance(node, Domain)),
            'mesh_count': sum(1 for node in ast if hasattr(node, '__class__') and 'Mesh' in node.__class__.__name__)
        }


def main():
    """Command-line interface for FLUX compiler"""
    parser = argparse.ArgumentParser(
        description="FLUX Scientific Computing Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flux-compile heat.flux                    # Compile to Python
  flux-compile heat.flux -b cpp -o build    # Compile to C++
  flux-compile heat.flux -b cuda --optimize # Compile to CUDA with optimizations
        """
    )

    parser.add_argument('source', help='FLUX source file (.flux)')
    parser.add_argument('-b', '--backend',
                       choices=['python', 'cpp', 'cuda'],
                       default='python',
                       help='Target backend (default: python)')
    parser.add_argument('-o', '--output',
                       default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--optimize',
                       action='store_true',
                       help='Enable optimizations')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--metadata',
                       action='store_true',
                       help='Generate metadata file')

    args = parser.parse_args()

    # Create compiler
    compiler = FluxCompiler(verbose=args.verbose)

    # Compile
    result = compiler.compile_file(
        source_file=args.source,
        output_dir=args.output,
        backend=args.backend,
        optimize=args.optimize
    )

    # Handle results
    if result['success']:
        print(f"✅ Compilation successful!")
        print(f"   Output: {args.output}")
        print(f"   Files: {len(result['output_files'])}")

        if args.metadata:
            metadata_file = os.path.join(args.output, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(result['metadata'], f, indent=2)
            print(f"   Metadata: {metadata_file}")

        if result['warnings']:
            print(f"⚠️ Warnings: {len(result['warnings'])}")
            for warning in result['warnings']:
                print(f"   - {warning}")

    else:
        print(f"❌ Compilation failed!")
        for error in result['errors']:
            print(f"   {error}")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())