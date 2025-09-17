"""
FLUX Code Generation
Generates C++/CUDA/Python code from FLUX PDE definitions
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from .pde_parser import *
from .codegen_python import PythonScientificGenerator

class CodeGenerator(ABC):
    """Base class for code generators"""
    
    @abstractmethod
    def generate(self, ast_nodes: List[ASTNode]) -> str:
        pass

class CUDAKernelGenerator(CodeGenerator):
    """Generate CUDA kernels from FLUX GPU kernel definitions"""
    
    def __init__(self):
        self.kernel_counter = 0
        
    def generate(self, ast_nodes: List[ASTNode]) -> str:
        code_sections = []
        
        # Add CUDA headers
        code_sections.append(self._generate_headers())
        
        for node in ast_nodes:
            if isinstance(node, KernelDefinition) and 'gpu' in (node.decorators or []):
                code_sections.append(self._generate_kernel(node))
        
        # Add kernel launcher functions
        code_sections.append(self._generate_launchers(ast_nodes))
        
        return '\n\n'.join(code_sections)
    
    def _generate_headers(self) -> str:
        return """// Auto-generated CUDA code from FLUX
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// FLUX runtime types
struct FluxField {
    float* data;
    int size;
    int dims[3];
};

struct FluxScalar {
    float value;
};

// CUDA error checking macro
#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        exit(EXIT_FAILURE); \\
    } \\
} while(0)"""
    
    def _generate_kernel(self, kernel: KernelDefinition) -> str:
        """Generate CUDA kernel from FLUX kernel definition"""
        kernel_name = f"flux_{kernel.name}"
        
        # Generate parameter list
        params = []
        for param in kernel.parameters:
            if param.type == 'Field':
                params.append(f"FluxField {param.name}")
            elif param.type == 'Scalar':
                params.append(f"FluxScalar {param.name}")
            else:
                params.append(f"float {param.name}")
        
        param_list = ', '.join(params)
        
        # Generate kernel body
        body_lines = []
        body_lines.append("    // Get thread index")
        body_lines.append("    int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        body_lines.append("")
        
        # Process kernel body statements
        for stmt in kernel.body:
            body_lines.append(f"    {self._translate_statement(stmt)}")
        
        kernel_body = '\n'.join(body_lines)
        
        return f"""__global__ void {kernel_name}({param_list}) {{
{kernel_body}
}}"""
    
    def _translate_statement(self, stmt: ASTNode) -> str:
        """Translate FLUX statement to CUDA C"""
        if isinstance(stmt, BinaryOp):
            left = self._translate_expression(stmt.left)
            right = self._translate_expression(stmt.right)
            
            if stmt.operator == '=':
                return f"{left} = {right};"
            else:
                return f"{left} {stmt.operator} {right};"
        
        elif isinstance(stmt, FunctionCall):
            if stmt.name == 'sqrt':
                arg = self._translate_expression(stmt.args[0])
                return f"sqrtf({arg})"
            elif stmt.name == 'abs':
                arg = self._translate_expression(stmt.args[0])
                return f"fabsf({arg})"
            else:
                args = [self._translate_expression(arg) for arg in stmt.args]
                return f"{stmt.name}({', '.join(args)});"
        
        return "// TODO: translate statement"
    
    def _translate_expression(self, expr: ASTNode) -> str:
        """Translate FLUX expression to CUDA C"""
        if isinstance(expr, Variable):
            if expr.indices:
                # Array access
                base = expr.name
                if len(expr.indices) == 1:
                    index = self._translate_expression(expr.indices[0])
                    return f"{base}.data[{index}]"
                else:
                    # Multi-dimensional access
                    indices = [self._translate_expression(idx) for idx in expr.indices]
                    return f"{base}.data[{' + '.join(indices)}]"
            return expr.name
        
        elif isinstance(expr, Number):
            if isinstance(expr.value, float):
                return f"{expr.value}f"
            else:
                return str(expr.value)
        
        elif isinstance(expr, BinaryOp):
            left = self._translate_expression(expr.left)
            right = self._translate_expression(expr.right)
            return f"({left} {expr.operator} {right})"
        
        return "0.0f"  # Default
    
    def _generate_launchers(self, ast_nodes: List[ASTNode]) -> str:
        """Generate kernel launcher functions"""
        launchers = []
        
        for node in ast_nodes:
            if isinstance(node, KernelDefinition) and 'gpu' in (node.decorators or []):
                launcher = self._generate_launcher(node)
                launchers.append(launcher)
        
        return '\n\n'.join(launchers)
    
    def _generate_launcher(self, kernel: KernelDefinition) -> str:
        """Generate C++ launcher function for CUDA kernel"""
        kernel_name = f"flux_{kernel.name}"
        launcher_name = f"launch_{kernel.name}"
        
        # Generate parameter list for launcher
        params = []
        for param in kernel.parameters:
            if param.type == 'Field':
                params.append(f"FluxField& {param.name}")
            elif param.type == 'Scalar':
                params.append(f"float {param.name}")
            else:
                params.append(f"float {param.name}")
        
        param_list = ', '.join(params)
        
        # Generate kernel call
        kernel_args = []
        for param in kernel.parameters:
            if param.type == 'Field':
                kernel_args.append(param.name)
            elif param.type == 'Scalar':
                kernel_args.append(f"{{.value = {param.name}}}")
            else:
                kernel_args.append(param.name)
        
        args_list = ', '.join(kernel_args)
        
        return f"""// Launcher for {kernel.name}
void {launcher_name}({param_list}) {{
    // Calculate grid dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    {kernel_name}<<<gridSize, blockSize>>>({args_list});
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}}"""

class CPPGenerator(CodeGenerator):
    """Generate C++ code from FLUX PDE definitions"""
    
    def generate(self, ast_nodes: List[ASTNode]) -> str:
        code_sections = []
        
        # Add headers
        code_sections.append(self._generate_headers())
        
        # Generate classes for PDEs
        for node in ast_nodes:
            if isinstance(node, PDEDefinition):
                code_sections.append(self._generate_pde_class(node))
        
        return '\n\n'.join(code_sections)
    
    def _generate_headers(self) -> str:
        return """// Auto-generated C++ code from FLUX
#include <vector>
#include <array>
#include <cmath>
#include <functional>

// FLUX runtime classes
template<typename T>
class Field {
public:
    std::vector<T> data;
    std::array<int, 3> dims;
    
    Field(int nx, int ny = 1, int nz = 1) : dims{nx, ny, nz} {
        data.resize(nx * ny * nz);
    }
    
    T& operator()(int i, int j = 0, int k = 0) {
        return data[k * dims[0] * dims[1] + j * dims[0] + i];
    }
    
    const T& operator()(int i, int j = 0, int k = 0) const {
        return data[k * dims[0] * dims[1] + j * dims[0] + i];
    }
};

using ScalarField = Field<double>;
using VectorField = Field<std::array<double, 3>>;"""
    
    def _generate_pde_class(self, pde: PDEDefinition) -> str:
        """Generate C++ class for PDE"""
        class_name = f"{pde.name}_solver"
        
        # Generate member variables
        members = []
        if pde.variables:
            for var in pde.variables:
                members.append(f"    ScalarField {var};")
        
        # Generate constructor
        constructor_params = "int nx, int ny, int nz"
        constructor_init = []
        if pde.variables:
            for var in pde.variables:
                constructor_init.append(f"{var}(nx, ny, nz)")
        
        constructor_init_list = ", ".join(constructor_init)
        
        # Generate equation functions
        equation_funcs = []
        for i, eq in enumerate(pde.equations):
            equation_funcs.append(f"    // Equation {i+1}: {self._equation_to_comment(eq)}")
            equation_funcs.append(f"    void compute_equation_{i+1}();")
        
        return f"""class {class_name} {{
public:
{chr(10).join(members)}

    {class_name}({constructor_params}) : {constructor_init_list} {{}}
    
{chr(10).join(equation_funcs)}
    
    void solve(double dt, int num_steps);
    
private:
    void apply_boundary_conditions();
    void time_step(double dt);
}};"""
    
    def _equation_to_comment(self, eq: Equation) -> str:
        """Convert equation to comment string"""
        return f"LHS = RHS"  # Simplified

class PythonGenerator(CodeGenerator):
    """Generate Python code from FLUX definitions"""

    def __init__(self):
        # Use the new scientific generator for better code generation
        self.scientific_gen = PythonScientificGenerator()

    def generate(self, ast_nodes: List[ASTNode]) -> str:
        # Delegate to the new comprehensive generator
        return self.scientific_gen.generate(ast_nodes)
    

class BackendManager:
    """Manages different code generation backends"""
    
    def __init__(self):
        self.generators = {
            'cuda': CUDAKernelGenerator(),
            'cpp': CPPGenerator(), 
            'python': PythonGenerator()
        }
    
    def generate_code(self, backend: str, ast_nodes: List[ASTNode]) -> str:
        """Generate code for specified backend"""
        if backend not in self.generators:
            raise ValueError(f"Unsupported backend: {backend}")
        
        generator = self.generators[backend]
        return generator.generate(ast_nodes)
    
    def add_generator(self, name: str, generator: CodeGenerator):
        """Add a custom code generator"""
        self.generators[name] = generator

def generate_makefile(targets: List[str]) -> str:
    """Generate Makefile for building generated code"""
    makefile_content = """# Auto-generated Makefile for FLUX project
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
NVCCFLAGS = -O3 -std=c++17

"""
    
    # Add targets
    for target in targets:
        if target.endswith('.cu'):
            # CUDA target
            obj_name = target.replace('.cu', '.o')
            makefile_content += f"{obj_name}: {target}\n"
            makefile_content += f"\t$(NVCC) $(NVCCFLAGS) -c $< -o $@\n\n"
        else:
            # C++ target
            obj_name = target.replace('.cpp', '.o')
            makefile_content += f"{obj_name}: {target}\n"
            makefile_content += f"\t$(CXX) $(CXXFLAGS) -c $< -o $@\n\n"
    
    makefile_content += """clean:
\trm -f *.o *.exe

.PHONY: clean
"""
    
    return makefile_content

def generate_cmake(project_name: str, sources: List[str]) -> str:
    """Generate CMakeLists.txt for building"""
    cmake_content = f"""# Auto-generated CMakeLists.txt for FLUX project
cmake_minimum_required(VERSION 3.18)
project({project_name} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(CUDA REQUIRED)

# Source files
set(SOURCES
"""
    
    for source in sources:
        cmake_content += f"    {source}\n"
    
    cmake_content += f""")

# Create executable
add_executable({project_name} ${{SOURCES}})

# Link libraries
target_link_libraries({project_name} 
    ${{CUDA_LIBRARIES}}
    ${{CUDA_CUBLAS_LIBRARIES}}
)

# Set properties
set_target_properties({project_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
"""
    
    return cmake_content