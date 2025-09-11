#!/usr/bin/env python3
"""
FLUX Scientific Computing Language
Command-line interface for PDE solving and scientific computing
"""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pde_lexer import FluxPDELexer
from src.pde_parser import FluxPDEParser
from src.codegen import BackendManager
from src.mesh import create_mesh

def compile_flux(source_file: str, backend: str = 'python', output_dir: str = 'output'):
    """Compile FLUX source to target backend"""
    try:
        # Read source file
        with open(source_file, 'r') as f:
            source = f.read()
        
        print(f"Compiling {source_file} to {backend}...")
        
        # Tokenize
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()
        print(f"Tokenized: {len(tokens)} tokens")
        
        # Parse
        parser = FluxPDEParser(tokens)
        ast_nodes = parser.parse()
        print(f"Parsed: {len(ast_nodes)} AST nodes")
        
        # Generate code
        backend_mgr = BackendManager()
        generated_code = backend_mgr.generate_code(backend, ast_nodes)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Write generated code
        if backend == 'cuda':
            output_file = os.path.join(output_dir, 'generated.cu')
        elif backend == 'cpp':
            output_file = os.path.join(output_dir, 'generated.cpp')
        else:
            output_file = os.path.join(output_dir, 'generated.py')
        
        with open(output_file, 'w') as f:
            f.write(generated_code)
        
        print(f"Generated code written to {output_file}")
        
        # Generate build files if needed
        if backend in ['cuda', 'cpp']:
            from src.codegen import generate_cmake, generate_makefile
            
            # CMakeLists.txt
            cmake_content = generate_cmake('flux_solver', [os.path.basename(output_file)])
            with open(os.path.join(output_dir, 'CMakeLists.txt'), 'w') as f:
                f.write(cmake_content)
            
            # Makefile
            makefile_content = generate_makefile([os.path.basename(output_file)])
            with open(os.path.join(output_dir, 'Makefile'), 'w') as f:
                f.write(makefile_content)
            
            print("Build files generated (CMakeLists.txt, Makefile)")
        
        print("Compilation successful!")
        
    except FileNotFoundError:
        print(f"Error: File '{source_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Compilation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def interactive_mode():
    """Interactive FLUX scientific computing environment"""
    print("FLUX Scientific Computing Language v0.1.0")
    print("Interactive Mode - Type 'exit' to quit")
    print("Available commands:")
    print("  compile <file> [backend] - Compile FLUX file")
    print("  mesh <type> [params]     - Create mesh")
    print("  plot <field>            - Plot results")
    print()
    
    backend_mgr = BackendManager()
    
    while True:
        try:
            cmd = input("flux-sci> ").strip()
            
            if cmd == 'exit':
                break
            
            parts = cmd.split()
            if not parts:
                continue
            
            command = parts[0]
            
            if command == 'compile':
                if len(parts) < 2:
                    print("Usage: compile <file> [backend]")
                    continue
                
                source_file = parts[1]
                backend = parts[2] if len(parts) > 2 else 'python'
                compile_flux(source_file, backend)
            
            elif command == 'mesh':
                if len(parts) < 2:
                    print("Usage: mesh <type> [nx] [ny]")
                    continue
                
                mesh_type = parts[1]
                nx = int(parts[2]) if len(parts) > 2 else 10
                ny = int(parts[3]) if len(parts) > 3 else 10
                
                mesh = create_mesh(mesh_type, "interactive_mesh", nx=nx, ny=ny)
                print(f"Created {mesh_type} with {mesh.get_node_count()} nodes, {mesh.get_cell_count()} cells")
            
            elif command == 'help':
                print("Available backends: python, cpp, cuda")
                print("Available mesh types: StructuredGrid, UnstructuredMesh, AdaptiveMesh")
            
            else:
                print(f"Unknown command: {command}")
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt")
            continue
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

def run_benchmarks():
    """Run FLUX scientific computing benchmarks"""
    print("FLUX Scientific Computing Benchmarks")
    print("=" * 40)
    
    # Heat equation benchmark
    print("\n1. Heat Equation (100x100 grid)")
    try:
        compile_flux("examples/heat_equation.flux", "python", "benchmark_output")
        print("✓ Heat equation compiled successfully")
    except:
        print("✗ Heat equation compilation failed")
    
    # CFD benchmark
    print("\n2. Navier-Stokes Cavity Flow (100x100 grid)")
    try:
        compile_flux("examples/navier_stokes_cavity.flux", "python", "benchmark_output")
        print("✓ Navier-Stokes compiled successfully")
    except:
        print("✗ Navier-Stokes compilation failed")
    
    # GPU benchmark
    print("\n3. GPU-Accelerated CFD")
    try:
        compile_flux("examples/gpu_accelerated_cfd.flux", "cuda", "benchmark_output")
        print("✓ GPU CFD compiled successfully")
    except:
        print("✗ GPU CFD compilation failed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FLUX Scientific Computing Language")
    parser.add_argument('file', nargs='?', help='FLUX source file to compile')
    parser.add_argument('-b', '--backend', choices=['python', 'cpp', 'cuda'], 
                       default='python', help='Target backend')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmarks()
    elif args.interactive or args.file is None:
        interactive_mode()
    else:
        compile_flux(args.file, args.backend, args.output)

if __name__ == "__main__":
    main()