#!/usr/bin/env python3
"""
FLUX Scientific Computing Language
Command-line interface for PDE solving and scientific computing
"""

import sys
import os
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime

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
    parser = argparse.ArgumentParser(
        description="FLUX Scientific Computing Language - Domain-Specific Language for Scientific Computing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flux-sci heat.flux                  # Compile to Python (default)
  flux-sci heat.flux -b cuda          # Compile to CUDA
  flux-sci heat.flux -o my_output     # Custom output directory
  flux-sci -i                         # Interactive mode
  flux-sci --benchmark                # Run performance benchmarks
  flux-sci --validate heat.flux       # Validate syntax without compilation
  flux-sci --run heat.flux            # Compile and execute immediately

For more information, visit: https://github.com/MichaelCrowe11/flux-sci-lang
        """
    )

    parser.add_argument('file', nargs='?', help='FLUX source file to compile')

    # Compilation options
    compile_group = parser.add_argument_group('compilation options')
    compile_group.add_argument('-b', '--backend',
                              choices=['python', 'cpp', 'cuda', 'julia', 'fortran'],
                              default='python',
                              help='Target backend (default: python)')
    compile_group.add_argument('-o', '--output',
                              default='output',
                              help='Output directory (default: output)')
    compile_group.add_argument('--optimize',
                              choices=['O0', 'O1', 'O2', 'O3'],
                              default='O2',
                              help='Optimization level (default: O2)')

    # Execution options
    exec_group = parser.add_argument_group('execution options')
    exec_group.add_argument('--run',
                           action='store_true',
                           help='Compile and execute immediately')
    exec_group.add_argument('--profile',
                           action='store_true',
                           help='Profile execution performance')
    exec_group.add_argument('--parallel',
                           type=int,
                           metavar='N',
                           help='Number of parallel threads/processes')

    # Mode options
    mode_group = parser.add_argument_group('mode options')
    mode_group.add_argument('-i', '--interactive',
                           action='store_true',
                           help='Interactive FLUX environment')
    mode_group.add_argument('--benchmark',
                           action='store_true',
                           help='Run performance benchmarks')
    mode_group.add_argument('--validate',
                           action='store_true',
                           help='Validate syntax without compilation')
    mode_group.add_argument('--ast',
                           action='store_true',
                           help='Print abstract syntax tree')

    # Debugging options
    debug_group = parser.add_argument_group('debugging options')
    debug_group.add_argument('-v', '--verbose',
                            action='count',
                            default=0,
                            help='Increase verbosity (-v, -vv, -vvv)')
    debug_group.add_argument('--debug',
                            action='store_true',
                            help='Enable debug mode')
    debug_group.add_argument('--log',
                            metavar='FILE',
                            help='Log output to file')

    # Visualization options
    viz_group = parser.add_argument_group('visualization options')
    viz_group.add_argument('--plot',
                          action='store_true',
                          help='Generate plots after execution')
    viz_group.add_argument('--animate',
                          action='store_true',
                          help='Create animation of solution')
    viz_group.add_argument('--save-plots',
                          metavar='DIR',
                          help='Save plots to directory')

    # Info options
    info_group = parser.add_argument_group('information')
    info_group.add_argument('--version',
                           action='version',
                           version='FLUX Scientific Computing Language v0.1.0')
    info_group.add_argument('--list-examples',
                           action='store_true',
                           help='List available examples')
    info_group.add_argument('--list-backends',
                           action='store_true',
                           help='List available backends')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.debug, args.log)

    # Handle different modes
    if args.list_examples:
        list_examples()
    elif args.list_backends:
        list_backends()
    elif args.benchmark:
        run_benchmarks()
    elif args.interactive:
        interactive_mode()
    elif args.file:
        if args.validate:
            validate_flux(args.file)
        elif args.ast:
            print_ast(args.file)
        elif args.run:
            compile_and_run(args.file, args.backend, args.output, args)
        else:
            compile_flux(args.file, args.backend, args.output)
    else:
        # No file specified and not in special mode
        parser.print_help()

def setup_logging(verbose: int, debug: bool, log_file: Optional[str]):
    """Setup logging configuration"""
    if debug:
        level = logging.DEBUG
    elif verbose >= 3:
        level = logging.DEBUG
    elif verbose >= 2:
        level = logging.INFO
    elif verbose >= 1:
        level = logging.WARNING
    else:
        level = logging.ERROR

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=format_str)

def list_examples():
    """List available example files"""
    examples_dir = Path("examples")
    if examples_dir.exists():
        print("Available FLUX examples:")
        print("-" * 40)
        for example in sorted(examples_dir.glob("*.flux")):
            # Read first line of description if available
            with open(example, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith("//"):
                    description = first_line[2:].strip()
                else:
                    description = "No description"
            print(f"  {example.name:<30} {description}")
        print("\nRun with: flux-sci examples/<filename>")
    else:
        print("No examples directory found")

def list_backends():
    """List available compilation backends"""
    backends = {
        'python': 'NumPy/SciPy implementation (default)',
        'cpp': 'C++ with OpenMP parallelization',
        'cuda': 'NVIDIA GPU acceleration',
        'julia': 'Julia language (experimental)',
        'fortran': 'Fortran 90 (experimental)'
    }

    print("Available compilation backends:")
    print("-" * 40)
    for name, description in backends.items():
        print(f"  {name:<10} {description}")

def validate_flux(source_file: str):
    """Validate FLUX syntax without compilation"""
    try:
        with open(source_file, 'r') as f:
            source = f.read()

        print(f"Validating {source_file}...")

        # Tokenize
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()
        print(f"  ✓ Tokenization successful ({len(tokens)} tokens)")

        # Parse
        parser = FluxPDEParser(tokens)
        ast = parser.parse()
        print(f"  ✓ Parsing successful ({len(ast)} AST nodes)")

        print("✓ Syntax validation passed")
        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

def print_ast(source_file: str):
    """Print abstract syntax tree"""
    try:
        with open(source_file, 'r') as f:
            source = f.read()

        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        parser = FluxPDEParser(tokens)
        ast = parser.parse()

        print("Abstract Syntax Tree:")
        print("-" * 40)
        for node in ast:
            print_ast_node(node, 0)

    except Exception as e:
        print(f"Error generating AST: {e}")

def print_ast_node(node, indent: int):
    """Recursively print AST node"""
    prefix = "  " * indent
    print(f"{prefix}{node.__class__.__name__}")

    # Print node attributes
    for key, value in node.__dict__.items():
        if not key.startswith('_'):
            if isinstance(value, list):
                print(f"{prefix}  {key}: [{len(value)} items]")
            else:
                print(f"{prefix}  {key}: {value}")

def compile_and_run(source_file: str, backend: str, output_dir: str, args):
    """Compile and immediately execute"""
    # First compile
    compile_flux(source_file, backend, output_dir)

    # Then execute based on backend
    print(f"\nExecuting compiled code...")

    if backend == 'python':
        output_file = Path(output_dir) / 'generated.py'
        if output_file.exists():
            import subprocess
            result = subprocess.run([sys.executable, str(output_file)],
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
    else:
        print(f"Direct execution not yet implemented for {backend} backend")

if __name__ == "__main__":
    main()