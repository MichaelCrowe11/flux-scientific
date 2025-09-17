"""
Integration tests for FLUX example files
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestExampleCompilation:
    """Test that example FLUX files compile correctly"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.examples_dir = Path("examples")
        self.flux_script = Path("flux_scientific.py")
        self.output_dir = Path("test_output")

    def run_flux_compiler(self, source_file, backend="python"):
        """Helper to run the FLUX compiler"""
        cmd = [
            sys.executable,
            str(self.flux_script),
            str(source_file),
            "-b", backend,
            "-o", str(self.output_dir)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        return result

    @pytest.mark.integration
    def test_heat_equation_python(self):
        """Test heat equation compiles to Python"""
        source = self.examples_dir / "heat_equation.flux"
        if not source.exists():
            pytest.skip(f"Example file {source} not found")

        result = self.run_flux_compiler(source, "python")
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        # Check output file was created
        output_file = self.output_dir / "generated.py"
        assert output_file.exists(), "Python output file not generated"

    @pytest.mark.integration
    def test_heat_equation_cpp(self):
        """Test heat equation compiles to C++"""
        source = self.examples_dir / "heat_equation.flux"
        if not source.exists():
            pytest.skip(f"Example file {source} not found")

        result = self.run_flux_compiler(source, "cpp")
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        # Check output files were created
        output_file = self.output_dir / "generated.cpp"
        cmake_file = self.output_dir / "CMakeLists.txt"
        makefile = self.output_dir / "Makefile"

        assert output_file.exists(), "C++ output file not generated"
        assert cmake_file.exists(), "CMakeLists.txt not generated"
        assert makefile.exists(), "Makefile not generated"

    @pytest.mark.integration
    def test_navier_stokes_python(self):
        """Test Navier-Stokes cavity flow compiles to Python"""
        source = self.examples_dir / "navier_stokes_cavity.flux"
        if not source.exists():
            pytest.skip(f"Example file {source} not found")

        result = self.run_flux_compiler(source, "python")
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        output_file = self.output_dir / "generated.py"
        assert output_file.exists(), "Python output file not generated"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gpu_cfd_cuda(self):
        """Test GPU-accelerated CFD compiles to CUDA"""
        source = self.examples_dir / "gpu_accelerated_cfd.flux"
        if not source.exists():
            pytest.skip(f"Example file {source} not found")

        result = self.run_flux_compiler(source, "cuda")
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        # Check CUDA output file was created
        output_file = self.output_dir / "generated.cu"
        assert output_file.exists(), "CUDA output file not generated"

    @pytest.mark.integration
    def test_structural_analysis(self):
        """Test structural analysis example compiles"""
        source = self.examples_dir / "structural_analysis.flux"
        if not source.exists():
            pytest.skip(f"Example file {source} not found")

        result = self.run_flux_compiler(source, "python")
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        output_file = self.output_dir / "generated.py"
        assert output_file.exists(), "Python output file not generated"

    @pytest.mark.integration
    def test_electromagnetic_scattering(self):
        """Test electromagnetic scattering example compiles"""
        source = self.examples_dir / "electromagnetic_scattering.flux"
        if not source.exists():
            pytest.skip(f"Example file {source} not found")

        result = self.run_flux_compiler(source, "python")
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        output_file = self.output_dir / "generated.py"
        assert output_file.exists(), "Python output file not generated"


class TestInteractiveMode:
    """Test interactive FLUX environment"""

    @pytest.mark.slow
    def test_interactive_commands(self):
        """Test basic interactive mode commands"""
        # Create input for interactive mode
        commands = [
            "mesh StructuredGrid 10 10",
            "help",
            "exit"
        ]
        input_text = "\n".join(commands)

        # Run interactive mode
        cmd = [sys.executable, "flux_scientific.py", "-i"]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Check it ran successfully
        assert result.returncode == 0
        assert "FLUX Scientific Computing Language" in result.stdout
        assert "Created StructuredGrid" in result.stdout


class TestBenchmarks:
    """Test benchmark execution"""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_run_benchmarks(self):
        """Test that benchmarks can be executed"""
        cmd = [sys.executable, "flux_scientific.py", "--benchmark"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check benchmarks ran
        assert result.returncode == 0
        assert "FLUX Scientific Computing Benchmarks" in result.stdout

    @pytest.mark.benchmark
    def test_simple_benchmark(self):
        """Test simple benchmark script"""
        benchmark_file = Path("benchmarks/simple_benchmark.py")
        if not benchmark_file.exists():
            pytest.skip("Benchmark file not found")

        cmd = [sys.executable, str(benchmark_file)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Just check it runs without error
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])