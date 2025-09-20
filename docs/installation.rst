Installation
============

Requirements
------------

FLUX requires Python 3.7 or later. The following packages are required:

**Core Dependencies**
   - numpy >= 1.18.0
   - scipy >= 1.5.0
   - matplotlib >= 3.3.0

**Optional Dependencies**
   - cupy-cuda11x (for GPU acceleration)
   - jupyter (for notebook support)
   - vtk (for advanced visualization)

Install from PyPI
-----------------

The easiest way to install FLUX is using pip:

.. code-block:: bash

   pip install flux-sci-lang

This will install FLUX and all required dependencies.

Install with GPU Support
-------------------------

For GPU-accelerated computations, install with CUDA support:

.. code-block:: bash

   pip install flux-sci-lang[gpu]

This includes CuPy for NVIDIA GPU acceleration.

Install Development Version
---------------------------

To install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/MichaelCrowe11/flux-sci-lang.git

Developer Installation
----------------------

For developers who want to modify FLUX:

.. code-block:: bash

   git clone https://github.com/MichaelCrowe11/flux-sci-lang.git
   cd flux-sci-lang
   pip install -e .[dev]

This installs FLUX in editable mode with development dependencies.

Verify Installation
-------------------

Test your installation:

.. code-block:: bash

   flux-sci --version
   flux-sci --list-examples

Or in Python:

.. code-block:: python

   import flux_sci
   print(flux_sci.__version__)

Docker Installation
-------------------

FLUX is also available as a Docker image:

.. code-block:: bash

   docker pull fluxscilang/flux-sci-lang:latest
   docker run -it fluxscilang/flux-sci-lang:latest

VS Code Extension
-----------------

Install the VS Code extension for enhanced development experience:

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "FLUX Scientific Computing Language"
4. Click Install

Or install from command line:

.. code-block:: bash

   code --install-extension flux-sci-lang.flux-sci-lang

Conda Installation
------------------

FLUX will be available on conda-forge soon:

.. code-block:: bash

   conda install -c conda-forge flux-sci-lang

Backend-Specific Requirements
-----------------------------

**C++ Backend**
   - CMake >= 3.12
   - C++14 compatible compiler (GCC 7+, Clang 6+, MSVC 2017+)
   - OpenMP (for parallelization)

**CUDA Backend**
   - NVIDIA CUDA Toolkit >= 11.0
   - Compatible NVIDIA GPU with compute capability 3.5+

**Julia Backend**
   - Julia >= 1.6
   - DifferentialEquations.jl package

**Fortran Backend**
   - Modern Fortran compiler (gfortran 9+, ifort)

Troubleshooting
---------------

**Import Errors**
   If you encounter import errors, ensure all dependencies are installed:

   .. code-block:: bash

      pip install --upgrade flux-sci-lang

**GPU Issues**
   For CUDA-related problems:

   .. code-block:: bash

      # Check CUDA installation
      nvidia-smi
      python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"

**Performance Issues**
   For optimal performance, ensure BLAS libraries are properly configured:

   .. code-block:: bash

      python -c "import numpy; numpy.show_config()"

Platform-Specific Notes
------------------------

**Windows**
   - Visual Studio Build Tools may be required for C++ compilation
   - Use Anaconda or Miniconda for easier dependency management

**macOS**
   - Xcode Command Line Tools required for C++ compilation
   - Use Homebrew for additional dependencies

**Linux**
   - Build-essential package recommended
   - CUDA installation varies by distribution

Getting Help
------------

If you encounter installation issues:

1. Check the `FAQ <https://flux-sci-lang.readthedocs.io/en/latest/faq.html>`_
2. Search existing `issues <https://github.com/MichaelCrowe11/flux-sci-lang/issues>`_
3. Create a new issue with system information