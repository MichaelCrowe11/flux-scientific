FLUX Scientific Computing Language Documentation
===============================================

Welcome to FLUX, a domain-specific language designed for solving partial differential equations (PDEs) and scientific computing tasks. FLUX provides an intuitive syntax for describing mathematical problems and automatically generates efficient numerical solvers.

.. image:: _static/flux_banner.png
   :alt: FLUX Scientific Computing Language
   :align: center
   :width: 600px

Quick Start
-----------

Install FLUX using pip:

.. code-block:: bash

   pip install flux-sci-lang

Create your first FLUX program:

.. code-block:: flux

   // Heat equation solver
   domain heat_domain {
       rectangle(0, 1, 0, 1)
       grid(50, 50)
   }

   equation heat_eq {
       dt(u) = alpha * laplacian(u)
   }

   boundary heat_bc {
       dirichlet(0.0) on left, right, top, bottom
   }

   solver heat_solver {
       method: crank_nicolson
       timestep: 0.001
       max_time: 1.0
   }

Compile and run:

.. code-block:: bash

   flux-sci heat_equation.flux --run

Key Features
------------

🔥 **Intuitive Syntax**
   Express complex PDEs using natural mathematical notation

⚡ **Multiple Backends**
   Generate Python, C++, CUDA, Julia, or Fortran code

🧮 **Validated Solvers**
   Production-ready numerical methods with proven accuracy

🚀 **GPU Acceleration**
   Automatic GPU optimization for compatible hardware

📊 **Built-in Visualization**
   Integrated plotting and animation capabilities

🔧 **VS Code Integration**
   Full IDE support with syntax highlighting and IntelliSense

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial/index
   examples/index
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Language Reference

   language/syntax
   language/domains
   language/equations
   language/boundaries
   language/solvers
   language/functions

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/backends
   advanced/optimization
   advanced/gpu_acceleration
   advanced/parallel_computing
   advanced/custom_solvers

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Examples

   notebooks/01_getting_started
   notebooks/02_advanced_pdes
   examples/heat_equation
   examples/wave_equation
   examples/navier_stokes
   examples/custom_pde

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   development/architecture
   development/extending
   changelog

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/flux_compiler
   api/solvers
   api/mesh
   api/visualization

Supported PDE Types
-------------------

FLUX supports a wide range of partial differential equations:

**Parabolic PDEs**
   - Heat/Diffusion equations
   - Reaction-diffusion systems
   - Black-Scholes equations

**Hyperbolic PDEs**
   - Wave equations
   - Transport equations
   - Shallow water equations

**Elliptic PDEs**
   - Poisson equations
   - Laplace equations
   - Helmholtz equations

**Mixed Systems**
   - Navier-Stokes equations
   - Maxwell's equations
   - Coupled multiphysics problems

Community
---------

- **GitHub**: `flux-sci-lang <https://github.com/MichaelCrowe11/flux-sci-lang>`_
- **Issues**: `Report bugs <https://github.com/MichaelCrowe11/flux-sci-lang/issues>`_
- **Discussions**: `Community forum <https://github.com/MichaelCrowe11/flux-sci-lang/discussions>`_

License
-------

FLUX is released under the MIT License. See the `LICENSE <https://github.com/MichaelCrowe11/flux-sci-lang/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`