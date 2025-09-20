Quick Start Guide
=================

This guide will get you up and running with FLUX in minutes. We'll solve a simple heat equation to demonstrate the basic workflow.

Your First FLUX Program
------------------------

Create a file called ``heat_example.flux``:

.. code-block:: flux

   // Simple 2D heat equation
   domain heat_domain {
       rectangle(0, 1, 0, 1)
       grid(50, 50)
   }

   equation heat_eq {
       dt(temperature) = 0.1 * laplacian(temperature)
   }

   boundary heat_bc {
       dirichlet(100.0) on left
       dirichlet(0.0) on right, top, bottom
   }

   solver heat_solver {
       method: crank_nicolson
       timestep: 0.01
       max_time: 1.0
   }

Compile and Run
---------------

.. code-block:: bash

   flux-sci heat_example.flux --run --plot

This will:
1. Compile the FLUX code to Python
2. Execute the numerical solver
3. Display a plot of the temperature field

Understanding the Code
----------------------

Let's break down each section:

**Domain Definition**
   .. code-block:: flux

      domain heat_domain {
          rectangle(0, 1, 0, 1)
          grid(50, 50)
      }

   - Defines a rectangular domain from (0,0) to (1,1)
   - Creates a 50×50 computational grid

**Equation**
   .. code-block:: flux

      equation heat_eq {
          dt(temperature) = 0.1 * laplacian(temperature)
      }

   - The heat equation: ∂T/∂t = α∇²T
   - ``dt(temperature)`` = time derivative
   - ``laplacian(temperature)`` = spatial Laplacian
   - Thermal diffusivity α = 0.1

**Boundary Conditions**
   .. code-block:: flux

      boundary heat_bc {
          dirichlet(100.0) on left
          dirichlet(0.0) on right, top, bottom
      }

   - Fixed temperature of 100°C on left boundary
   - Fixed temperature of 0°C on other boundaries

**Solver Configuration**
   .. code-block:: flux

      solver heat_solver {
          method: crank_nicolson
          timestep: 0.01
          max_time: 1.0
      }

   - Uses Crank-Nicolson method (2nd order accurate)
   - Time step of 0.01 seconds
   - Simulate for 1.0 second total

Interactive Mode
----------------

FLUX includes an interactive mode for experimentation:

.. code-block:: bash

   flux-sci -i

.. code-block:: text

   flux-sci> compile heat_example.flux
   flux-sci> mesh rectangle 100 100
   flux-sci> help

Command Line Options
--------------------

Common command line options:

.. code-block:: bash

   # Compile to different backends
   flux-sci heat.flux -b python     # Default
   flux-sci heat.flux -b cpp        # C++ with OpenMP
   flux-sci heat.flux -b cuda       # GPU acceleration

   # Optimization levels
   flux-sci heat.flux --optimize O3

   # Validation and debugging
   flux-sci heat.flux --validate    # Check syntax only
   flux-sci heat.flux --ast         # Show parse tree
   flux-sci heat.flux --profile     # Performance profiling

More Examples
-------------

**Wave Equation**
   .. code-block:: flux

      domain wave_domain {
          rectangle(-1, 1, -1, 1)
          grid(100, 100)
      }

      equation wave_eq {
          d2dt(amplitude) = 0.5^2 * laplacian(amplitude)
      }

      boundary wave_bc {
          dirichlet(0.0) on all
      }

      solver wave_solver {
          method: explicit
          timestep: 0.001
          max_time: 2.0
      }

**Poisson Equation**
   .. code-block:: flux

      domain poisson_domain {
          rectangle(0, 1, 0, 1)
          grid(64, 64)
      }

      equation poisson_eq {
          laplacian(potential) = -source_term
      }

      boundary poisson_bc {
          dirichlet(0.0) on all
      }

      solver poisson_solver {
          method: conjugate_gradient
          tolerance: 1e-8
      }

Python Integration
------------------

You can also use FLUX from Python:

.. code-block:: python

   from flux_sci import FluxCompiler

   # Compile FLUX code
   compiler = FluxCompiler()
   solver = compiler.compile_file('heat_example.flux')

   # Run simulation
   result = solver.solve()

   # Visualize results
   result.plot()
   result.animate()

Next Steps
----------

Now that you've created your first FLUX program:

1. **Learn the Language**: Read the :doc:`tutorial/index` for comprehensive coverage
2. **Explore Examples**: Check out :doc:`examples/index` for more complex problems
3. **Advanced Features**: Learn about :doc:`advanced/gpu_acceleration` and :doc:`advanced/parallel_computing`
4. **VS Code Setup**: Install the VS Code extension for better development experience

Common Workflows
----------------

**Research & Prototyping**
   1. Write FLUX code with domain-specific notation
   2. Use Python backend for rapid iteration
   3. Validate against analytical solutions
   4. Generate publication-quality plots

**Production Deployment**
   1. Develop in FLUX with Python backend
   2. Optimize with C++ backend
   3. Scale with CUDA for large problems
   4. Deploy with containerization

**Educational Use**
   1. Start with simple 1D problems
   2. Progress to 2D and 3D
   3. Explore different numerical methods
   4. Compare stability and accuracy

Getting Help
------------

- **Documentation**: This documentation covers all features
- **Examples**: Built-in examples demonstrate best practices
- **Community**: GitHub discussions for questions and ideas
- **Issues**: Report bugs on GitHub issue tracker