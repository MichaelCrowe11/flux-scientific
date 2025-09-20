FLUX Language Syntax
====================

FLUX uses a clean, mathematical syntax designed to naturally express partial differential equations and their numerical solution. This section covers the complete language syntax.

Basic Structure
---------------

A FLUX program consists of four main components:

.. code-block:: flux

   // Domain definition
   domain name { ... }

   // Equation specification
   equation name { ... }

   // Boundary conditions
   boundary name { ... }

   // Solver configuration
   solver name { ... }

Comments
--------

FLUX supports both line and block comments:

.. code-block:: flux

   // Single line comment

   /*
    * Multi-line comment
    * can span multiple lines
    */

Identifiers and Keywords
------------------------

**Identifiers**
   - Must start with a letter or underscore
   - Can contain letters, numbers, and underscores
   - Case-sensitive

.. code-block:: flux

   temperature    // valid
   velocity_x     // valid
   field_2d       // valid
   _internal      // valid
   2D_field       // invalid (starts with number)

**Reserved Keywords**
   - ``domain``, ``equation``, ``boundary``, ``solver``
   - ``field``, ``function``, ``let``, ``const``
   - ``if``, ``else``, ``for``, ``while``, ``return``
   - ``real``, ``int``, ``bool``, ``vector``, ``matrix``

Data Types
----------

**Scalar Types**
   .. code-block:: flux

      real temperature = 273.15
      int grid_points = 100
      bool is_steady = false

**Vector Types**
   .. code-block:: flux

      vector velocity = [1.0, 0.0, 0.0]
      vector position = [x, y, z]

**Field Types**
   .. code-block:: flux

      field scalar temperature
      field vector velocity
      field tensor stress

Operators
---------

**Arithmetic Operators**
   .. code-block:: flux

      +    // addition
      -    // subtraction
      *    // multiplication
      /    // division
      ^    // exponentiation
      %    // modulo

**Comparison Operators**
   .. code-block:: flux

      ==   // equal
      !=   // not equal
      <    // less than
      >    // greater than
      <=   // less than or equal
      >=   // greater than or equal

**Logical Operators**
   .. code-block:: flux

      &&   // logical AND
      ||   // logical OR
      !    // logical NOT

**Differential Operators**
   .. code-block:: flux

      dt(u)           // ∂u/∂t (time derivative)
      dx(u)           // ∂u/∂x (spatial derivative)
      dy(u)           // ∂u/∂y
      dz(u)           // ∂u/∂z
      d2dx(u)         // ∂²u/∂x² (second derivative)
      d2dt(u)         // ∂²u/∂t²
      laplacian(u)    // ∇²u (Laplacian)
      gradient(u)     // ∇u (gradient)
      divergence(v)   // ∇·v (divergence)
      curl(v)         // ∇×v (curl)

Mathematical Functions
----------------------

**Built-in Functions**
   .. code-block:: flux

      sin(x), cos(x), tan(x)      // trigonometric
      exp(x), log(x), sqrt(x)     // exponential/logarithmic
      abs(x), min(x,y), max(x,y)  // utility
      pow(x,n), mod(x,y)          // power and modulo

**Special Functions**
   .. code-block:: flux

      integral(f, a, b)           // numerical integration
      diff(f, x)                  // numerical differentiation
      sum(array)                  // array summation
      norm(vector)                // vector norm

Control Flow
------------

**Conditional Statements**
   .. code-block:: flux

      if condition {
          // statements
      } else if other_condition {
          // statements
      } else {
          // statements
      }

**Loops**
   .. code-block:: flux

      for i in 0..n {
          // loop body
      }

      while condition {
          // loop body
      }

**Functions**
   .. code-block:: flux

      function real gaussian(real x, real sigma) {
          return exp(-x^2 / (2 * sigma^2))
      }

Expressions and Precedence
--------------------------

Operator precedence (highest to lowest):

1. Function calls: ``f(x)``
2. Exponentiation: ``^``
3. Unary: ``+``, ``-``, ``!``
4. Differential operators: ``dt()``, ``dx()``, ``laplacian()``
5. Multiplication/Division: ``*``, ``/``, ``%``
6. Addition/Subtraction: ``+``, ``-``
7. Comparison: ``<``, ``>``, ``<=``, ``>=``
8. Equality: ``==``, ``!=``
9. Logical AND: ``&&``
10. Logical OR: ``||``

**Examples**
   .. code-block:: flux

      dt(u) + v * dx(u)           // (dt(u)) + (v * (dx(u)))
      a + b * c^2                 // a + (b * (c^2))
      laplacian(u) - k * u        // (laplacian(u)) - (k * u)

String Literals
---------------

.. code-block:: flux

   "double quoted string"
   'single quoted string'
   "string with \\escape sequences\\n"

Arrays and Indexing
-------------------

.. code-block:: flux

   real array[10]                  // 1D array
   real matrix[10][20]             // 2D array

   array[i] = value                // indexing
   matrix[i][j] = value            // 2D indexing

   real slice[5:10]                // array slice

Physical Units (Optional)
-------------------------

FLUX supports optional unit annotations:

.. code-block:: flux

   real temperature [K] = 273.15
   real velocity [m/s] = 10.0
   real pressure [Pa] = 101325.0

Constants and Variables
-----------------------

.. code-block:: flux

   const real pi = 3.14159
   const real gravity = 9.81 [m/s^2]

   let real density = 1000.0 [kg/m^3]
   let int iterations = 100

Error Handling
--------------

.. code-block:: flux

   try {
       // potentially failing code
   } catch error {
       // error handling
   }

Preprocessing
-------------

**Include Directives**
   .. code-block:: flux

      #include "common_functions.flux"
      #include <standard_library.flux>

**Conditional Compilation**
   .. code-block:: flux

      #ifdef GPU_ENABLED
          // GPU-specific code
      #else
          // CPU fallback
      #endif

**Macros**
   .. code-block:: flux

      #define THERMAL_DIFFUSIVITY 0.1
      #define GRID_SIZE 100

Language Extensions
-------------------

**Custom Operators**
   Users can define custom differential operators:

   .. code-block:: flux

      operator biharmonic(u) = laplacian(laplacian(u))

**Domain-Specific Libraries**
   .. code-block:: flux

      import fluid_dynamics
      import electromagnetics
      import structural_mechanics

Best Practices
--------------

1. **Naming Conventions**
   - Use descriptive names: ``temperature`` not ``T``
   - Use snake_case for variables: ``thermal_diffusivity``
   - Use CamelCase for types: ``FluidDomain``

2. **Code Organization**
   - Group related equations together
   - Use functions for repeated expressions
   - Add comments for complex mathematical formulations

3. **Performance Considerations**
   - Avoid deep nesting in expressions
   - Use built-in operators when possible
   - Consider vectorized operations

Example: Complete Syntax Usage
------------------------------

.. code-block:: flux

   // Navier-Stokes equations with complete syntax
   #include "fluid_constants.flux"

   const real reynolds = 100.0
   const real nu = 1.0 / reynolds

   domain flow_domain {
       rectangle(0, 1, 0, 1)
       grid(128, 128)
   }

   field vector velocity
   field scalar pressure

   function real inlet_profile(real y) {
       return 6.0 * y * (1.0 - y)
   }

   equation momentum_x {
       dt(velocity.x) + velocity.x * dx(velocity.x) + velocity.y * dy(velocity.x)
           = -dx(pressure) + nu * laplacian(velocity.x)
   }

   equation momentum_y {
       dt(velocity.y) + velocity.x * dx(velocity.y) + velocity.y * dy(velocity.y)
           = -dy(pressure) + nu * laplacian(velocity.y)
   }

   equation continuity {
       dx(velocity.x) + dy(velocity.y) = 0
   }

   boundary flow_bc {
       dirichlet([inlet_profile(y), 0.0]) on left
       dirichlet([0.0, 0.0]) on top, bottom
       neumann([0.0, 0.0]) on right
   }

   solver navier_solver {
       method: fractional_step
       timestep: 0.001
       max_time: 10.0
       tolerance: 1e-6
   }