"""
FLUX Scientific Computing Language
A production-ready domain-specific language for scientific computing with validated PDE solvers
"""

__version__ = "0.1.0"
__author__ = "Michael Crowe"
__email__ = "michael@flux-sci.org"

# Import main components
from .pde_lexer import FluxPDELexer
from .pde_parser import FluxPDEParser
from .codegen import BackendManager
from .heat_solver import HeatEquationSolver

# Import solvers
from .solvers import (
    FiniteDifferenceSolver,
    ValidationSuite,
    AnalyticalSolutions
)

__all__ = [
    "FluxPDELexer",
    "FluxPDEParser",
    "BackendManager",
    "HeatEquationSolver",
    "FiniteDifferenceSolver",
    "ValidationSuite",
    "AnalyticalSolutions",
]