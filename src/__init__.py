"""
FLUX Language - Future Language for Universal eXecution
An AI-native programming language with quantum computing support
"""

from .lexer import FLUXLexer, Token, TokenType
from .parser import FLUXParser
from .interpreter import FLUXInterpreter

__version__ = "0.1.0"
__all__ = ["FLUXLexer", "FLUXParser", "FLUXInterpreter", "Token", "TokenType"]