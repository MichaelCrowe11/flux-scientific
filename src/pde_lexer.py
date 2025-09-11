"""
FLUX Scientific Computing Lexer
Extended tokenizer for PDE syntax and scientific operators
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Any

class TokenType(Enum):
    # Numbers and Identifiers
    NUMBER = auto()
    IDENTIFIER = auto()
    
    # PDE-specific keywords
    PDE = auto()
    DOMAIN = auto()
    MESH = auto()
    SOLVER = auto()
    BOUNDARY = auto()
    INITIAL = auto()
    VARIABLES = auto()
    REGIONS = auto()
    INTERFACE = auto()
    
    # Scientific keywords
    GRADIENT = auto()
    DIVERGENCE = auto()
    CURL = auto()
    LAPLACIAN = auto()
    PARTIAL = auto()
    
    # Mesh types
    STRUCTURED_GRID = auto()
    UNSTRUCTURED_MESH = auto()
    ADAPTIVE_MESH = auto()
    
    # Solver types
    IMPLICIT_EULER = auto()
    EXPLICIT_EULER = auto()
    RUNGE_KUTTA = auto()
    FINITE_VOLUME = auto()
    FINITE_ELEMENT = auto()
    SPECTRAL = auto()
    
    # Mathematical operators
    NABLA = auto()          # ∇
    PARTIAL_DERIV = auto()  # ∂
    DOT_PRODUCT = auto()    # ·
    CROSS_PRODUCT = auto()  # ×
    TENSOR_PRODUCT = auto() # ⊗
    
    # Standard operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    POWER = auto()
    ASSIGN = auto()
    
    # Comparison
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    
    # Special
    IN = auto()
    ON = auto()
    OVER = auto()
    AT = auto()
    WITH = auto()
    
    # Decorators
    GPU = auto()
    BACKEND = auto()
    VECTORIZE = auto()
    PARALLEL = auto()
    
    # Types
    FIELD = auto()
    VECTOR_FIELD = auto()
    TENSOR_FIELD = auto()
    SCALAR = auto()
    MATRIX = auto()
    
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

class FluxPDELexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # Keywords mapping
        self.keywords = {
            'pde': TokenType.PDE,
            'domain': TokenType.DOMAIN,
            'mesh': TokenType.MESH,
            'solver': TokenType.SOLVER,
            'boundary': TokenType.BOUNDARY,
            'initial': TokenType.INITIAL,
            'variables': TokenType.VARIABLES,
            'regions': TokenType.REGIONS,
            'interface': TokenType.INTERFACE,
            
            # Mesh types
            'StructuredGrid': TokenType.STRUCTURED_GRID,
            'UnstructuredMesh': TokenType.UNSTRUCTURED_MESH,
            'AdaptiveMesh': TokenType.ADAPTIVE_MESH,
            
            # Solver types
            'ImplicitEuler': TokenType.IMPLICIT_EULER,
            'ExplicitEuler': TokenType.EXPLICIT_EULER,
            'RungeKutta': TokenType.RUNGE_KUTTA,
            'FiniteVolume': TokenType.FINITE_VOLUME,
            'FiniteElement': TokenType.FINITE_ELEMENT,
            'Spectral': TokenType.SPECTRAL,
            
            # Mathematical functions
            'gradient': TokenType.GRADIENT,
            'grad': TokenType.GRADIENT,
            'divergence': TokenType.DIVERGENCE,
            'div': TokenType.DIVERGENCE,
            'curl': TokenType.CURL,
            'rot': TokenType.CURL,
            'laplacian': TokenType.LAPLACIAN,
            
            # Types
            'Field': TokenType.FIELD,
            'VectorField': TokenType.VECTOR_FIELD,
            'TensorField': TokenType.TENSOR_FIELD,
            'Scalar': TokenType.SCALAR,
            'Matrix': TokenType.MATRIX,
            
            # Special keywords
            'in': TokenType.IN,
            'on': TokenType.ON,
            'over': TokenType.OVER,
            'at': TokenType.AT,
            'with': TokenType.WITH,
            
            # Decorators
            'gpu': TokenType.GPU,
            'backend': TokenType.BACKEND,
            'vectorize': TokenType.VECTORIZE,
            'parallel': TokenType.PARALLEL,
        }
        
        # Unicode mathematical operators
        self.unicode_ops = {
            '∇': TokenType.NABLA,
            '∂': TokenType.PARTIAL_DERIV,
            '·': TokenType.DOT_PRODUCT,
            '×': TokenType.CROSS_PRODUCT,
            '⊗': TokenType.TENSOR_PRODUCT,
            '∇²': TokenType.LAPLACIAN,
        }
    
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        char = self.source[self.position]
        self.position += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '/' and self.peek_char() == '/':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_number(self) -> Token:
        start_line = self.line
        start_col = self.column
        num_str = ''
        
        # Handle scientific notation
        while self.current_char() and (self.current_char().isdigit() or 
                                      self.current_char() in '.eE+-'):
            if self.current_char() in 'eE':
                num_str += self.advance()
                if self.current_char() in '+-':
                    num_str += self.advance()
            else:
                num_str += self.advance()
        
        try:
            value = float(num_str) if '.' in num_str or 'e' in num_str.lower() else int(num_str)
        except ValueError:
            value = num_str  # Return as string if parsing fails
            
        return Token(TokenType.NUMBER, value, start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line = self.line
        start_col = self.column
        identifier = ''
        
        # Allow Greek letters and subscripts in identifiers
        while self.current_char() and (self.current_char().isalnum() or 
                                      self.current_char() in '_αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ₀₁₂₃₄₅₆₇₈₉'):
            identifier += self.advance()
        
        # Check if it's a keyword
        token_type = self.keywords.get(identifier, TokenType.IDENTIFIER)
        
        return Token(token_type, identifier, start_line, start_col)
    
    def read_partial_derivative(self) -> Token:
        start_line = self.line
        start_col = self.column
        
        # Read ∂ or partial derivative notation
        self.advance()  # Skip ∂
        
        return Token(TokenType.PARTIAL_DERIV, '∂', start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        while self.position < len(self.source):
            self.skip_whitespace()
            
            if self.current_char() is None:
                break
            
            # Skip comments
            if self.current_char() == '/' and self.peek_char() == '/':
                self.skip_comment()
                continue
            
            start_line = self.line
            start_col = self.column
            char = self.current_char()
            
            # Newlines
            if char == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\\n', start_line, start_col))
                continue
            
            # Numbers (including scientific notation)
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Unicode mathematical operators
            if char in self.unicode_ops:
                token_type = self.unicode_ops[char]
                self.advance()
                # Check for ∇²
                if char == '∇' and self.current_char() == '²':
                    self.advance()
                    self.tokens.append(Token(TokenType.LAPLACIAN, '∇²', start_line, start_col))
                else:
                    self.tokens.append(Token(token_type, char, start_line, start_col))
                continue
            
            # Partial derivative ∂
            if char == '∂':
                self.tokens.append(self.read_partial_derivative())
                continue
            
            # Identifiers and keywords (including Greek letters)
            if char.isalpha() or char == '_' or char in 'αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ':
                self.tokens.append(self.read_identifier())
                continue
            
            # Operators
            # Two-character operators
            if char == '=' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '==', start_line, start_col))
            elif char == '!' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NEQ, '!=', start_line, start_col))
            elif char == '<' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LTE, '<=', start_line, start_col))
            elif char == '>' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GTE, '>=', start_line, start_col))
            elif char == '*' and self.peek_char() == '*':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.POWER, '**', start_line, start_col))
            # Single-character operators
            elif char == '+':
                self.advance()
                self.tokens.append(Token(TokenType.PLUS, '+', start_line, start_col))
            elif char == '-':
                self.advance()
                self.tokens.append(Token(TokenType.MINUS, '-', start_line, start_col))
            elif char == '*':
                self.advance()
                self.tokens.append(Token(TokenType.STAR, '*', start_line, start_col))
            elif char == '/':
                self.advance()
                self.tokens.append(Token(TokenType.SLASH, '/', start_line, start_col))
            elif char == '=':
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, '=', start_line, start_col))
            elif char == '<':
                self.advance()
                self.tokens.append(Token(TokenType.LT, '<', start_line, start_col))
            elif char == '>':
                self.advance()
                self.tokens.append(Token(TokenType.GT, '>', start_line, start_col))
            elif char == '@':
                self.advance()
                self.tokens.append(Token(TokenType.AT, '@', start_line, start_col))
            # Delimiters
            elif char == '(':
                self.advance()
                self.tokens.append(Token(TokenType.LPAREN, '(', start_line, start_col))
            elif char == ')':
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', start_line, start_col))
            elif char == '{':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', start_line, start_col))
            elif char == '}':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACE, '}', start_line, start_col))
            elif char == '[':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACKET, '[', start_line, start_col))
            elif char == ']':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACKET, ']', start_line, start_col))
            elif char == ',':
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ',', start_line, start_col))
            elif char == '.':
                self.advance()
                self.tokens.append(Token(TokenType.DOT, '.', start_line, start_col))
            elif char == ':':
                self.advance()
                self.tokens.append(Token(TokenType.COLON, ':', start_line, start_col))
            elif char == ';':
                self.advance()
                self.tokens.append(Token(TokenType.SEMICOLON, ';', start_line, start_col))
            else:
                raise SyntaxError(f"Unexpected character '{char}' at line {self.line}, column {self.column}")
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens