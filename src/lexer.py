"""
FLUX Language Lexer
Tokenizes FLUX source code with support for AI-native and quantum types
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Any

class TokenType(Enum):
    # Literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    CHAR = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Keywords
    LET = auto()
    VAR = auto()
    CONST = auto()
    FUNCTION = auto()
    ASYNC = auto()
    AWAIT = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    MATCH = auto()
    TEMPLATE = auto()
    RENDER = auto()
    STATE = auto()
    ON = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    
    # Type Keywords
    INT_TYPE = auto()
    FLOAT_TYPE = auto()
    BOOL_TYPE = auto()
    STRING_TYPE = auto()
    VECTOR = auto()
    TENSOR = auto()
    PROBABILITY = auto()
    UNCERTAIN = auto()
    SEMANTIC = auto()
    QUBIT = auto()
    QUANTUM = auto()
    ENTANGLED = auto()
    UNIQUE = auto()
    BORROWED = auto()
    SHARED = auto()
    
    # AI/ML Keywords
    LLM = auto()
    EMBED = auto()
    AI = auto()
    MODEL = auto()
    PROMPT = auto()
    GRADIENT = auto()
    DIFFERENTIABLE = auto()
    
    # Quantum Keywords
    CIRCUIT = auto()
    MEASURE = auto()
    HADAMARD = auto()
    CLASSICAL = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()
    
    # Comparison
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Assignment
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    
    # Special Operators
    ARROW = auto()
    FAT_ARROW = auto()
    PIPE = auto()
    SEMANTIC_SIMILAR = auto()  # ~~
    MATRIX_MUL = auto()  # @
    
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
    DECORATOR = auto()  # @decorator
    TEMPLATE_START = auto()  # {
    TEMPLATE_END = auto()  # }
    COMMENT = auto()
    NEWLINE = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    
class FLUXLexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # Keyword mapping
        self.keywords = {
            'let': TokenType.LET,
            'var': TokenType.VAR,
            'const': TokenType.CONST,
            'function': TokenType.FUNCTION,
            'async': TokenType.ASYNC,
            'await': TokenType.AWAIT,
            'return': TokenType.RETURN,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'match': TokenType.MATCH,
            'template': TokenType.TEMPLATE,
            'render': TokenType.RENDER,
            'state': TokenType.STATE,
            'on': TokenType.ON,
            'import': TokenType.IMPORT,
            'from': TokenType.FROM,
            'as': TokenType.AS,
            'true': TokenType.BOOL,
            'false': TokenType.BOOL,
            # Types
            'int': TokenType.INT_TYPE,
            'float': TokenType.FLOAT_TYPE,
            'bool': TokenType.BOOL_TYPE,
            'string': TokenType.STRING_TYPE,
            'vector': TokenType.VECTOR,
            'tensor': TokenType.TENSOR,
            'probability': TokenType.PROBABILITY,
            'uncertain': TokenType.UNCERTAIN,
            'semantic': TokenType.SEMANTIC,
            'qubit': TokenType.QUBIT,
            'quantum': TokenType.QUANTUM,
            'entangled': TokenType.ENTANGLED,
            'unique': TokenType.UNIQUE,
            'borrowed': TokenType.BORROWED,
            'shared': TokenType.SHARED,
            # AI/ML
            'llm': TokenType.LLM,
            'embed': TokenType.EMBED,
            'ai': TokenType.AI,
            'model': TokenType.MODEL,
            'prompt': TokenType.PROMPT,
            'gradient': TokenType.GRADIENT,
            # Quantum
            'circuit': TokenType.CIRCUIT,
            'measure': TokenType.MEASURE,
            'hadamard': TokenType.HADAMARD,
            'classical': TokenType.CLASSICAL,
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
    
    def read_string(self) -> str:
        quote = self.advance()  # Skip opening quote
        value = ''
        while self.current_char() and self.current_char() != quote:
            if self.current_char() == '\\':
                self.advance()
                next_char = self.advance()
                # Handle escape sequences
                if next_char == 'n':
                    value += '\n'
                elif next_char == 't':
                    value += '\t'
                elif next_char == '\\':
                    value += '\\'
                elif next_char == quote:
                    value += quote
                else:
                    value += next_char
            else:
                value += self.advance()
        self.advance()  # Skip closing quote
        return value
    
    def read_number(self) -> Token:
        start_line = self.line
        start_col = self.column
        num_str = ''
        is_float = False
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            if self.current_char() == '.':
                if is_float:
                    break
                is_float = True
            num_str += self.advance()
        
        if is_float:
            return Token(TokenType.FLOAT, float(num_str), start_line, start_col)
        else:
            return Token(TokenType.INT, int(num_str), start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line = self.line
        start_col = self.column
        identifier = ''
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            identifier += self.advance()
        
        # Check if it's a keyword
        token_type = self.keywords.get(identifier, TokenType.IDENTIFIER)
        
        # Handle boolean values
        if identifier == 'true':
            return Token(TokenType.BOOL, True, start_line, start_col)
        elif identifier == 'false':
            return Token(TokenType.BOOL, False, start_line, start_col)
        
        return Token(token_type, identifier, start_line, start_col)
    
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
            
            # Newlines
            if self.current_char() == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\\n', start_line, start_col))
                continue
            
            # String literals
            if self.current_char() in '"\'':
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, value, start_line, start_col))
                continue
            
            # Numbers
            if self.current_char().isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Operators and punctuation
            char = self.current_char()
            
            # Two-character operators
            if char == '-' and self.peek_char() == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '->', start_line, start_col))
            elif char == '=' and self.peek_char() == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.FAT_ARROW, '=>', start_line, start_col))
            elif char == '=' and self.peek_char() == '=':
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
            elif char == '~' and self.peek_char() == '~':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.SEMANTIC_SIMILAR, '~~', start_line, start_col))
            elif char == '&' and self.peek_char() == '&':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.AND, '&&', start_line, start_col))
            elif char == '|' and self.peek_char() == '|':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.OR, '||', start_line, start_col))
            elif char == '+' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', start_line, start_col))
            elif char == '-' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', start_line, start_col))
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
            elif char == '%':
                self.advance()
                self.tokens.append(Token(TokenType.PERCENT, '%', start_line, start_col))
            elif char == '@':
                self.advance()
                # Check if it's a decorator or matrix multiplication
                if self.current_char() and self.current_char().isalpha():
                    self.tokens.append(Token(TokenType.DECORATOR, '@', start_line, start_col))
                else:
                    self.tokens.append(Token(TokenType.MATRIX_MUL, '@', start_line, start_col))
            elif char == '=':
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, '=', start_line, start_col))
            elif char == '<':
                self.advance()
                self.tokens.append(Token(TokenType.LT, '<', start_line, start_col))
            elif char == '>':
                self.advance()
                self.tokens.append(Token(TokenType.GT, '>', start_line, start_col))
            elif char == '!':
                self.advance()
                self.tokens.append(Token(TokenType.NOT, '!', start_line, start_col))
            elif char == '|':
                self.advance()
                self.tokens.append(Token(TokenType.PIPE, '|', start_line, start_col))
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