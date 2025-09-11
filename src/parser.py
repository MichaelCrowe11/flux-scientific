"""
FLUX Language Parser
Builds an Abstract Syntax Tree from tokenized FLUX code
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union
from .lexer import Token, TokenType, FLUXLexer

# AST Node Classes
@dataclass
class ASTNode:
    pass

@dataclass 
class Program(ASTNode):
    statements: List[ASTNode]

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class IntLiteral(ASTNode):
    value: int

@dataclass
class FloatLiteral(ASTNode):
    value: float

@dataclass
class StringLiteral(ASTNode):
    value: str

@dataclass
class BoolLiteral(ASTNode):
    value: bool

@dataclass
class VectorLiteral(ASTNode):
    elements: List[ASTNode]

@dataclass
class TensorLiteral(ASTNode):
    shape: List[int]
    data: List[ASTNode]

@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode

@dataclass
class Assignment(ASTNode):
    target: str
    value: ASTNode

@dataclass
class VariableDecl(ASTNode):
    kind: str  # 'let', 'var', 'const'
    name: str
    type_annotation: Optional[ASTNode]
    value: Optional[ASTNode]

@dataclass
class FunctionDecl(ASTNode):
    name: str
    params: List['Parameter']
    return_type: Optional[ASTNode]
    body: List[ASTNode]
    is_async: bool = False
    decorators: List[str] = None

@dataclass
class Parameter(ASTNode):
    name: str
    type_annotation: Optional[ASTNode]
    default: Optional[ASTNode] = None

@dataclass
class FunctionCall(ASTNode):
    name: str
    args: List[ASTNode]

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    then_branch: List[ASTNode]
    else_branch: Optional[List[ASTNode]]

@dataclass
class ForLoop(ASTNode):
    variable: str
    iterable: ASTNode
    body: List[ASTNode]

@dataclass
class WhileLoop(ASTNode):
    condition: ASTNode
    body: List[ASTNode]

@dataclass
class ReturnStatement(ASTNode):
    value: Optional[ASTNode]

@dataclass
class MatchStatement(ASTNode):
    expr: ASTNode
    cases: List['MatchCase']

@dataclass
class MatchCase(ASTNode):
    pattern: ASTNode
    guard: Optional[ASTNode]
    body: List[ASTNode]

@dataclass
class TypeAnnotation(ASTNode):
    base_type: str
    type_params: Optional[List[ASTNode]] = None

@dataclass
class TemplateDecl(ASTNode):
    name: str
    state: Optional[List[ASTNode]]
    render: Optional[ASTNode]
    handlers: List['EventHandler']

@dataclass
class EventHandler(ASTNode):
    event: str
    params: List[str]
    body: List[ASTNode]

@dataclass
class AIOperation(ASTNode):
    operation: str  # 'llm', 'embed', 'generate'
    params: dict
    body: Optional[ASTNode]

@dataclass
class QuantumCircuit(ASTNode):
    name: str
    params: List[Parameter]
    classical_pre: Optional[List[ASTNode]]
    quantum_ops: List[ASTNode]
    classical_post: Optional[List[ASTNode]]

@dataclass
class MemberAccess(ASTNode):
    object: ASTNode
    member: str

@dataclass
class IndexAccess(ASTNode):
    object: ASTNode
    index: ASTNode

class FLUXParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        
    def current_token(self) -> Optional[Token]:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]
    
    def peek_token(self, offset: int = 1) -> Optional[Token]:
        pos = self.position + offset
        if pos >= len(self.tokens):
            return None
        return self.tokens[pos]
    
    def advance(self) -> Token:
        token = self.current_token()
        self.position += 1
        # Skip newlines unless they're significant
        while self.current_token() and self.current_token().type == TokenType.NEWLINE:
            self.position += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if not token or token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type if token else 'EOF'}")
        return self.advance()
    
    def match(self, *token_types: TokenType) -> bool:
        token = self.current_token()
        return token and token.type in token_types
    
    def parse(self) -> Program:
        statements = []
        while self.current_token() and self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return Program(statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        # Skip newlines
        while self.match(TokenType.NEWLINE):
            self.advance()
            
        if not self.current_token() or self.current_token().type == TokenType.EOF:
            return None
            
        # Decorators
        decorators = []
        while self.match(TokenType.DECORATOR):
            self.advance()
            decorator_name = self.expect(TokenType.IDENTIFIER).value
            decorators.append(decorator_name)
            # Handle decorator arguments if present
            if self.match(TokenType.LPAREN):
                self.parse_function_call_args()  # Parse but ignore for now
                
        # Variable declarations
        if self.match(TokenType.LET, TokenType.VAR, TokenType.CONST):
            return self.parse_variable_decl()
        
        # Function declarations
        if self.match(TokenType.FUNCTION) or (decorators and self.match(TokenType.FUNCTION)):
            return self.parse_function_decl(decorators=decorators)
        
        # Async function
        if self.match(TokenType.ASYNC):
            self.advance()
            if self.match(TokenType.FUNCTION):
                return self.parse_function_decl(is_async=True, decorators=decorators)
        
        # Template declaration
        if self.match(TokenType.TEMPLATE):
            return self.parse_template_decl()
        
        # Quantum circuit
        if self.match(TokenType.QUANTUM):
            self.advance()
            if self.match(TokenType.CIRCUIT):
                return self.parse_quantum_circuit()
        
        # Control flow
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        
        if self.match(TokenType.FOR):
            return self.parse_for_loop()
        
        if self.match(TokenType.WHILE):
            return self.parse_while_loop()
        
        if self.match(TokenType.MATCH):
            return self.parse_match_statement()
        
        if self.match(TokenType.RETURN):
            return self.parse_return_statement()
        
        # Expression statement
        expr = self.parse_expression()
        return expr
    
    def parse_variable_decl(self) -> VariableDecl:
        kind = self.advance().value  # let, var, or const
        name = self.expect(TokenType.IDENTIFIER).value
        
        type_annotation = None
        if self.match(TokenType.COLON):
            self.advance()
            type_annotation = self.parse_type_annotation()
        
        value = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            value = self.parse_expression()
        
        return VariableDecl(kind, name, type_annotation, value)
    
    def parse_function_decl(self, is_async=False, decorators=None) -> FunctionDecl:
        self.expect(TokenType.FUNCTION)
        name = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.LPAREN)
        params = self.parse_parameter_list()
        self.expect(TokenType.RPAREN)
        
        return_type = None
        if self.match(TokenType.ARROW):
            self.advance()
            return_type = self.parse_type_annotation()
        
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)
        
        return FunctionDecl(name, params, return_type, body, is_async, decorators)
    
    def parse_parameter_list(self) -> List[Parameter]:
        params = []
        
        if not self.match(TokenType.RPAREN):
            params.append(self.parse_parameter())
            
            while self.match(TokenType.COMMA):
                self.advance()
                if self.match(TokenType.RPAREN):
                    break
                params.append(self.parse_parameter())
        
        return params
    
    def parse_parameter(self) -> Parameter:
        name = self.expect(TokenType.IDENTIFIER).value
        
        type_annotation = None
        if self.match(TokenType.COLON):
            self.advance()
            type_annotation = self.parse_type_annotation()
        
        default = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            default = self.parse_expression()
        
        return Parameter(name, type_annotation, default)
    
    def parse_type_annotation(self) -> TypeAnnotation:
        # Accept any type keyword or identifier
        if self.match(TokenType.IDENTIFIER, TokenType.INT_TYPE, TokenType.FLOAT_TYPE,
                     TokenType.BOOL_TYPE, TokenType.STRING_TYPE, TokenType.VECTOR,
                     TokenType.TENSOR, TokenType.QUBIT, TokenType.QUANTUM):
            base_type = self.advance().value
        else:
            raise SyntaxError(f"Expected type annotation, got {self.current_token()}")
        
        type_params = None
        if self.match(TokenType.LT):
            self.advance()
            type_params = []
            type_params.append(self.parse_type_annotation())
            
            while self.match(TokenType.COMMA):
                self.advance()
                type_params.append(self.parse_type_annotation())
            
            self.expect(TokenType.GT)
        
        return TypeAnnotation(base_type, type_params)
    
    def parse_block(self) -> List[ASTNode]:
        statements = []
        
        while not self.match(TokenType.RBRACE) and self.current_token():
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        return statements
    
    def parse_if_statement(self) -> IfStatement:
        self.expect(TokenType.IF)
        
        condition = self.parse_expression()
        
        self.expect(TokenType.LBRACE)
        then_branch = self.parse_block()
        self.expect(TokenType.RBRACE)
        
        else_branch = None
        if self.match(TokenType.ELSE):
            self.advance()
            if self.match(TokenType.IF):
                # else if
                else_branch = [self.parse_if_statement()]
            else:
                self.expect(TokenType.LBRACE)
                else_branch = self.parse_block()
                self.expect(TokenType.RBRACE)
        
        return IfStatement(condition, then_branch, else_branch)
    
    def parse_for_loop(self) -> ForLoop:
        self.expect(TokenType.FOR)
        
        variable = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.IDENTIFIER)  # 'in' keyword (add to lexer if needed)
        iterable = self.parse_expression()
        
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)
        
        return ForLoop(variable, iterable, body)
    
    def parse_while_loop(self) -> WhileLoop:
        self.expect(TokenType.WHILE)
        
        condition = self.parse_expression()
        
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)
        
        return WhileLoop(condition, body)
    
    def parse_return_statement(self) -> ReturnStatement:
        self.expect(TokenType.RETURN)
        
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.SEMICOLON, TokenType.RBRACE):
            value = self.parse_expression()
        
        return ReturnStatement(value)
    
    def parse_match_statement(self) -> MatchStatement:
        self.expect(TokenType.MATCH)
        
        expr = self.parse_expression()
        
        self.expect(TokenType.LBRACE)
        cases = []
        
        while not self.match(TokenType.RBRACE):
            pattern = self.parse_expression()
            
            guard = None
            if self.match(TokenType.IF):
                self.advance()
                guard = self.parse_expression()
            
            self.expect(TokenType.FAT_ARROW)
            
            if self.match(TokenType.LBRACE):
                self.advance()
                body = self.parse_block()
                self.expect(TokenType.RBRACE)
            else:
                body = [self.parse_expression()]
            
            cases.append(MatchCase(pattern, guard, body))
            
            if self.match(TokenType.COMMA):
                self.advance()
        
        self.expect(TokenType.RBRACE)
        
        return MatchStatement(expr, cases)
    
    def parse_template_decl(self) -> TemplateDecl:
        self.expect(TokenType.TEMPLATE)
        name = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.LBRACE)
        
        state = None
        render = None
        handlers = []
        
        while not self.match(TokenType.RBRACE):
            if self.match(TokenType.STATE):
                self.advance()
                # Parse state declarations
                state = []  # Simplified for now
                
            elif self.match(TokenType.RENDER):
                self.advance()
                self.expect(TokenType.LBRACE)
                render = self.parse_block()
                self.expect(TokenType.RBRACE)
                
            elif self.match(TokenType.ON):
                self.advance()
                event = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.LPAREN)
                params = []  # Parse event params
                self.expect(TokenType.RPAREN)
                self.expect(TokenType.LBRACE)
                body = self.parse_block()
                self.expect(TokenType.RBRACE)
                handlers.append(EventHandler(event, params, body))
        
        self.expect(TokenType.RBRACE)
        
        return TemplateDecl(name, state, render, handlers)
    
    def parse_quantum_circuit(self) -> QuantumCircuit:
        name = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.LPAREN)
        params = self.parse_parameter_list()
        self.expect(TokenType.RPAREN)
        
        self.expect(TokenType.LBRACE)
        
        classical_pre = None
        quantum_ops = []
        classical_post = None
        
        while not self.match(TokenType.RBRACE):
            if self.match(TokenType.CLASSICAL):
                self.advance()
                self.expect(TokenType.LBRACE)
                if classical_pre is None:
                    classical_pre = self.parse_block()
                else:
                    classical_post = self.parse_block()
                self.expect(TokenType.RBRACE)
            elif self.match(TokenType.QUANTUM):
                self.advance()
                self.expect(TokenType.LBRACE)
                quantum_ops = self.parse_block()
                self.expect(TokenType.RBRACE)
            else:
                self.advance()  # Skip unknown sections for now
        
        self.expect(TokenType.RBRACE)
        
        return QuantumCircuit(name, params, classical_pre, quantum_ops, classical_post)
    
    def parse_expression(self) -> ASTNode:
        return self.parse_assignment()
    
    def parse_assignment(self) -> ASTNode:
        expr = self.parse_logical_or()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN):
            op = self.advance()
            right = self.parse_assignment()
            
            if isinstance(expr, Identifier):
                return Assignment(expr.name, right)
        
        return expr
    
    def parse_logical_or(self) -> ASTNode:
        left = self.parse_logical_and()
        
        while self.match(TokenType.OR):
            op = self.advance().value
            right = self.parse_logical_and()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_logical_and(self) -> ASTNode:
        left = self.parse_equality()
        
        while self.match(TokenType.AND):
            op = self.advance().value
            right = self.parse_equality()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_equality(self) -> ASTNode:
        left = self.parse_comparison()
        
        while self.match(TokenType.EQ, TokenType.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_semantic()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self.advance().value
            right = self.parse_semantic()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_semantic(self) -> ASTNode:
        left = self.parse_addition()
        
        while self.match(TokenType.SEMANTIC_SIMILAR):
            op = self.advance().value
            right = self.parse_addition()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_addition(self) -> ASTNode:
        left = self.parse_multiplication()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplication()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_multiplication(self) -> ASTNode:
        left = self.parse_matrix_mul()
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_matrix_mul()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_matrix_mul(self) -> ASTNode:
        left = self.parse_unary()
        
        while self.match(TokenType.MATRIX_MUL):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.NOT, TokenType.MINUS):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> ASTNode:
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.DOT):
                self.advance()
                member = self.expect(TokenType.IDENTIFIER).value
                expr = MemberAccess(expr, member)
            elif self.match(TokenType.LBRACKET):
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(expr, index)
            elif self.match(TokenType.LPAREN):
                # Function call
                args = self.parse_function_call_args()
                if isinstance(expr, Identifier):
                    expr = FunctionCall(expr.name, args)
            else:
                break
        
        return expr
    
    def parse_function_call_args(self) -> List[ASTNode]:
        self.expect(TokenType.LPAREN)
        args = []
        
        if not self.match(TokenType.RPAREN):
            args.append(self.parse_expression())
            
            while self.match(TokenType.COMMA):
                self.advance()
                if self.match(TokenType.RPAREN):
                    break
                args.append(self.parse_expression())
        
        self.expect(TokenType.RPAREN)
        return args
    
    def parse_primary(self) -> ASTNode:
        # Literals
        if self.match(TokenType.INT):
            value = self.advance().value
            return IntLiteral(value)
        
        if self.match(TokenType.FLOAT):
            value = self.advance().value
            return FloatLiteral(value)
        
        if self.match(TokenType.STRING):
            value = self.advance().value
            return StringLiteral(value)
        
        if self.match(TokenType.BOOL):
            value = self.advance().value
            return BoolLiteral(value)
        
        # Identifiers
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            return Identifier(name)
        
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        # Array/Vector literal
        if self.match(TokenType.LBRACKET):
            self.advance()
            elements = []
            
            if not self.match(TokenType.RBRACKET):
                elements.append(self.parse_expression())
                
                while self.match(TokenType.COMMA):
                    self.advance()
                    if self.match(TokenType.RBRACKET):
                        break
                    elements.append(self.parse_expression())
            
            self.expect(TokenType.RBRACKET)
            return VectorLiteral(elements)
        
        # AI operations
        if self.match(TokenType.AI, TokenType.LLM, TokenType.EMBED):
            op = self.advance().value
            if self.match(TokenType.DOT):
                self.advance()
                method = self.expect(TokenType.IDENTIFIER).value
                args = self.parse_function_call_args() if self.match(TokenType.LPAREN) else []
                return AIOperation(f"{op}.{method}", {}, None)
        
        raise SyntaxError(f"Unexpected token: {self.current_token()}")