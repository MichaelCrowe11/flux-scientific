"""
FLUX Scientific Computing Parser
Parses PDE definitions and scientific computing constructs
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union
from .pde_lexer import Token, TokenType, FluxPDELexer

# AST Node Classes for Scientific Computing
@dataclass
class ASTNode:
    pass

@dataclass
class PDEDefinition(ASTNode):
    name: str
    equations: List['Equation']
    boundary_conditions: List['BoundaryCondition']
    initial_conditions: List['InitialCondition']
    variables: Optional[List[str]] = None

@dataclass
class Equation(ASTNode):
    lhs: 'Expression'
    rhs: 'Expression'
    domain: Optional[str] = None

@dataclass
class BoundaryCondition(ASTNode):
    expression: 'Expression'
    location: str
    type: str = 'dirichlet'  # dirichlet, neumann, robin

@dataclass
class InitialCondition(ASTNode):
    variable: str
    expression: 'Expression'
    time: float = 0.0

@dataclass
class Expression(ASTNode):
    pass

@dataclass
class BinaryOp(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryOp(Expression):
    operator: str
    operand: Expression

@dataclass
class PartialDerivative(Expression):
    variable: str
    with_respect_to: str
    order: int = 1

@dataclass
class Gradient(Expression):
    field: Expression

@dataclass
class Divergence(Expression):
    field: Expression

@dataclass
class Laplacian(Expression):
    field: Expression

@dataclass
class Curl(Expression):
    field: Expression

@dataclass
class Variable(Expression):
    name: str
    indices: Optional[List[Expression]] = None

@dataclass
class Number(Expression):
    value: Union[int, float]

@dataclass
class FunctionCall(Expression):
    name: str
    args: List[Expression]

@dataclass
class Domain(ASTNode):
    name: str
    shape: str
    parameters: Dict[str, Any]

@dataclass
class Mesh(ASTNode):
    name: str
    type: str
    domain: str
    parameters: Dict[str, Any]

@dataclass
class Solver(ASTNode):
    name: str
    type: str
    parameters: Dict[str, Any]

@dataclass
class KernelDefinition(ASTNode):
    name: str
    backend: str
    parameters: List['Parameter']
    body: List[ASTNode]
    decorators: List[str] = None

@dataclass
class Parameter(ASTNode):
    name: str
    type: str
    default: Optional[Any] = None

class FluxPDEParser:
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
    
    def parse(self) -> List[ASTNode]:
        """Parse the entire program"""
        nodes = []
        while self.current_token() and self.current_token().type != TokenType.EOF:
            node = self.parse_top_level()
            if node:
                nodes.append(node)
        return nodes
    
    def parse_top_level(self) -> Optional[ASTNode]:
        """Parse top-level constructs"""
        # Skip newlines
        while self.match(TokenType.NEWLINE):
            self.advance()
        
        if not self.current_token() or self.current_token().type == TokenType.EOF:
            return None
        
        # PDE definition
        if self.match(TokenType.PDE):
            return self.parse_pde_definition()
        
        # Domain definition
        if self.match(TokenType.DOMAIN):
            return self.parse_domain()
        
        # Mesh definition
        if self.match(TokenType.MESH):
            return self.parse_mesh()
        
        # Solver definition
        if self.match(TokenType.SOLVER):
            return self.parse_solver()
        
        # GPU kernel
        if self.match(TokenType.AT):
            return self.parse_kernel()
        
        # Default: try to parse as expression
        return self.parse_expression()
    
    def parse_pde_definition(self) -> PDEDefinition:
        """Parse PDE definition block"""
        self.expect(TokenType.PDE)
        name = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.LBRACE)
        
        equations = []
        boundary_conditions = []
        initial_conditions = []
        variables = None
        
        while not self.match(TokenType.RBRACE):
            # Variables declaration
            if self.match(TokenType.VARIABLES):
                self.advance()
                self.expect(TokenType.COLON)
                variables = self.parse_variable_list()
            
            # Boundary conditions
            elif self.match(TokenType.BOUNDARY):
                self.advance()
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    bc = self.parse_boundary_condition()
                    boundary_conditions.append(bc)
                self.expect(TokenType.RBRACE)
            
            # Initial conditions
            elif self.match(TokenType.INITIAL):
                self.advance()
                self.expect(TokenType.COLON)
                ic = self.parse_initial_condition()
                initial_conditions.append(ic)
            
            # Equations
            else:
                eq = self.parse_equation()
                if eq:
                    equations.append(eq)
        
        self.expect(TokenType.RBRACE)
        
        return PDEDefinition(name, equations, boundary_conditions, initial_conditions, variables)
    
    def parse_equation(self) -> Optional[Equation]:
        """Parse a PDE equation"""
        # Check for domain specifier (e.g., "in domain:")
        domain = None
        if self.match(TokenType.IN):
            self.advance()
            domain = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
        
        # Parse LHS
        lhs = self.parse_expression()
        
        # Expect equals sign
        if not self.match(TokenType.ASSIGN):
            return None
        self.advance()
        
        # Parse RHS
        rhs = self.parse_expression()
        
        return Equation(lhs, rhs, domain)
    
    def parse_boundary_condition(self) -> BoundaryCondition:
        """Parse boundary condition"""
        expr = self.parse_expression()
        
        # Expect "on" keyword
        self.expect(TokenType.ON)
        location = self.expect(TokenType.IDENTIFIER).value
        
        # Determine BC type (simplified)
        bc_type = 'dirichlet'  # Default
        if isinstance(expr, BinaryOp) and expr.operator == '∂':
            bc_type = 'neumann'
        
        return BoundaryCondition(expr, location, bc_type)
    
    def parse_initial_condition(self) -> InitialCondition:
        """Parse initial condition"""
        var = self.expect(TokenType.IDENTIFIER).value
        
        # Handle function notation like u(x,y,0)
        if self.match(TokenType.LPAREN):
            self.advance()
            # Skip spatial variables
            while not self.match(TokenType.RPAREN):
                self.advance()
            self.expect(TokenType.RPAREN)
        
        self.expect(TokenType.ASSIGN)
        expr = self.parse_expression()
        
        return InitialCondition(var, expr)
    
    def parse_variable_list(self) -> List[str]:
        """Parse comma-separated variable list"""
        variables = []
        
        # Parse first variable
        var = self.expect(TokenType.IDENTIFIER).value
        variables.append(var)
        
        # Parse additional variables
        while self.match(TokenType.COMMA):
            self.advance()
            var = self.expect(TokenType.IDENTIFIER).value
            variables.append(var)
        
        return variables
    
    def parse_domain(self) -> Domain:
        """Parse domain definition"""
        self.expect(TokenType.DOMAIN)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        
        # Parse domain constructor (e.g., Rectangle(0, 1, 0, 1))
        shape = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LPAREN)
        
        parameters = {}
        param_count = 0
        while not self.match(TokenType.RPAREN):
            if self.match(TokenType.IDENTIFIER):
                # Named parameter
                param_name = self.advance().value
                self.expect(TokenType.ASSIGN)
                param_value = self.parse_primary()
                parameters[param_name] = param_value
            else:
                # Positional parameter
                param_value = self.parse_primary()
                parameters[f'arg{param_count}'] = param_value
                param_count += 1
            
            if self.match(TokenType.COMMA):
                self.advance()
        
        self.expect(TokenType.RPAREN)
        
        return Domain(name, shape, parameters)
    
    def parse_mesh(self) -> Mesh:
        """Parse mesh definition"""
        self.expect(TokenType.MESH)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        
        # Parse mesh type
        mesh_type = None
        if self.match(TokenType.STRUCTURED_GRID):
            mesh_type = 'StructuredGrid'
            self.advance()
        elif self.match(TokenType.UNSTRUCTURED_MESH):
            mesh_type = 'UnstructuredMesh'
            self.advance()
        elif self.match(TokenType.ADAPTIVE_MESH):
            mesh_type = 'AdaptiveMesh'
            self.advance()
        else:
            mesh_type = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.LPAREN)
        
        # First parameter is usually the domain
        domain = self.expect(TokenType.IDENTIFIER).value
        
        parameters = {'domain': domain}
        
        # Parse additional parameters
        while self.match(TokenType.COMMA):
            self.advance()
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.ASSIGN)
            param_value = self.parse_primary()
            parameters[param_name] = param_value
        
        self.expect(TokenType.RPAREN)
        
        return Mesh(name, mesh_type, domain, parameters)
    
    def parse_solver(self) -> Solver:
        """Parse solver definition"""
        self.expect(TokenType.SOLVER)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        
        # Parse solver type
        solver_type = None
        if self.match(TokenType.IMPLICIT_EULER):
            solver_type = 'ImplicitEuler'
            self.advance()
        elif self.match(TokenType.EXPLICIT_EULER):
            solver_type = 'ExplicitEuler'
            self.advance()
        elif self.match(TokenType.RUNGE_KUTTA):
            solver_type = 'RungeKutta'
            self.advance()
        else:
            solver_type = self.expect(TokenType.IDENTIFIER).value
        
        parameters = {}
        
        # Parse parameters if present
        if self.match(TokenType.LPAREN):
            self.advance()
            while not self.match(TokenType.RPAREN):
                param_name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.ASSIGN)
                param_value = self.parse_primary()
                parameters[param_name] = param_value
                
                if self.match(TokenType.COMMA):
                    self.advance()
            self.expect(TokenType.RPAREN)
        
        return Solver(name, solver_type, parameters)
    
    def parse_kernel(self) -> KernelDefinition:
        """Parse GPU kernel definition"""
        decorators = []
        
        # Parse decorators
        while self.match(TokenType.AT):
            self.advance()
            decorator = self.expect(TokenType.IDENTIFIER).value
            decorators.append(decorator)
            
            # Parse decorator arguments
            if self.match(TokenType.LPAREN):
                self.advance()
                # Skip arguments for now
                while not self.match(TokenType.RPAREN):
                    self.advance()
                self.expect(TokenType.RPAREN)
        
        # Expect kernel keyword or function
        if not self.match(TokenType.IDENTIFIER):
            raise SyntaxError("Expected kernel or function definition")
        
        keyword = self.advance().value
        name = self.expect(TokenType.IDENTIFIER).value
        
        # Parse parameters
        self.expect(TokenType.LPAREN)
        parameters = []
        while not self.match(TokenType.RPAREN):
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            param_type = self.expect(TokenType.IDENTIFIER).value
            parameters.append(Parameter(param_name, param_type))
            
            if self.match(TokenType.COMMA):
                self.advance()
        self.expect(TokenType.RPAREN)
        
        # Parse body
        self.expect(TokenType.LBRACE)
        body = []
        while not self.match(TokenType.RBRACE):
            stmt = self.parse_expression()
            if stmt:
                body.append(stmt)
        self.expect(TokenType.RBRACE)
        
        return KernelDefinition(name, 'gpu', parameters, body, decorators)
    
    def parse_expression(self) -> Expression:
        """Parse mathematical expression"""
        return self.parse_additive()
    
    def parse_additive(self) -> Expression:
        """Parse addition and subtraction"""
        left = self.parse_multiplicative()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_multiplicative(self) -> Expression:
        """Parse multiplication and division"""
        left = self.parse_power()
        
        while self.match(TokenType.STAR, TokenType.SLASH):
            op = self.advance().value
            right = self.parse_power()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_power(self) -> Expression:
        """Parse exponentiation"""
        left = self.parse_unary()
        
        if self.match(TokenType.POWER):
            op = self.advance().value
            right = self.parse_power()  # Right associative
            return BinaryOp(left, op, right)
        
        return left
    
    def parse_unary(self) -> Expression:
        """Parse unary operations"""
        # Gradient operator ∇
        if self.match(TokenType.NABLA):
            self.advance()
            
            # Check for Laplacian ∇²
            if self.current_token() and self.current_token().value == '²':
                self.advance()
                operand = self.parse_postfix()
                return Laplacian(operand)
            else:
                operand = self.parse_postfix()
                return Gradient(operand)
        
        # Partial derivative ∂
        if self.match(TokenType.PARTIAL_DERIV):
            return self.parse_partial_derivative()
        
        # Divergence
        if self.match(TokenType.DIVERGENCE):
            self.advance()
            if self.match(TokenType.LPAREN):
                self.advance()
                field = self.parse_expression()
                self.expect(TokenType.RPAREN)
                return Divergence(field)
            else:
                field = self.parse_postfix()
                return Divergence(field)
        
        # Curl
        if self.match(TokenType.CURL):
            self.advance()
            if self.match(TokenType.LPAREN):
                self.advance()
                field = self.parse_expression()
                self.expect(TokenType.RPAREN)
                return Curl(field)
            else:
                field = self.parse_postfix()
                return Curl(field)
        
        # Laplacian
        if self.match(TokenType.LAPLACIAN):
            self.advance()
            operand = self.parse_postfix()
            return Laplacian(operand)
        
        # Unary minus
        if self.match(TokenType.MINUS):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        
        return self.parse_postfix()
    
    def parse_partial_derivative(self) -> PartialDerivative:
        """Parse partial derivative notation ∂u/∂x"""
        self.expect(TokenType.PARTIAL_DERIV)
        
        # Parse variable being differentiated
        variable = self.expect(TokenType.IDENTIFIER).value
        
        # Expect division symbol
        self.expect(TokenType.SLASH)
        
        # Expect another ∂
        self.expect(TokenType.PARTIAL_DERIV)
        
        # Parse variable with respect to
        wrt = self.expect(TokenType.IDENTIFIER).value
        
        # Check for higher order derivatives (e.g., ∂²u/∂x²)
        order = 1
        if self.current_token() and self.current_token().value in ['²', '2']:
            self.advance()
            order = 2
        
        return PartialDerivative(variable, wrt, order)
    
    def parse_postfix(self) -> Expression:
        """Parse postfix operations"""
        expr = self.parse_primary()
        
        while True:
            # Function call or array indexing
            if self.match(TokenType.LPAREN):
                self.advance()
                args = []
                while not self.match(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    if self.match(TokenType.COMMA):
                        self.advance()
                self.expect(TokenType.RPAREN)
                
                if isinstance(expr, Variable):
                    expr = FunctionCall(expr.name, args)
                else:
                    # Array indexing
                    expr = Variable(str(expr), args)
            
            # Array/tensor indexing
            elif self.match(TokenType.LBRACKET):
                self.advance()
                indices = []
                indices.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    self.advance()
                    indices.append(self.parse_expression())
                self.expect(TokenType.RBRACKET)
                
                if isinstance(expr, Variable):
                    expr.indices = indices
                else:
                    expr = Variable(str(expr), indices)
            
            # Dot product
            elif self.match(TokenType.DOT_PRODUCT):
                op = self.advance().value
                right = self.parse_unary()
                expr = BinaryOp(expr, op, right)
            
            # Cross product
            elif self.match(TokenType.CROSS_PRODUCT):
                op = self.advance().value
                right = self.parse_unary()
                expr = BinaryOp(expr, op, right)
            
            # Tensor product
            elif self.match(TokenType.TENSOR_PRODUCT):
                op = self.advance().value
                right = self.parse_unary()
                expr = BinaryOp(expr, op, right)
            
            else:
                break
        
        return expr
    
    def parse_primary(self) -> Expression:
        """Parse primary expressions"""
        # Numbers
        if self.match(TokenType.NUMBER):
            value = self.advance().value
            return Number(value)
        
        # Variables and functions
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            return Variable(name)
        
        # Mathematical functions
        if self.match(TokenType.GRADIENT, TokenType.DIVERGENCE, 
                     TokenType.CURL, TokenType.LAPLACIAN):
            func_type = self.current_token().type
            self.advance()
            
            # Parse argument
            if self.match(TokenType.LPAREN):
                self.advance()
                arg = self.parse_expression()
                self.expect(TokenType.RPAREN)
            else:
                arg = self.parse_primary()
            
            if func_type == TokenType.GRADIENT:
                return Gradient(arg)
            elif func_type == TokenType.DIVERGENCE:
                return Divergence(arg)
            elif func_type == TokenType.CURL:
                return Curl(arg)
            elif func_type == TokenType.LAPLACIAN:
                return Laplacian(arg)
        
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        raise SyntaxError(f"Unexpected token: {self.current_token()}")