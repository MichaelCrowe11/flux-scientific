"""
FLUX Language Interpreter
Executes FLUX Abstract Syntax Trees with support for AI and quantum operations
"""

import random
import math
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from .parser import *

class Environment:
    """Environment for variable and function scoping"""
    def __init__(self, parent=None):
        self.parent = parent
        self.variables = {}
        self.functions = {}
        
    def get(self, name: str):
        if name in self.variables:
            return self.variables[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NameError(f"Variable '{name}' not defined")
    
    def set(self, name: str, value: Any):
        self.variables[name] = value
    
    def define_function(self, name: str, func):
        self.functions[name] = func
    
    def get_function(self, name: str):
        if name in self.functions:
            return self.functions[name]
        elif self.parent:
            return self.parent.get_function(name)
        else:
            raise NameError(f"Function '{name}' not defined")

@dataclass
class FluxVector:
    """Vector type for FLUX with semantic operations"""
    data: np.ndarray
    
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data, dtype=float)
        else:
            self.data = data
    
    def similarity(self, other: 'FluxVector') -> float:
        """Cosine similarity for semantic comparison"""
        dot = np.dot(self.data, other.data)
        norm1 = np.linalg.norm(self.data)
        norm2 = np.linalg.norm(other.data)
        return dot / (norm1 * norm2) if norm1 * norm2 != 0 else 0

@dataclass
class FluxTensor:
    """Tensor type for FLUX"""
    data: np.ndarray
    shape: tuple
    
    def __init__(self, data, shape=None):
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data
        self.shape = shape or self.data.shape

@dataclass
class FluxProbability:
    """Probabilistic value with confidence"""
    value: Any
    confidence: float
    
    def __init__(self, value, confidence=1.0):
        self.value = value
        self.confidence = max(0.0, min(1.0, confidence))

@dataclass 
class FluxQubit:
    """Quantum bit representation"""
    alpha: complex
    beta: complex
    
    def __init__(self, alpha=1.0, beta=0.0):
        # Normalize the state
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm if norm != 0 else 1.0
        self.beta = beta / norm if norm != 0 else 0.0
    
    def measure(self) -> int:
        """Measure the qubit, collapsing to 0 or 1"""
        prob_zero = abs(self.alpha) ** 2
        return 0 if random.random() < prob_zero else 1

class FLUXInterpreter:
    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self.setup_builtins()
        
    def setup_builtins(self):
        """Setup built-in functions"""
        # Math functions
        self.global_env.define_function('sin', lambda x: math.sin(x))
        self.global_env.define_function('cos', lambda x: math.cos(x))
        self.global_env.define_function('sqrt', lambda x: math.sqrt(x))
        self.global_env.define_function('abs', lambda x: abs(x))
        
        # IO functions
        self.global_env.define_function('print', lambda *args: print(*args))
        self.global_env.define_function('input', lambda prompt="": input(prompt))
        
        # Type conversion
        self.global_env.define_function('int', lambda x: int(x))
        self.global_env.define_function('float', lambda x: float(x))
        self.global_env.define_function('str', lambda x: str(x))
        
        # Vector/Tensor operations
        self.global_env.define_function('vector', lambda *args: FluxVector(list(args)))
        self.global_env.define_function('tensor', lambda data: FluxTensor(data))
        self.global_env.define_function('zeros', lambda *shape: FluxTensor(np.zeros(shape)))
        self.global_env.define_function('ones', lambda *shape: FluxTensor(np.ones(shape)))
        self.global_env.define_function('random', lambda *shape: FluxTensor(np.random.random(shape)))
        
        # AI operations (mock implementations)
        self.global_env.define_function('embed', self.mock_embed)
        self.global_env.define_function('generate', self.mock_generate)
        
        # Quantum operations
        self.global_env.define_function('qubit', lambda: FluxQubit())
        self.global_env.define_function('hadamard', self.hadamard_gate)
        
    def mock_embed(self, text: str) -> FluxVector:
        """Mock text embedding function"""
        # Simple hash-based embedding for demonstration
        hash_val = hash(text)
        random.seed(hash_val)
        embedding = [random.random() for _ in range(768)]  # 768-dim like BERT
        return FluxVector(embedding)
    
    def mock_generate(self, prompt: str) -> str:
        """Mock text generation function"""
        return f"[Generated response for: {prompt}]"
    
    def hadamard_gate(self, qubit: FluxQubit) -> FluxQubit:
        """Apply Hadamard gate to a qubit"""
        h_factor = 1 / math.sqrt(2)
        new_alpha = h_factor * (qubit.alpha + qubit.beta)
        new_beta = h_factor * (qubit.alpha - qubit.beta)
        return FluxQubit(new_alpha, new_beta)
    
    def interpret(self, ast: ASTNode) -> Any:
        """Main interpreter entry point"""
        if isinstance(ast, Program):
            return self.interpret_program(ast)
        else:
            return self.evaluate(ast)
    
    def interpret_program(self, program: Program) -> Any:
        result = None
        for statement in program.statements:
            result = self.evaluate(statement)
        return result
    
    def evaluate(self, node: ASTNode) -> Any:
        """Evaluate an AST node"""
        if node is None:
            return None
            
        # Literals
        if isinstance(node, IntLiteral):
            return node.value
        elif isinstance(node, FloatLiteral):
            return node.value
        elif isinstance(node, StringLiteral):
            return node.value
        elif isinstance(node, BoolLiteral):
            return node.value
        elif isinstance(node, VectorLiteral):
            elements = [self.evaluate(e) for e in node.elements]
            return FluxVector(elements)
        elif isinstance(node, TensorLiteral):
            data = [self.evaluate(e) for e in node.data]
            return FluxTensor(data, node.shape)
        
        # Variables and identifiers
        elif isinstance(node, Identifier):
            return self.current_env.get(node.name)
        
        # Variable declaration
        elif isinstance(node, VariableDecl):
            value = self.evaluate(node.value) if node.value else None
            self.current_env.set(node.name, value)
            return value
        
        # Assignment
        elif isinstance(node, Assignment):
            value = self.evaluate(node.value)
            self.current_env.set(node.target, value)
            return value
        
        # Binary operations
        elif isinstance(node, BinaryOp):
            return self.evaluate_binary_op(node)
        
        # Unary operations
        elif isinstance(node, UnaryOp):
            return self.evaluate_unary_op(node)
        
        # Function declaration
        elif isinstance(node, FunctionDecl):
            self.current_env.define_function(node.name, node)
            return None
        
        # Function call
        elif isinstance(node, FunctionCall):
            return self.evaluate_function_call(node)
        
        # Control flow
        elif isinstance(node, IfStatement):
            return self.evaluate_if_statement(node)
        elif isinstance(node, ForLoop):
            return self.evaluate_for_loop(node)
        elif isinstance(node, WhileLoop):
            return self.evaluate_while_loop(node)
        elif isinstance(node, ReturnStatement):
            return self.evaluate(node.value) if node.value else None
        elif isinstance(node, MatchStatement):
            return self.evaluate_match_statement(node)
        
        # Member access
        elif isinstance(node, MemberAccess):
            obj = self.evaluate(node.object)
            if hasattr(obj, node.member):
                return getattr(obj, node.member)
            elif isinstance(obj, dict):
                return obj.get(node.member)
            else:
                raise AttributeError(f"Object has no attribute '{node.member}'")
        
        # Index access
        elif isinstance(node, IndexAccess):
            obj = self.evaluate(node.object)
            index = self.evaluate(node.index)
            if isinstance(obj, (list, FluxVector, FluxTensor)):
                if isinstance(obj, FluxVector):
                    return obj.data[index]
                elif isinstance(obj, FluxTensor):
                    return obj.data[index]
                else:
                    return obj[index]
            else:
                raise TypeError(f"Object of type {type(obj)} is not subscriptable")
        
        # AI operations
        elif isinstance(node, AIOperation):
            return self.evaluate_ai_operation(node)
        
        # Quantum operations
        elif isinstance(node, QuantumCircuit):
            return self.evaluate_quantum_circuit(node)
        
        else:
            raise NotImplementedError(f"Evaluation not implemented for {type(node)}")
    
    def evaluate_binary_op(self, node: BinaryOp) -> Any:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        
        # Arithmetic operations
        if node.op == '+':
            return left + right
        elif node.op == '-':
            return left - right
        elif node.op == '*':
            return left * right
        elif node.op == '/':
            return left / right
        elif node.op == '%':
            return left % right
        
        # Comparison operations
        elif node.op == '==':
            return left == right
        elif node.op == '!=':
            return left != right
        elif node.op == '<':
            return left < right
        elif node.op == '>':
            return left > right
        elif node.op == '<=':
            return left <= right
        elif node.op == '>=':
            return left >= right
        
        # Logical operations
        elif node.op == '&&':
            return left and right
        elif node.op == '||':
            return left or right
        
        # Semantic similarity
        elif node.op == '~~':
            if isinstance(left, FluxVector) and isinstance(right, FluxVector):
                return left.similarity(right)
            else:
                raise TypeError("Semantic similarity requires vectors")
        
        # Matrix multiplication
        elif node.op == '@':
            if isinstance(left, FluxTensor) and isinstance(right, FluxTensor):
                return FluxTensor(np.matmul(left.data, right.data))
            elif isinstance(left, FluxVector) and isinstance(right, FluxVector):
                return np.dot(left.data, right.data)
            else:
                raise TypeError("Matrix multiplication requires tensors or vectors")
        
        else:
            raise NotImplementedError(f"Binary operation '{node.op}' not implemented")
    
    def evaluate_unary_op(self, node: UnaryOp) -> Any:
        operand = self.evaluate(node.operand)
        
        if node.op == '-':
            return -operand
        elif node.op == '!':
            return not operand
        else:
            raise NotImplementedError(f"Unary operation '{node.op}' not implemented")
    
    def evaluate_function_call(self, node: FunctionCall) -> Any:
        # Check for built-in functions
        try:
            func = self.current_env.get_function(node.name)
        except NameError:
            # Try as a regular variable (for first-class functions)
            func = self.current_env.get(node.name)
        
        args = [self.evaluate(arg) for arg in node.args]
        
        if callable(func):
            return func(*args)
        elif isinstance(func, FunctionDecl):
            # User-defined function
            return self.execute_function(func, args)
        else:
            raise TypeError(f"'{node.name}' is not callable")
    
    def execute_function(self, func: FunctionDecl, args: List[Any]) -> Any:
        # Create new environment for function
        func_env = Environment(self.current_env)
        
        # Bind parameters
        for i, param in enumerate(func.params):
            if i < len(args):
                func_env.set(param.name, args[i])
            elif param.default:
                func_env.set(param.name, self.evaluate(param.default))
            else:
                raise TypeError(f"Missing argument for parameter '{param.name}'")
        
        # Execute function body
        prev_env = self.current_env
        self.current_env = func_env
        
        result = None
        for statement in func.body:
            result = self.evaluate(statement)
            if isinstance(statement, ReturnStatement):
                break
        
        self.current_env = prev_env
        return result
    
    def evaluate_if_statement(self, node: IfStatement) -> Any:
        condition = self.evaluate(node.condition)
        
        if condition:
            for stmt in node.then_branch:
                result = self.evaluate(stmt)
                if isinstance(stmt, ReturnStatement):
                    return result
        elif node.else_branch:
            for stmt in node.else_branch:
                result = self.evaluate(stmt)
                if isinstance(stmt, ReturnStatement):
                    return result
        
        return None
    
    def evaluate_for_loop(self, node: ForLoop) -> Any:
        iterable = self.evaluate(node.iterable)
        
        if isinstance(iterable, FluxVector):
            iterable = iterable.data
        elif isinstance(iterable, FluxTensor):
            iterable = iterable.data.flatten()
        
        for item in iterable:
            self.current_env.set(node.variable, item)
            for stmt in node.body:
                result = self.evaluate(stmt)
                if isinstance(stmt, ReturnStatement):
                    return result
        
        return None
    
    def evaluate_while_loop(self, node: WhileLoop) -> Any:
        while self.evaluate(node.condition):
            for stmt in node.body:
                result = self.evaluate(stmt)
                if isinstance(stmt, ReturnStatement):
                    return result
        
        return None
    
    def evaluate_match_statement(self, node: MatchStatement) -> Any:
        expr_value = self.evaluate(node.expr)
        
        for case in node.cases:
            pattern_value = self.evaluate(case.pattern)
            
            if expr_value == pattern_value:
                if case.guard:
                    if not self.evaluate(case.guard):
                        continue
                
                for stmt in case.body:
                    result = self.evaluate(stmt)
                    if isinstance(stmt, ReturnStatement):
                        return result
                break
        
        return None
    
    def evaluate_ai_operation(self, node: AIOperation) -> Any:
        if node.operation == 'ai.generate':
            # Mock AI text generation
            return self.mock_generate("Sample prompt")
        elif node.operation == 'embed':
            # Mock embedding
            return self.mock_embed("Sample text")
        else:
            raise NotImplementedError(f"AI operation '{node.operation}' not implemented")
    
    def evaluate_quantum_circuit(self, node: QuantumCircuit) -> Any:
        # Simplified quantum circuit evaluation
        result = {}
        
        # Execute classical preprocessing
        if node.classical_pre:
            for stmt in node.classical_pre:
                self.evaluate(stmt)
        
        # Execute quantum operations (simplified)
        qubits = []
        for op in node.quantum_ops:
            if isinstance(op, FunctionCall):
                if op.name == 'allocate':
                    # Allocate qubits
                    n = self.evaluate(op.args[0]) if op.args else 1
                    qubits = [FluxQubit() for _ in range(n)]
                elif op.name == 'hadamard' and qubits:
                    # Apply Hadamard to all qubits
                    qubits = [self.hadamard_gate(q) for q in qubits]
                elif op.name == 'measure' and qubits:
                    # Measure all qubits
                    result['measurements'] = [q.measure() for q in qubits]
        
        # Execute classical postprocessing
        if node.classical_post:
            for stmt in node.classical_post:
                self.evaluate(stmt)
        
        return result