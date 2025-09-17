"""
Tests for FLUX Scientific Computing Lexer
"""

import pytest
from src.pde_lexer import FluxPDELexer, TokenType


class TestFluxPDELexer:
    """Test PDE-specific lexer functionality"""

    def test_tokenize_pde_operators(self):
        """Test tokenization of PDE operators"""
        source = "∂u/∂t = α * ∇²u"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        # Check that partial derivative and laplacian are recognized
        token_types = [t.type for t in tokens]
        assert TokenType.PARTIAL_DERIV in token_types
        assert TokenType.IDENTIFIER in token_types

    def test_mesh_keywords(self):
        """Test mesh-related keywords"""
        source = "mesh M = StructuredGrid(nx=100, ny=100)"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        token_types = [t.type for t in tokens]
        assert TokenType.MESH in token_types
        assert TokenType.STRUCTURED_GRID in token_types

    def test_boundary_conditions(self):
        """Test boundary condition keywords"""
        source = "boundary { u = 0 on walls }"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        token_types = [t.type for t in tokens]
        assert TokenType.BOUNDARY in token_types

    def test_solver_types(self):
        """Test solver type keywords"""
        source = "solver = ImplicitEuler(dt=0.01)"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        token_types = [t.type for t in tokens]
        assert TokenType.SOLVER in token_types
        assert TokenType.IMPLICIT_EULER in token_types

    def test_scientific_numbers(self):
        """Test scientific notation"""
        source = "const α = 1.5e-3"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        # Find the number token
        numbers = [t for t in tokens if t.type == TokenType.NUMBER]
        assert len(numbers) > 0
        assert numbers[0].value == 1.5e-3

    def test_vector_field_notation(self):
        """Test vector field operations"""
        source = "∇·v = 0"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        token_types = [t.type for t in tokens]
        assert TokenType.DIVERGENCE in token_types or TokenType.NABLA in token_types

    def test_pde_definition(self):
        """Test PDE definition syntax"""
        source = """
        pde heat_equation {
            ∂u/∂t = α * ∇²u
        }
        """
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        token_types = [t.type for t in tokens]
        assert TokenType.PDE in token_types

    def test_domain_definition(self):
        """Test domain definition"""
        source = "domain Ω = Rectangle(0, 1, 0, 1)"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()

        token_types = [t.type for t in tokens]
        assert TokenType.DOMAIN in token_types


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_source(self):
        """Test empty source code"""
        lexer = FluxPDELexer("")
        tokens = lexer.tokenize()
        assert len(tokens) == 0 or (len(tokens) == 1 and tokens[0].type == TokenType.EOF)

    def test_unicode_operators(self):
        """Test various Unicode mathematical operators"""
        operators = ["∇", "∂", "×", "⊗", "∇²"]
        for op in operators:
            lexer = FluxPDELexer(op)
            tokens = lexer.tokenize()
            assert len(tokens) > 0

    def test_mixed_notation(self):
        """Test mixing standard and Unicode notation"""
        source = "grad(u) = ∇u"
        lexer = FluxPDELexer(source)
        tokens = lexer.tokenize()
        assert len(tokens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])