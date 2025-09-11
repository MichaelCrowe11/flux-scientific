# Contributing to FLUX

Thank you for your interest in contributing to FLUX! We welcome contributions from the scientific computing community.

## How to Contribute

### Reporting Issues
- Check if the issue already exists
- Provide a minimal reproducible example
- Include your system information (OS, Python version, CUDA version if applicable)

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Run existing tests
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to all functions and classes
- Keep functions focused and small

### Areas Where We Need Help

#### Phase 0 (Current Focus)
- **Solver implementations**: FEM, FVM, spectral methods
- **Mesh generators**: Improving unstructured mesh generation
- **Examples**: CFD, electromagnetics, structural mechanics
- **Documentation**: Tutorials, API docs, theory guides

#### Future Phases
- **GPU optimization**: CUDA kernel optimization
- **MPI support**: Distributed computing
- **Verification suite**: Method of Manufactured Solutions
- **GUI/Visualization**: Post-processing tools

### Testing
```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_mesh.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Documentation
- Code should be self-documenting
- Add examples for new features
- Update README.md if adding major features

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/flux-lang.git
cd flux-lang

# Install in development mode
pip install -e .
pip install pytest pytest-cov

# Install optional dependencies
pip install cupy  # For GPU development
pip install mpi4py  # For MPI development
```

## Community

- Discord: [Join our server](https://discord.gg/flux-sci)
- Discussions: Use GitHub Discussions for questions
- Academic partnerships: Contact partnerships@flux-lang.io

## License

By contributing, you agree that your contributions will be licensed under the MIT License for the open-source components and that commercial features may be subject to the commercial license.