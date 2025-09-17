# 📦 Publishing FLUX Scientific to PyPI

## Prerequisites

1. **Install build tools**:
```bash
pip install --upgrade pip
pip install --upgrade build twine
```

2. **Configure PyPI token** (if not already done):
Create `~/.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = YOUR_PYPI_TOKEN

[testpypi]
username = __token__
password = YOUR_TEST_PYPI_TOKEN
```

## Step 1: Push to GitHub

```bash
# Push the production-ready code
git push origin main
```

## Step 2: Build Distribution Packages

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build source and wheel distributions
python -m build

# Verify the build
twine check dist/*
```

## Step 3: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ flux-sci-lang
```

## Step 4: Publish to PyPI

```bash
# If using saved token
twine upload dist/*

# Or with token directly (replace YOUR_TOKEN)
twine upload -u __token__ -p YOUR_TOKEN dist/*
```

## Step 5: Verify Installation

```bash
# Install from PyPI
pip install flux-sci-lang

# Test the installation
python -c "import flux_scientific; print(flux_scientific.__version__)"
```

## Step 6: Test the Package

```python
# Test basic functionality
from flux_scientific.solvers import FiniteDifferenceSolver
import numpy as np

solver = FiniteDifferenceSolver()
result = solver.solve_heat_equation(
    domain=((0, 1),),
    grid_points=(101,),
    initial_condition=np.sin(np.pi * np.linspace(0, 1, 101)),
    boundary_conditions={'left': 0, 'right': 0},
    thermal_diffusivity=1.0,
    time_final=0.1
)
print(f"Solver working! Max temperature: {np.max(result['solution']):.4f}")
```

## Package Details

- **Package Name**: `flux-sci-lang`
- **Version**: 0.1.0 (as per updated pyproject.toml)
- **Python Versions**: 3.8+
- **Dependencies**: numpy, scipy, matplotlib

## What's Included

✅ Complete finite difference PDE solver
✅ Validation suite with analytical solutions
✅ Enhanced Python code generator
✅ FLUX language lexer and parser
✅ Heat equation solver
✅ Comprehensive examples
✅ Full documentation

## GitHub Release

After publishing to PyPI, create a GitHub release:

```bash
# Tag the release
git tag -a v0.1.0 -m "Production-ready FLUX with validated PDE solvers"
git push origin v0.1.0
```

Then create a release on GitHub with the changelog:
- Visit: https://github.com/MichaelCrowe11/flux-sci-lang/releases
- Click "Create a new release"
- Select the tag v0.1.0
- Add release notes from the commit message

## Troubleshooting

If you encounter issues:

1. **Name conflict**: The package name might already exist. Try `flux-scientific` or `fluxlang`
2. **Authentication error**: Ensure your PyPI token is correctly configured
3. **Build errors**: Make sure all dependencies are installed: `pip install build twine`
4. **Import errors**: Verify the package structure with: `python -m build --sdist`

## Next Steps

After successful publication:

1. **Documentation**: Consider setting up ReadTheDocs
2. **CI/CD**: Add GitHub Actions for automated testing and publishing
3. **Badge**: Add PyPI badge to README:
   ```markdown
   [![PyPI version](https://badge.fury.io/py/flux-sci-lang.svg)](https://badge.fury.io/py/flux-sci-lang)
   ```

## Support

For issues or questions:
- GitHub Issues: https://github.com/MichaelCrowe11/flux-sci-lang/issues
- Email: michael@flux-sci.org