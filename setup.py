"""
Setup script for FLUX Scientific Computing Language
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flux-scientific",
    version="0.1.0",
    author="FLUX Development Team",
    author_email="dev@flux-lang.io",
    description="Domain-Specific Language for PDEs, CFD, and Computational Physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flux-lang/flux-scientific",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "gpu": ["cupy>=10.0.0"],
        "mpi": ["mpi4py>=3.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "vtk>=9.0",
            "mayavi>=4.7",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flux=flux_scientific:main",
            "flux-sci=flux_scientific:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["examples/*.flux", "templates/*.flux"],
    },
)