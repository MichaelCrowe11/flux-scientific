# FLUX Scientific Computing Language - Development Container
FROM python:3.11-slim

# Set working directory
WORKDIR /flux

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-dev \
    libnetcdf-dev \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY setup.py .
COPY README.md .
COPY . .

# Install FLUX with all development dependencies
RUN pip install --no-cache-dir -e ".[dev,mpi,viz]"

# Create examples and workspace directories
RUN mkdir -p /flux/workspace /flux/output

# Set environment variables
ENV PYTHONPATH="/flux:$PYTHONPATH"
ENV FLUX_HOME="/flux"

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command
CMD ["bash"]