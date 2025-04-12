# TileLang ROCm Installation Guide

This guide details how to install and configure TileLang with ROCm support for AMD GPUs.

## Prerequisites

- Ubuntu 22.04 or compatible Linux distribution
- ROCm 6.3+ installed (check with `rocminfo`)
- Python 3.10+
- Git

## Step 1: Install System Dependencies

First, install the necessary system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libstdc++-12-dev cmake
```

## Step 2: Set Up a Virtual Environment (Recommended)

Create and activate a Python virtual environment:

```bash
python3 -m venv ~/myenv
source ~/myenv/bin/activate
```

## Step 3: Clone the Repository

```bash
git clone https://github.com/your-org/tilelang.git
cd tilelang
```

## Step 4: Install Python Dependencies

Update the requirements files to ensure ROCm compatibility:

### requirements.txt
```
# runtime requirements
Cython>=3.0.0
decorator
numpy>=1.23.5
tqdm>=4.62.3
typing_extensions>=4.10.0
attrs
cloudpickle
ml_dtypes
psutil
# For ROCm support, install PyTorch with ROCm bindings
```

### requirements-build.txt
```
# Should be mirrored in pyproject.toml
build
cmake>=3.26
packaging
setuptools>=61
wheel
tox
auditwheel
patchelf
# PyTorch will be installed separately with ROCm support
```

## Step 5: Install PyTorch with ROCm Support

```bash
pip uninstall -y torch torchvision torchaudio  # Remove any existing PyTorch installation
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Note: Use the ROCm version that matches your installation (e.g., `rocm6.3` for ROCm 6.3.x).

## Step 6: Install Other Python Requirements

```bash
pip install -r requirements-build.txt
pip install -r requirements.txt
```

## Step 7: Run the Installation Script

Use the updated `install_rocm.sh` script:

```bash
chmod +x install_rocm.sh
./install_rocm.sh
```

### Updated install_rocm.sh
```bash
#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

echo "Starting installation script..."

# Check for ROCm installation
if [ ! -d "/opt/rocm" ]; then
    echo "Error: ROCm installation not found at /opt/rocm"
    echo "Please install ROCm before continuing."
    exit 1
fi

# install requirements
pip install -r requirements-build.txt
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python requirements."
    exit 1
else
    echo "Python requirements installed successfully."
fi

# determine if root
USER_IS_ROOT=false
if [ "$EUID" -eq 0 ]; then
    USER_IS_ROOT=true
fi

if $USER_IS_ROOT; then
    # Fetch the GPG key for the LLVM repository and add it to the trusted keys
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

    # Check if the repository is already present in the sources.list
    if ! grep -q "http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" /etc/apt/sources.list; then
        # Add the LLVM repository to sources.list
        echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
        echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
    else
        # Print a message if the repository is already added
        echo "The repository is already added."
    fi

    # Update package lists and install llvm-16
    apt-get update
    apt-get install -y llvm-16 build-essential libstdc++-12-dev
else
    # Fetch the GPG key for the LLVM repository and add it to the trusted keys using sudo
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

    # Check if the repository is already present in the sources.list
    if ! grep -q "http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" /etc/apt/sources.list; then
        # Add the LLVM repository to sources.list using sudo
        echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee -a /etc/apt/sources.list
        echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee -a /etc/apt/sources.list
    else
        # Print a message if the repository is already added
        echo "The repository is already added."
    fi

    # Update package lists and install llvm-16 using sudo
    sudo apt-get update
    sudo apt-get install -y llvm-16 build-essential libstdc++-12-dev
fi

# Set up environment variables for ROCm
export CPATH=/opt/rocm/include:$CPATH
export LIBRARY_PATH=/opt/rocm/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Clone and build TVM
echo "Cloning TVM repository and initializing submodules..."
# clone and build tvm
git submodule update --init --recursive

if [ -d build ]; then
    rm -rf build
fi

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build

echo "Configuring TVM build with LLVM and ROCm paths..."
echo "set(USE_LLVM llvm-config-16)" >> config.cmake
echo "set(USE_ROCM /opt/rocm)" >> config.cmake

echo "Running CMake for TileLang..."
/usr/bin/cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TileLang with make..."
make -j
if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

cd ..

# Define the lines to be added
TILELANG_PATH="$(pwd)"
echo "Configuring environment variables for TVM..."

TVM_HOME_ENV="export TVM_HOME=${TILELANG_PATH}/3rdparty/tvm"
TILELANG_PYPATH_ENV="export PYTHONPATH=\$TVM_HOME/python:${TILELANG_PATH}:\$PYTHONPATH"
ROC_DEVICE_ENV="export HIP_VISIBLE_DEVICES=0,1,2,3"  # Adjust based on your GPU configuration

# Check and add the lines if not already present
if ! grep -qxF "$TVM_HOME_ENV" ~/.bashrc; then
    echo "$TVM_HOME_ENV" >> ~/.bashrc
    echo "Added TVM_HOME to ~/.bashrc"
else
    echo "TVM_HOME is already set in ~/.bashrc"
fi

if ! grep -qxF "$TILELANG_PYPATH_ENV" ~/.bashrc; then
    echo "$TILELANG_PYPATH_ENV" >> ~/.bashrc
    echo "Added PYTHONPATH to ~/.bashrc"
else
    echo "PYTHONPATH is already set in ~/.bashrc"
fi

if ! grep -qxF "$ROC_DEVICE_ENV" ~/.bashrc; then
    echo "$ROC_DEVICE_ENV" >> ~/.bashrc
    echo "Added HIP_VISIBLE_DEVICES to ~/.bashrc"
else
    echo "HIP_VISIBLE_DEVICES is already set in ~/.bashrc"
fi

# Add ROCm environment variables
if ! grep -q "CPATH=/opt/rocm/include" ~/.bashrc; then
    echo "export CPATH=/opt/rocm/include:\$CPATH" >> ~/.bashrc
    echo "Added CPATH to ~/.bashrc"
fi

if ! grep -q "LIBRARY_PATH=/opt/rocm/lib" ~/.bashrc; then
    echo "export LIBRARY_PATH=/opt/rocm/lib:\$LIBRARY_PATH" >> ~/.bashrc
    echo "Added LIBRARY_PATH to ~/.bashrc"
fi

if ! grep -q "LD_LIBRARY_PATH=/opt/rocm/lib" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added LD_LIBRARY_PATH to ~/.bashrc"
fi

# Reload ~/.bashrc to apply the changes
source ~/.bashrc

echo "Installation script completed successfully."
```

## Step 8: Verify Installation

Create or modify an example script to use ROCm for testing:

```python
import tilelang
import torch

# Use hip target for TileLang
kernel = tilelang.compile(func, out_idx=[2], target="hip", execution_backend="cython")

# Use cuda device for PyTorch (yes, even for AMD GPUs)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
```

## Common Issues and Solutions

### Issue: CMake Not Found
Solution: Install cmake with `sudo apt-get install -y cmake`

### Issue: Missing C++ Headers
Solution: Install build essentials with `sudo apt-get install -y build-essential libstdc++-12-dev`

### Issue: PyTorch with CUDA Dependencies
Solution: Uninstall the CUDA version and install the ROCm version:
```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### Issue: PyTorch Error "No NVIDIA driver found"
Solution: Make sure you have the ROCm version of PyTorch installed, not the CUDA version.

### Issue: ROCm Compatibility with PyTorch
Note: When using PyTorch with ROCm, continue to use `device="cuda"` in PyTorch code, but use `target="hip"` for TileLang compilation.
