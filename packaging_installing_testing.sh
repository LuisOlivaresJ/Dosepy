#!/bin/bash

# How to use this script on Linux:
# 1. Run this script from the root directory of the project as ./packaging_installing_testing.sh
# 2. The script will create a source distribution of the package and install it.
# 3. The script will also uninstall the package if it is already installed.


# Delete the dist directory
rm -rf dist

# Delete __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} +

# Update pip
python3 -m pip install --upgrade pip
python3 -m pip install uv

# Uninstall the package
python3 -m pip uninstall -y Dosepy

# This script is used to package the application for distribution.
#pyphon3 -m pip install --upgrade build
#python3 -m pip install --upgrade twine

# Install build
python3 -m pip install build

# Create a source distribution
python3 -m build

# Install the package from the source distribution
cd dist
#python3 -m pip install *.tar.gz
python3 -m pip install *.tar.gz

# Delete __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} +

# Testing the application
python3 -m pip install pytest
cd ..
python3 -m pytest -v tests/unit/