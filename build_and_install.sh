#!/bin/bash

# How to use this script:
# 1. Run this script from the root directory of the project as ./_build_and_install.sh
# 2. The script will create a source distribution of the package and install it.
# 3. The script will also uninstall the package if it is already installed.


# Delete the dist directory
rm -rf dist

# Uninstall the package
python3 -m pip uninstall -y Dosepy

# This script is used to package the application for distribution.
python3 -m pip install --upgrade pip
pyphon3 -m pip install --upgrade build
#python3 -m pip install --upgrade twine

# Create a source distribution
python3 -m build

# Install the package from the source distribution
cd dist
python3 -m pip install *.tar.gz
