#!/bin/bash

# Remove previous distribution files
rm -rf dist/*

# Build the distribution
python3 setup.py sdist bdist_wheel

# Check the operating system
OS="$(uname)"
if [ "$OS" = "Darwin" ]; then
    # macOS specific commands
    delocate-wheel -w fixed_wheels -v dist/*.whl
    mv fixed_wheels/*.whl dist/
elif [ "$OS" = "Linux" ]; then
    # Linux specific commands (if any can be added here)
    echo "Linux OS detected. No additional steps required for Linux."
fi

# Upload the distribution to PyPI
twine upload dist/*