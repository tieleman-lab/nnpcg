"""
NNPCG: NNP Coarse-Grained Simulations
=====================================

:Authors: Daniel P. Ramirez 
:Year: 2023
:Copyright: MIT License

Python interface to set up simulations with state-of-the-art neural network potentials.
"""

import sys
from setuptools import setup, find_packages

short_description = "Python interface to set up simulations with state-of-the-art neural network potentials.".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None

setup(
    # Self-descriptive entries which should always be present
    name='nnpcg',
    author='Daniel P. Ramirez',
    author_email='daniel.ramirezecheme@ucalgary.ca',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.1.0',
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    python_requires=">=3.9",  # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)
