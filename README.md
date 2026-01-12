# NumPES

[![PyPI version](https://img.shields.io/pypi/v/numpes)](https://pypi.org/project/numpes/)
![Tests](https://github.com/bwolleswinkel/numpes/actions/workflows/run-package-tests.yml/badge.svg)

NumPES is a controls-oriented Python package for performings numerical operations on polytopes, ellipsoids, and subspaces. Its classes are implemented as emulating numeric types, enabling a breadth of operations to be performed and evaluated. A modern scientific computing package implemented in Python, using C/C++ as a backend for efficient computation. 

## Installation

The easiest way to instal NumPES is by running

``` zsh
pip install numpes
```

in your terminal or dedicated virtual environment. Then, import NumPES using

``` py
import numpes as pes
```

at the top of your script to make its functionalities available.

## Quick start

NumPES allows you to construct polytope, ellipsoid, and subspace objects and manipulate these. For example, start by initializing a polytope represented by its half-space representation (H-representation).

``` py
import numpy as np
import numpes as pes

A = np.array([[ 0,  1],
              [-2,  0],
              [ 1,  1],
              [ 0, -1]])
b = np.array([1, 1, 1, 0])

poly = pes.poly(A, b)  # Defines a polytope A x <= b

poly.plot(show=True)
```

## Contribute to NumPES

We wholeheartedly invite collaborators to make contributions to NumPES. The easiest way to contribute is to fork this repository, implement your new functionality/improvement, and then create a pull request to merge this functionality in a new release of NumPES.