"""
NumPES
======

Provides
  1. Numeric operations on polytopes, ellipsoids, and subspaces
  2. Visualization tools for these objects, and plotting functions in 1D-3D
  3. Implementation of several algorithms from the field of control theory

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided with the code, 
and a `Read the Docs<https://numpes.readthedocs.io/en/latest/>`_ hosted version with 
more elaborate explanations and examples.

The docstring examples assume that `numpes` has been imported as `pes`:
>>> import numpes as pes

Code snippets are indicated by three greater-than signs:
>>> P = pes.poly(n=3)
>>> print(P.is_empty)
True

Use the built-in `help` function to view a function's docstring:
>>> help(pes.poly)

Sub-packages
------------
control
    Provides several algorithms from control theory
utils
    Provides utility functions from linear algebra, polyhedral computations, and more

Modules
-------
polytope
    Provides the Polytope and Zonotope class for representing 
    convex polytopes and zonotopes, respectively, and related functions
ellipsoid
    Provides the Ellipsoid class for representing ellipsoids, and related functions
subspace
    Provides the Subspace class for representing linear subspaces, and related functions

Utilities
---------
__version__
    NumPES version string

"""

from importlib.metadata import PackageNotFoundError

from numpes import control, utils
from numpes._config import algo_options, get_config, reset_config, set_algo_options
from numpes.exceptions import ConversionError, InvalidCombinationOfArguments, InvalidRepresentation, NumpesException
from numpes.polytope import Polytope, poly, poly_ambient, poly_empty, poly_from_bounds, poly_from_ineq, poly_from_verts

try:
    from importlib.metadata import version
    __version__: str = version('numpes')
except (ImportError, PackageNotFoundError) as _:
    __version__ = "unknown"
