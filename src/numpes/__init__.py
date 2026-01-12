"""
NumPES
======

Provides
  1. Numeric operations on polytopes, ellipsoids, and subspaces
  2. Visualization tools for these objects, and plotting functions in 1D-3D
  3. Implementation of several algorithms from the field of control theory

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided with the code, and a `Read the Docs<https://numpes.readthedocs.io/en/latest/>`_ hosted version with more elaborate explanations and examples.

The docstring examples assume that `numpes` has been imported as `pes`:
>>> import numpes as pes

Code snippets are indicated by three greater-than signs:
>>> P = pes.poly(n=3)
>>> print(P.is_empty)
True

Use the built-in `help` function to view a function's docstring:
>>> help(pes.poly)
... # doctest: +SKIP

Available sub-packages
----------------------
control
    Provides several algorithms from control theory

Utilities
---------
__version__
    NumPES version string

"""

from .polytope import *

# FIXME: This does not seem to be working properly for version numbering
try: 
    from ._version import version as __version__
except ImportError:
    __version__: str = "unknown"
