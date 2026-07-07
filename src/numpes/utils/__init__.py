"""The utils subpackage contains utility functions for the NumPES package. These functions are 
not specific to any particular control algorithm, but are used by multiple algorithms in the package.

Modules
-------
linalg
    Provides several additional linear algebra functionalities.
spatial
    Provides several spatial functionalities related to polyhedral computations.
timeout
    Provides a context manager to set a timeout for executing code.
linprog
    Provides handlers for solving linear programs using either SciPy, CVXPY, or PuLP as a backend.
"""

from numpes.utils.linalg import *
from numpes.utils.linprog import *
from numpes.utils.spatial import *
from numpes.utils.plot import *
from numpes.utils.timeout import *
