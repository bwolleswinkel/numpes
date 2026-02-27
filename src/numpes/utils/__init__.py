"""The utils subpackage contains utility functions for the NumPES package. These functions are not specific to any particular control algorithm, but are used by multiple algorithms in the package.

Modules
-------
linalg
    Provides several linear algebra functionalities that are not available in the packages NumPy and SciPy.
spatial
    Provides several spatial functionalities, mainly related to polytopes, leveraging the `pycddlib` package as a backend.
timeout
    Provides a context manager to set a timeout for a block of code.
linprog
    Provides handlers for solving linear programs using either SciPy or CVXPY as a backend.

"""