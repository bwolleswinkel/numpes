"""This module contains the internal sub-package, which contains modules that are used for the internal structuring of the code, but are not intended to be used by users.

Modules
-------
exceptions
    This module defines the custom exceptions used in NumPES.
multipledispatch
    This module defines the multiple dispatch functionality used in NumPES, allowing for function overloading based on argument types in the Polytope constructor.
typing
    Contains type hints for NumPy arrays, specifically NDArray, used in type annotations throughout the codebase.
wraps
    This module defines a decorator for wrapping functions, used in for instance factory patterns.

"""