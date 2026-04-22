"""This module contains the internal sub-package, which contains modules that are used for the 
internal structuring of the code, but are not intended to be used by users.

Modules
-------
multipledispatch
    This module defines the multiple dispatch functionality used in NumPES, allowing for function 
    overloading based on argument types in the Polytope constructor.
wraps
    This module defines a decorator for wrapping functions, used in for instance factory patterns.

"""

from .multipledispatch import *
from .wraps import *
