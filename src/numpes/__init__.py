# FIXME: Add docstring

from .polytope import *

try: 
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
