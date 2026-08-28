"""Module containing custom exceptions"""


class NumpesException(Exception):
    """Base exception for all exceptions raised by the NumPES package"""


class InvalidCombinationOfArgumentsError(NumpesException, TypeError):
    """An invalid combination of arguments is provided to a function, method, or constructor"""


class InvalidRepresentationError(NumpesException, AssertionError):
    """The object has an invalid or unresolvable representation"""


class ConversionError(NumpesException, RuntimeError):
    """Object converts from one representation to another, but the conversion is not implemented or not allowed according to the global configuration"""


class DimensionError(NumpesException, ValueError):
    """Two objects have incompatible dimensions for the operation to be performed"""
