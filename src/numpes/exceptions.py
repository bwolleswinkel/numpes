"""Module containing the custom exceptions defined in the NumPES package"""


class NumpesException(Exception):
    """Base exception class for all exceptions raised by the NumPES package"""


class InvalidCombinationOfArguments(NumpesException, TypeError):
    """Exception raised when an invalid combination of arguments is provided to a function, method, or constructor in the NumPES package"""


class InvalidRepresentation(NumpesException, AssertionError):
    """Exception raised when an object has an invalid representation in the NumPES package"""


class ConversionError(NumpesException, RuntimeError):
    """Exception raised when an object converts from one representation to another, but the conversion is not implemented or not allowed according to the global configuration"""
