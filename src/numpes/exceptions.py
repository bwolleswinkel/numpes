"""Module containing custom exceptions"""


class NumpesException(Exception):
    """Base exception for all exceptions raised by the NumPES package"""


class InvalidCombinationOfArgumentsError(NumpesException, TypeError):
    """An invalid combination of arguments is provided to a function, method, or constructor.

    Examples
    --------
    >>> try:
    ...     pes.poly([[1,  2],
    ...               [0, -1]], n=2)
    ... except pes.InvalidCombinationOfArgumentsError as e:
    ...     print(f"Caught {type(e).__name__}: {e}")
    Caught InvalidCombinationOfArgumentsError: Cannot provide 'n' when initializing from vertices
    """


class InvalidOperationError(NumpesException, ValueError):
    """The (order of) operation between these two objects is invalid or undefined"""


class InvalidRepresentationError(NumpesException, AssertionError):
    """The object has an invalid or unresolvable representation.

    Examples
    --------
    >>> try:
    ...     poly = pes.poly([[1,  2],
    ...                      [0, -1]])
    ...     poly._vrepr = None
    ...     print(poly)
    ... except pes.InvalidRepresentationError as e:
    ...     print(f"Caught {type(e).__name__}: {e}")
    Caught InvalidRepresentationError: Polytope is not properly initialized with either V-representation or H-representation
    """


class ConversionError(NumpesException, RuntimeError):
    """Object converts from one representation to another, but the conversion is not implemented or not allowed according to the global configuration.

    Examples
    --------
    >>> with pes.algo_options(on_poly_convert='error'):
    ...     try: 
    ...         poly = pes.poly([[1,  2],
    ...                          [0, -1]])
    ...         print(f"{poly:h}")
    ...     except pes.ConversionError as e:
    ...         print(f"Caught {type(e).__name__}: {e}")
    Caught ConversionError: The value of 'CFG.on_poly_convert' is set to 'error', so conversion between polytope representations is not allowed
    """


class DimensionError(NumpesException, ValueError):
    """Two objects have incompatible dimensions for the operation to be performed"""
