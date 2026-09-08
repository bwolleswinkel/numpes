"""Tests for functionality inside `src/numpes/exceptions.py`"""

import inspect

import numpes as pes


class TestNumpesException:
    """Tests for the `pes.NumpesException` exception class"""

    def test_is_exception(self) -> None:
        """Test if the class is a subclass of Exception"""
        assert issubclass(pes.NumpesException, Exception), \
            "NumpesException should be a subclass of the built-in Exception"

    def test_docstring(self) -> None:
        """Test the docstring formulation"""
        expected_docstring = "Base exception for all exceptions raised by the NumPES package"
        assert pes.NumpesException.__doc__ == expected_docstring, \
            f"Expected docstring to read '{expected_docstring}', but received '{pes.NumpesException.__doc__}'"


class TestInvalidCombinationOfArgumentsError:
    """Tests for the `pes.InvalidCombinationOfArgumentsError` exception class"""

    def test_is_exception(self) -> None:
        """Test if the class is a subclass of Exception"""
        assert issubclass(pes.InvalidCombinationOfArgumentsError, Exception), \
            "InvalidCombinationOfArgumentsError should be a subclass of the built-in Exception"

    def test_is_numpes_exception(self) -> None:
        """Test if the class is a subclass of NumpesException"""
        assert issubclass(pes.InvalidCombinationOfArgumentsError, pes.NumpesException), \
            "InvalidCombinationOfArgumentsError should be a subclass of NumpesException"

    def test_is_type_error(self) -> None:
        """Test if the class is a subclass of the built-in TypeError"""
        assert issubclass(pes.InvalidCombinationOfArgumentsError, TypeError), \
            "InvalidCombinationOfArgumentsError should be a subclass of TypeError"

    def test_docstring(self) -> None:
        """Test the docstring formulation"""
        expected_docstring = \
            r"""An invalid combination of arguments is provided to a function, method, or constructor.

            Examples
            --------
            >>> try:
            ...     pes.poly([[1,  2],
            ...               [0, -1]], n=2)
            ... except pes.InvalidCombinationOfArgumentsError as e:
            ...     print(f"Caught {type(e).__name__}: {e}")
            Caught InvalidCombinationOfArgumentsError: Cannot provide 'n' when initializing from vertices
            """
        assert inspect.cleandoc(pes.InvalidCombinationOfArgumentsError.__doc__) == inspect.cleandoc(expected_docstring), \
            f"Expected docstring to read '{expected_docstring}', but received '{pes.InvalidCombinationOfArgumentsError.__doc__}'"


class TestInvalidOperationError:
    """Tests for the `pes.InvalidOperationError` exception class"""

    def test_is_exception(self) -> None:
        """Test if the class is a subclass of Exception"""
        assert issubclass(pes.InvalidOperationError, Exception), \
            "InvalidOperationError should be a subclass of the built-in Exception"

    def test_is_numpes_exception(self) -> None:
        """Test if the class is a subclass of NumpesException"""
        assert issubclass(pes.InvalidOperationError, pes.NumpesException), \
            "InvalidOperationError should be a subclass of NumpesException"

    def test_is_value_error(self) -> None:
        """Test if the class is a subclass of the built-in ValueError"""
        assert issubclass(pes.InvalidOperationError, ValueError), \
            "InvalidOperationError should be a subclass of ValueError"

    def test_docstring(self) -> None:
        """Test the docstring formulation"""
        expected_docstring = "The (order of) operation between these two objects is invalid or undefined"
        assert pes.InvalidOperationError.__doc__ == expected_docstring, \
            f"Expected docstring to read '{expected_docstring}', but received '{pes.InvalidOperationError.__doc__}'"


class TestDimensionError:
    """Tests for the `pes.DimensionError` exception class"""

    def test_is_exception(self) -> None:
        """Test if the class is a subclass of Exception"""
        assert issubclass(pes.DimensionError, Exception), \
            "DimensionError should be a subclass of the built-in Exception"

    def test_is_numpes_exception(self) -> None:
        """Test if the class is a subclass of NumpesException"""
        assert issubclass(pes.DimensionError, pes.NumpesException), \
            "DimensionError should be a subclass of NumpesException"

    def test_is_value_error(self) -> None:
        """Test if the class is a subclass of the built-in ValueError"""
        assert issubclass(pes.DimensionError, ValueError), \
            "DimensionError should be a subclass of ValueError"

    def test_docstring(self) -> None:
        """Test the docstring formulation"""
        expected_docstring = "Two objects have incompatible dimensions for the operation to be performed"
        assert pes.DimensionError.__doc__ == expected_docstring, \
            f"Expected docstring to read '{expected_docstring}', but received '{pes.DimensionError.__doc__}'"


class TestInvalidRepresentationError:
    """Tests for the `pes.InvalidRepresentationError` exception class"""

    def test_is_exception(self) -> None:
        """Test if the class is a subclass of Exception"""
        assert issubclass(pes.InvalidRepresentationError, Exception), \
            "InvalidRepresentationError should be a subclass of the built-in Exception"

    def test_is_numpes_exception(self) -> None:
        """Test if the class is a subclass of NumpesException"""
        assert issubclass(pes.InvalidRepresentationError, pes.NumpesException), \
            "InvalidRepresentationError should be a subclass of NumpesException"

    def test_is_assertion_error(self) -> None:
        """Test if the class is a subclass of the built-in AssertionError"""
        assert issubclass(pes.InvalidRepresentationError, AssertionError), \
            "InvalidRepresentationError should be a subclass of AssertionError"

    def test_docstring(self) -> None:
        """Test the docstring formulation"""
        expected_docstring = \
            r"""The object has an invalid or unresolvable representation.

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
        assert inspect.cleandoc(pes.InvalidRepresentationError.__doc__) == inspect.cleandoc(expected_docstring), \
            f"Expected docstring to read '{expected_docstring}', but received '{pes.InvalidRepresentationError.__doc__}'"


class TestConversionError:
    """Tests for the `pes.ConversionError` exception class"""

    def test_is_exception(self) -> None:
        """Test if the class is a subclass of Exception"""
        assert issubclass(pes.ConversionError, Exception), \
            "ConversionError should be a subclass of the built-in Exception"

    def test_is_numpes_exception(self) -> None:
        """Test if the class is a subclass of NumpesException"""
        assert issubclass(pes.ConversionError, pes.NumpesException), \
            "ConversionError should be a subclass of NumpesException"

    def test_is_runtime_error(self) -> None:
        """Test if the class is a subclass of the built-in RuntimeError"""
        assert issubclass(pes.ConversionError, RuntimeError), \
            "ConversionError should be a subclass of RuntimeError"

    def test_docstring(self) -> None:
        """Test the docstring formulation"""
        expected_docstring = \
            r"""Object converts from one representation to another, but the conversion is not implemented or not allowed according to the global configuration.

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
        assert inspect.cleandoc(pes.ConversionError.__doc__) == inspect.cleandoc(expected_docstring), \
            f"Expected docstring to read '{expected_docstring}', but received '{pes.ConversionError.__doc__}'"
