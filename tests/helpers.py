"""Script containing helper functions for testing"""

import importlib.util
import inspect
from functools import wraps
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.conftest import ATOL, RTOL

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray


@wraps(pytest.approx)
def approx(*args, rtol: Optional[float] = None, atol: Optional[float] = None, **kwargs):
    """Wrapper function of `pytest.approx` with sensible defaults"""
    kwargs.setdefault('rel', RTOL if rtol is None else rtol)
    kwargs.setdefault('abs', ATOL if atol is None else atol)
    return pytest.approx(*args, **kwargs)


@wraps(np.lexsort)
def lsort(arr: NDArray) -> NDArray:
    """Lexicographical sort the rows of a 2D array"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, but got '{type(arr).__name__}' instead")
    if arr.ndim > 2:
        raise ValueError(f"Input array must not have more than 2 dimensions, but got {arr.ndim} dimensions instead")
    return arr[np.lexsort(arr.T[::-1])]


@wraps(np.rad2deg)
def rad2deg(arr: NDArray) -> NDArray:
    """Convert radians to degrees"""
    return np.rad2deg(arr)


@wraps(np.deg2rad)
def deg2rad(arr: NDArray) -> NDArray:
    """Convert degrees to radians"""
    return np.deg2rad(arr)


def normalize(Ab: NDArray, eq: bool = False) -> NDArray:
    """Normalize constraint matrix rows by coefficient vector norm.
    
    Each row of [A, b] represents the constraint `A @ x <= b` (inequality)
    or `A @ x = b` (equality). Normalizes by ||A|| to make constraints 
    comparable up to positive scaling.
    
    For equality constraints, applies canonical sign normalization to handle
    the fact that `A @ x = b` and `(-A) @ x = (-b)` are equivalent.
    
    Parameters
    ----------
    Ab : NDArray
        Constraint matrix [A, b]
    eq : bool, default=False
        If True, applies canonical sign normalization for equality constraints
        
    Returns
    -------
    NDArray
        Normalized constraint matrix
    """
    if Ab.size == 0:
        return Ab
    normalized = Ab.copy().astype(float)
    for idx in range(Ab.shape[0]):
        coeffs = Ab[idx, :-1]
        norm_coeffs = np.linalg.norm(coeffs)
        if not np.isclose(norm_coeffs, 0):
            normalized[idx] = Ab[idx] / norm_coeffs
            if eq:
                first_nonzero_idx = np.where(~np.isclose(normalized[idx, :-1], 0))[0]
                if len(first_nonzero_idx) > 0:
                    first_nonzero = normalized[idx, first_nonzero_idx[0]]
                    if first_nonzero < 0:
                        normalized[idx] *= -1
        else:
            normalized[idx] = Ab[idx]
    return normalized


# FROM: GitHub Copilot Raptor mini (Preview) | 2026/04/26[untested/unverified]
def requires(import_name: str, exception: type[BaseException] = ImportError, match: str | None = None):
    """Decorator for optional dependency tests.

    If the named module is unavailable, the wrapped test is executed under
    ``pytest.raises`` and must raise the expected exception with an optional
    regular expression match.

    If the module is available, the decorated test runs normally.

    Supports decorating pytest test functions and test classes by wrapping
    all class attributes whose names start with ``test``.
    """

    def decorator(test_obj):
        if inspect.isclass(test_obj):
            for attr_name in dir(test_obj):
                if not attr_name.startswith('test'):
                    continue
                attr_value = getattr(test_obj, attr_name)
                if not callable(attr_value):
                    continue
                if getattr(attr_value, '__requires_wrapped__', False):
                    continue
                setattr(test_obj, attr_name, decorator(attr_value))
            return test_obj

        @wraps(test_obj)
        def wrapper(*args, **kwargs):
            if importlib.util.find_spec(import_name) is None:
                with pytest.raises(exception, match=match):
                    test_obj(*args, **kwargs)
                return None
            return test_obj(*args, **kwargs)

        wrapper.__requires_wrapped__ = True
        return wrapper

    return decorator


# FROM: GitHub Copilot Raptor mini (Preview) | 2026/04/28[untested/unverified]
def close_figures(test_obj):
    """Decorator that closes matplotlib figures after each test execution.

    When applied to a class, wraps all methods whose names start with ``test``
    so each test automatically closes any matplotlib figures it created.

    For Hypothesis-decorated tests, wraps the underlying ``inner_test`` so
    cleanup happens after every generated example instead of only after the
    whole Hypothesis run.
    """
    def close_all_figures():
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError as _:
            pass

    def wrap_test_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                close_all_figures()

        return wrapper

    if inspect.isclass(test_obj):
        for attr_name, attr_value in vars(test_obj).items():
            if attr_name.startswith('test') and callable(attr_value):
                if hasattr(attr_value, 'hypothesis') and hasattr(attr_value.hypothesis, 'inner_test'):
                    inner = attr_value.hypothesis.inner_test
                    if callable(inner):
                        attr_value.hypothesis.inner_test = wrap_test_function(inner)
                else:
                    setattr(test_obj, attr_name, close_figures(attr_value))
        return test_obj

    return wrap_test_function(test_obj)