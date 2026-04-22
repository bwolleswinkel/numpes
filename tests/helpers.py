"""Script containing helper functions for testing"""

from typing import TYPE_CHECKING
from functools import wraps

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