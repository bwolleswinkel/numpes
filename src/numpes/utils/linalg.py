"""Module containing linear algebra functionality for matrices such as 
removing linearly dependent rows, removing redundant inequalities, and 
small utilities such as checking if a matrix is singular

Functions
---------
minimize_hrepr
    Minimize an H-representation by removing redundant inequalities and finding implicit equalities
reduce_eq
    Remove linearly dependent rows from an equality constraint matrix
reduce_ineq
    Remove redundant inequalities from a constraint matrix that are implied by other inequalities and equalities
find_implicit
    Find implicit equalities using slack analysis from a single LP solve
span
    Remove linearly dependent columns from a matrix
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from numpes._config import CFG
from numpes.utils.linprog import Status, solve_lp

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/04/19[untested/unverified]
def minimize_hrepr(Ab: NDArray, Ab_eq: Optional[NDArray] = None) -> tuple[NDArray, NDArray]:
    """Minimize an H-representation by removing redundant inequalities
    and finding implicit equalities"""
    if Ab_eq is None:
        Ab_eq = np.empty((0, Ab.shape[1]))
    # Step 1: Find implicit equalities in the inequalities, move them to the equality matrix,
    # and remove redundant equalities; repeat until no more implicit equalities are found
    while True:
        # Remove redundant/duplicate/trivial ([0, 0, ..., 0]) equalities
        Ab_eq = reduce_eq(Ab_eq)
        n = Ab_eq.shape[1] - 1

        Ab, Ab_eq_new = find_implicit(Ab, Ab_eq)  # pylint: disable=invalid-name
        # Check if find_implicit returned the infeasible marker
        if Ab.shape[0] == 1 and Ab_eq_new.size == 0 and np.array_equal(Ab[0], [0] * n + [-1]):
            return Ab, Ab_eq_new

        if Ab_eq_new.size == 0:
            break
        Ab_eq = np.vstack([Ab_eq, Ab_eq_new]) if Ab_eq.size > 0 else Ab_eq_new

    # Step 2: Global redundancy check for inequalities
    Ab = reduce_ineq(Ab, Ab_eq)

    return Ab, Ab_eq


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/04/18[unverified]
def reduce_eq(Ab_eq: NDArray) -> NDArray:
    """Remove linearly dependent rows from equality constraint matrix"""
    if Ab_eq.shape[0] <= 1:
        return Ab_eq
    rank = np.linalg.matrix_rank(Ab_eq, tol=CFG.atol)
    if rank == Ab_eq.shape[0]:
        return Ab_eq
    selected = []
    current_matrix = np.empty((0, Ab_eq.shape[1]))
    for row in range(Ab_eq.shape[0]):
        test_matrix = np.vstack([current_matrix, Ab_eq[row, :]])
        test_rank = np.linalg.matrix_rank(test_matrix, tol=CFG.atol)
        if test_rank > current_matrix.shape[0]:
            selected.append(row)
            current_matrix = test_matrix
        if len(selected) == rank:
            break
    return Ab_eq[selected, :]


def reduce_ineq(Ab: NDArray, Ab_eq: Optional[NDArray] = None) -> NDArray:
    """Remove redundant inequalities from the constraint matrix that are implied by other inequalities and equalities"""
    if Ab.shape[0] == 0:
        return Ab
    m, n = Ab.shape[0], Ab.shape[1] - 1
    if Ab_eq is None:
        Ab_eq = np.empty((0, n + 1))
    redundant = np.zeros(m, dtype=bool)
    for idx in range(m):
        redundant[idx] = True
        res = solve_lp(-Ab[idx, :-1], Ab[~redundant, :-1], Ab[~redundant, -1], Ab_eq[:, :-1], Ab_eq[:, -1], None)
        # If the LP is successful and the max value is <= b_i, it's redundant
        if (res.success
            and res.status != Status.UNBOUNDED
            and -res.value < Ab[idx, -1]  # type: ignore[operator]
            and not np.isclose(-res.value, Ab[idx, -1], rtol=CFG.rtol, atol=CFG.atol)):  # type: ignore[operator]
            redundant[idx] = True
        else:
            redundant[idx] = False
    return Ab[~redundant]


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/04/20[untested/unverified]
def find_implicit(Ab: NDArray, Ab_eq: NDArray) -> tuple[NDArray, NDArray]:
    """Find implicit equalities by checking whether each inequality is tight everywhere"""
    m, n = Ab.shape[0], Ab.shape[1] - 1
    if m == 0:
        return Ab, np.empty((0, n + 1))

    # Remove trivial all-zero inequalities before solving LPs.
    # If 0*x <= b with b < 0, the system is infeasible.
    zero_rows = np.all(np.isclose(Ab[:, :-1], 0, rtol=CFG.rtol, atol=CFG.atol), axis=1)
    if np.any(zero_rows):
        bounds = Ab[zero_rows, -1]
        if np.any(bounds <= -CFG.atol):
            return np.array([[0] * n + [-1]]), np.empty((0, n + 1))
        Ab = Ab[~zero_rows, :]
        m = Ab.shape[0]
        if m == 0:
            return Ab, np.empty((0, n + 1))

    # Check feasibility of the full system first
    A_eq, b_eq = (Ab_eq[:, :-1], Ab_eq[:, -1]) if Ab_eq.size > 0 else (None, None)
    feasibility = solve_lp(np.zeros(n), Ab[:, :-1], Ab[:, -1], A_eq, b_eq, None)
    if not feasibility.success:
        return np.array([[0] * n + [-1]]), np.empty((0, n + 1))

    implicit_mask = np.zeros(m, dtype=bool)
    for idx in range(m):
        # If the minimum of a_i x over the feasible set equals b_i, then
        # the inequality a_i x <= b_i is tight for every feasible x.
        res = solve_lp(Ab[idx, :-1], Ab[:, :-1], Ab[:, -1], A_eq, b_eq, None)
        if (res.success
            and res.status != Status.UNBOUNDED
            and res.value is not None
            and np.isclose(res.value, Ab[idx, -1], rtol=CFG.rtol, atol=CFG.atol)):
            implicit_mask[idx] = True

    Ab_eq_new = Ab[implicit_mask, :]  # pylint: disable=invalid-name
    return Ab[~implicit_mask], Ab_eq_new


def span(A: NDArray) -> NDArray:
    """Remove linearly dependent columns from a matrix. The columns are preserved in a left-to-right order."""
    if A.ndim != 2:
        raise ValueError(f"Parameter 'A' must be a matrix of size `(m, n)`, but recieved {A.shape}")
    if np.isnan(A).any() or not np.isfinite(A).all():
        raise ValueError("Array 'A' must not contain NaN or inf values")
    
    if np.all(np.abs(A) <= CFG.atol):
        return np.empty((0, A.shape[1]))
    if A.shape[0] <= 1:
        return A

    # FIXME: For some reason, left-to-right ordering is NOT preserved; the returned columns are always in order,
    # but if columns i and i + delta are lin dependent, sometimes the i + delta column is returned instead of
    # the 'first' encountered i column, which is a bit arbitrary. Off course this is not a huge problem, but might
    # be nice to look into if we can actually preseve this ordering.
    _, R, P = sp.linalg.qr(A, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > CFG.atol)
    return A[:, sorted(P[:rank])]
