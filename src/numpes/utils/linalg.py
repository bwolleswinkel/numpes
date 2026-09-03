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


def is_square(A: NDArray) -> bool:
    """Check whether a matrix `A` is square"""
    if not A.ndim == 2:
        return False
    if not A.shape[0] == A.shape[1]:
        return False
    return True


def is_sing(A: NDArray) -> bool:
    """Check whether the matrix is singular (i.e., non-invertible)"""
    if not is_square(A):
        return True
    if np.linalg.matrix_rank(A, rtol=CFG.rtol) < A.shape[0]:
        return True
    return False


# FROM: Google Gemini Pro | 2026/08/31[untested/unverified]
def minimize_vrepr(verts: NDArray, rays: Optional[NDArray] = None) -> tuple[NDArray, NDArray]:
    """Minimize the V-representation by removing redundant rays and vertices"""

    def spans_ambient_space(rays: NDArray) -> bool:
        """Check whether the rays span the ambient space"""
        k_rays, n = rays.shape
        if k_rays < n + 1:
            return False
        for basis_vec in np.vstack((np.eye(n), -np.eye(n))):
            res = solve_lp(np.zeros(k_rays), A_eq=rays.T, b_eq=basis_vec, bounds=[(0, None)] * k_rays)
            if not res.success:
                return False
        return True

    if not isinstance(verts, np.ndarray) or (rays is not None and not isinstance(rays, np.ndarray)):
        raise TypeError(f"Expected verts and rays to be NumPy arrays, but got verts of type {type(verts)} and rays of type {type(rays)}")
    if verts.ndim != 2:
        raise ValueError(f"Expected verts to be a 2D array, but got an array with shape {verts.shape}")
    if rays is not None and rays.ndim != 2:
        raise ValueError(f"Expected rays to be a 2D array, but got an array with shape {rays.shape}")
    if rays is not None and verts.shape[1] != rays.shape[1]:
        raise ValueError(f"Expected verts and rays to have the same number of columns, but got verts with shape {verts.shape} and rays with shape {rays.shape}")
    n = verts.shape[1]

    if rays is None:
        rays = np.empty((0, n))
    # 0: Check for the special zero rays case, which should be mapped to a zero vertex
    if verts.size == 0 and np.allclose(rays, 0, rtol=CFG.rtol, atol=CFG.atol):
        return np.zeros((1, n)), np.empty((0, n))
    # 1: Find redundant rays
    rays = reduce_rays(rays)
    # 1-A: Check of the rays span the entire space
    if spans_ambient_space(rays):
        return np.empty((0, n)), np.vstack((np.eye(n), -np.ones(n)))
    # 1-B: Check if the vertices are empty and rays are not (non-pointed; add the zero vertex)
    if verts.size == 0 and rays.size > 0:
        return np.zeros((1, n)), rays
    # 2: Find redundant vertices
    verts = reduce_verts(verts, rays)

    return verts, rays


# FROM: Google Gemini Pro | 2026/08/31[untested/unverified]
# FIXME: Should these also/only return the indices? That's more performant, right?
def reduce_rays(rays: NDArray) -> NDArray:
    """Minimize a collection of rays by removing directions already captured by other rays"""
    if (k_rays := rays.shape[0]) <= 1:
        return rays
    redundant = np.zeros(k_rays, dtype=bool)
    for idx in range(k_rays):
        other = ~redundant.copy()
        other[idx] = False
        A = rays[other, :].T
        b = rays[idx]
        num_other_rays = np.count_nonzero(other)
        if num_other_rays == 0:
            continue
        c = np.zeros(num_other_rays)
        bounds = [(0, None) for _ in range(num_other_rays)]
        res = solve_lp(c, A_eq=A, b_eq=b, bounds=bounds)
        if res.success:
            redundant[idx] = True
    return rays[~redundant]


# FROM: Google Gemini Pro | 2026/08/31[untested/unverified]
# FIXME: Should these also/only return the indices? That's more performant, right?
# pylint: disable=invalid-name
def reduce_verts(verts: NDArray, rays: NDArray) -> NDArray:
    """Minimize a collection of vertices by removing directions already implied by other vertices and rays"""
    if (k := verts.shape[0]) <= 1:
        return verts
    redundant = np.zeros(k, dtype=bool)
    for idx in range(k):
        # Gather all other non-redundant vertices
        # Construct equality constraints for:
        # Sum(lambda_i * v_i) + Sum(mu_j * r_j) == v_k
        # Sum(lambda_i) == 1
        # Top part of matrix: coordinates matching v_k
        other = ~redundant.copy()
        other[idx] = False
        A_top_v = verts[other].T
        A_top_r = rays.T
        A_top = np.column_stack((A_top_v, A_top_r))
        # Bottom part of matrix: sum of lambdas must equal 1 (mus are ignored here)
        A_bot_v = np.ones((1, np.count_nonzero(other)))
        A_bot_r = np.zeros((1, (k_rays := rays.shape[0])))
        A_bot = np.column_stack((A_bot_v, A_bot_r))

        A_eq = np.vstack((A_top, A_bot))
        b_eq = np.append(verts[idx], 1.0)

        # LP formulation:
        # Minimize 0
        # Subject to: A_eq * [lambda, mu] == b_eq, lambda >= 0, mu >= 0
        num_vars = np.count_nonzero(other) + k_rays
        c = np.zeros(num_vars)
        bounds = [(0, None) for _ in range(num_vars)]

        res = solve_lp(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if res.success:
            redundant[idx] = True

    return verts[~redundant]


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


# [untested/unverified]
def span(A: NDArray) -> NDArray:
    """Remove linearly dependent columns from a matrix. The columns are preserved in a left-to-right order."""
    # FIXME: Should we just make this row-major ordering instead? to fix the "transpose-hell"?
    if A.ndim != 2:
        raise ValueError(f"Parameter 'A' must be a matrix of size `(m, n)`, but recieved {A.shape}")
    if np.isnan(A).any() or not np.isfinite(A).all():
        raise ValueError("Array 'A' must not contain NaN or inf values")

    if np.all(np.abs(A) <= CFG.atol):
        return np.empty((A.shape[0], 0))
    if A.shape[0] <= 1:
        return A

    # FIXME: For some reason, left-to-right ordering is NOT preserved; the returned columns are always in order,
    # but if columns i and i + delta are lin dependent, sometimes the i + delta column is returned instead of
    # the 'first' encountered i column, which is a bit arbitrary. Off course this is not a huge problem, but might
    # be nice to look into if we can actually preseve this ordering.
    _, R, P = sp.linalg.qr(A, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > CFG.atol)
    return A[:, sorted(P[:rank])]
