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
import warnings

import numpy as np
import scipy as sp

from numpes._config import CFG
from numpes.utils.linprog import Status, solve_lp

if TYPE_CHECKING:
    from typing import Optional, Literal

    from numpy.typing import NDArray


def is_square(A: NDArray) -> bool:
    """Check whether a matrix `A` is square"""
    if not A.ndim == 2:
        return False
    if not A.shape[0] == A.shape[1]:
        return False
    return True


def is_sym(A: NDArray) -> bool:
    """Check if a matrix `A` is symmetric"""
    if not is_square(A):
        return False
    if not np.allclose(A, A.T, rtol=CFG.rtol, atol=CFG.atol):
        return False
    return True


def is_sing(A: NDArray) -> bool:
    """Check whether the matrix is singular (i.e., non-invertible)"""
    if not is_square(A):
        return True
    if np.linalg.matrix_rank(A, rtol=CFG.rtol) < A.shape[0]:
        return True
    return False


def is_posdef(A: NDArray, semi_def: bool = False) -> bool:
    """Check if a matrix `A` is positive definite or positive semi-definite"""
    if semi_def:
        if not is_sym(A):
            return False
        if not np.all(np.linalg.eigvalsh(A) >= -CFG.atol):
            return False
        return True
    try:
        _ = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as _:
        return False
    return True


def is_rot_mat(R: NDArray) -> bool:
    """Check if a matrix `R` is a valid rotation matrix"""
    if not is_square(R) or R.size <= 1:  # NOTE: This is not a valid rotation matrix due to the geometric definition (no rotation in 1D)
        return False
    if not np.allclose(R @ R.T, np.eye(R.shape[0]), rtol=CFG.rtol, atol=CFG.atol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, rtol=CFG.rtol, atol=CFG.atol):
        return False
    return True


# [untested/unverified]
def rot_mat(angles: list[float]) -> NDArray:
    """Construct a rotation matrix from a sequence of Givens angles.
    
    Parameters
    ----------
    angles : list[float]
        A list of Givens angles in radians in QR-like adjacent-plane sweep order
        
    Returns
    -------
    R : NDArray
        An n x n rotation matrix corresponding to the given Givens angles

    Raises
    ------
    ValueError
        If the length of `angles` is not compatible with a valid rotation matrix size
    """
    # Check if the length of angles is compatible with a valid rotation matrix size (1 = 2d, 3 = 3d, 6 = 4d, 10 = 5d, etc.)
    m = len(angles)
    if m <= 0:
        raise ValueError(f"The length={m} of `angles` is not compatible with a valid rotation matrix size (see notes). Nearest valid length is 1 (2d).")
    if not (n := ((1 + np.sqrt(1 + 8 * m)) / 2)).is_integer():
        raise ValueError(f"The length={m} of 'angles' is not compatible with a valid rotation matrix size (see notes). Nearest valid lengths are {int(n) * (int(n) - 1) // 2} ({int(n)}d) or {(int(n) + 1) * int(n) // 2} ({int(n) + 1}d).")
    n = int(n)
    R = np.eye(n)
    for k, angle in enumerate(angles):
        i, j = _idx_plane_ij(k, n)
        G = givens_mat(i, j, angle, n)
        R = R @ G
    return R


# [untested/unverified]
def rot_mat_2d(angle: float) -> NDArray:
    """Create a 2D rotation matrix from a single angle in radians.
    
    Parameters
    ----------
    angle : float
        The rotation angle in radians
        
    Returns
    -------
    R : NDArray
        A 2 x 2 rotation matrix corresponding to the given angle
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return R


# [untested/unverified]
def rot_mat_3d(angles: list[float],
               convention: str | Literal['givens', 'yaw_pitch_roll'] = 'givens',
               ) -> NDArray:
    # FIXME: Instead of using 'proper_euler' and 'tait_bryan', I should use the much more clear 'xyz', 'XYZ', etc., for intrinsic and ectrinsit rotation, and just keep 'givens' and 'yaw_pitch_roll' as special cases. 
    """Create a 3D rotation matrix from a sequence of angles based on the specified convention.
    
    Parameters
    ----------
    angles : list[float]
        A list of length 3 of angles in radians
    convention : str | 'yaw_pitch_roll' | 'givens', default='givens'
        The axis-order convention to use for constructing the rotation matrix (see also notes). Must be 3 characters belonging to the set {'X', 'Y', 'Z'} (for intrinsic rotations) or {'x', 'y', 'z'} (for extrinsic rotations). Extrinsic and intrinsic rotations cannot be mixed teh character sequence. Two special cases are provided for convenience:
        - 'yaw_pitch_roll': Yaw-Pitch-Roll angles (ZYX intrinsic). Identical to 'ZYX'.
        - 'givens': QR-like adjacent-plane Givens sweep (xzx extrinsic). Identical to 'xzx'.

    Returns
    -------
    R : NDArray 
        A 3 x 3 rotation matrix corresponding to the given angles and convention

    Examples
    --------
    >>> angles = np.deg2rad([0, -45, 90])
    >>> print(pes.utils.rot_mat_3d(angles, convention='yaw_pitch_roll').round(2))
    [[ 0.    1.    0.  ]
     [-0.71  0.   -0.71]
     [-0.71  0.    0.71]]
    >>> print(pes.utils.rot_mat_3d(angles, convention='xzx').round(2))
    [[ 0.71  0.   -0.71]
     [ 0.71  0.    0.71]
     [ 0.   -1.    0.  ]]

    Use `pes.utils.angles_3d_convert` to convert between different conventions of 3D rotation angles.

    >>> angles = np.deg2rad([-30, 60, 45])  # Proper Euler angles in YZY convention
    >>> angles_converted = pes.utils.angles_3d_convert(angles, from_convention='yzy', to_convention='givens')
    >>> print(pes.utils.rot_mat_3d(angles_converted))
    [[ 0.66 -0.75  0.05]
     [ 0.61  0.5  -0.61]
     [ 0.44  0.43  0.79]]

    Notes
    -----
    Note that extrinsic rotations are equivalent to intrinsic rotations in the reverse order. For example, a zyx extrinsic rotation is equivalent to an XYZ intrinsic rotation.
    """
    angles = angles_3d_convert(angles, from_convention=convention, to_convention='givens')
    return rot_mat(angles)


def givens_mat(i: int, j: int, angle: float, n: int) -> NDArray:
    """Construct a Givens rotation matrix for the plane spanned by axes i and j"""
    if i < 0 or j < 0 or i >= n or j >= n or i >= j:
        raise ValueError(f"Invalid indices i={i}, j={j} for Givens rotation matrix of size {n} (must satisfy 0 <= i < j < n)")
    G = np.eye(n)
    c, s = np.cos(angle), np.sin(angle)
    G[[i, i], [i, j]] = c, -s
    G[[j, j], [i, j]] = s, c
    return G


# FROM: Gemini 1.5 Flash | 26/09/01[untested/unverified]
def angles_givens(R: NDArray) -> list[float]:
    """Convert an n-dimensional rotation matrix `R` into n * (n - 1) / 2 Givens angles

    Parameters
    ----------
    R : NDArray
        An n x n rotation matrix

    Returns
    -------
    angles : list[float]
        A list of Givens angles in radians, with length n * (n - 1) / 2

    Raises
    ------
    ValueError
        If `R` is not valid rotation matrix
    
    Notes
    -----
    The returned order is the QR-like adjacent plane sweep used by `rot_mat`: for each
    column `col = 0 .. n-2`, rows are traversed bottom-to-top (`row = n-1 .. col+1`) and
    plane `(row-1, row)` is used. This function is the inverse of that construction order. The plane sequence index `k` can be converted to the adjacent plane indices `(i, j)` using `_idx_plane_i_j(k, n)`.
    """
    if not is_square(R):
        raise ValueError(f"Input matrix R must be square, received matrix with shape {R.shape}")
    if not is_rot_mat(R):
        raise ValueError(f"Input matrix R must be a valid rotation matrix (orthogonal with determinant 1), received R @ R.T = {R @ R.T} and det(R) = {np.linalg.det(R)}")

    n = R.shape[0]
    R_copy = np.array(R, dtype=float, copy=True)
    angles = []

    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            y, x = R_copy[row, col], R_copy[row - 1, col]

            theta = np.arctan2(y, x).item()
            angles.append(theta)

            c, s = np.cos(-theta), np.sin(-theta)  # Perform the inverse rotation
            G = np.array([[c, -s],
                          [s,  c]])

            R_copy[[row - 1, row], :] = G @ R_copy[[row - 1, row], :]

    return angles


def angle_2d(R: NDArray) -> float:
    """Compute the rotation angle from a 2D rotation matrix `R`.

    Parameters
    ----------
    R : NDArray
        A 2 x 2 rotation matrix

    Returns
    -------
    angle : float
        The rotation angle in radians

    Raises
    ------
    ValueError
        If `R` is not a valid 2D rotation matrix
    """
    if R.shape != (2, 2):
        raise ValueError(f"Input matrix R must be a 2 x 2 rotation matrix, received shape={R.shape}")
    if not is_rot_mat(R):
        raise ValueError(f"Input matrix R must be a valid rotation matrix (orthogonal with determinant 1), received R @ R.T = {R @ R.T} and det(R) = {np.linalg.det(R)}")
    angle = np.arctan2(R[1, 0], R[0, 0])
    return angle


def angles_3d(R: NDArray,
              convention: str | Literal['yaw_pitch_roll', 'givens'] = 'givens',
              ) -> list[float]:
    if R.shape != (3, 3):
        raise ValueError(f"Input matrix R must be a 3 x 3 rotation matrix, received shape={R.shape}")
    if not is_rot_mat(R):
        raise ValueError(f"Input matrix R must be a valid rotation matrix (orthogonal with determinant 1), received R @ R.T = {R @ R.T} and det(R) = {np.linalg.det(R)}")
    angles = angles_givens(R)
    angles_converted = angles_3d_convert(angles, from_convention='givens', to_convention=convention)
    return angles_converted


# [untested/unverified]
def angles_3d_convert(angles: list[float],
                      from_convention: str | Literal['yaw_pitch_roll', 'givens'] = 'givens',
                      to_convention: str | Literal['yaw_pitch_roll', 'givens'] = 'givens',
                      ) -> list[float]:
    """Convert a sequence of 3D rotation angles from one convention to another.
        
    Parameters
    ----------
    angles : list[float]
        A list of length 3 of angles in radians
    from_convention : str | 'yaw_pitch_roll' | 'givens', default='givens'
        The axis-order convention of the input angles. Must be 3 characters belonging to the set {'X', 'Y', 'Z'} (for intrinsic rotations) or {'x', 'y', 'z'} (for extrinsic rotations). Extrinsic and intrinsic rotations cannot be mixed in the character sequence. Two special cases are provided for convenience:
        - 'yaw_pitch_roll': Yaw-Pitch-Roll angles (ZYX intrinsic). Identical to 'ZYX'.
        - 'givens': QR-like adjacent-plane Givens sweep (xzx extrinsic). Identical to 'xzx'.
    to_convention : str | 'yaw_pitch_roll' | 'givens', default='givens'
        The axis-order convention to convert the angles to. Must be 3 characters belonging to the set {'X', 'Y', 'Z'} (for intrinsic rotations) or {'x', 'y', 'z'} (for extrinsic rotations). Extrinsic and intrinsic rotations cannot be mixed in the character sequence. Two special cases are provided for convenience:
        - 'yaw_pitch_roll': Yaw-Pitch-Roll angles (ZYX intrinsic). Identical to 'ZYX'.
        - 'givens': QR-like adjacent-plane Givens sweep (xzx extrinsic). Identical to 'xzx'.

    Returns
    -------
    angles_converted : list[float]
        A list of length 3 of angles in radians corresponding to the converted convention
    """

    def _validate_convention(convention: str | Literal['yaw_pitch_roll', 'givens']) -> None:
        if convention in {'yaw_pitch_roll', 'givens'}:
            return
        if len(convention) != 3:
            raise ValueError(f"Invalid convention '{convention}' (must be a string of length 3 or one of the special cases 'yaw_pitch_roll' or 'givens')")
        if set(convention) - {'X', 'Y', 'Z', 'x', 'y', 'z'}:
            raise ValueError(f"Invalid convention '{convention}' (must only contain characters from {{'X', 'Y', 'Z'}} or {{'x', 'y', 'z'}})")
        if any(c.isupper() for c in convention) and any(c.islower() for c in convention):
            raise ValueError(f"Invalid convention '{convention}'. Cannot mix intrinsic (uppercase) and extrinsic (lowercase) rotations.")
        if convention[0] == convention[1] or convention[1] == convention[2]:
            raise ValueError(f"Invalid convention '{convention}' (consecutive axes must be different)")

    _validate_convention(from_convention)
    _validate_convention(to_convention)

    convention_map = {
        'yaw_pitch_roll': 'zyx',
        'givens': 'XZX',
    }
    src_seq = convention_map.get(from_convention, from_convention)
    tgt_seq = convention_map.get(to_convention, to_convention)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        angles_converted = sp.spatial.transform.Rotation.from_euler(src_seq, angles).as_euler(tgt_seq)
    return angles_converted


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


# FROM: Gemini 1.5 Flash | 26/09/02[untested/unverified]
def _idx_plane_ij(k: int, n: int) -> tuple[int, int]:
    """Convert plane-sequence index k to adjacent plane indices (i, j) for QR-like sweep order

    Parameters
    ----------
    k : int
        The plane sequence index, where 0 <= k < n * (n - 1) / 2
    n : int
        The dimension of the rotation matrix

    Returns
    -------
    i : int
        The first index of the adjacent pair
    j : int
        The second index of the adjacent pair

    Raises
    ------
    ValueError
        If `k` is out of bounds for the given `n`
    """
    m = n * (n - 1) // 2
    if k < 0 or k >= m:
        raise ValueError(f"Index k={k} is out of bounds for n={n} (must satisfy 0 <= k < {m})")

    # FIXME: Actually, I think there is a very natural ordering with tranches (add to notes):
    # 2d: (0, 1) -> [[I]]
    # 3d: (0, 1), (1, 2), (0, 2) -> [[1, 2], [1]]
    # 3d: (2, 3), (1, 2), (0, 1), (2, 3), (1, 2), (2, 3) -> [[1, 2, 3], [1, 2], [1]]
    # See the pattern? The first tranche is some length, then it repeats on shorter, then two shorter, etc.: now, how to get the ordering of the first tranche? I do not know, so I need to still figure this out

    count = 0
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            if count == k:
                return row - 1, row
            count += 1

    raise ValueError(f"Could not map k={k} to a plane for n={n}")
