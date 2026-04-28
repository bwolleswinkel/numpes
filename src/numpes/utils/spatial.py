"""Module containing spatial algorithms for polytopes, such as convex hull and enumeration of vertices and facets.

Functions
---------
enum_gens
    Enumerate the vertices and rays of a polytope given its inequalities and equalities
enum_facets
    Enumerate the inequalities and equalities of a polytope given its vertices and rays
conv
    Compute the convex hull for a set of points
signed_angle
    Compute the signed angle between two vectors given a look direction
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

try:
    import cdd
    CDD_INSTALLED: bool = True
except ImportError as _:
    CDD_INSTALLED = False

from .._config import CFG

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray


# FROM: GitHub Copilot Claude Sonnet 4.5 | 2026/02/08[unverified]
def enum_gens(Ab: NDArray, Ab_eq: Optional[NDArray] = None) -> tuple[NDArray, NDArray]:
    """Enumerate the vertices and rays of a polytope defined by its facets using the double description method.
    
    Parameters
    ----------
    Ab : NDArray
        A matrix of size `(m, n + 1)` containing `m` facet normals.
    Ab_eq : NDArray | None, default=None
        A n_eq x n + 1 array of m_eq facet normals corresponding to equalities.
    
    Returns
    -------
    verts : NDArray
        A (k, n) array of k vertices in n-dimensional space.
    rays : NDArray
        A (k_rays, n) array of k_rays rays in n-dimensional space.
    
    """
    if not CDD_INSTALLED:
        raise ImportError("The package 'pycddlib' is not installed. Please install it to enable converting from H-representation to V-representation.")

    # Validate dimension consistency between Ab and Ab_eq
    if Ab_eq is not None and Ab.shape[0] > 0 and Ab_eq.shape[0] > 0 and Ab.shape[1] != Ab_eq.shape[1]:
        raise ValueError(f"Both Ab and Ab_eq should have the same number of columns n + 1," \
        f" but received Ab.shape={Ab.shape} and Ab_eq.shape={Ab_eq.shape}")

    # Special case: Unconstrained space (entire plane/space)
    if Ab.size == 0 and (Ab_eq is None or Ab_eq.size == 0):
        n = Ab.shape[1] - 1
        return np.empty((0, n)), np.vstack([np.eye(n), -np.eye(n)])

    # Convert to cdd format (b, -A)
    n_cols_ab, n_cols_eq = (Ab.shape[1]
                            if Ab.ndim >= 2
                            else None,
                            Ab_eq.shape[1]
                            if Ab_eq is not None and Ab_eq.ndim >= 2
                            else None)
    cdd_ineq = (np.column_stack((Ab[:, -1], -Ab[:, :-1]))
                if Ab.shape[0] > 0
                else np.empty((0, n_cols_ab or n_cols_eq or 0)))

    if Ab_eq is None:
        cdd_eq = np.empty((0, (Ab.shape[1] - 1 if Ab.size > 0 else 0) + 1))
    elif Ab_eq.shape[0] > 0:
        cdd_eq = np.column_stack((Ab_eq[:, -1], -Ab_eq[:, :-1]))
    else:
        cdd_eq = np.empty((0, Ab_eq.shape[1] + (1 if Ab_eq.shape[1] == Ab.shape[1] - 1 else 0)))

    # Combine constraints and create cdd matrix
    cdd_input = (np.vstack((cdd_ineq, cdd_eq))
                 if cdd_ineq.shape[0] > 0 and cdd_eq.shape[0] > 0
                 else (cdd_ineq
                       if cdd_ineq.shape[0] > 0
                       else (cdd_eq
                             if cdd_eq.shape[0] > 0
                             else (cdd_ineq
                                   if cdd_ineq.shape[1] > 0
                                   else cdd_eq))))
    mat = cdd.matrix_from_array(cdd_input.astype(float).tolist(), rep_type=cdd.RepType.INEQUALITY)
    if cdd_eq.shape[0] > 0:
        mat.lin_set = set(range(cdd_ineq.shape[0], cdd_ineq.shape[0] + cdd_eq.shape[0]))

    # Get generators from cdd
    gen_obj = cdd.copy_generators(cdd.polyhedron_from_matrix(mat))
    gmat, lin_set = np.array(gen_obj.array).astype(float), gen_obj.lin_set

    # Special case: Empty polytope (infeasible constraints)
    if (gmat.ndim == 1 or gmat.shape[0] == 0) and cdd_input.shape[0] > 0:
        n = cdd_input.shape[1] - 1 if cdd_input.shape[1] > 0 else 0
        return np.empty((0, n)), np.empty((0, n))

    # BUG: It seems pycddlib does NOT always return exactly 0 or 1 in it's first column
    # to indicate vertices vs rays, but can return -5.30825384e-16
    verts_mask = np.abs(gmat[:, 0] - 1.0) < 1E-12
    ray_mask = np.abs(gmat[:, 0]) < 1E-12
    verts = gmat[verts_mask, 1:] if np.any(verts_mask) else np.empty((0, gmat.shape[1]-1))
    rays_and_lines = gmat[ray_mask, 1:] if np.any(ray_mask) else np.empty((0, gmat.shape[1]-1))

    # Handle bidirectional lines (convert to pairs of opposite rays)
    if lin_set:
        ray_list = []
        for i, ray_idx in enumerate(np.where(ray_mask)[0]):
            if ray_idx in lin_set:
                ray_list.extend([rays_and_lines[i], -rays_and_lines[i]])
            else:
                ray_list.append(rays_and_lines[i])
        rays = np.array(ray_list) if ray_list else np.empty((0, gmat.shape[1]-1))
    else:
        rays = rays_and_lines

    # Ensure origin vertex exists if only rays found
    if verts.shape[0] == 0 and rays.shape[0] > 0:
        verts = np.zeros((1, gmat.shape[1] - 1))

    return verts, rays


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/04/14[untested/unverified]
def enum_facets(verts: NDArray, rays: Optional[NDArray] = None) -> tuple[NDArray, NDArray]:
    """Enumerate the facets of a polytope defined by its vertices using the double description method.
    
    Parameters
    ----------
    verts : NDArray
        A (k, n) array of k vertices in n-dimensional space.
    rays : NDArray | None, default=None
        A (k_rays, n) array of k_rays rays in n-dimensional space.
        
    Returns
    -------
    Ab : NDArray
        An (m, n + 1) array of m facet normals.
    Ab_eq : NDArray
        An (m_eq, n + 1) array of m_eq facet normals corresponding to equalities.
    """
    if not CDD_INSTALLED:
        raise ImportError("The package 'pycddlib' is not installed. Please install it to enable converting from H-representation to V-representation.")
    
    # Validate input dimensions
    if verts.ndim != 2:
        raise ValueError(f"Verts should be a 2D array of shape (k, n), but received verts.shape={verts.shape}")
    if rays is not None and rays.ndim != 2:
        raise ValueError(f"Rays should be a 2D array of shape (k_rays, n), but received rays.shape={rays.shape}")

    # Validate dimension consistency between verts and rays
    if rays is not None and verts.shape[1] != rays.shape[1]:
        raise ValueError(f"Both verts and rays should have the same number of columns n," \
                         f" but received verts.shape={verts.shape} and rays.shape={rays.shape}")

    # Special case: Empty polytope (no vertices and no rays)
    if verts.size == 0 and (rays is None or rays.size == 0):
        n = verts.shape[1] if verts.ndim >= 2 else (rays.shape[1] if rays is not None and rays.ndim >= 2 else 1)
        return np.array([[0.0] * n + [-1.0]]), np.empty((0, n + 1))  # Inconsistent constraint: 0 @ x <= -1

    # Create cdd matrix from generators
    if rays is not None and rays.size > 0:
        generators = np.vstack((verts, rays)) if verts.size > 0 else rays
        ray_flags = np.concatenate((np.ones(verts.shape[0]),
                                    np.zeros(rays.shape[0]))) if verts.size > 0 else np.zeros(rays.shape[0])
        mat = cdd.matrix_from_array(
            np.column_stack((ray_flags, generators)).astype(float).tolist(),
            rep_type=cdd.RepType.GENERATOR,
        )
    else:
        mat = cdd.matrix_from_array(
            np.column_stack((np.ones(verts.shape[0]), verts)).astype(float).tolist(),
            rep_type=cdd.RepType.GENERATOR,
        )

    # Get inequalities from cdd
    ineq = cdd.copy_inequalities(cdd.polyhedron_from_matrix(mat))
    hmat, lin_set, n_cols = np.array(ineq.array).astype(float), ineq.lin_set, verts.shape[1] + 1

    # Separate equalities and inequalities
    eq_mask = np.array([idx in lin_set for idx in range(hmat.shape[0])], dtype=bool)
    h_ineq, h_eq = ((hmat[~eq_mask], hmat[eq_mask])
                    if np.any(~eq_mask) and np.any(eq_mask)
                    else ((hmat[~eq_mask], np.empty((0, n_cols)))
                          if np.any(~eq_mask)
                          else (np.empty((0, n_cols)), hmat[eq_mask])
                          if np.any(eq_mask)
                          else (np.empty((0, n_cols)), np.empty((0, n_cols)))))

    # Convert format: cdd [b, -A] -> our [A, b] and filter trivial constraints
    Ab = np.column_stack((-h_ineq[:, 1:], h_ineq[:, 0])) if h_ineq.size else np.empty((0, n_cols))
    if Ab.size:
        A, b = Ab[:, :-1], Ab[:, -1]
        trivial = np.isclose(A, 0, rtol=CFG.rtol, atol=CFG.atol).all(axis=1) \
            & ((b >= 0) | np.isclose(b, 0, rtol=CFG.rtol, atol=CFG.atol))
        Ab = Ab[~trivial]

    Ab_eq = np.column_stack((h_eq[:, 1:], -h_eq[:, 0])) if h_eq.size else np.empty((0, n_cols))

    return Ab, Ab_eq


# FROM: GitHub Copilot ChatGPT 4.1 | 2026/01/28[unverified]
def conv(verts: NDArray) -> NDArray:
    """Compute the convex hull of a set of points given by `verts`
    
    Parameters
    ----------
    verts : NDArray
        A (k, n) array of k points in n-dimensional space.
        
    Returns
    -------
    hull_verts : NDArray
        A (k', n) array of k' <= k vertices that form the convex hull of the input points.
    """
    # Special case: zero or one point
    if verts.shape[0] <= 1:
        return verts

    # Zero tiny values for robustness
    verts_clean = np.array(verts, dtype=np.float64, copy=True)
    verts_clean[np.abs(verts_clean) < CFG.atol] = 0.0

    centered = verts_clean - np.mean(verts_clean, axis=0)
    rank = np.linalg.matrix_rank(centered, tol=CFG.atol)
    if rank == 0:
        return verts[0:1, :]  # To maintain 2D shape, use original
    if rank < verts_clean.shape[1]:
        # Points lie on a lower-dimensional manifold - project to intrinsic dimension
        _, _, Vh = np.linalg.svd(centered, full_matrices=False)
        coords_proj = centered @ Vh[:rank, :].T  # Project to rank-dimensional subspace
        if rank == 1:
            # For collinear points, find extremes along the line
            idx_min, idx_max = np.argmin(coords_proj[:, 0]), np.argmax(coords_proj[:, 0])
            return verts[np.unique([idx_min, idx_max])]
        proj_rank = np.linalg.matrix_rank(coords_proj, tol=CFG.atol)
        n_unique_points = len(np.unique(coords_proj, axis=0))
        if proj_rank == rank and n_unique_points > rank and coords_proj.shape[0] > rank:
            # Projected coordinates are well-conditioned for ConvexHull
            hull = sp.spatial.ConvexHull(coords_proj)  # pylint: disable=no-member
            return verts[hull.vertices]
        # Projected coordinates are still degenerate - use extremes fallback
        extremes = []
        for i in range(rank):
            idx_min, idx_max = np.argmin(coords_proj[:, i]), np.argmax(coords_proj[:, i])
            extremes.extend([idx_min, idx_max])
        return verts[np.unique(extremes)]
    # 1D case
    if verts_clean.shape[1] == 1:
        # For 1D case, find min and max points
        idx_min, idx_max = np.argmin(verts_clean[:, 0]), np.argmax(verts_clean[:, 0])
        return verts[np.unique([idx_min, idx_max])]
    hull = sp.spatial.ConvexHull(verts_clean)  # pylint: disable=no-member
    return verts[hull.vertices]


def signed_angle(v_1: NDArray, v_2: NDArray, look: Optional[NDArray] = None) -> float:
    """Compute the signed angle between two vectors `v_1` and `v_2` in 2D or 3D space.
    
    Parameters
    ----------
    v_1 : NDArray
        First vector of shape (2,) or (3,).
    v_2 : NDArray
        Second vector of shape (2,) or (3,).
    look : Optional[NDArray], default=None
        Reference direction for determining sign. If not provided, defaults to
        [0, 0, 1] for both 2D and 3D vectors. Must be non-zero if provided.

    Returns
    -------
    signed_angle : float
        Signed angle in radians, range (-π, π]. If either `v_1` or `v_2` is zero, the result
        is undefined and `np.nan` is returned. Counter-clockwise rotation from `v_1` to `v_2`
        is positive when viewed from the `look` direction.
        
    Raises
    ------
    ValueError
        If vectors have different sizes, are not 2D/3D, are zero vectors, if look vector is not 
        3-dimensional when provided, or if look vector is zero.
    """
    if v_1.size != v_2.size:
        raise ValueError("Both input vectors must have the same size")
    if v_1.size not in {2, 3}:
        raise ValueError("Signed angle is only implemented for 2D and 3D vectors")
    if look is not None and look.size != 3:
        raise ValueError("Look vector must be 3-dimensional if provided")
    if look is not None and np.allclose(look, 0):
        raise ValueError("Look vector must be non-zero")
    if np.allclose(v_1, 0) or np.allclose(v_2, 0):
        return np.nan  # Undefined angle if either vector is zero

    v_1_norm, v_2_norm = v_1 / np.linalg.norm(v_1), v_2 / np.linalg.norm(v_2)
    dot_prod = np.clip(np.dot(v_1_norm, v_2_norm), -1.0, 1.0)
    angle = np.arccos(dot_prod)

    # Robust sign computation
    if v_1.size == 2:
        sign_val = v_1_norm[0] * v_2_norm[1] - v_1_norm[1] * v_2_norm[0]
        sign = np.sign(sign_val)
        # If look is provided in 2D, check if it points in negative z direction
        if look is not None and look.size >= 3 and look[2] < 0:
            sign = -sign
        # If sign is zero due to collinearity, always return +|angle|
        if np.isclose(sign_val, 0, atol=CFG.atol, rtol=CFG.rtol):
            sign = 1
    else:
        if look is None:
            look = np.array([0.0, 0.0, 1.0])
        cross_prod = np.cross(v_1_norm, v_2_norm)
        cross_dot = np.dot(cross_prod, look / np.linalg.norm(look))
        # If cross product is nearly zero, treat as collinear and force sign=1
        if np.isclose(cross_dot, 0, atol=CFG.atol, rtol=CFG.rtol):
            sign = 1
        else:
            sign = np.sign(cross_dot)

    return float(sign * angle)
