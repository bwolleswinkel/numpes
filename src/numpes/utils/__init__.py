"""The utils subpackage contains utility functions for the NumPES package. These functions are 
not specific to any particular control algorithm, but are used by multiple algorithms in the package.

Modules
-------
linalg
    Provides several additional linear algebra functionalities.
spatial
    Provides several spatial functionalities related to polyhedral computations.
timeout
    Provides a context manager to set a timeout for executing code.
linprog
    Provides handlers for solving linear programs using either SciPy, CVXPY, or PuLP as a backend.
"""

from numpes.utils.linalg import angle_2d, angles_3d, angles_3d_convert, angles_givens, find_implicit, givens_mat, is_posdef, is_rot_mat, is_sing, is_square, is_sym, minimize_hrepr, minimize_vrepr, reduce_eq, reduce_ineq, reduce_rays, reduce_verts, rot_mat, rot_mat_2d, rot_mat_3d, span
from numpes.utils.linprog import OptimizationProgramResult, Status, solve_lp
from numpes.utils.plot import add_1d_subplot, plot_box, plot_line, plot_plane, plot_vector
from numpes.utils.spatial import conv, enum_facets, enum_gens, signed_angle
