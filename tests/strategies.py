"""Script containing strategies from hypothesis used for testing"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import numpes as pes
from hypothesis import reject
from hypothesis import strategies as st
from hypothesis.strategies import integers, floats, lists, sampled_from
from hypothesis.extra.numpy import arrays
from tests.conftest import RTOL, ATOL, N_MAX

if TYPE_CHECKING:
    from typing import Literal, Optional
    from collections.abc import Collection
    from numbers import Number
    from numpes import Polytope


class Decimal:
    """Marker type for metric submultiple fractions partitioning the unit interval into `RESOLUTION` numbers"""
    RES = 1_000


@st.composite
def poly_rand(draw, repr: Literal['vrepr', 'hrepr', 'both'], n: int, exclude_degen: bool = False) -> Polytope:
    # TODO: Implement a check that the generated polytope is not degenerate if `exclude_degen=True`
    match repr:
        case 'vrepr':
            num_verts = draw(st.integers(n + 1, n + 10))
            verts = draw(arrays(float, (num_verts, n), elements=st.floats(-100, 100, allow_infinity=False, allow_nan=False)))
            try:  # FIXME: I don't know if this is a very good strategy; this was a fix for a QHull error
                poly = pes.Polytope(verts)  # FIXME: Maybe I should do `with pes.algo_options(on_prop_assign='pass')` instead
            except RuntimeError as _:
                reject()
        case 'hrepr':
            num_facets = draw(st.integers(n + 1, n + 10))
            A = draw(arrays(float, (num_facets, n), elements=st.floats(-100, 100, allow_infinity=False, allow_nan=False)))
            b = draw(arrays(float, (num_facets,), elements=st.floats(-100, 100, allow_infinity=False, allow_nan=False)))
            try:  # FIXME: I don't know if this is a very good strategy; this was a fix for a QHull error
                poly = pes.Polytope(A, b)
            except (RuntimeError, ValueError) as _:  # FIXME: Can we log this error instead?
                reject()
            if exclude_degen:  # FIXME: We need more checks here
                if poly.is_empty:
                    reject()
        case 'both':
            num_verts = draw(st.integers(n + 1, n + 10))
            verts = draw(arrays(float, (num_verts, n), elements=st.floats(-100, 100, allow_infinity=False, allow_nan=False)))
            try:
                Ab, _ = pes.utils.enum_facets(verts)
            except RuntimeError as _:
                reject()
            poly = pes.Polytope(n=n)  # FIXME: This should NOT be the way to initialize! This sets a lot of private variables
            poly._vrepr = (verts, np.empty((0, n)))
            poly._hrepr = (Ab, np.empty((0, n + 1)))
        case _:
            raise ValueError(f"Unknown representation type '{repr}' specified for polytope strategy (must be one of 'both', 'vrepr', or 'hrepr')")
    return poly


@st.composite
def polytopes(draw,
              n: Optional[int | Collection[int]] = None,
              which_repr: Optional[Literal['vrepr', 'hrepr', 'both']] = None,
              dtype: type[Number] | type[Decimal] = Decimal,
              *,
              is_bounded: bool = True,
              is_full_dim: bool = True,
              is_minimal: bool = True,
              dim: Optional[int] = None,
              ) -> Polytope:
    """Strategy for generating random polytopes. Polytopes can be specified for any dimensions `n >= 1`, and in either V-representation, H-representation, or both.
    
    Parameters
    ----------
    n : int, optional
        Dimension of the  ambient space. If `None`, a dimension will be chosen in the range [1, N_MAX].
    which_repr : 'vrepr', 'hrepr', or 'both', optional
        Which representation to use for defining the polytope. If `None`, either 'vrepr' or 'hrepr' will be chosen. When 'both' is selected, a V-representation is constructed first from which a (minimal) H-representation is computed. Note that for the latter, the package pycddlib must be installed.
    dtype : int, float, or Microdecimal, default=Microdecimal
        Which numerical representation to use in the arrays. Default is Microdecimal, meaning fixed step sizes of 1E-6 are taken.
    is_bounded : bool, default=True
        Whether to only generate bounded polytopes (see Warnings)
    is_full_dim : bool, default=True
        Whether to only generate full dimensional polytopes (see Warnings)
    is_minimal : bool, default=True
        Whether to minimize the representation or not
    dim : int or None, default=None
        The specific dimension of the polytope to generate (see Warnings). If None, the full dimension `n` or a random lower dimension `n'` in [1, ..., n - 1] is chosen, for `is_full_dim=True` and `is_full_dim=False`, respectively.

    Raises
    ------
    ValueError
        If `n` is not a positive integer, if `which_repr` is not 'vrepr', 'hrepr', or 'both', or if `dim` is not a positive integer or larger then `n` 
    TypeError
        If `dtype` is not a valid numeric datatype

    Warnings
    --------
    # FIXME: I don't think this is actually correct anymore?
    By default (`enforce = False`), some assumptions on the polytope are not actually enforced, but statistically highly unlikely to occur during testing (due to the method of construction). The chance of occurrence increases (but should still be small) when `dtype=int` is selected. As these properties can be relatively expensive to check *a-priori*, this is not done by default. The (exhaustive list of) combinations for which this applies are:
    - `which_repr='vrepr', is_full_dim='True', ...` (and similarly for `which_repr='both', is_full_dim='True', ...`)
    - `which_repr='hrepr', is_bounded='True', ...`
    - `which_repr='hrepr', is_bounded='False', ...`
    - `which_repr='hrepr', is_full_dim='True', ...`
    - `dim=a`, where `a: int`

    If one would still like to enforce this assumption to hold for every generated polytope, one can select `enforce = True`. Alternatively, the following can be added at the beginning of each test:
    >>> from hypothesis import assume  # doctest: +SKIP
    >>> @given(poly=polytopes(...))  # doctest: +SKIP
    >>> def test_foo(poly: Polytope) -> None:  # doctest: +SKIP
    ...     assume(poly.is_full_dim)  # Or similar, like `poly.is_bounded` or `not poly.is_bounded`

    Otherwise, for better performance, one can only check this assumption after a test case has failed, performing the expensive property verification only when it might have caused a failing test result. Note that this should be used at your own risk, as it will not catch polytopes that did not satisfy the assumption in the first place, yet still passed the test (when they possibly should not). The following should also not block hypothesis shrinking behavior when looking for a counter example, although this is not confirmed.
    >>> from hypothesis.errors import UnsatisfiedAssumption  # doctest: +SKIP
    >>> @given(poly=polytopes(...))  # doctest: +SKIP
    >>> def test_foo(poly: Polytope) -> None:  # doctest: +SKIP
    ...     try:
    ...         assert ..., "Test case did not pass"
    ...     except AssertionError:
    ...         if not poly.is_full_dim:  # Or similar, like `not poly.is_bounded` or `poly.is_bounded`
    ...             raise UnsatisfiedAssumption()
    ...         raise  # Otherwise, re-raise the original assertion error

    Notes 
    -----
    For vertices, the probability of the resulting polytope not being full dimensional happens when all points happen to lie on a lower-dimensional manifold. This depends on two things, namely the number of possible options for a point to choose from, and the total number of points sampled. Based on the datatype and the default bounds [-100, 100], the total number of options is resolution^n, where resolution ~ 200 for `dtype=int`, and resolution ~ 10^18 for `dtype=float`.

    One extremely important caveat, however, is that `hypothesis.extra.numpy.arrays` does not sample points uniformly from this domain. As such, one might encounter far more violations of the assumption in practice then one would expect by uniform-like sampling.
    """
    # FIXME: Make it such that doctest actually doesn't test the above code.
    if n is not None:
         if not isinstance(n, int):
             n = draw(sampled_from(n))
         elif n < 1:
            raise ValueError(f"'n' must be a positive integer, received {n} or type '{type(n)}'")
    else:
        n = draw(integers(1, N_MAX))
    if dim is not None:
        if dim < n and is_full_dim:
            raise ValueError(f"Dimension 'dim' cannot be set smaller then 'n' when is_full_dim=True, received n={n}, dim={dim}")
        if dim == n and not is_full_dim:
            raise ValueError(f"Dimension 'dim' cannot be set equal to 'n' when is_full_dim=False, received n={n}, dim={dim}")
        if dim == 0 and not is_bounded:
            raise ValueError(f"Dimension 'dim' cannot be set to 0 when is_bounded=False")
    else:
        dim = draw(integers(0, n - 1)) if not is_full_dim else n
    if not is_full_dim and not is_bounded and dim == 0:
        reject()
    if which_repr is None:
        which_repr = draw(sampled_from(('vrepr', 'hrepr')))
    MAX_VAL = 100
    match which_repr:
        case 'vrepr' | 'both':
            num_verts = (dim + draw(integers(1, dim ** 2))
                         if dim != 0
                         else 1)
            bound = (MAX_VAL
                     if is_full_dim
                     else np.sqrt(MAX_VAL / max((dim, 1))))
            if issubclass(dtype, int):
                bound = int(bound)
            elements = (integers(int(-bound * Decimal.RES),
                                 int( bound * Decimal.RES)).map(lambda value: value / Decimal.RES)
                        if dtype is Decimal
                        else floats(-bound, bound)
                        if issubclass(dtype, float)
                        else integers(-bound, bound))
            verts = draw(arrays(float if dtype is Decimal else dtype,
                                (num_verts, max((dim, 1))),
                                elements=elements)
                                .filter(lambda verts: np.linalg.matrix_rank(verts[0] - verts[1:], tol=ATOL) == dim
                                        if dim != 0
                                        else True))
            if not is_full_dim:
                M = draw(arrays(float if dtype is Decimal else dtype,
                                (n, max((dim, 1))),
                                elements=elements)
                                .filter(lambda M: np.linalg.matrix_rank(M, tol=ATOL) == dim
                                        if dim != 0
                                        else True))
                verts = verts @ M.T
            if not is_bounded:
                num_rays = draw(integers(1, dim))
                if not is_full_dim:
                    rays = M[:, draw(lists(integers(0, dim - 1),
                                           min_size=num_rays,
                                           max_size=num_rays))].T
                else:
                    rays = draw(arrays(float if dtype is Decimal else dtype,
                                       (num_rays, n),
                                       elements=elements)
                                       .filter(lambda rays: not np.allclose(rays, 0, rtol=RTOL, atol=ATOL)))
            else:
                rays = np.empty((0, n))
            with pes.algo_options(on_property_assign='minimal' if is_minimal else 'pass'):
                poly = pes.poly(verts, rays=rays)
            if which_repr == 'both':
                try:  # FIXME: This is really not working; FAR to many polytopes are rejected, especially for higher dimensions
                    _ = poly.hrepr
                except RuntimeError as e:
                    if str(e).strip() == \
                        "*Error: Numerical inconsistency is found.  Use the GMP exact arithmetic.":
                        reject()
                    raise
        case 'hrepr':
            repr_dtype = int if issubclass(dtype, int) else float
            elements = (integers(int(-MAX_VAL * Decimal.RES),
                                 int(MAX_VAL * Decimal.RES)).map(lambda value: value / Decimal.RES)
                                 if dtype is Decimal
                                 else floats(-MAX_VAL, MAX_VAL)
                                 if issubclass(dtype, float)
                                 else integers(-MAX_VAL, MAX_VAL))
            if dim == 0:
                Ab = np.empty((0, n + 1))
            else:
                num_planes = draw(integers(n, n ** 3))
                Ab_box = np.block([[ np.eye(n, dtype=repr_dtype), np.full(n, MAX_VAL, dtype=repr_dtype)[:, np.newaxis]],
                                   [-np.eye(n, dtype=repr_dtype), np.full(n, MAX_VAL, dtype=repr_dtype)[:, np.newaxis]]])
                coordinate, sign = 0, 0
                if not is_bounded:  # Remove some planes from the bounding box
                    selected = draw(lists(integers(0, 2 * n - 1), min_size=1, max_size=n, unique=True))
                    Ab_box = Ab_box[selected]
                    coordinate, sign = draw(sampled_from([(j, 1) for j in range(n) if np.all(Ab_box[:, j] <= ATOL)]
                                                         + [(j, -1) for j in range(n) if np.all(Ab_box[:, j] >= -ATOL)]))
                A_extra = draw(arrays(repr_dtype,
                                      (num_planes, n),
                                      elements=elements)
                                      .filter(lambda A: np.all(np.linalg.norm(A, axis=1) > (0 if dtype is int else ATOL)))
                                      .map((lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
                                            if dtype is not int
                                            else A))  # Normalize to direction vector on the unit hyperspere
                                      .filter(lambda A: np.all(sign * A[:, coordinate] <= (0 if dtype is int else ATOL))
                                              if not is_bounded
                                              else True))
                b_extra = draw(arrays(repr_dtype,
                                      num_planes,
                                      elements=(integers(1, MAX_VAL)
                                                if dtype is int
                                                else floats(0.1, 0.9))))
                Ab = np.vstack((Ab_box, np.column_stack((A_extra, b_extra))))
            if not is_full_dim:
                if dim == 0:
                    A_eq = np.eye(n, dtype=repr_dtype)
                    b_eq = draw(arrays(repr_dtype, n, elements=elements))
                else:
                    num_eq_columns = n - 1 if not is_bounded else n
                    A_eq = draw(arrays(repr_dtype,
                                       (n - dim, num_eq_columns),
                                       elements=elements)
                                       .filter(lambda A: np.linalg.matrix_rank(A) == n - dim))
                    if not is_bounded:
                        A_eq = np.insert(A_eq, coordinate, 0, axis=1)
                    b_eq = np.zeros(n - dim, dtype=repr_dtype)
                Ab_eq = np.column_stack((A_eq, b_eq))
            else:
                Ab_eq = np.empty((0, n + 1))
            with pes.algo_options(on_property_assign='minimal' if is_minimal else 'pass'):  # FIXME: I think 'minimal' is extremely slow here; we need to look into that
                poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1])
        case _:
            raise ValueError(f"Unknown representation type '{which_repr}' specified for polytope strategy (must be one of 'both', 'vrepr', or 'hrepr')")
    return poly


@st.composite
def poly_rand_pair(draw, repr: Literal['vrepr', 'hrepr', 'both'], n: int | tuple[int, int], same_n: bool = True, same_repr: bool = True) -> tuple[Polytope, Polytope]:
    if isinstance(n, tuple):
        if not same_n:
            n_1, n_2 = draw(st.integers(n[0], n[1])), draw(st.integers(n[0], n[1]))
        else:
            n = draw(st.integers(n[0], n[1]))
            n_1, n_2 = n, n
    poly_1 = draw(poly_rand(repr=repr, n=n_1))
    poly_2 = draw(poly_rand(repr=repr, n=n_2))
    return poly_1, poly_2