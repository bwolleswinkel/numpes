"""Script containing strategies from hypothesis used for testing"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import numpes as pes
from hypothesis import reject
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

if TYPE_CHECKING:
    from typing import Literal
    from numpes import Polytope


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
        case 'both':
            num_verts = draw(st.integers(n + 1, n + 10))
            verts = draw(arrays(float, (num_verts, n), elements=st.floats(-100, 100, allow_infinity=False, allow_nan=False)))
            try:
                Ab, _ = pes.utils.enum_facets(verts)
            except RuntimeError as _:
                reject()
            poly = pes.Polytope(n=n)
            poly._vrepr = (verts, np.empty((0, n)))
            poly._hrepr = (Ab, np.empty((0, n + 1)))
        case _:
            raise ValueError(f"Unknown representation type '{repr}' specified for polytope strategy (must be one of 'both', 'vrepr', or 'hrepr')")
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