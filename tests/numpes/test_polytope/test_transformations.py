"""Tests for transformations of the Polytope class"""

import numpy as np
from numpes._config import CFG

from tests.helpers import lsort, approx


# FIXME: Change to all archetypes
def test_polytope_hrepr_archetypes_enum_verts(poly_arch_nondegen_hrepr_2d):
    poly, poly_data = poly_arch_nondegen_hrepr_2d
    assert lsort(poly.verts) == approx(lsort(poly_data.verts)), \
        f"Expected vertices\n{poly_data.verts}\nto be equal to\n{poly.verts}\n(same rows, order does not matter)"


# FIXME: Change to all archetypes
def test_polytope_vrepr_archetypes_enum_facets(poly_arch_nondegen_vrepr_2d):    
    poly, poly_data = poly_arch_nondegen_vrepr_2d
    excepted_Ab = np.column_stack([poly_data.A, poly_data.b])
    assert lsort(poly.Ab) == approx(lsort(excepted_Ab)), \
        f"Expected matrix\n{poly_data.A}\nto be equal to\n{poly.A}\n, and vector\n{poly_data.b}\nto be equal to\n{poly.b}\n(same rows, order does not matter)"