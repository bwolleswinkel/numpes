"""Test functions for the spatial module of the `utils` subpackage"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, tuples

import numpes as pes
from tests.conftest import ATOL, N_MAX
from tests.helpers import approx, deg2rad, normalize, rad2deg, lsort

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from numpes import Polytope
    from tests.data.archetypes_polytope import PolytopeData

# Mark all tests in this file as utils tests
pytestmark = [pytest.mark.utils, pytest.mark.unit]

# =====================
# (function) enum_verts
# =====================

# TODO: Do I also want to test for np.inf input in the arrays? Or np.nan inputs in the arrays? Or incorrect type of inputs, for instance, `Ab = [[1, 2, 3]]` or `Ab = ...`?

# TODO: Implement check when Ab.ndims != 2 as raises the error


@pytest.mark.parametrize('Ab, Ab_eq', [
    (np.array([[1, 0, 0]]),
     np.array([[0, 0]])),
    (np.array([[0, 0]]),
     np.array([[1, 0, 1]])),
    (np.array([[0, 0, 1, 2]]),
     np.array([[1, 0, 1],
               [0, 1, 1]]))
])
def test_enum_verts_parameterize_inconsistent_dimensions_value_error(Ab: NDArray, Ab_eq: NDArray):
    with pytest.raises(ValueError, match=re.escape(f"Both Ab and Ab_eq should have the same number of columns n + 1, but received Ab.shape={Ab.shape} and Ab_eq.shape={Ab_eq.shape}")):
        _ = pes.utils.enum_gens(Ab, Ab_eq)


@pytest.mark.enum_verts
@pytest.mark.parametrize('Ab, expected_verts', [
    (np.array([[ 1,  0, 1],
               [ 0,  1, 1],
               [-1,  0, 0],
               [ 0, -1, 0]]),
     np.array([[0, 0],
               [1, 0],
               [0, 1],
               [1, 1]])),
    (np.array([[ 1,  1,  2  ],
               [-1,  0, -0.5],
               [ 0, -1, -0.5]]),
     np.array([[0.5, 0.5],
               [0.5, 1.5],
               [1.5, 0.5]]))
])
def test_enum_verts_nondegen_2d(Ab: NDArray, expected_verts: NDArray):
    verts, rays = pes.utils.enum_gens(Ab)
    assert lsort(verts) == approx(lsort(expected_verts)), \
        f"Expected vertices\n{expected_verts}\nto be equal to\n{verts}\n(same rows, order does not matter)"
    assert rays.size == 0, \
        f"Expected no rays, but got {rays}"


@pytest.mark.enum_verts
@pytest.mark.parametrize('Ab, Ab_eq, expected_verts, expected_rays', [
    (np.array([[-1,  0, 0],
               [ 0, -1, 0]]), 
     np.empty((0, 3)), 
     np.array([[0, 0]]),
     np.array([[1, 0],
               [0, 1]])), 
    (np.array([[-1,  0, 0]]), 
     np.array([[1, -1, 0]]),
     np.array([[0, 0]]),
     np.array([[1, 1]]))
])
def test_enum_verts_unbounded_2d(Ab: NDArray, Ab_eq: NDArray, expected_verts: NDArray, expected_rays: NDArray):
    verts, rays = pes.utils.enum_gens(Ab, Ab_eq)
    assert lsort(verts) == approx(lsort(expected_verts)), \
        f"Expected vertices\n{expected_verts}\nto be equal to\n{verts}\n(same rows, order does not matter)"
    assert lsort(rays) == approx(lsort(expected_rays)), \
        f"Expected rays\n{expected_rays}\nto be equal to\n{rays}\n(same rows, order does not matter)"


@pytest.mark.enum_verts
@pytest.mark.parametrize('Ab, Ab_eq, k, expected_rays', [
    (np.array([[ 1, 0, 1],
               [-1, 0, 1]]), 
     np.empty((0, 2)),
     2,
     np.array([[0,  1],
               [0, -1]])) 
])
def test_enum_verts_non_unique_2d(Ab: NDArray, Ab_eq: NDArray, k: int, expected_rays: NDArray):
    verts, rays = pes.utils.enum_gens(Ab, Ab_eq)
    assert verts.shape == (k, Ab.shape[1] - 1), \
        f"Expected {k} vertices, but got {verts.shape[0]}"
    assert lsort(rays) == approx(lsort(expected_rays)), \
        f"Expected rays\n{expected_rays}\nto be equal to\n{rays}\n(same rows, order does not matter)"


@pytest.mark.enum_verts
@pytest.mark.parametrize('Ab, Ab_eq, expected_verts', [
    (np.empty((0, 2)), 
     np.array([[1, 0, 0],
               [0, 1, 0]]),
     np.array([[0, 0]])),
    (np.array([[ 1,  0, 0],
               [ 0,  1, 0],
               [-1,  0, 0],
               [ 0, -1, 0]]),
     np.empty((0, 2)),
     np.array([[0, 0]])),
    (np.empty((0, 2)), 
     np.array([[1, 0, -0.2],
               [0, 1,  0.5]]),
     np.array([[-0.2, 0.5]]))
])
def test_enum_verts_singleton_2d(Ab: NDArray, Ab_eq: NDArray, expected_verts: NDArray):
    verts, rays = pes.utils.enum_gens(Ab, Ab_eq)
    assert lsort(verts) == approx(lsort(expected_verts)), \
        f"Expected vertices\n{expected_verts}\nto be equal to\n{verts}\n(same rows, order does not matter)"
    assert rays.size == 0, \
        f"Expected no rays, but got {rays.size}"


@pytest.mark.enum_verts
@given(n=integers(min_value=1, max_value=N_MAX))
def test_enum_verts_empty_nd(n: int):
    Ab, Ab_eq = np.column_stack([np.zeros((1, n)), [-1]]), np.empty((0, n + 1))
    verts, rays = pes.utils.enum_gens(Ab, Ab_eq)
    assert verts.size == 0, \
        f"Expected no vertices, but got {verts} (verts.shape={verts.shape})"
    assert rays.size == 0, \
        f"Expected no rays, but got {rays} (rays.shape={rays.shape})"


@pytest.mark.enum_verts
@given(n=integers(min_value=1, max_value=N_MAX))
def test_enum_verts_full_space_nd(n: int):
    Ab, Ab_eq = np.empty((0, n + 1)), np.empty((0, n + 1))
    verts, rays = pes.utils.enum_gens(Ab, Ab_eq)
    I_min_I = np.vstack([np.eye(n), -np.eye(n)])
    assert verts.size == 0, \
        f"Expected no vertices, but got {verts} (verts.shape={verts.shape})"
    assert lsort(rays) == approx(lsort(I_min_I)), \
        f"Expected rays to be row-equal to [I, -I] (unit vectors in all cardinal directions), but got\n{rays}"


@pytest.mark.enum_verts
def test_enum_verts_archetypes(poly_arch_all: tuple[Polytope, PolytopeData]):
    _, poly_data = poly_arch_all
    Ab, Ab_eq = np.column_stack([poly_data.A, poly_data.b]), np.column_stack([poly_data.A_eq, poly_data.b_eq])
    verts, rays = pes.utils.enum_gens(Ab, Ab_eq)
    assert lsort(verts) == approx(lsort(poly_data.verts)), \
        f"Expected vertices\n{poly_data.verts}\nto be equal to\n{verts}\n(same rows, order does not matter)"
    assert lsort(rays) == approx(lsort(poly_data.rays)), \
        f"Expected rays\n{poly_data.rays}\nto be equal to\n{rays}\n(same rows, order does not matter)"
    

def test_enum_verts_archetypes_2d_nondegen_none_equivalent_empty(poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
    _, poly_data = poly_arch_nondegen_2d
    Ab, n = np.column_stack([poly_data.A, poly_data.b]), poly_data.n
    assert lsort(pes.utils.enum_gens(Ab)[0]) == approx(lsort(pes.utils.enum_gens(Ab, np.empty((0, n + 1)))[0])), \
        f"Expected enum_verts(Ab) and enum_verts(Ab, np.empty((0, n + 1))) to give the same vertices, but got\n{pes.utils.enum_gens(Ab)}\nand\n{pes.utils.enum_gens(Ab, np.empty((0, n + 1)))}"
    assert lsort(pes.utils.enum_gens(Ab)[0]) == approx(lsort(pes.utils.enum_gens(Ab, None)[0])), \
        f"Expected enum_verts(Ab) and enum_verts(Ab, None) to give the same vertices, but got\n{pes.utils.enum_gens(Ab)}\nand\n{pes.utils.enum_gens(Ab, None)}"


# ======================
# (function) enum_facets
# ======================

# TODO: Implement check when verts.ndims != 2 as raises the error


@pytest.mark.parametrize('verts, rays', [
    (np.array([[1, 0, 1]]),
     np.array([[0, 0]])),
    (np.array([[0, 0]]),
     np.array([[1, 0, 1]])),
    (np.array([[0, 0, 1, 2]]),
     np.array([[1, 0, 1],
               [0, 1, 1]]))
])
def test_enum_facets_parameterize_inconsistent_dimensions_value_error(verts: NDArray, rays: NDArray):
    with pytest.raises(ValueError, match=re.escape(f"Both verts and rays should have the same number of columns n, but received verts.shape={verts.shape} and rays.shape={rays.shape}")):
        _ = pes.utils.enum_facets(verts, rays)


@pytest.mark.parametrize('verts, expected_Ab', [
    (np.array([[0, 0],
               [1, 0],
               [0, 1],
               [1, 1]]),
     np.array([[ 1,  0, 1],
               [ 0,  1, 1],
               [-1,  0, 0],
               [ 0, -1, 0]])),
    (np.array([[0.5, 0.5],
               [0.5, 1.5],
               [1.5, 0.5]]),
     np.array([[ 1,  1,  2  ],
               [-1,  0, -0.5],
               [ 0, -1, -0.5]]))
])  
def test_enum_facets_nondegen_2d(verts: NDArray, expected_Ab: NDArray):
    Ab, Ab_eq = pes.utils.enum_facets(verts, np.empty((0, verts.shape[1])))
    assert lsort(normalize(Ab)) == approx(lsort(normalize(expected_Ab))), \
        f"Expected inequalities\n{expected_Ab}\nto be equal to Ab=\n{Ab}\n(same rows, order does not matter)"
    assert Ab_eq.size == 0, \
        f"Expected no equality constraints, but got {Ab_eq} (Ab_eq.shape={Ab_eq.shape})"


@pytest.mark.parametrize('verts, rays, expected_Ab, expected_Ab_eq', [
    (np.array([[0, 0]]),
     np.array([[1, 0],
               [0, 1]]), 
     np.array([[-1,  0, 0],
               [ 0, -1, 0]]),
     np.empty((0, 3))), 
    (np.array([[0, 0]]), 
     np.array([[1, 1]]),
     np.array([[-1,  0, 0]]),
     np.array([[1, -1, 0]]))
])
def test_enum_facets_unbounded_2d(verts: NDArray, rays: NDArray, expected_Ab: NDArray, expected_Ab_eq: NDArray):
    Ab, Ab_eq = pes.utils.enum_facets(verts, rays)
    assert lsort(normalize(Ab)) == approx(lsort(normalize(expected_Ab))), \
        f"Expected inequalities\n{expected_Ab}\nto be equal to Ab=\n{Ab}\n(same rows, order does not matter)"
    assert lsort(normalize(Ab_eq, eq=True)) == approx(lsort(normalize(expected_Ab_eq, eq=True))), \
        f"Expected equality constraints\n{expected_Ab_eq}\nto be equal to Ab_eq=\n{Ab_eq}\n(same rows, order does not matter)"


@pytest.mark.parametrize('verts, rays, expected_Ab_eq', [
    (np.array([[1,  np.nan]]),
     np.array([[0,  1],
               [0, -1]]),
     np.array([[1, 0, 1]]))
])
@given(y=floats(min_value=-1E6, max_value=1E6, allow_nan=False, allow_infinity=False))
def test_enum_facets_non_unique_2d(verts: NDArray, rays: NDArray, expected_Ab_eq: NDArray, y: float):
    verts[0, 1] = y
    Ab, Ab_eq = pes.utils.enum_facets(verts, rays)
    assert Ab.size == 0, \
        f"Expected no inequalities, but got Ab={Ab} (Ab.shape={Ab.shape})"
    assert lsort(normalize(Ab_eq, eq=True)) == approx(lsort(normalize(expected_Ab_eq, eq=True))), \
        f"Expected equality constraints\n{expected_Ab_eq}\nto be equal to Ab_eq=\n{Ab_eq}\n(same rows, order does not matter)"


@pytest.mark.parametrize('verts, expected_Ab_eq', [
    (np.array([[0, 0]]),
     np.array([[1, 0, 0],
               [0, 1, 0]])),
    (np.array([[-0.2, 0.5]]), 
     np.array([[1, 0, -0.2],
               [0, 1,  0.5]]))
])
def test_enum_facets_singleton_2d(verts: NDArray, expected_Ab_eq: NDArray):
    Ab, Ab_eq = pes.utils.enum_facets(verts)
    assert Ab.size == 0, \
        f"Expected no inequalities, but got Ab={Ab} (Ab.shape={Ab.shape})"
    assert lsort(normalize(Ab_eq, eq=True)) == approx(lsort(normalize(expected_Ab_eq, eq=True))), \
        f"Expected equality constraints\n{expected_Ab_eq}\nto be equal to Ab_eq=\n{Ab_eq}\n(same rows, order does not matter)"


@pytest.mark.parametrize('Ab, expected_verts', [
    (..., ...)
])  
def test_enum_facets_nondegen_3d(Ab: NDArray, expected_verts: NDArray):
    ...


@pytest.mark.parametrize('verts, rays, expected_Ab, expected_Ab_eq', [
    (..., ..., ..., ...)
])
def test_enum_facets_unbounded_3d(verts: NDArray, rays: NDArray, expected_Ab: NDArray, expected_Ab_eq: NDArray):
    ...


@pytest.mark.parametrize('verts, rays, k, expected_Ab', [
    (..., ..., ..., ...)
])
def test_enum_facets_non_unique_3d(verts: NDArray, rays: NDArray, k: int, expected_Ab: NDArray):
    ...


@pytest.mark.parametrize('verts, expected_Ab_eq', [
    (..., ...)
])
def test_enum_facets_singleton_3d(verts: NDArray, expected_Ab_eq: NDArray):
    ...


@given(n=integers(min_value=1, max_value=N_MAX))
def test_enum_facets_empty_nd(n: int):
    verts, rays = np.empty((0, n)), np.empty((0, n))
    Ab, Ab_eq = pes.utils.enum_facets(verts, rays)
    assert Ab == approx(np.column_stack([np.zeros((1, n)), [-1]])), \
        f"Expected a single inequality [0, ..., 0, -1], but got Ab={Ab} (Ab.shape={Ab.shape})"
    assert Ab_eq.size == 0, \
        f"Expected no equality constraints, but got {Ab_eq} (Ab_eq.shape={Ab_eq.shape})"


@pytest.mark.enum_facets
@given(n=integers(min_value=1, max_value=N_MAX))
def test_enum_facets_full_space_nd(n: int):
    verts, rays = np.empty((0, n)), np.vstack([np.eye(n), -np.eye(n)])
    Ab, Ab_eq = pes.utils.enum_facets(verts, rays)
    assert Ab.size == 0, \
        f"Expected no inequalities, but got Ab={Ab} (Ab.shape={Ab.shape})"
    assert Ab.shape == (0, n + 1), \
        f"Expected shape of Ab to be (0, n + 1), but got (Ab.shape={Ab.shape})"
    assert Ab_eq.size == 0, \
        f"Expected no equalities, but got Ab_eq={Ab_eq} (Ab_eq.shape={Ab_eq.shape})"
    assert Ab_eq.shape == (0, n + 1), \
        f"Expected shape of Ab_eq to be (0, n + 1), but got (Ab_eq.shape={Ab_eq.shape})"


def test_enum_facets_archetypes(poly_arch_all: tuple[Polytope, PolytopeData]):
    _, poly_data = poly_arch_all
    verts, rays = poly_data.verts, poly_data.rays
    Ab, Ab_eq = pes.utils.enum_facets(verts, rays)
    excepted_Ab, expected_Ab_eq = np.column_stack([poly_data.A, poly_data.b]), np.column_stack([poly_data.A_eq, poly_data.b_eq])
    assert lsort(normalize(Ab)) == approx(lsort(normalize(excepted_Ab))), \
        f"Expected inequalities\n{excepted_Ab}\nto be equal to Ab=\n{Ab}\n(same rows, order does not matter)"
    assert lsort(normalize(Ab_eq)) == approx(lsort(normalize(expected_Ab_eq))), \
        f"Expected equalities\n{expected_Ab_eq}\nto be equal to Ab_eq=\n{Ab_eq}\n(same rows, order does not matter)"
    

def test_enum_facets_archetypes_2d_nondegen_none_equivalent_empty(poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
    _, poly_data = poly_arch_nondegen_2d
    verts, n = poly_data.verts, poly_data.n
    assert lsort(pes.utils.enum_facets(verts)[0]) == approx(lsort(pes.utils.enum_facets(verts, np.empty((0, n)))[0])), \
        f"Expected enum_facets(verts) and enum_facets(verts, np.empty((0, n))) to give the same inequalities, but got\n{pes.utils.enum_facets(verts)}\nand\n{pes.utils.enum_facets(verts, np.empty((0, n)))}"
    assert lsort(pes.utils.enum_facets(verts)[0]) == approx(lsort(pes.utils.enum_facets(verts, None)[0])), \
        f"Expected enum_facets(verts) and enum_facets(verts, None) to give the same inequalities, but got\n{pes.utils.enum_facets(verts)}\nand\n{pes.utils.enum_facets(verts, None)}"
    

def test_invariance_enum_verts_enum_facets_minimal() -> None:
    """Tests whether enumerating the vertices and then enumerating the facets from those vertices leaves the input unchanged when there are no redundant constraints"""
    ...
    

# TODO: Add integration test where `@given(ndarray)`, we test `pes.utils.sort_rows(pes.utils.conv(verts)) == pes.utils.sort_rows(pes.utils.conv(pes.utils.enum_verts(pes.utils.enum_facets(verts, ...))))`


# ===============
# (function) conv
# ===============


class TestConv:
    """Tests for the function `pes.utils.conv`"""

    # TODO: Implement check when verts.ndims != 2 as raises the error

    @pytest.mark.parametrize('verts', [
        (np.array([[0],
                   [1]])),
        (np.array([[-1.5],
                   [ 2  ]]))
    ])
    def test_parameterize_1d_nondegen_nonredundant(self, verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[0  ],
                   [0.5],
                   [1  ]]),
         np.array([[0  ],
                   [1  ]])),
        (np.array([[-1.0],
                   [-1.5],
                   [ 2  ],
                   [ 0  ]]),
         np.array([[-1.5],
                   [ 2  ]]))
    ])
    def test_parameterize_1d_nondegen_redundant_inside(self, verts: NDArray, expected_verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(expected_verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[0  ],
                   [0  ],
                   [1  ]]),
         np.array([[0  ],
                   [1  ]])),
        (np.array([[-1.5],
                   [ 2  ],
                   [-1.5],
                   [ 2  ]]),
         np.array([[-1.5],
                   [ 2  ]]))
    ])
    def test_parameterize_1d_nondegen_redundant_on_edge(self, verts: NDArray, expected_verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(expected_verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts', [
        (np.array([[0]])),
        (np.array([[-2.5]])),
    ])
    def test_parameterize_1d_degen_singleton(self, verts: NDArray) -> None:
        assert pes.utils.conv(verts) == approx(verts), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts', [
        (np.array([[0, 0],
                   [1, 0],
                   [0, 1]])),
        (np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]]))
    ])
    def test_parameterize_2d_nondegen_nonredundant(self, verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"
        
    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[0  , 0  ],
                   [0.1, 0.1],
                   [1  , 0  ],
                   [0  , 1  ]]),
         np.array([[0, 0],
                   [1, 0],
                   [0, 1]])),
        (np.array([[0.5, 0.5],
                   [0  , 0  ],
                   [1  , 0  ],
                   [0.1, 0.2],
                   [0  , 1  ],
                   [1  , 1  ]]),
         np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]]))
    ])
    def test_parameterize_2d_nondegen_redundant_inside(self, verts: NDArray, expected_verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(expected_verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[0  , 0  ],
                   [0.5, 0.5],
                   [1  , 0  ],
                   [0  , 1  ]]),
         np.array([[0, 0],
                   [1, 0],
                   [0, 1]])),
        (np.array([[0  , 0  ],
                   [0  , 0  ],
                   [1  , 0  ],
                   [1  , 0.5],
                   [0  , 1  ],
                   [1  , 1  ]]),
         np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]]))
    ])
    def test_parameterize_2d_nondegen_redundant_on_edge(self, verts: NDArray, expected_verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(expected_verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts', [
        (np.array([[0, 0],
                   [0, 1]])),
        (np.array([[-1, -1],
                   [ 1,  1]]))
    ])
    def test_parameterize_2d_degen_nonredundant_line(self, verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[0, 0  ],
                   [0, 0.5],
                   [0, 1  ]]),
         np.array([[0, 0],
                   [0, 1]])),
        (np.array([[-1, -1],
                   [-1, -1],
                   [ 1,  1],
                   [ 0,  0]]),
         np.array([[-1, -1],
                   [ 1,  1]]))
    ])
    def test_parameterize_2d_degen_redundant_line(self, verts: NDArray, expected_verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(expected_verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts', [
        (np.array([[0, 0]])),
        (np.array([[-0.5, 1.8]]))
    ])
    def test_parameterize_2d_degen_singleton(self, verts: NDArray) -> None:
        assert pes.utils.conv(verts) == approx(verts), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    def test_parameterize_3d_nondegen_nonredundant(self) -> None:
        ...

    def test_parameterize_3d_nondegen_redundant_inside(self) -> None:
        ...

    def test_parameterize_3d_nondegen_redundant_on_edge(self) -> None:
        ...

    @pytest.mark.parametrize('verts', [
        (np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [1, 1, 0]]))
    ])
    def test_parameterize_3d_degen_nonredundant_plane(self, verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[0  , 0  , 0],
                   [0.3, 0.3, 0],
                   [1  , 0  , 0],
                   [0  , 1  , 0],
                   [1  , 1  , 0]]),
         np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [1, 1, 0]]))
    ])
    def test_parameterize_3d_degen_redundant_plane(self, verts: NDArray, expected_verts: NDArray) -> None:
        assert lsort(pes.utils.conv(verts)) == approx(lsort(expected_verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"
    
    def test_parameterize_3d_degen_line(self) -> None:
        ...

    def test_parameterize_3d_degen_singleton(self) -> None:
        ...

    @given(n=integers(min_value=1, max_value=N_MAX))
    def test_random_nd_empty(self, n: int) -> None:
        assert pes.utils.conv(np.empty((0, n))).shape == (0, n), \
            f"Expected convex hull of empty array with shape (0, n) to be empty array with shape (0, n), but got {pes.utils.conv(np.empty((0, n)))} with shape {pes.utils.conv(np.empty((0, n))).shape}"

    @given(verts=integers(
        min_value=1, max_value=N_MAX).flatmap(lambda n: arrays(float, (1, n), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))),
        num_copies=integers(min_value=1, max_value=5)
    )
    def test_random_nd_single_repeated_point(self, verts: NDArray, num_copies: int) -> None:
        verts_repeated = np.repeat(verts, num_copies, axis=0)
        assert lsort(pes.utils.conv(verts_repeated)) == approx(lsort(verts)), \
            f"Expected convex hull of\n{verts_repeated}\nto be equal to\n{pes.utils.conv(verts_repeated)}\n(same rows, order does not matter)" 
        
    @given(verts=integers(
        min_value=1, max_value=N_MAX).flatmap(lambda n: arrays(float, (n, n), elements=floats(-100, 100, allow_infinity=False, allow_nan=False)))
    )
    def test_random_nd_points_on_lower_dimensional_hyperplane(self, verts: NDArray) -> None:
        assume(np.linalg.matrix_rank(verts, tol=ATOL) == verts.shape[0])  # Skip cases where some points are within the subspace spanned by the other points
        assert lsort(pes.utils.conv(verts)) == approx(lsort(verts)), \
            f"Expected convex hull of\n{verts}\nto be equal to\n{pes.utils.conv(verts)}\n(same rows, order does not matter)"
        
    # scipy.spatial._qhull.QhullError: QH6154 Qhull precision error: Initial simplex is flat (facet 5 is coplanar with the interior point)
    # E   
    # E   While executing:  | qhull i Qx Qt
    # E   Options selected for Qhull 2020.2.r 2020/08/31:
    # E     run-id 908544612  incidence  Qxact-merge  Qtriangulate  _zero-centrum
    # E     _max-width  1  Error-roundoff 2.7e-15  _one-merge 3e-14  _near-inside 1.5e-13
    # E     Visible-distance 1.6e-14  U-max-coplanar 1.6e-14  Width-outside 3.3e-14
    # E     _wide-facet 9.8e-14  _maxoutside 3.3e-14
    # E   
    # E   precision problems (corrected unless 'Q0' or an error)
    # E         4 nearly singular or axis-parallel hyperplanes
    # E         2 zero divisors during back substitute
    # E         3 zero divisors during gaussian elimination
    # E   
    # E   The input to qhull appears to be less than 5 dimensional, or a
    # E   computation has overflowed.
    # E   
    # E   Qhull could not construct a clearly convex simplex from points:
    # E   - p5(v6): 1.5e-103 1.5e-103 1.5e-103 1.5e-103     1
    # E   - p3(v5): 1.5e-103 1.5e-103     0 1.5e-103 1.5e-103
    # E   - p1(v4): 1.5e-103 1.5e-103 1.5e-103     1 1.5e-103
    # E   - p0(v3): 1.5e-103     1 1.5e-103 1.5e-103 1.5e-103
    # E   - p4(v2):     1 1.5e-103 1.5e-103 1.5e-103 1.5e-103
    # E   - p2(v1):     0 1.5e-103     1 1.5e-103 1.5e-103
    # E   
    # E   The center point is coplanar with a facet, or a vertex is coplanar
    # E   with a neighboring facet.  The maximum round off error for
    # E   computing distances is 2.7e-15.  The center point, facets and distances
    # E   to the center point are as follows:
    # E   
    # E   center point   0.1667   0.1667   0.1667   0.1667   0.1667
    # E   
    # E   facet p3 p1 p0 p4 p2 distance= -0.17
    # E   facet p5 p1 p0 p4 p2 distance= -0.075
    # E   facet p5 p3 p0 p4 p2 distance= -0.17
    # E   facet p5 p3 p1 p4 p2 distance= -0.17
    # E   facet p5 p3 p1 p0 p2 distance=    0
    # E   facet p5 p3 p1 p0 p4 distance= -0.17
    # E   
    # E   These points either have a maximum or minimum x-coordinate, or
    # E   they maximize the determinant for k coordinates.  Trial points
    # E   are first selected from points that maximize a coordinate.
    # E   
    # E   The min and max coordinates for each dimension are:
    # E     0:         0         1  difference=    1
    # E     1:  1.454e-103         1  difference=    1
    # E     2:         0         1  difference=    1
    # E     3:  1.454e-103         1  difference=    1
    # E     4:  1.454e-103         1  difference=    1
    # E   
    # E   If the input should be full dimensional, you have several options that
    # E   may determine an initial simplex:
    # E     - use 'QJ'  to joggle the input and make it full dimensional
    # E     - use 'QbB' to scale the points to the unit cube
    # E     - use 'QR0' to randomly rotate the input for different maximum points
    # E     - use 'Qs'  to search all points for the initial simplex
    # E     - use 'En'  to specify a maximum roundoff error less than 2.7e-15.
    # E     - trace execution with 'T3' to see the determinant for each point.
    # E   
    # E   If the input is lower dimensional:
    # E     - use 'QJ' to joggle the input and make it full dimensional
    # E     - use 'Qbk:0Bk:0' to delete coordinate k from the input.  You should
    # E       pick the coordinate with the least range.  The hull will have the
    # E       correct topology.
    # E     - determine the flat containing the points, rotate the points
    # E       into a coordinate plane, and delete the other coordinates.
    # E     - add one or more points to make the input full dimensional.
    # E   
    # E   Falsifying example: test_random_nd_k_prime_inequality(
    # E       self=<tests.numpes.utils.test_spatial.TestConv object at 0x11aaa36b0>,
    # E       verts=array([[1.45413421e-103, 1.00000000e+000, 1.45413421e-103,
    # E               1.45413421e-103, 1.45413421e-103],
    # E              [1.45413421e-103, 1.45413421e-103, 1.45413421e-103,
    # E               1.00000000e+000, 1.45413421e-103],
    # E              [0.00000000e+000, 1.45413421e-103, 1.00000000e+000,
    # E               1.45413421e-103, 1.45413421e-103],
    # E              [1.45413421e-103, 1.45413421e-103, 0.00000000e+000,
    # E               1.45413421e-103, 1.45413421e-103],
    # E              [1.00000000e+000, 1.45413421e-103, 1.45413421e-103,
    # E               1.45413421e-103, 1.45413421e-103],
    # E              [1.45413421e-103, 1.45413421e-103, 1.45413421e-103,
    # E               1.45413421e-103, 1.00000000e+000]]),
    # E   )

    # scipy/spatial/_qhull.pyx:356: QhullError
    @pytest.mark.skip(reason="For some reason, this test is now failing. We need to find why. See error message in comment above")
    @given(verts=integers(min_value=1, max_value=N_MAX).flatmap(
        lambda n: integers(min_value=1, max_value=n + 10).flatmap(
            lambda k: arrays(float, (k, n), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))
        )
    ))
    def test_random_nd_k_prime_inequality(self, verts: NDArray) -> None:
        assert pes.utils.conv(verts).shape[0] <= verts.shape[0], \
            f"Expected convex hull of k points in n-dimensional space to have at most k vertices, but got {pes.utils.conv(verts).shape[0]} vertices for {verts.shape[0]} input points"


# =======================
# (function) signed_angle
# =======================


@pytest.mark.parametrize('v_1, v_2, look, expected', [
    (np.array([1, 0]), 
     np.array([0, 1]), None, 90),
    (np.array([0, 1]), 
     np.array([1, 0]), None, -90),
    (np.array([1, 0]), 
     np.array([0, 1]), np.array([0, 0, 1]), 90),
    (np.array([1, 0]), 
     np.array([0, 1]), np.array([0, 0, -1]), -90),
    (np.array([1, 0]), 
     np.array([1, 1]), None, 45),
    (np.array([-1, 1]), 
     np.array([ 1, 1]), None, -90),
    (np.array([1E-6, 0]), 
     np.array([   0, 1]), None, 90),
    (np.array([1,  0]), 
     np.array([0, -1]), None, -90)
])
def test_signed_angle_parametrize_2d(v_1: NDArray, v_2: NDArray, look: NDArray, expected: float):
    assert pes.utils.signed_angle(v_1, v_2, look=look) == approx(deg2rad(expected)), \
        f"Expected signed angle between {v_1} and {v_2} to be {expected} degrees, but got {rad2deg(pes.utils.signed_angle(v_1, v_2, look=look))} degrees"


@pytest.mark.parametrize('v_1, v_2, look, expected', [
    (np.array([1, 0, 0]), 
     np.array([0, 1, 0]), None, 90),
    (np.array([0, 1, 0]), 
     np.array([1, 0, 0]), None, -90)
])
def test_signed_angle_parametrize_3d(v_1: NDArray, v_2: NDArray, look: NDArray | None, expected: float):
    assert pes.utils.signed_angle(v_1, v_2, look=look) == approx(deg2rad(expected)), \
        f"Expected signed angle between {v_1} and {v_2} to be {expected} degrees, but got {rad2deg(pes.utils.signed_angle(v_1, v_2, look=look))} degrees"


@given(vector_nd_pair=integers(min_value=1, max_value=N_MAX).flatmap(lambda n: tuples(
    arrays(float, (n,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False)), 
    arrays(float, (n,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))
)))
def test_signed_angle_random_nd_not_2d_or_3d_value_error(vector_nd_pair: tuple[NDArray, NDArray]):
    v_1, v_2 = vector_nd_pair
    assume(not v_1.size in {2, 3})
    with pytest.raises(ValueError):
        _ = pes.utils.signed_angle(v_1, v_2)


@given(vector_2d_pair=tuples(
    arrays(float, (2,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False)), 
    arrays(float, (2,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))
))
def test_signed_angle_random_2d_in_range(vector_2d_pair: tuple[NDArray, NDArray]):
    v_1, v_2 = vector_2d_pair
    assume(not v_1 == approx(0) and not v_2 == approx(0))
    assert -180 <= rad2deg(pes.utils.signed_angle(v_1, v_2)) <= 180, \
        f"Expected signed angle between {v_1} and {v_2} to be between -180 and 180 degrees, but got {np.degrees(pes.utils.signed_angle(v_1, v_2))} degrees"


@given(vector_2d_pair=tuples(
    arrays(float, (2,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False)), 
    arrays(float, (2,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))
))
def test_signed_angle_random_2d_commutative(vector_2d_pair: tuple[NDArray, NDArray]):
    v_1, v_2 = vector_2d_pair
    assume(not v_1 == approx(0) and not v_2 == approx(0))  # Skip cases where one of the vectors is zero
    assume(not np.linalg.matrix_rank(np.column_stack([v_1, v_2])) < 2)  # Skip cases where the vectors are (anti)parallel
    assert pes.utils.signed_angle(v_1, v_2) == approx(-pes.utils.signed_angle(v_2, v_1)), \
        f"Expected signed angle between {v_1} and {v_2} to be the negative of the signed angle between {v_2} and {v_1}, but got {rad2deg(pes.utils.signed_angle(v_1, v_2))} degrees and {rad2deg(pes.utils.signed_angle(v_2, v_1))} degrees respectively"


@pytest.mark.parametrize('v_1, v_2, look', [
    (np.array([1]), 
     np.array([1]), None),
    (np.ones(4), 
     np.ones(4), None),
    (np.array([1, 2]), 
     np.array([3, 4, 5]), None),
    (np.array([1, 2]), 
     np.array([3, 4]), np.array([0, 1])),
    (np.array([[1], [2]]),
     np.array([[3], [4]]), None),
    (np.array([1, 0]), 
     np.array([0, 1]), np.zeros(3))  # NOTE: Look vector cannot be zero
])
def test_signed_angle_parametrize_value_error(v_1: NDArray, v_2: NDArray, look: NDArray):
    with pytest.raises(ValueError):
        pes.utils.signed_angle(v_1, v_2, look=look)


@pytest.mark.parametrize('v_1, v_2, look', [
    ([1, 2],
     [3, 4], None)
])
def test_signed_angle_parametrize_attribute_error(v_1: NDArray, v_2: NDArray, look: NDArray):
    with pytest.raises(AttributeError):
        pes.utils.signed_angle(v_1, v_2, look=look)


@given(vector_2d_or_3d=integers(
        min_value=2, max_value=3).flatmap(lambda n: arrays(float, (n,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))), scalar=floats(0, 100, allow_infinity=False, allow_nan=False)
)
def test_signed_angle_random_2d_or_3d_parallel(vector_2d_or_3d: NDArray, scalar: float):
    v_1 = vector_2d_or_3d
    v_2 = scalar * v_1
    assume(not v_1 == approx(0) and not v_2 == approx(0))  # Skip cases where any vector is zero
    assert pes.utils.signed_angle(v_1, v_2) == approx(0, atol=1E-6), \
        f"Expected signed angle between {v_1} and {v_2} to be 0 degrees (parallel), but got {rad2deg(pes.utils.signed_angle(v_1, v_2))} degrees"
    

@given(vector_2d_or_3d=integers(
        min_value=2, max_value=3).flatmap(lambda n: arrays(float, (n,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))), scalar=floats(-100, 0, allow_infinity=False, allow_nan=False)
)
def test_signed_angle_random_2d_or_3d_antiparallel(vector_2d_or_3d: NDArray, scalar: float):
    v_1 = vector_2d_or_3d
    v_2 = scalar * v_1
    assume(not v_1 == approx(0) and not v_2 == approx(0))  # Skip cases where any vector is zero
    assert abs(pes.utils.signed_angle(v_1, v_2)) == approx(deg2rad(180)), \
        f"Expected signed angle between {v_1} and {v_2} to be ±180 degrees (antiparallel), but got {rad2deg(pes.utils.signed_angle(v_1, v_2))} degrees"


@given(vector_3d_pair=tuples(
    arrays(float, (3,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False)), 
    arrays(float, (3,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False))
))
def test_signed_angle_random_3d_cross_product_perpendicular(vector_3d_pair):
    v_1, v_2 = vector_3d_pair
    cross_product = np.cross(v_1, v_2)
    assume(not v_1 == approx(0) and not v_2 == approx(0))  # Skip cases where one of the vectors is zero
    assume(not np.linalg.matrix_rank(np.column_stack([v_1, v_2])) < 2)  # Skip cases where the vectors are (anti)parallel
    # FIXME: I should be able to remove the line below, but sometimes hypothesis suddenly generates a numerically ill-conditioned counterexample.
    assume(not np.linalg.norm(cross_product) == approx(0))  # Skip cases where cross product is numerically zero
    assert abs(pes.utils.signed_angle(v_1, cross_product)) == approx(deg2rad(90)), \
        f"Expected signed angle between {v_1} and {cross_product} to be ±90 degrees when one is the cross product of the other, but got {rad2deg(pes.utils.signed_angle(v_1, cross_product))} degrees"


@given(vector_2d_or_3d=integers(
    min_value=2, max_value=3).flatmap(lambda n: arrays(float, (n,), elements=floats(-100, 100, allow_infinity=False, allow_nan=False)))
)
def test_signed_angle_random_2d_or_3d_identical_inputs(vector_2d_or_3d):
    v = vector_2d_or_3d
    assume(not v == approx(0))  # Skip cases where the vector is numerically zero
    assert pes.utils.signed_angle(v, v) == approx(0, atol=1E-6), f"The angle between a vector and itself should be zero, received {rad2deg(pes.utils.signed_angle(v, v))}"


@pytest.mark.parametrize('v_1, v_2, look', [
    (np.array([1, 0]), 
     np.array([0, 0]), None),
    (np.array([0, 0]), 
     np.array([0, 0]), None),
    (np.array([1E-12, 0, 0]), 
     np.array([    0, 0, 1]), None)
])
def test_signed_angle_parametrize_2d_or_3d_zero_vector_nan(v_1: NDArray, v_2: NDArray, look: NDArray):
    assert pes.utils.signed_angle(v_1, v_2, look=look) is np.nan, f"Expected signed angle between {v_1} and {v_2} to be undefined (np.nan) as one of the vectors is zero, but got {pes.utils.signed_angle(v_1, v_2, look=look)}"


@pytest.mark.parametrize('v_1, v_2, look', [
    (np.array([ 1, 0]), 
     np.array([-1, 0]), None),
    (np.array([1000, 0]), 
     np.array([  -1, 0]), None),
    (np.array([ 1,  1E-6]), 
     np.array([-1,     0]), None),
    (np.array([ 1, -1E-6]), 
     np.array([-1,     0]), None),
])
def test_signed_angle_parametrize_2d_or_3d_opposite(v_1: NDArray, v_2: NDArray, look: NDArray):
    assert np.isclose(np.abs(pes.utils.signed_angle(v_1, v_2, look=look)), np.radians(180)), f"Expected signed angle between opposite vectors {v_1} and {v_2} to be 180 degrees, but got {np.degrees(pes.utils.signed_angle(v_1, v_2, look=look))} degrees"


def test_signed_angle_arbitrary():
    ...


def test_signed_angle_look():
    ...


def test_signed_angle_look_parallel():
    ...


def test_signed_angle_look_perpendicular():
    ...


def test_signed_angle_look_opposite():
    ...
