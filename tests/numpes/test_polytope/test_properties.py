"""Tests for the properties of the Polytope class"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpes as pes
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, tuples, sampled_from, booleans

from tests.strategies import poly_rand, poly_rand_pair
from tests.helpers import lsort, approx
from tests.conftest import N_MAX

if TYPE_CHECKING:
    from typing import Callable
    from numpes import Polytope
    from numpy.typing import ArrayLike, NDArray
    from tests.conftest import PolytopeData


@pytest.mark.parametrize('params, expected', [
    ({'n': 1}, 1),
    ({'n': 2}, 2),
    ({'n': 3}, 3),
    ({'n': 5}, 5),
    ({'n': 10}, 10),
    ({'n': 20_000}, 20_000)
])
def test_polytope_n_empty(params: dict[str, int], expected: int):
    assert pes.Polytope(**params).n == expected, \
        f"Expected property 'n'={expected} but got poly.n={pes.Polytope(**params).n}"


def test_polytope_n_archetypes(poly_arch_all: tuple[Polytope, PolytopeData]):
    poly, poly_data = poly_arch_all
    assert poly.n == poly_data.n, \
        f"Expected property 'n'={poly_data.n} but got poly.n={poly.n}"


@pytest.mark.parametrize('generator', [
    'unit_hypercube',
    'centered_hypercube',
    'simplex',
    'cross_polytope'
])
@pytest.mark.parametrize('n', range(5, 10))
def test_polytope_n_generators(poly_gen_factory: Callable[[str, int], tuple[Polytope, PolytopeData]], generator: str, n: int):
    poly, poly_data = poly_gen_factory(generator, n)
    assert poly.n == n, \
        f"Expected polytope '{poly_data.name}' to have dimension n={n}"


@given(poly=integers(
    min_value=5, max_value=10).flatmap(lambda n: poly_rand(repr='vrepr', n=n, exclude_degen=False))
)
def test_polytope_n_random_in_range(poly: Polytope):
    assert poly.n in range(5, 10 + 1), \
        f"Expected property 'n' to be in the interval [5, 10] but got {poly.n}"


@given(poly_pair=poly_rand_pair(repr='vrepr', n=(5, 10), same_n=True))
def test_polytope_n_random_pair_equal(poly_pair: tuple[Polytope, Polytope]):
    poly_1, poly_2 = poly_pair
    assert poly_1.n == poly_2.n, \
        f"Expected property 'n' to be equal for both polytopes in the pair but got {poly_1.n} and {poly_2.n}"


def test_polytope_verts_archetypes_vrepr(poly_arch_all: tuple[Polytope, PolytopeData]):
    poly, poly_data = poly_arch_all
    assert poly.verts.shape == poly_data.verts.shape, \
        f"Expected shape number of vertices {poly_data.verts.shape} but got {poly.verts.shape}"


def test_polytope_rays_archetypes_vrepr(poly_arch_all: tuple[Polytope, PolytopeData]):
    poly, poly_data = poly_arch_all
    assert poly.rays.shape == poly_data.rays.shape, \
        f"Expected shape number of rays {poly_data.rays.shape} but got {poly.rays.shape}"


def test_polytope_verts_archetypes_from_hrepr(poly_arch_hrepr_all: tuple[Polytope, PolytopeData]):
    poly, poly_data = poly_arch_hrepr_all
    assert lsort(poly.verts) == approx(lsort(poly_data.verts)), \
        f"Expected vertices\n{poly_data.verts}\nto be equal to\n{poly.verts}\n(same rows, order does not matter)"
    assert lsort(poly.rays) == approx(lsort(poly_data.rays)), \
        f"Expected rays\n{poly_data.rays}\nto be equal to\n{poly.rays}\n(same rows, order does not matter)"


class TestPolytopeIsEmpty:
    """Tests for the `pes.Polytope.is_empty` property"""

    @given(poly=tuples(
        integers(min_value=1, max_value=N_MAX),
        sampled_from(['vrepr', 'hrepr']),
        ).flatmap(lambda pair: poly_rand(repr=pair[1], n=pair[0], exclude_degen=True))
    )
    def test_random_nondegen_minimal(self, poly: Polytope) -> None:
        """Test whether non-degenerate, minimal polytopes are correctly classified as non-empty"""
        assert not poly.is_empty, \
            f"Expected `poly.is_empty` to return False for poly={poly:r}, but received True"

    @pytest.mark.parametrize('verts, rays', [
        ([0, 0, 0], None),
        ([0, 0], [0, 0]),
        (None, [0, 0, 0]),
        (None, [1, 1, 1]),
        ([[0, 0], [0, 0], [0, 1]], np.empty((0, 2))),
    ])
    def test_parameterize_vrepr_not_empty(self, verts: ArrayLike, rays: ArrayLike | None) -> None:
        """Test whether polytopes initialized from V-representation gets marked as non-empty"""
        if verts is None:
            verts = np.empty((0, len(rays)))
        if rays is None:
            rays = np.empty((0, len(verts)))
        poly = pes.poly(verts, rays=rays)
        assert not poly.is_empty, \
            f"Expected `poly.is_empty` to return False for poly={poly:r}, but received True"

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.empty((0, 3)), np.empty((0, 3))),
        (np.empty((0, 8)), np.empty((0, 8))),
        (np.array([[0, 0, 1]]), np.empty((0, 3))),
        (np.empty((0, 3)), np.array([[1, 0, 1]])),
        (np.array([[0, 0, 0, 0]]), np.empty((0, 4))),
    ])
    def test_parameterize_hrepr_not_empty(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test whether polytopes initialized from H-representation gets marked as non-empty"""
        poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1])
        assert not poly.is_empty, \
            f"Expected `poly.is_empty` to return False for poly={poly:r}, but received True"

    @pytest.mark.parametrize('verts, rays', [
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.empty((0, 10)), np.empty((0, 10))),
    ])
    def test_parameterize_vrepr_empty(self, verts: ArrayLike, rays: ArrayLike | None) -> None:
        """Test whether empty polytopes initialized from V-representation gets marked as empty"""
        poly = pes.poly(verts, rays=rays)
        assert poly.is_empty, \
            f"Expected `poly.is_empty` to return True for poly={poly:r}, but received False"

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.array([[0, 0, -1]]), np.empty((0, 3))),
        (np.array([[0, 0, -2]]), np.empty((0, 3))),
        (np.array([[ 1, 0,  1],
                   [-1, 0, -2]]), np.empty((0, 3))),
        (np.empty((0, 4)), np.array([[1, 0, 1, 2], 
                                     [1, 0, 1, 3]])),
        (np.empty((0, 3)), np.array([[0, 0, 1]])),
    ])
    def test_parameterize_hrepr_empty(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test whether empty polytopes initialized from H-representation gets marked as empty"""
        poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1],)
        assert poly.is_empty, \
            f"Expected `poly.is_empty` to return True for poly={poly:r}, but received False"

    @given(poly=tuples(
        integers(min_value=1, max_value=N_MAX),
        sampled_from(['vrepr', 'hrepr']),
        ).flatmap(lambda pair: poly_rand(repr=pair[1], n=pair[0], exclude_degen=True)),
        is_empty_override=booleans(),
    )
    def test_random_access_private_attr(self, poly: Polytope, is_empty_override: bool) -> None:
        """Test whether overwriting the `Polytope._is_empty` attribute gets correctly run through"""
        poly._is_empty = is_empty_override
        assert poly.is_empty == is_empty_override, \
            f"Expected is_empty_override={is_empty_override} after override, but received poly.is_empty={poly.is_empty}"

    @pytest.mark.coupled('pes.poly_empty')
    @pytest.mark.parametrize('n', [
        1,
        2,
        3,
        5,
        10,
        100,
    ])
    def test_parameterize_poly_empty(self, n: int) -> None:
        """Test that initializing a polytope with `pes.poly_empty(n)` results in is_empty=True"""
        poly = pes.poly_empty(n)
        assert poly.is_empty, \
            f"Expected empty polytope initialization with n={n}, poly={poly} to result in True, but received False"

    @pytest.mark.coupled('pes.poly_ambient')
    @pytest.mark.parametrize('n', [
        1,
        2,
        3,
        5,
        10,
        100,
    ])
    def test_parameterize_poly_ambient(self, n: int) -> None:
        """Test that initializing a polytope with `pes.poly_ambient(n)` results in is_empty=False"""
        poly = pes.poly_ambient(n)
        assert not poly.is_empty, \
            f"Expected ambient polytope initialization with n={n}, poly={poly} to result in False, but received True"
