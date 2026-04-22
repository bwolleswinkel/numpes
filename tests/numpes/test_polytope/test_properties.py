"""Tests for the properties of the Polytope class"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpes as pes
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from tests.strategies import poly_rand, poly_rand_pair
from tests.helpers import lsort, approx

if TYPE_CHECKING:
    from typing import Callable
    from numpes import Polytope
    from ...conftest import PolytopeData


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
