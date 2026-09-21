"""Tests for the properties of the Polytope class"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpes as pes
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, floats, tuples, sampled_from, booleans
from hypothesis.extra.numpy import arrays

from tests.strategies import poly_rand, poly_rand_pair
from tests.helpers import lsort, approx
from tests.conftest import N_MAX, ATOL

if TYPE_CHECKING:
    from typing import Callable, Literal
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


class TestPolytopeIsSingleton:
    """Tests for the `Polytope.is_singleton` property"""

    @pytest.mark.skip("Option 'exclude_degen=True' is not actually working")
    @given(poly=tuples(
        integers(min_value=1, max_value=N_MAX),
        sampled_from(['vrepr', 'hrepr']),
        ).flatmap(lambda pair: poly_rand(repr=pair[1], n=pair[0], exclude_degen=True))
    )
    def test_random_nondegen_minimal(self, poly: Polytope) -> None:
        """Test whether non-degenerate, minimal polytopes are correctly classified as not singletons"""
        assert not poly.is_singleton, \
            f"Expected `poly.is_singleton` to return False for poly={poly:r}, but received True"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('verts, rays', [
        ([[0, 0, 0], [0, 0, 1]], None),
        ([0, 0], [1, 0]),
        (None, [1, 2, 3]),
        ([0, -1, 0], [0, 0, 0]),
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_vrepr_no_singleton(self, verts: ArrayLike, rays: ArrayLike | None, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether polytopes initialized from V-representation gets marked as not a singleton"""
        if verts is None:
            verts = np.empty((0, np.atleast_2d(rays).shape[1]))
        if rays is None:
            rays = np.empty((0, np.atleast_2d(verts).shape[1]))
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(verts, rays=rays)
        assert not poly.is_singleton, \
            f"Expected `poly.is_singleton` to return False for poly={poly:r}, but received True"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.empty((0, 3)), np.empty((0, 3))),  # Ambient space
        (np.empty((0, 8)), np.empty((0, 8))),  # Ambient space
        (np.array([[0, 0, 0, 0]]), np.empty((0, 4))),  # Ambient space
        (np.array([[0, 0,  1]]), np.empty((0, 3))),
        (np.array([[0, 0,  1],
                   [1, 2, 3]]), np.empty((0, 3))),
        (np.array([[0, 0, -1]]), np.empty((0, 3))),  # Empty
        (np.empty((0, 4)), np.array([[1, 0, 1, 2],
                                     [1, 0, 1, 3]])),  # Empty
        (np.empty((0, 3)), np.array([[1, 1, 2]])),  # Line
        (np.array([[ 1, 0,  2],
                   [-1, 0, -2]]), np.empty((0, 3))),  # Line
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_hrepr_no_singleton(self, Ab: NDArray, Ab_eq: NDArray, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether polytopes initialized from H-representation gets marked as not a singleton"""
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1])
        assert not poly.is_singleton, \
            f"Expected `poly.is_singleton` to return False for poly={poly:r}, but received True"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('verts, rays', [
        ([0, 0, 1], None),
        ([-0.5, 0.6, 7.1], None),
        (np.ones(100), None),
        ([[1, 2, 3],
          [1, 2, 3]], None),
        ([[1, 2 + ATOL / 2, 3],
          [1, 2           , 3]], None),
        (None, [0, 0, 0, 0]),  # Zero vertex
        (None, [[0, 0, 0, 0],
                [0, 0, 0, 0]]),  # Zero vertex
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_vrepr_singleton(self, verts: ArrayLike, rays: ArrayLike | None, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether singleton polytopes initialized from V-representation gets marked as a singleton"""
        if verts is None:
            verts = np.empty((0, np.atleast_2d(rays).shape[1]))
        if rays is None:
            rays = np.empty((0, np.atleast_2d(verts).shape[1]))
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(verts, rays=rays)
        assert poly.is_singleton, \
            f"Expected `poly.is_singleton` to return True for poly={poly:r}, but received False"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.empty((0, 4)), np.array([[1, 0, 0, 2],
                                     [0, 1, 0, 3],
                                     [0, 0, 1, 4]])),
        (np.array([[ 1,  0,  2],
                   [ 0,  1,  3],
                   [-1,  0, -2],
                   [ 0, -1, -3]]), np.empty((0, 3))),
        (np.array([[ 1,  0,  4.0],
                   [-1,  0, -4.0]]), np.array([[0, 1, -2]])),
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_hrepr_singleton(self, Ab: NDArray, Ab_eq: NDArray, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether singleton polytopes initialized from H-representation gets marked as a singleton"""
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1])
        assert poly.is_singleton, \
            f"Expected `poly.is_singleton` to return True for poly={poly:r}, but received False"

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
        """Test that initializing a polytope with `pes.poly_empty(n)` does not result in a singleton"""
        poly = pes.poly_empty(n)
        assert not poly.is_singleton, \
            f"Expected empty polytope initialization with n={n}, poly={poly} to result in False, but received True"

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
        """Test that initializing a polytope with `pes.poly_ambient(n)` does not result in a singleton"""
        poly = pes.poly_ambient(n)
        assert not poly.is_singleton, \
            f"Expected ambient polytope initialization with n={n}, poly={poly} to result in False, but received True"

    @pytest.mark.coupled('pes.poly_from_point')
    @given(
        point=tuples(sampled_from([int, float]), integers(min_value=1, max_value=N_MAX)).flatmap(
            lambda args: arrays(args[0],
                                args[1],
                                elements=(floats(-100, 100, allow_infinity=False, allow_nan=False)
                                          if issubclass(args[0], float)
                                          else integers(-100, 100)))
        )
    )
    def test_random_poly_from_point(self, point: NDArray) -> None:
        """Test that initializing a polytope with `pes.poly_from_point(point)` does result in a singleton"""
        poly = pes.poly_from_point(point)
        assert poly.is_singleton, \
            f"Expected polytope initialization with point={point}, poly={poly} to result in True, but received False"


class TestPolytopeIsFullDim:
    """Tests for the `Polytope.is_full_dim` property"""

    @pytest.mark.skip("Option 'exclude_degen=True' is not actually working")
    @given(poly=tuples(
        integers(min_value=1, max_value=N_MAX),
        sampled_from(['vrepr', 'hrepr']),
        ).flatmap(lambda pair: poly_rand(repr=pair[1], n=pair[0], exclude_degen=True))
    )
    def test_random_nondegen_minimal(self, poly: Polytope) -> None:
        """Test whether non-degenerate, minimal polytopes are correctly classified as full dimensional"""
        assert poly.is_full_dim, \
            f"Expected `poly.is_full_dim` to return True for poly={poly:r}, but received False"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('verts, rays', [
        ([[1, 0],
          [0, 1],
          [1, 1]], None),
        ([[  0,   0],
          [  1,   0],
          [  0,   1],
          [  1,   1],
          [0.5, 0.5]], None),
        ([[0, 0],
          [1, 1]], [1, 0]),
        ([[0, 0]], [[1, 0],
                    [1, 1]]),
        ([[  0,        0],
          [ 10,        0],
          [100,        0],
          [  0, 2 * ATOL]], None),
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_vrepr_full_dim(self, verts: ArrayLike, rays: ArrayLike | None, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether polytopes initialized from V-representation gets marked as full dimensional"""
        if verts is None:
            verts = np.empty((0, np.atleast_2d(rays).shape[1]))
        if rays is None:
            rays = np.empty((0, np.atleast_2d(verts).shape[1]))
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(verts, rays=rays)
        assert poly.is_full_dim, \
            f"Expected `poly.is_full_dim` to return True for poly={poly:r}, but received False"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('verts, rays', [
        ([[1, 0],
          [0, 1],
          [0, 1]], None),
        ([0.4, 0.8], None),
        (None, [1.2, -3.1]),
        (None, [[0,  1],
                [0,  2],
                [0, -3]]),
        ([[  1,   0],
          [  0,   1],
          [0.5, 0.5]], None),
        ([[0, 0],
          [1, 1]], [1, 1]),
        ([[1, 1],
          [2, 2]], [0, 0]),
        ([[0, 0]], [[1, 1]]),
        ([[  0,        0],
          [ 10,        0],
          [100,        0],
          [  0, ATOL / 2]], None),
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_vrepr_lower_dim(self, verts: ArrayLike, rays: ArrayLike | None, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether lower-dimensional polytopes initialized from V-representation get marked as not full-dimensional"""
        if verts is None:
            verts = np.empty((0, np.atleast_2d(rays).shape[1]))
        if rays is None:
            rays = np.empty((0, np.atleast_2d(verts).shape[1]))
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(verts, rays=rays)
        assert not poly.is_full_dim, \
            f"Expected `poly.is_full_dim` to return False for poly={poly:r}, but received True"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.empty((0, 3)), np.empty((0, 3))),  # Ambient space
        (np.empty((0, 8)), np.empty((0, 8))),  # Ambient space
        (np.array([[0, 0, 0, 0]]), np.empty((0, 4))),  # Ambient space
        (np.array([[0, 0, 0, 10]]), np.empty((0, 4))),  # Ambient space, trivial inequality constraint
        (np.array([[0, 0,  1]]), np.empty((0, 3))),
        (np.array([[0, 0, 1],
                   [1, 2, 3]]), np.empty((0, 3))),
        (np.empty((0, 3)), np.array([[0, 0, 0]])),  # Trivial equality constraint 
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_hrepr_full_dim(self, Ab: NDArray, Ab_eq: NDArray, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether polytopes initialized from H-representation gets marked as full-dimensional"""
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1])
        assert poly.is_full_dim, \
            f"Expected `poly.is_full_dim` to return True for poly={poly:r}, but received False"

    @pytest.mark.coupled('pes.algo_options')
    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.array([[0, 0, -1]]), np.empty((0, 3))),  # Empty
        (np.array([[0, 0, -5]]), np.empty((0, 3))),  # Empty
        (np.empty((0, 4)), np.array([[1, 0, 1, 2],
                                     [1, 0, 1, 3]])),  # Empty
        (np.array([[1, 2, 3]]), np.array([[0, 0, -10]])),  # Empty
        (np.array([[1, 2, 3]]), np.array([[0, 0, 10]])),  # Empty
        (np.empty((0, 3)), np.array([[1, 1, 2]])),  # Line
        (np.array([[ 1, 0,  2],
                   [-1, 0, -2]]), np.empty((0, 3))),  # Line
        (np.array([[ 1, 0,  0, 1],
                   [-1, 0,  0, 0],
                   [0,  1,  0, 1],
                   [0, -1,  0, 0],
                   [0,  0,  1, 0],
                   [0,  0, -1, 0]]), np.empty((0, 4))),  # Plane, implicit equalities
        (np.array([[1, 0,  0, 1],
                   [0, 1,  0, 1],
                   [0, 0,  1, 0],
                   [0, 0, -1, 0]]), np.empty((0, 4))),  # Plane, unbounded, implicit equalities
        (np.array([[1, 0,  0, 1],
                   [0, 1,  0, 1]]), np.array([[0, 0, 1, 2]])),  # Plane, unbounded, explicit equalities
    ])
    @pytest.mark.parametrize('on_property_assign', [
        'minimal',
        'pass',
    ])
    def test_parameterize_hrepr_lower_dim(self, Ab: NDArray, Ab_eq: NDArray, on_property_assign: Literal['minimal', 'pass']) -> None:
        """Test whether lower dimensional polytopes initialized from H-representation gets marked as not full-dimensional"""
        with pes.algo_options(on_property_assign=on_property_assign):
            poly = pes.poly(Ab[:, :-1], Ab[:, -1], A_eq=Ab_eq[:, :-1], b_eq=Ab_eq[:, -1])
        assert not poly.is_full_dim, \
            f"Expected `poly.is_full_dim` to return False for poly={poly:r}, but received True"

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
        """Test that initializing a polytope with `pes.poly_empty(n)` does not result in a full dimensional polytope"""
        poly = pes.poly_empty(n)
        assert not poly.is_full_dim, \
            f"Expected empty polytope initialization with n={n}, poly={poly} to result in False, but received True"

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
        """Test that initializing a polytope with `pes.poly_ambient(n)` does result in a full-dimensional polytope"""
        poly = pes.poly_ambient(n)
        assert poly.is_full_dim, \
            f"Expected ambient polytope initialization with n={n}, poly={poly} to result in True, but received False"

    @pytest.mark.coupled('pes.poly_from_point')
    @given(
        point=tuples(sampled_from([int, float]), integers(min_value=1, max_value=N_MAX)).flatmap(
            lambda args: arrays(args[0],
                                args[1],
                                elements=(floats(-100, 100, allow_infinity=False, allow_nan=False)
                                          if issubclass(args[0], float)
                                          else integers(-100, 100)))
        )
    )
    def test_random_poly_from_point(self, point: NDArray) -> None:
        """Test that initializing a polytope with `pes.poly_from_point(point)` does not result in a full-dimensional polytope"""
        poly = pes.poly_from_point(point)
        assert not poly.is_full_dim, \
            f"Expected singleton polytope initialization with point={point}, poly={poly} to result in False, but received True"
