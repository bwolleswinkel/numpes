"""Tests for operations on polytopes"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from tests.helpers import lsort, approx
from tests.strategies import poly_rand_pair

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpes import Polytope


class TestPolytopeMinkSum:
    """Tests for the Minkowski sum operation on polytopes"""

    @pytest.mark.skip(reason="Method 'mink_sum' is currently not yet implemented")
    @pytest.mark.parametrize('poly_1_name, poly_2_name, expected_verts', [
        ('poly_arch_unit_square_2d', 'poly_arch_unit_square_2d',
        np.array([[0, 0],
                  [2, 0],
                  [2, 2],
                  [0, 2]])),
        ('poly_arch_unit_square_2d', 'poly_arch_centered_square_2d',
        np.array([[-1, -1],
                  [ 2, -1],
                  [ 2,  2],
                  [-1,  2]])),
        ('poly_arch_unit_square_2d', 'poly_arch_triangle_2d',
        np.array([[0, 0],
                  [2, 0],
                  [0, 2],
                  [1, 2],
                  [2, 1]]))
    ])
    def test_archetypes_2d_nondegen(self, poly_1_name: str, poly_2_name: str, expected_verts: NDArray, request: pytest.FixtureRequest) -> None:
        """Test Minkowski sum on several combinations of non-degenerate 2D archetypes"""
        poly_1, poly_2 = request.getfixturevalue(poly_1_name), request.getfixturevalue(poly_2_name)
        assert lsort(poly_1.mink_sum(poly_2).verts) == lsort(expected_verts), \
            f"Minkowski sum of {poly_1_name} and {poly_2_name} should have vertices {expected_verts}, but got {poly_1.mink_sum(poly_2).verts} instead"


    @pytest.mark.skip(reason="Method 'mink_sum' is currently not yet implemented, and equality `==` is also not implemented")
    @given(poly_pair=integers(
        min_value=1, max_value=10).flatmap(lambda n: poly_rand_pair(repr='vrepr', n=n, same_n=True, same_repr=True))
    )
    def test_commutative(self, poly_pair: tuple[Polytope, Polytope]) -> None:
        poly_1, poly_2 = poly_pair
        poly_sum_1 = poly_1.mink_sum(poly_2)
        poly_sum_2 = poly_2.mink_sum(poly_1)
        assert poly_sum_1 == poly_sum_2, \
            f"Minkowski sum should be commutative, but got {poly_sum_1} and {poly_sum_2} for polytopes {poly_1} and {poly_2}"

    @pytest.mark.skip(reason="Method 'mink_sum' is currently not yet implemented, and `.vol` is also not yet implemented")
    @given(poly_pair=integers(
        min_value=1, max_value=10).flatmap(lambda n: poly_rand_pair(repr='vrepr', n=n, same_n=True, same_repr=True))
    )
    def test_volume_lesser_equal(self, poly_pair: tuple[Polytope, Polytope]) -> None:
        poly_1, poly_2 = poly_pair
        poly_sum = poly_1.mink_sum(poly_2)
        assert poly_sum.vol < poly_1.vol + poly_2.vol or poly_sum.vol == approx(poly_1.vol + poly_2.vol), \
            f"Volume of Minkowski sum should be at most the sum of the volumes, but got {poly_sum.vol} > {poly_1.vol} + {poly_2.vol} for polytopes {poly_1} and {poly_2}"
        
    @pytest.mark.skip(reason="Method 'mink_sum' is currently not yet implemented, and `.vol` is also not yet implemented")
    @given(poly_pair=integers(
        min_value=1, max_value=10).flatmap(lambda n: poly_rand_pair(repr='vrepr', n=n, same_n=True, same_repr=True))
    )
    def test_volume_greater_equal_power(self, poly_pair: tuple[Polytope, Polytope]) -> None:
        """Test whether the Minkowski inequality vol(P + Q)^(1/n) >= vol(P)^(1/n) + vol(Q)^(1/n) holds"""
        poly_1, poly_2 = poly_pair
        poly_sum = poly_1.mink_sum(poly_2)
        assert poly_sum.vol ** (1 / poly_1.n) > poly_1.vol ** (1 / poly_1.n) + poly_2.vol ** (1 / poly_2.n) or poly_sum.vol ** (1 / poly_1.n) == approx(poly_1.vol ** (1 / poly_1.n) + poly_2.vol ** (1 / poly_2.n)), \
            f"Volume of Minkowski sum should be at most the sum of the volumes, but got {poly_sum.vol} > {poly_1.vol} + {poly_2.vol} for polytopes {poly_1} and {poly_2}"


class TestPolytopeCopy:
    """Tests for the `Polytope.copy()` method"""


class TestPolytopeMatMul:
    """Tests for the `Polytope.mat_mul()` method"""
