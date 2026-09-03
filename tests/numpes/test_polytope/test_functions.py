"""Test module for the function in `polytope.py`"""

from __future__ import annotations
from typing import TYPE_CHECKING
import re

import numpy as np
import numpes as pes
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, floats

from tests.conftest import ATOL, N_MAX
from tests.helpers import lsort, approx

if TYPE_CHECKING:
    from typing import EllipsisType
    from numpy.typing import ArrayLike


class TestPolyFromName:
    """Tests for the `pes.poly_from_name` factory function"""

    @pytest.fixture(params=[
        # FIXME: Uncomment these tests when the corresponding polytopes are implemented
        # 'triangle',
        # 'square',
        # 'pentagon',
        # 'hexagon',
        # 'heptagon',
        # 'octagon',
        # 'tetrahedron',
        # 'simplex',
        # 'cube',
        # 'octahedron',
        # 'dodecahedron',
        # 'icosahedron',
        'house',
        'pyramid',
    ])
    def name(self, request) -> str:
        """Fixture that provides all valid polytope name"""
        return request.param

    def test_parameterize_all_names_return_polytope(self, name: str) -> None:
        """Test that all names on the list are valid and return a polytope"""
        poly = pes.poly_from_name(name)
        assert isinstance(poly, pes.Polytope), \
            f"Expected a Polytope object for name '{name}', but got {type(poly)} instead."

    def test_parameterize_both_representations_set(self, name: str) -> None:
        """Test that both representations of the polytope are set (for use in the 'base' package)"""

    def test_parameterize_both_representations_equal(self, name: str) -> None:
        """Test that both representations of the polytope are define the same polytope"""

    def test_parameterize_no_attribute_none(self, name: str) -> None:
        """Test that returned polytopes have no attributes which are still None"""
        poly = pes.poly_from_name(name)
        attrs = [attr for attr in dir(poly) if (attr.startswith('_') and not attr.startswith('__'))]
        for attr in attrs:
            assert getattr(poly, attr) is not None, \
                f"Attribute '{attr}' of polytope '{name}' is None"

    @pytest.mark.parametrize('name', [
        'prism',
        '...',
        'poly',
        'none',
        'None',
        'kegel',
        'flat',
        'random',
        '1',
        '',
        None,
        ...,
    ])
    def test_invalid_name_raises_value_error(self, name: str | None | EllipsisType) -> None:
        """Test that invalid names raise a ValueError"""
        with pytest.raises(ValueError, match=re.escape(
            f"Unrecognized name '{name}'"
            )):
            pes.poly_from_name(name)

    def test_house_private_attributes(self) -> None:
        """Test that the private attributes of the 'house' polytope are correct"""
        inv_sqrt2 = 1 / np.sqrt(2)
        expected_attrs = {
            '_vrepr': (np.array([[  0,   0],
                                 [  1,   0],
                                 [  1,   1],
                                 [  0,   1],
                                 [0.5, 1.5]]), np.empty((0, 2))),
            '_hrepr': (np.array([[       0.0,      -1.0,           0.0],
                                 [      -1.0,       0.0,           0.0],
                                 [-inv_sqrt2, inv_sqrt2,     inv_sqrt2],
                                 [       1.0,       0.0,           1.0],
                                 [ inv_sqrt2, inv_sqrt2, 1 / inv_sqrt2]]), np.empty((0, 3))),
            '_is_degen': False,
            '_is_bounded': True,
            '_is_full_dim': True,
            '_is_pointed': True,
            '_is_singleton': False,
            '_dim': 2,
            '_vol': 5 / 4,
            '_diam': np.sqrt(0.5 ** 2 + 1.5 ** 2),
            '_width': 1,
            '_chebcr': (np.array([0.5, 0.5]), 0.5),
        }
        poly = pes.poly_from_name('house')
        for key, expected_value in expected_attrs.items():
            assert hasattr(poly, key), \
                f"Polytope 'house' is missing attribute '{key}'"
            value = getattr(poly, key)
            match expected_value:
                case np.ndarray() | float():
                    assert value == approx(expected_value), \
                        f"Polytope 'house' has incorrect value for attribute '{key}': expected {expected_value}, got {getattr(poly, key)}"
                case int() | bool():
                    assert value == expected_value, \
                        f"Polytope 'house' has incorrect value for attribute '{key}': expected {expected_value}, got {getattr(poly, key)}"
                case tuple():
                    for v, e in zip(value, expected_value):
                        if isinstance(e, np.ndarray):
                            assert lsort(v) == approx(lsort(e)), \
                                f"Polytope 'house' has incorrect value for attribute '{key}': expected\n{expected_value}, got\n{getattr(poly, key)}"
                        else:
                            assert v == approx(e), \
                                f"Polytope 'house' has incorrect value for attribute '{key}': expected {expected_value}, got {getattr(poly, key)}"
                case _:
                    raise TypeError(f"Unexpected type for expected value of attribute '{key}': {type(expected_value)}")

    def test_house_properties(self) -> None:
        """Test that the properties of the 'house' polytope when accessed are correct"""

    def test_house_properties_recalculated(self) -> None:
        """Test that the properties of the 'house' polytope agree with the recalculated values from the vertices and half-space representations"""

    def test_pyramid_private_attributes(self) -> None:
        """Test that the private attributes of the 'pyramid' polytope are correct"""
        expected_attrs = {
            '_vrepr': (np.array([[  0,   0,  0],
                                 [  0,   1,  0],
                                 [  1,   0,  0],
                                 [  1,   1,  0],
                                 [0.5, 0.5,  1]]) , np.empty((0, 3))),
            '_hrepr': (np.array([[ 0,  0, -1, 0],
                                 [-2,  0,  1, 0],
                                 [ 0, -2,  1, 0],
                                 [ 2,  0,  1, 2],
                                 [ 0,  2,  1, 2]]), np.empty((0, 4))),
            '_is_degen': False,
            '_is_bounded': True,
            '_is_full_dim': True,
            '_is_pointed': True,
            '_is_singleton': False,
            '_dim': 3,
            '_vol': 1 / 3,
            '_diam': np.sqrt(2),
            '_width': 1,
            '_chebcr': (np.array([0.5, 0.5, 0.25]), 0.75),
        }
        poly = pes.poly_from_name('pyramid')
        for key, expected_value in expected_attrs.items():
            assert hasattr(poly, key), \
                f"Polytope 'pyramid' is missing attribute '{key}'"
            value = getattr(poly, key)
            match expected_value:
                case np.ndarray() | float():
                    assert value == approx(expected_value), \
                        f"Polytope 'pyramid' has incorrect value for attribute '{key}': expected {expected_value}, got {getattr(poly, key)}"
                case int() | bool():
                    assert value == expected_value, \
                        f"Polytope 'pyramid' has incorrect value for attribute '{key}': expected {expected_value}, got {getattr(poly, key)}"
                case tuple():
                    for v, e in zip(value, expected_value):
                        if isinstance(e, np.ndarray):
                            assert lsort(v) == approx(lsort(e)), \
                                f"Polytope 'pyramid' has incorrect value for attribute '{key}': expected\n{expected_value}, got\n{getattr(poly, key)}"
                        else:
                            assert v == approx(e), \
                                f"Polytope 'pyramid' has incorrect value for attribute '{key}': expected {expected_value}, got {getattr(poly, key)}"
                case _:
                    raise TypeError(f"Unexpected type for expected value of attribute '{key}': {type(expected_value)}")

    def test_pyramid_properties(self) -> None:
        """Test that the properties of the 'pyramid' polytope when accessed are correct"""

    def test_pyramid_properties_recalculated(self) -> None:
        """Test that the properties of the 'pyramid' polytope agree with the recalculated values from the vertices and half-space representations"""


class TestPolyFromPoint:
    """Tests for the `pes.poly_from_point` factory function"""

    @pytest.mark.parametrize('point', [
        np.array([0]),
        np.array([1]),
        np.array([-1]),
        np.array([-6.4, 4.1]),
        np.array([1, 2]),
        np.array([-1, -2]),
        np.array([0, 0, 0]),
        np.array([1.1, 2.2, 33.3]),
        np.array([-1, -2, -3]),
        np.arange(100),
        [0],
        (0, 0, 0),
        [1, -1, -100],
    ])
    def test_parameterize_polytope(self, point: ArrayLike) -> None:
        """Test that the polytope created from a point is a polytope"""
        poly = pes.poly_from_point(point)
        assert isinstance(poly, pes.Polytope), \
            f"Expected a Polytope object for point '{point}', but got {type(poly)} instead."

    @pytest.mark.parametrize('point', [
        [0],
        [1],
        [-2.5],
        [0, 0],
        [1.5, 0.8, 1.4],
        [-1, -2, -3, -4, -5],
        np.full((10,), 0.25),
    ])
    def test_parameterize_private_attributes(self, point: ArrayLike) -> None:
        """Test that the polytope created from a point has the correct properties"""
        poly = pes.poly_from_point(point)
        assert isinstance(poly, pes.Polytope), \
            f"Expected a Polytope object for point '{point}', but got {type(poly)} instead."
        assert poly._is_empty == False, \
            f"Polytope created from point '{point}' should not be empty."
        assert poly._is_singleton == True, \
            f"Polytope created from point '{point}' should be a singleton."
        assert poly._is_bounded == True, \
            f"Polytope created from point '{point}' should be bounded."
        assert poly._is_degen == True, \
            f"Polytope created from point '{point}' should be degenerate."
        assert poly._is_full_dim == False, \
            f"Polytope created from point '{point}' should not be full-dimensional."
        assert poly._is_pointed == True, \
            f"Polytope created from point '{point}' should be pointed."
        assert poly._dim == 0, \
            f"Polytope created from point '{point}' should have dimension 0."
        assert poly._vol == 0, \
            f"Polytope created from point '{point}' should have volume 0."
        assert poly._diam == 0, \
            f"Polytope created from point '{point}' should have diameter 0."
        assert poly._width == 0, \
            f"Polytope created from point '{point}' should have width 0."
        assert poly._chebcr[0] == approx(point), \
            f"Polytope created from point '{point}' should have Chebyshev center equal to the point itself."
        assert poly._chebcr[1] == 0, \
            f"Polytope created from point '{point}' should have Chebyshev radius equal to 0."

    def test_parameterize_properties(self) -> None:
        """Test that the properties of the polytope created from a point are correct"""

    def test_parameterize_properties_recalculated(self) -> None:
        """Test that the properties of the polytope created from a point agree with the recalculated values from the vertices and half-space representations"""

    @given(point=integers(min_value=1, max_value=N_MAX).flatmap(
        lambda n: lists(floats(min_value=-1E3, max_value=1E3, allow_nan=False, allow_infinity=False), min_size=n, max_size=n)
    ))
    def test_random_nd_polytope(self, point: ArrayLike) -> None:
        """Test that the polytope created from a random point in N dimensions is correct"""
        poly = pes.poly_from_point(point)
        assert isinstance(poly, pes.Polytope), \
            f"Expected a Polytope object for point '{point}', but got {type(poly)} instead."

    @pytest.mark.parametrize('point', [
        [np.nan],
        [np.inf],
        [-float('inf')],
        [0, float('nan')],
        [1.5, 0.8, np.nan],
    ])
    def test_parameterize_invalid_values(self, point: ArrayLike) -> None:
        """Test that the polytope created from a point with invalid values raises an error"""
        with pytest.raises(ValueError, match=re.escape(
            f"Point vector cannot contain inf or NaN values, received point={np.atleast_1d(point)}"
            )):
            _ = pes.poly_from_point(point)

    @pytest.mark.parametrize('point', [
        [[1, -2, 3, 4]],
        [[0]],
        np.array([0])[:, np.newaxis, np.newaxis, np.newaxis],
        [[0, 0]],
        [[1, 2, 3]],
        [[[1, 2], [3, 4]]],
        [[1, 2, 3], [4, 5, 6]],
        np.array([[[1, 2, 3], [4, 5, 6]]]),
    ])
    def test_parameterize_invalid_shape(self, point: ArrayLike) -> None:
        """Test that the polytope created from a point with invalid shape raises an error"""
        with pytest.raises(ValueError, match=re.escape(
            f"Point must be a 1D array, but received an array of shape {np.atleast_1d(point).shape}"
            )):
            _ = pes.poly_from_point(point)
