"""Test module for the function in `polytope.py`"""

from typing import TYPE_CHECKING
import re

import numpy as np
import numpes as pes
from numpes import InvalidCombinationOfArguments
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from tests.conftest import ATOL, N_MAX
from tests.helpers import lsort, normalize, approx

if TYPE_CHECKING:
    from typing import Any, EllipsisType
    from numpy.typing import NDArray, ArrayLike


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