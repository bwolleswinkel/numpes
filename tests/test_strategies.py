import importlib.util
import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.errors import NonInteractiveExampleWarning

from tests.strategies import poly_rand, poly_rand_pair
from numpes import Polytope


class TestPolyRand:
    """Tests for the `poly_rand` strategy"""

    @given(poly_rand(repr='vrepr', n=3))
    def test_random_vrepr_3d(self, poly: Polytope):
        assert isinstance(poly, Polytope), \
            f"Expected a Polytope instance, got {type(poly)}"
        assert isinstance(poly._vrepr, tuple), \
            f"Expected _vrepr to be a tuple, got {type(poly._vrepr)}"
        assert poly._hrepr is None, \
            f"Expected _hrepr to be None for vrepr, got {type(poly._hrepr)}"
        assert poly.n == 3, \
            f"Expected dimension n=3, got n={poly.n}"

    @given(poly_rand(repr='hrepr', n=3))
    def test_random_hrepr_3d(self, poly: Polytope):
        assert isinstance(poly, Polytope), \
            f"Expected a Polytope instance, got {type(poly)}"
        assert poly._vrepr is None, \
            f"Expected _vrepr to be None for hrepr, got {type(poly._vrepr)}"
        assert isinstance(poly._hrepr, tuple), \
            f"Expected _hrepr to be a tuple, got {type(poly._hrepr)}"
        assert poly.n == 3, \
            f"Expected dimension n=3, got n={poly.n}"

    @pytest.mark.skipif(
        importlib.util.find_spec('cdd') is None,
        reason='cdd is not installed, skipping both-representation strategy test',
    )
    @given(poly_rand(repr='both', n=3))
    def test_random_both_3d(self, poly: Polytope):
        assert isinstance(poly, Polytope), \
            f"Expected a Polytope instance, got {type(poly)}"
        assert isinstance(poly._vrepr, tuple), \
            f"Expected _vrepr to be a tuple, got {type(poly._vrepr)}"
        assert isinstance(poly._hrepr, tuple), \
            f"Expected _hrepr to be a tuple, got {type(poly._hrepr)}"
        assert poly.n == 3, \
            f"Expected dimension n=3, got n={poly.n}"

    def test_random_both_3d_consistent(self):
        ...

    @given(poly=integers(1, 10).flatmap(lambda n: poly_rand(repr='vrepr', n=n)))
    def test_random_vrepr_nd_in_range(self, poly: Polytope):
        assert isinstance(poly, Polytope), \
            f"Expected a Polytope instance, got {type(poly)}"
        assert 1 <= poly.n <= 10, \
            f"Expected dimension n in range [1, 10], got n={poly.n}"

    def test_3d_invalid_repr_value_error(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NonInteractiveExampleWarning)
            with pytest.raises(ValueError, match="Unknown representation type"):
                poly_rand(repr='invalid', n=3).example()

    def test_random_nd_exclude_degen(self):
        ...


class TestPolyRandPair:
    """Tests for the `poly_rand_pair` strategy"""

    @given(poly_rand_pair(repr='vrepr', n=(1, 10), same_n=True))
    def test_random_vrepr_same_n_range(self, polys):
        poly_1, poly_2 = polys
        assert poly_1.n == poly_2.n, \
            f"Expected both polytopes to have the same dimension, got n_1={poly_1.n} and n_2={poly_2.n}"
        assert 1 <= poly_1.n <= 10, \
            f"Expected dimension n in range [1, 10], got n={poly_1.n}"

    @given(poly_rand_pair(repr='hrepr', n=(1, 10), same_n=False))
    def test_random_hrepr_same_n_false_range(self, polys):
        poly_1, poly_2 = polys
        assert 1 <= poly_1.n <= 10, \
            f"Expected dimension n in range [1, 10], got n={poly_1.n}"
        assert 1 <= poly_2.n <= 10, \
            f"Expected dimension n in range [1, 10], got n={poly_2.n}"
        
    def test_random_vrepr_nd_same_repr(self):
        ...

    def test_random_hrepr_nd_same_repr(self):
        ...

    def test_random_both_nd_same_repr(self):
        ...

    def test_random_both_exclude_degen(self):
        ...
