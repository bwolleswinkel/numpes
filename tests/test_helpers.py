from typing import TYPE_CHECKING
import importlib.util

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats

from tests.helpers import approx, deg2rad, lsort, normalize, rad2deg, requires, close_figures, wrap_angle
from tests.conftest import ATOL, RTOL

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


class TestApprox:
    """Tests for the `approx` helper function"""

    @pytest.mark.parametrize("actual, expected", [
        ([1.0, 2.0], [1.0 + RTOL / 2, 2.0 + ATOL / 2]),
        (ATOL / 2, 0),
    ])
    def test_parameterize_default_tol(self, actual: ArrayLike, expected: ArrayLike):
        assert actual == approx(expected), \
            f"Expected {actual} to be approximately equal to {expected} with default tolerances"
        
    def test_rtol(self):
        ...

    def test_atol(self):
        ...


class TestLsort:
    """Tests for the `lsort` helper function"""

    @pytest.mark.parametrize('arr, expected', [
        (np.array([[1, 2],
                   [0, 9],
                   [1, 1]]),
         np.array([[0, 9],
                   [1, 1],
                   [1, 2]])),
    ])
    def test_parameterize(self, arr: NDArray, expected: NDArray):
        assert np.array_equal(lsort(arr), expected), \
            f"Expected lsort to lexicographically sort the rows of {arr}, got {lsort(arr)} instead"

    def test_invalid_ndims_value_error(self):
        ...


class TestRadiansDegreesConversion:
    """Tests for the `rad2deg` and `deg2rad` helper functions"""

    @pytest.mark.parametrize("rad, deg", [
        (0, 0),
        (np.pi / 2, 90),
        (np.pi, 180),
        (3 * np.pi / 2, 270),
        (2 * np.pi, 360),
        (np.pi / 4, 45),
    ])
    def test_parameterize_conversion(self, rad: float, deg: float):
        assert rad == pytest.approx(deg2rad(deg)), \
            f"Expected {rad} radians to be approximately equal to {deg2rad(deg)} radians"
        assert deg == pytest.approx(rad2deg(rad)), \
            f"Expected {deg} degrees to be approximately equal to {rad2deg(rad)} degrees"


class TestWrapAngle:
    """Tests for the `wrap_angle` helper function"""

    @pytest.mark.parametrize('angle, unit, expected', [
        (0.0, 'deg', 0.0),
        (180.0, 'deg', 180.0),
        (-180.0, 'deg', 180.0),
        (181.0, 'deg', -179.0),
        (np.pi, 'rad', np.pi),
        (-np.pi, 'rad', np.pi),
        (3 * np.pi / 2, 'rad', -np.pi / 2),
    ])
    def test_parameterize_wrap_angle(self, angle: float, unit: str, expected: float):
        assert wrap_angle(angle, unit=unit) == pytest.approx(expected), \
            f"Expected wrap_angle({angle}, unit='{unit}') to be {expected}"

    @given(val=floats(min_value=-1E10, max_value=1E10, allow_infinity=False, allow_nan=False))
    def test_random_rad2deg_deg2rad_inverse(self, val: float):
        assert val == pytest.approx(rad2deg(deg2rad(val))), \
            f"Expected {val} degrees to be approximately equal to {rad2deg(deg2rad(val))} degrees"

    @given(val=floats(min_value=-1E10, max_value=1E10, allow_infinity=False, allow_nan=False))
    def test_random_deg2rad_rad2deg_inverse(self, val: float):
        assert val == pytest.approx(deg2rad(rad2deg(val))), \
            f"Expected {val} radians to be approximately equal to {deg2rad(rad2deg(val))} radians"


class TestNormalize:
    """Tests for the `normalize` helper function"""

    @pytest.mark.parametrize("arr, expected", [
        (np.array([[3.0, 4.0, 5.0],
                   [0.0, 0.0, 1.0]]),
         np.array([[0.6, 0.8, 1.0],
                   [0.0, 0.0, 1.0]])),
    ])
    def test_parameterize_normalize(self, arr: NDArray, expected: NDArray):
        assert np.allclose(normalize(arr), expected)

    def test_parameterize_normalize(self):
        ...


class TestRequires:
    """Tests for the `@requires` decorator"""

    def test_missing_dependency_import_error(self):
        @requires('foo',
                  ImportError,
                  match="Module 'foo' is required but not installed"
        )
        def bar():
            raise ImportError("Module 'foo' is required but not installed")

        bar()

    def test_missing_dependency_fail_import_error_incorrect_match(self):
        @requires('foo',
                  ImportError,
                  match="Module 'foo' is required but not installed"
        )
        def bar():
            raise ImportError("Module 'baz' is required but not installed")

        with pytest.raises(AssertionError, match='Regex pattern did not match.\n  '
                           'Expected regex: "Module \'foo\' is required but not installed"\n  '
                           'Actual message: "Module \'baz\' is required but not installed"'):
            bar()

    def test_runs_dependency_present(self):
        @requires('math',
                  ImportError,
                  match="..."
        )
        def bar():
            return 42

        assert bar() == 42


class TestCloseFigures:
    """Tests for the `@close_figures` decorator"""

    @pytest.mark.skipif(
        importlib.util.find_spec('matplotlib') is None,
        reason='matplotlib is not installed, skipping close_figures decorator test',
    )
    def test_single_figure(self):
        import matplotlib.pyplot as plt

        @close_figures
        def foo():
            plt.figure()
            assert len(plt.get_fignums()) == 1

        foo()
        assert len(plt.get_fignums()) == 0
