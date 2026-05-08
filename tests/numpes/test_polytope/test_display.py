"""Tests for displaying Polytope objects"""

from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

import numpy as np
import numpes as pes
import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    matplotlib.use('Agg')  # Use a non-interactive backend for testing
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

from tests.conftest import N_MAX
from tests.strategies import poly_rand
from tests.helpers import close_figures, requires

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from numpes import Polytope
    from tests.data.archetypes_polytope import PolytopeData


class TestPolytopeRepr:
    """Tests for the `Polytope.__repr__` method"""

    @pytest.mark.parametrize('poly, expected_str', [
        (pes.poly([[1, 0],
                   [0, 1],
                   [0, 0]]),
         "Polytope(_vrepr=(NDArray[shape=(3, 2), dtype=int64], NDArray[shape=(0, 2), dtype=float64]), _hrepr=None, _is_empty=None, _is_degen=None, _is_bounded=None, _is_full_dim=None, _is_pointed=None, _is_singleton=None, _dim=None, _vol=None, _chebcr=None)"),
    ])
    def test_parameterize_str(self, poly: Polytope, expected_str: str) -> None:
        assert repr(poly) == expected_str, \
            f"Expected __repr__ to return\n{expected_str}\nbut got \n{repr(poly)}\n instead"

    @pytest.mark.parametrize('poly, expected_str', [
        (pes.poly([[1, 0],
                   [0, 1],
                   [0, 0]]),
         "Polytope(_vrepr=(NDArray[shape=(3, 2), dtype=int64], NDArray[shape=(0, 2), dtype=float64]), _hrepr=None, _is_empty=None, _is_degen=None, _is_bounded=None, _is_full_dim=None, _is_pointed=None, _is_singleton=None, _dim=None, _vol=None, _chebcr=None)"),
        (pes.poly([[-1,  0],
                   [ 0, -1],
                   [ 1,  0],
                   [ 0,  1]], [0, 0, 1.0, 1.0]),
         "Polytope(_vrepr=None, _hrepr=(NDArray[shape=(4, 3), dtype=float64], NDArray[shape=(0, 3), dtype=float64]), _is_empty=None, _is_degen=None, _is_bounded=None, _is_full_dim=None, _is_pointed=None, _is_singleton=None, _dim=None, _vol=None, _chebcr=None)"),
    ])
    def test_parameterize_print(self, poly: Polytope, expected_str: str, capsys) -> None:
        print(repr(poly))
        captured = capsys.readouterr()
        assert captured.out == f"{expected_str}\n", \
            f"Expected print(repr(poly)) to output\n{expected_str}\nbut got\n{captured.out}\n instead"

    @pytest.mark.skip(reason="Formatting is not yet implemented")
    def test_parameterize_format(self) -> None:
        ...
        
    @given(n=integers(min_value=1, max_value=N_MAX))
    def test_random_empty(self, n: int) -> None:
        poly = pes.poly_empty(n)
        expected_str = f"Polytope(_vrepr=(NDArray[shape=(0, {n}), dtype=float64], NDArray[shape=(0, {n}), dtype=float64]), _hrepr=(NDArray[shape=(1, {n + 1}), dtype=int64], NDArray[shape=(0, {n + 1}), dtype=float64]), _is_empty=True, _is_degen=True, _is_bounded=True, _is_full_dim=False, _is_pointed=True, _is_singleton=False, _dim=0, _vol=0, _chebcr=(NDArray[shape=({n},), dtype=float64], nan))"
        assert repr(poly) == expected_str, \
            f"Expected __repr__ to return\n{expected_str}\nbut got \n{repr(poly)}\n instead"


@requires(
    'matplotlib',
    ImportError,
    "Matplotlib is required for plotting. Please install it with 'pip install matplotlib' and try again.",
)
@close_figures
class TestPolytopePlot:
    """Tests for the `Polytope.plot` method"""

    @pytest.mark.display
    def test_polytope_plot_archetypes_2d_nondegen(self, poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
        """Test plotting of all non-degenerate 2D archetypes"""
        poly, _ = poly_arch_nondegen_2d
        ax = poly.plot(show=False)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"

    @pytest.mark.parametrize('verts, Ab', [
        (np.array([[1, 0],
                   [0, 1],
                   [0, 0]]),
         np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1]])),
    ])
    def test_parameterize_poly_both_repr(self, verts: NDArray, Ab: NDArray) -> None:
        """Test whether parameterizing both H- and V-representation works without error"""
        poly = pes.Polytope(verts)
        poly._hrepr = (Ab, np.empty((0, Ab.shape[1])))
        ax = poly.plot(show=False)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"

    # @pytest.mark.display
    # @given(poly=poly_rand(repr='both', n=2, exclude_degen=True))
    # def test_polytope_plot_random_2d_nondegen(self, poly: Polytope) -> None:
    #     """Test plotting random 2D non-degenerate polytopes"""
    #     ax = poly.plot(show=False)
    #     assert isinstance(ax, Axes), \
    #         f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"

    @pytest.mark.display
    def test_polytope_plot_archetypes_3d_nondegen(self, poly_arch_nondegen_3d: tuple[Polytope, PolytopeData]) -> None:
        """Test plotting of all non-degenerate 3D archetypes"""
        poly, _ = poly_arch_nondegen_3d
        ax = poly.plot(show=False)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"

    # @pytest.mark.skip(reason="Runs into numerically unstable behavior when `exclude_degen=True` is not implemented")
    # @pytest.mark.display
    # @given(poly=poly_rand(repr='both', n=3, exclude_degen=True))
    # def test_polytope_plot_random_3d_nondegen(self, poly: Polytope) -> None:
    #     """Test plotting random 3D non-degenerate polytopes"""
    #     ax = poly.plot(show=False)
    #     assert isinstance(ax, Axes), \
    #         f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"

    # @pytest.mark.display
    # @given(poly=integers(
    #     min_value=1, max_value=N_MAX).flatmap(lambda n: poly_rand(repr='both', n=n, exclude_degen=True))
    # )
    # def test_polytope_plot_random_nd_value_error(self, poly: Polytope) -> None:
    #     """Test whether plotting random n-dimensional polytopes raises a ValueError"""
    #     assume(poly.n not in {1, 2, 3})
    #     with pytest.raises(ValueError):
    #         poly.plot(show=False)

    # def test_passing_ax_arg_valid(self) -> None:
    #     ...

    @pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason="Matplotlib is required for creating the 2D axes")
    def test_polytope_plot_invalid_2d_ax(self) -> None:
        """Test whether providing an invalid `ax` argument for 2D plotting raises a ValueError"""
        poly = pes.Polytope(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        _, ax = plt.subplots()
        with pytest.raises(ValueError):
            poly.plot(show=False, ax=ax)

    @pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason="Matplotlib is required for creating the 3D axes")
    def test_polytope_plot_invalid_3d_ax(self) -> None:
        """Test whether providing an invalid `ax` argument for 3D plotting raises a ValueError"""
        poly = pes.Polytope(np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        with pytest.raises(ValueError):
            poly.plot(show=False, ax=ax)
