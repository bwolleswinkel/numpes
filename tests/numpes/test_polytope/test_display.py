"""Tests for displaying Polytope objects"""

from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

import numpy as np
import numpes as pes
import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers

from tests.strategies import poly_rand

if TYPE_CHECKING:
    from numpes import Polytope
    from tests.data.archetypes_polytope import PolytopeData

    
@pytest.mark.skipif(
    not importlib.util.find_spec('matplotlib'), reason="requires the matplotlib library"
)
@pytest.mark.display
def test_polytope_plot_archetypes_2d_nondegen(poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
    """Test plotting of all non-degenerate 2D archetypes"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    matplotlib.use('Agg')  # Use a non-interactive backend for testing
    poly, _ = poly_arch_nondegen_2d
    _, ax = plt.subplots()
    try:
        ax = poly.plot(show=False, ax=ax)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"
    finally:
        plt.close()
    

@pytest.mark.display
@given(poly=poly_rand(repr='both', n=2, exclude_degen=True))
def test_polytope_plot_random_2d_nondegen(poly: Polytope) -> None:
    """Test plotting random 2D non-degenerate polytopes"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    matplotlib.use('Agg')  # Use a non-interactive backend for testing
    _, ax = plt.subplots()
    try:
        ax = poly.plot(show=False, ax=ax)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"
    finally:
        plt.close()


@pytest.mark.display
def test_polytope_plot_archetypes_3d_nondegen(poly_arch_nondegen_3d: tuple[Polytope, PolytopeData]) -> None:
    """Test plotting of all non-degenerate 3D archetypes"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    matplotlib.use('Agg')  # Use a non-interactive backend for testing
    poly, _ = poly_arch_nondegen_3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax = poly.plot(show=False, ax=ax)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"
    finally:
        plt.close()


@pytest.mark.skip(reason="Runs into numerically unstable behavior when `exclude_degen=True` is not implemented")
@pytest.mark.display
@given(poly=poly_rand(repr='both', n=3, exclude_degen=True))
def test_polytope_plot_random_3d_nondegen(poly: Polytope) -> None:
    """Test plotting random 3D non-degenerate polytopes"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    matplotlib.use('Agg')  # Use a non-interactive backend for testing
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax = poly.plot(show=False, ax=ax)
        assert isinstance(ax, Axes), \
            f"Expected plot to return a matplotlib Axes object, but got {type(ax)} instead"
    finally:
        plt.close()


@pytest.mark.display
@given(poly=integers(
    min_value=1, max_value=10).flatmap(lambda n: poly_rand(repr='both', n=n, exclude_degen=True))
)
def test_polytope_plot_random_nd_value_error(poly: Polytope) -> None:
    """Test whether plotting random n-dimensional polytopes raises a ValueError"""
    import matplotlib.pyplot as plt
    assume(poly.n not in {1, 2, 3})
    _, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            poly.plot(show=False, ax=ax)
    finally:
        plt.close()


@pytest.mark.display
def test_polytope_plot_invalid_2d_ax() -> None:
    """Test whether providing an invalid `ax` argument for 2D plotting raises a ValueError"""
    import matplotlib.pyplot as plt
    poly = pes.Polytope(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    _, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            poly.plot(show=False, ax=ax)
    finally:
        plt.close()


@pytest.mark.display
def test_polytope_plot_invalid_3d_ax() -> None:
    """Test whether providing an invalid `ax` argument for 3D plotting raises a ValueError"""
    import matplotlib.pyplot as plt
    poly = pes.Polytope(np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        with pytest.raises(ValueError):
            poly.plot(show=False, ax=ax)
    finally:
        plt.close()
