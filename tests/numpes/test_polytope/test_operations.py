"""Tests for operations on polytopes"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, tuples

import numpes as pes

from tests.helpers import lsort, approx, normalize, requires
from tests.strategies import poly_rand, poly_rand_pair

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray, ArrayLike
    from numpes import Polytope

    from tests.data.archetypes_polytope import PolytopeData


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

    def test_arch_2d_shear_unit_horizontal(self, poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
        """Test 2D archetype polytopes sheared along the x-axis by 1 unit for their vertex and halfspace representations"""
        M_shear_x = np.array([[1, 1],
                              [0, 1]])
        poly, poly_data = poly_arch_nondegen_2d
        poly_res = M_shear_x @ poly
        match poly_data.name:
            case 'unit_square_2d':
                expected_verts = np.array([[0, 0],
                                           [1, 0],
                                           [2, 1],
                                           [1, 1]])
                expected_A = np.array([[ 0,  1],
                                       [ 0, -1],
                                       [-1,  1],
                                       [ 1, -1]])
                expected_b = np.array([1, 0, 0, 1])
            case 'centered_square_2d':
                expected_verts = expected_verts = np.array([[ 0,  1],
                                                            [ 2,  1],
                                                            [ 0, -1],
                                                            [-2, -1]])
                expected_A = np.array([[ 0,  1],
                                       [ 0, -1],
                                       [ 1, -1],
                                       [-1,  1]])
                expected_b = np.array([1, 1, 1, 1])
            case 'slim_beam_2d':
                expected_verts = np.array([[  0, 0],
                                           [100, 0],
                                           [101, 1],
                                           [  1, 1]])
                expected_A = np.array([[ 0,  1],
                                       [ 0, -1],
                                       [-1,  1],
                                       [ 1, -1]])
                expected_b = np.array([1, 0, 0, 100])
            case 'triangle_2d':
                expected_verts = np.array([[0, 0],
                                           [1, 0],
                                           [1, 1]])
                expected_A = np.array([[ 1,  0],
                                       [ 0, -1],
                                       [-1,  1]])
                expected_b = np.array([1, 0, 0])
            case 'house_2d':
                expected_verts = np.array([[0, 0  ],
                                           [1, 0  ],
                                           [1, 1  ],
                                           [2, 1  ],
                                           [2, 1.5]])
                expected_A = np.array([[ 0, -1],
                                       [-1,  1],
                                       [ 1, -1],
                                       [-1,  2],
                                       [ 1,  0]])
                expected_b = np.array([0, 0, 1, 1, 2])
            case _:
                raise NotImplementedError(f"Test not implemented for archetype with name '{poly_data.name}', verts={poly.verts},\nAb={poly.Ab}")
        assert lsort(normalize(poly_res.verts)) == approx(lsort(normalize(expected_verts))), \
            f"Shearing '{poly_data.name}' with verts_org=\n{poly.verts}\nalong the (positive) x-axis by 1 unit should yield vertices\n{lsort(expected_verts)},\nbut got verts_res=\n{lsort(poly_res.verts)} instead"
        assert lsort(normalize(poly_res.Ab)) == approx(lsort(normalize(np.column_stack((expected_A, expected_b))))), \
            f"Shearing '{poly_data.name}' with Ab_org=\n{poly.Ab}\nalong the (positive) x-axis by 1 unit should yield Ab=\n{lsort(np.column_stack((expected_A, expected_b)))},\nbut got Ab_res=\n{lsort(poly_res.Ab)} instead"

    def test_arch_2d_proj_onto_x_verts(self, poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
        """Test 2D archetype polytopes projected onto the x-axis for their vertex and half-space representations"""
        M_proj_x = np.array([[1, 0]])
        poly, poly_data = poly_arch_nondegen_2d
        poly_res = M_proj_x @ poly
        match poly_data.name:
            case 'unit_square_2d':
                expected_verts = np.array([[0],
                                           [1]])
                expected_A = np.array([[ 1],
                                       [-1]])
                expected_b = np.array([1, 0])
            case 'centered_square_2d':
                expected_verts = np.array([[-1],
                                           [ 1]])
                expected_A = np.array([[ 1],
                                       [-1]])
                expected_b = np.array([1, 1])
            case 'slim_beam_2d':
                expected_verts = np.array([[  0],
                                           [100]])
                expected_A = np.array([[ 1],
                                       [-1]])
                expected_b = np.array([100, 0])
            case 'triangle_2d':
                expected_verts = np.array([[0],
                                           [1]])
                expected_A = np.array([[ 1],
                                       [-1]])
                expected_b = np.array([1, 0])
            case 'house_2d':
                expected_verts = np.array([[0],
                                           [1]])
                expected_A = np.array([[ 1],
                                       [-1]])
                expected_b = np.array([1, 0])
            case _:
                raise NotImplementedError(f"Test not implemented for archetype with name '{poly_data.name}', verts={poly.verts},\nAb={poly.Ab}")
        assert lsort(normalize(poly_res.verts)) == approx(lsort(normalize(expected_verts))), \
            f"Projecting '{poly_data.name}' with verts_org=\n{poly.verts}\nwith non-square matrix M={M_proj_x}\nshould yield vertices\n{lsort(expected_verts)},\nbut got verts_res=\n{lsort(poly_res.verts)} instead"

    @requires(
        'cdd',
        ImportError,
        "The package 'pycddlib' is not installed. Please install it to enable converting from H-representation to V-representation.",
    )
    def test_arch_2d_proj_onto_x_Ab(self, poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
        """Test 2D archetype polytopes projected onto the x-axis for their vertex and half-space representations"""
        M_proj_x = np.array([[1, 0]])
        poly, poly_data = poly_arch_nondegen_2d
        poly_res = M_proj_x @ poly
        match poly_data.name:
            case 'unit_square_2d':
                expected_A = np.array([[ 1],
                                        [-1]])
                expected_b = np.array([1, 0])
            case 'centered_square_2d':
                expected_A = np.array([[ 1],
                                        [-1]])
                expected_b = np.array([1, 1])
            case 'slim_beam_2d':
                expected_A = np.array([[ 1],
                                        [-1]])
                expected_b = np.array([100, 0])
            case 'triangle_2d':
                expected_A = np.array([[ 1],
                                        [-1]])
                expected_b = np.array([1, 0])
            case 'house_2d':
                expected_A = np.array([[ 1],
                                        [-1]])
                expected_b = np.array([1, 0])
            case _:
                raise NotImplementedError(f"Test not implemented for archetype with name '{poly_data.name}', verts={poly.verts},\nAb={poly.Ab}")
        assert lsort(normalize(poly_res.Ab)) == approx(lsort(normalize(np.column_stack((expected_A, expected_b))))), \
            f"Shearing '{poly_data.name}' with Ab_org=\n{poly.Ab}\nwith non-square matrix M={M_proj_x}\nshould yield Ab=\n{lsort(np.column_stack((expected_A, expected_b)))},\nbut got Ab_res=\n{lsort(poly_res.Ab)} instead"

    def test_arch_2d_proj_into_3d_on_xy_verts(self, poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
        """Test whether 2D archetype polytopes projected into 3D space (whilst remaining on the xy-plane) works"""
        M_proj_xy = np.array([[1, 0],
                                [0, 1],
                                [0, 0]])
        poly, poly_data = poly_arch_nondegen_2d
        poly_res = M_proj_xy @ poly
        assert poly_res.n == 3, \
            f"Projecting '{poly_data.name}' with verts_org=\n{poly.verts}\nwith non-square matrix M=\n{M_proj_xy}\nshould yield a polytope with n=3, but got n={poly_res.n} instead"
        assert lsort(poly_res.verts[:, :2]) == approx(lsort(poly.verts)), \
            f"Projecting '{poly_data.name}' with verts_org=\n{poly.verts}\nwith non-square matrix M=\n{M_proj_xy}\nshould yield a polytope with verts_res[:, :2]=\n{poly.verts},\nbut got verts_res[:, :2]=\n{poly_res.verts[:, :2]}\ninstead"

    @requires(
        'cdd',
        ImportError,
        "The package 'pycddlib' is not installed. Please install it to enable converting from H-representation to V-representation.",
    )
    def test_arch_2d_proj_into_3d_on_xy_Ab(self, poly_arch_nondegen_2d: tuple[Polytope, PolytopeData]) -> None:
        """Test whether 2D archetype polytopes projected into 3D space (whilst remaining on the xy-plane) works"""
        M_proj_xy = np.array([[1, 0],
                              [0, 1],
                              [0, 0]])
        poly, poly_data = poly_arch_nondegen_2d
        poly_res = M_proj_xy @ poly
        assert poly_res.n == 3, \
            f"Projecting '{poly_data.name}' with verts_org=\n{poly.verts}\nwith non-square matrix M=\n{M_proj_xy}\nshould yield a polytope with n=3, but got n={poly_res.n} instead"
        assert lsort(normalize(poly_res.Ab[:, [0, 1, 3]])) == approx(lsort(normalize(poly.Ab))), \
            f"Projecting '{poly_data.name}' with Ab_org=\n{poly.Ab}\nwith non-square matrix M=\n{M_proj_xy}\nshould yield a polytope with Ab_res[:, [0, 1, 3]]=poly.Ab=\n{poly.Ab},\nbut got Ab_res[:, [0, 1, 3]]=\n{poly_res.Ab[:, [0, 1, 3]]}\ninstead"
        assert poly_res.Ab_eq == approx(np.array([[0, 0, 1, 0]])), \
            f"Projecting '{poly_data.name}' with Ab_eq_org=\n{poly.Ab_eq}\nwith non-square matrix M=\n{M_proj_xy}\nshould yield a polytope with Ab_eq_res=[0, 0, 1, 0]=\n{np.array([[0, 0, 1, 0]])},\nbut got Ab_eq_res=\n{poly_res.Ab_eq}\ninstead"

    @given(poly=integers(
        min_value=5, max_value=10).flatmap(lambda n: poly_rand(repr='vrepr', n=n, exclude_degen=False))
    )
    def test_random_invalid_right_multiplication_raises_invalid_operation_error(self, poly: Polytope) -> None:
        """Test whether trying the right-multiply a polytope with a matrix raises a `InvalidOperationError`"""
        M = np.random.randn(poly.n, poly.n)
        with pytest.raises(pes.InvalidOperationError, match=re.escape(
            "Right-hand matrix multiplication is not defined for polytopes. Only left-hand matrix multiplication is allowed by reversing the order of operands (i.e., use 'M @ poly' instead of 'poly @ M')."
            )):
            _ = poly @ M

    @pytest.mark.parametrize('M', [
        1,
        0,
        'k',
        np.nan,
        True,
        {'x': [1, 2],
         'y': [-3, 0]},
        [1],
        [[ 1, 2], 
         [-3, 0]],
        ([1, 2],
         [-3, 0]),
        ...,
    ])
    @given(poly=integers(
        min_value=5, max_value=10).flatmap(lambda n: poly_rand(repr='vrepr', n=n, exclude_degen=False))
    )
    def test_parameterize_invalid_wrong_type_matrix(self, poly: Polytope, M: Any) -> None:
        """Test that passing in anything other then a NumPy array for M raises a TypeError"""
        with pytest.raises(TypeError, match=re.escape(
            f"Input 'M' must be a NumPy array, but received an object of type '{type(M).__name__}'"
            )):
            _ = M @ poly

    @pytest.mark.coupled('pes.poly_from_point')
    @pytest.mark.parametrize('point', [
        [0],
        [1],
        [-1.5],
        [1, 0],
        [0.01, -0.99],
        [1, 1, 1],
        np.arange(5),
    ])
    def test_parameterize_singleton_with_vandermonde(self, point: ArrayLike) -> None:
        """Test that matrix multiplication of a singleton polytope will result in the correct representation"""
        n = len(point)
        poly = pes.poly_from_point(point)
        M = np.vander(np.arange(1, n + 1), increasing=True)
        poly_res = M @ poly
        assert poly_res.verts.shape == (1, n), \
            f"Multiplying point={point} (n={len(point)}) with Vandermonde matrix\n{M}\n should yield a singleton polytope with shape (1, {n}), but got shape {poly_res.verts.shape} instead"
        assert poly_res.verts == approx(np.atleast_2d(M @ point)), \
            f"Multiplying point={point} (n={len(point)}) with Vandermonde matrix\n{M}\n should yield a singleton polytope with vertex\n{np.atleast_2d(M @ point)},\nbut got\n{poly_res.verts}\ninstead"

    @given(args=tuples(
        integers(min_value=1, max_value=10).flatmap(lambda n: poly_rand(repr='vrepr', n=n, exclude_degen=False)),
        integers(min_value=1, max_value=10),
        integers(min_value=1, max_value=10),
    ))
    def test_random_invalid_incorrect_dimensions_matrix(self, args: tuple[Polytope, int]) -> None:
        """Test that passing in a matrix with incorrect dimensions raises a `DimensionError`"""
        poly, nrows, ncols = args
        assume(ncols != poly.n)
        M = np.arange(nrows * ncols).reshape(nrows, ncols)
        with pytest.raises(pes.DimensionError, match=re.escape(
            f"Input matrix 'M' must be of size (m, {poly.n}), received shape={M.shape}"
            )):
            _ = M @ poly
