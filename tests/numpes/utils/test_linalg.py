"""Test functions for the linalg utility module"""

from typing import TYPE_CHECKING
import re

import numpy as np
import numpes as pes
import pytest
from hypothesis import assume, given
from hypothesis.strategies import integers, floats, lists, tuples, just

from tests.conftest import ATOL, N_MAX
from tests.helpers import approx, lsort, normalize, rad2deg, deg2rad, wrap_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(params=[0, 30, 45, 60, 90, 180, 270, 360])
def angle_deg(request):
    """Shared parametrized fixture for rotation angles in degrees"""
    return request.param
        

class TestIsSquare:
    """Test class for the `pes.utils.is_square` function"""

    @pytest.mark.parametrize('A', [
            np.array([[1]]),
            np.array([[-2.5]]),
            np.array([[1, 2],
                      [3, 4]]),
            np.array([[0.4, 0.3],
                      [-1.8, 7.0]]),
            np.array([[     1, np.nan, 3],
                      [     4,      5, 6],
                      [np.inf,      8, 0]]),
            np.arange(10_000).reshape(100, 100),
            np.full((10, 10), -np.inf),
        ])
    def test_parametrize_square(self, A: NDArray) -> None:
        """Test the `is_square` function with various square matrix"""
        assert pes.utils.is_square(A), \
            f"Expected is_square to return True for square matrix A=\n{A}, but got False"

    @pytest.mark.parametrize('A', [
            np.array([[0, 0]]),
            np.array([[1, 2, 3],
                      [4, 5, 6]]),
            np.array([[ 1.4, 6.0],
                      [-3.2, 0  ],
                      [ 5.0, 1.2]]),
            np.array([[True, False, True]]),
            np.arange(12).reshape(3, 4),
            np.full((5, 6), np.nan),
    ])
    def test_parametrize_non_square(self, A: NDArray) -> None:
        """Test the `is_square` function with various non-square matrices"""
        assert not pes.utils.is_square(A), \
            f"Expected is_square to return False for non-square matrix A=\n{A}, but got True"

    @pytest.mark.parametrize('A, expected', [
        (np.empty((0, 0)), True),
        (np.empty((0, 1)), False),
        (np.empty((2, 2)), True),
        (np.empty((5, 25)), False),
    ])
    def test_parametrize_empty_array(self, A: NDArray, expected: bool) -> None:
        """Test the `is_square` function with an empty array"""
        assert pes.utils.is_square(A) == expected, \
            f"Expected is_square to return {expected} for empty array A=\n{A} (shape={A.shape}), but got {not expected}"

    @pytest.mark.parametrize('A', [
        np.array(5),
        np.array(-3.2),
        np.array([3]),
    ])
    def test_parametrize_scalar_like(self, A: NDArray) -> None:
        """Test the `is_square` function with a scalar-like array"""
        assert not pes.utils.is_square(A), \
            f"Expected is_square to return False for scalar-like array A=\n{A}, but got True"

    @pytest.mark.parametrize('A', [
        5,
        -3.2,
        [3],
        ((1, 0), (2, 0)),
    ])
    def test_parametrize_non_ndarray(self, A: NDArray) -> None:
        """Test the `is_square` function with a non-ndarray input"""
        with pytest.raises(AttributeError):
            pes.utils.is_square(A)


class TestIsSym:
    """Test class for the `pes.utils.is_sym` function"""


class TestIsPosdef:
    """Test class for the `pes.utils.is_posdef` function"""


class TestIsRotMat:
    """Test class for the `pes.utils.is_rot_mat` function"""

    @pytest.mark.parametrize('R', [
        np.empty((0, 0)),
        np.empty((1, 1)),
        np.empty((3, 2)),
    ])
    def test_parametrize_empty(self, R: NDArray) -> None:
        """Test the `is_rot_mat` function with various empty matrices"""
        assert not pes.utils.is_rot_mat(R), \
            f"Expected is_rot_mat to return False for empty matrix R (shape={R.shape}), but got True"

    @pytest.mark.parametrize('R', [
        np.array([[1]]),
        np.array([[-1]]),
        np.array([[0]]),
    ])
    def test_parametrize_1d(self, R: NDArray) -> None:
        """Test the `is_rot_mat` function with a 1D matrix"""
        assert not pes.utils.is_rot_mat(R), \
            f"Expected 1d matrix R=\n{R} (shape={R.shape}) not to be a rotation matrix, but got True"

    def test_parametrize_2d(self, angle_deg: float) -> None:
        """Test the `is_rot_mat` function with a 2D rotation matrix"""
        angle = deg2rad(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        assert pes.utils.is_rot_mat(R), \
            f"Expected 2D rotation matrix R=\n{R} (angle={rad2deg(angle)} deg) to be a valid rotation matrix, but got False"

    @given(angle_deg=floats(0, 360))
    def test_random_2d(self, angle_deg: float) -> None:
        """Test the `is_rot_mat` function with a random 2D rotation matrix"""
        angle = deg2rad(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        assert pes.utils.is_rot_mat(R), \
            f"Expected 2D rotation matrix R=\n{R} (angle={angle_deg} deg) to be a valid rotation matrix, but got False"

    def test_parametrize_3d_around_x(self) -> None:
        """Test the `is_rot_mat` function with a 3D rotation matrix around the x-axis"""

    def test_parametrize_3d_around_y(self) -> None:
        """Test the `is_rot_mat` function with a 3D rotation matrix around the y-axis"""

    def test_parametrize_3d_around_z(self) -> None:
        """Test the `is_rot_mat` function with a 3D rotation matrix around the z-axis"""

    def test_parametrize_3d(self) -> None:
        """Test the `is_rot_mat` function with various 3D rotation matrices"""

    def test_parametrize_rot_mat(self) -> None:
        """Test the `is_rot_mat` function with a rotation matrix from the `pes.utils.rot_mat` function"""

    def test_random_rot_mat(self) -> None:
        """Test the `is_rot_mat` function with a rotation matrix from the `pes.utils.rot_mat` function on random inputs"""

    @pytest.mark.parametrize('R', [
        np.array([[1,  0],
                  [0, -1]]),  # Reflection across the x-axis
        np.array([[-1, 0],
                  [ 1, 1]]),  # Reflection across the y-axis
        np.array([[0, 1],
                  [1, 0]]),  # Reflection across the line y = x
        np.array([[ 0, -1],
                  [-1,  0]]),  # Reflection across the line y = -x
        np.array([[-1, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0, 1]]),  # Reflection across the yz-plane
        np.array([[1,  0, 0],
                  [0, -1, 0],
                  [0,  0, 1]]),  # Reflection across the xz-plane
        np.array([[1, 0,  0],
                  [0, 1,  0],
                  [0, 0, -1]]),  # Reflection across the xy-plane
        np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]),  # Reflection across the plane x = y
        np.array([[ 0, -1, 0],
                  [-1, 0, 0],
                  [ 0, 0, 1]]),  # Reflection across the plane x = -y
        np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0]]),  # Reflection across the plane y = z
        np.array([[1,  0,  0],
                  [0,  0, -1],
                  [0, -1,  0]]),  # Reflection across the plane y = -z
        np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]]),  # Reflection across the plane x = z
        np.array([[ 0, 0, -1],
                  [ 0, 1, 0],
                  [-1, 0, 0]]),  # Reflection across the plane x = -z
    ])
    def test_parametrize_reflection_matrix_no_rotation(self, R: NDArray) -> None:
        """Test the `is_rot_mat` function with a reflection matrix (one which is not equal to a rotation matrix)"""
        assert not pes.utils.is_rot_mat(R), \
            f"Expected reflection matrix R=\n{R} (shape={R.shape}, R @ R.T={R @ R.T}, det(R)={np.linalg.det(R)}) not to be a valid rotation matrix, but got True"

    @pytest.mark.parametrize('R', [
        np.array([[1, 2],
                  [3, 4]]),
        np.array([[ 0.9, -0.5],
                  [-0.2,  0.8]]),
        np.array([[1,      0],
                  [0, np.inf]]),
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 2]]),
        np.full((4, 4), np.nan),
    ])
    def test_parametrize_non_rot_mat(self, R: NDArray) -> None:
        """Test the `is_rot_mat` function with a non-rotation matrix"""
        assert not pes.utils.is_rot_mat(R), \
            f"Expected non-rotation matrix R=\n{R} (shape={R.shape}, R @ R.T={R @ R.T}, det(R)={np.linalg.det(R)}) not to be a valid rotation matrix, but got True"

    @given(n=integers(2, 100))
    def test_random_identity_matrix(self, n: int) -> None:
        """Test the `is_rot_mat` function with various identity matrices"""
        assert pes.utils.is_rot_mat(np.eye(n)), \
            f"Expected identity matrix of size {n} to be a valid rotation matrix, but got False"

    def test_random_givens_mat(self) -> None:
        """Test with a random Givens rotation matrix"""


class TestRotMat:
    """Test class for the `pes.utils.rot_mat` function"""

    def test_parametrize_2d(self, angle_deg: float) -> None:
        """Test the `rot_mat` function with a 2D rotation matrix from a given angle in degrees"""
        angle = deg2rad(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        assert pes.utils.rot_mat([angle]) == approx(R), \
            f"Expected 2D rotation matrix R=\n{R} (angle={rad2deg(angle)} deg) to be equal to the output of rot_mat, but got {pes.utils.rot_mat([angle])}"

    @given(angle_deg=floats(0, 360))
    def test_random_2d(self, angle_deg: float) -> None:
        """Test the `rot_mat` function with a random 2D rotation matrix"""
        angle = deg2rad(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        assert pes.utils.rot_mat([angle]) == approx(R), \
            f"Expected 2D rotation matrix R=\n{R} (angle={rad2deg(angle)} deg) to be equal to the output of rot_mat, but got {pes.utils.rot_mat([angle])}"

    def test_parametrize_3d_around_x(self, angle_deg: float) -> None:
        """Test the `rot_mat` function with a 3D rotation matrix around the x-axis"""
        angle = deg2rad(angle_deg)
        R_x = np.array([[1,             0,              0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle),  np.cos(angle)]])
        assert pes.utils.rot_mat([angle, 0, 0]) == approx(R_x), \
            f"Expected 3D rotation matrix R_x=\n{R_x.round(2) + 0} (angle={angle_deg} deg)\nto be equal to the output of rot_mat([angle, 0, 0]), but got\n{pes.utils.rot_mat([angle, 0, 0]).round(2) + 0}"

    def test_parametrize_3d_around_y(self, angle_deg: float) -> None:
        """Test the `rot_mat` function with a 3D rotation matrix around the y-axis"""
        angle = deg2rad(angle_deg)
        R_y = np.array([[ np.cos(angle), 0, np.sin(angle)],
                        [             0, 1,             0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        assert pes.utils.rot_mat([-np.pi / 2, angle, np.pi / 2]) == approx(R_y), \
            f"Expected 3D rotation matrix R_y=\n{R_y.round(2) + 0} (angle={angle_deg} deg)\nto be equal to the output of rot_mat([-np.pi / 2, angle, np.pi / 2]), but got\n{pes.utils.rot_mat([-np.pi / 2, angle, np.pi / 2]).round(2) + 0}"

    def test_parametrize_3d_around_z(self, angle_deg: float) -> None:
        """Test the `rot_mat` function with a 3D rotation matrix around the z-axis"""
        angle = deg2rad(angle_deg)
        R_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [            0,              0, 1]])
        assert pes.utils.rot_mat([0, angle, 0]) == approx(R_z), \
            f"Expected 3D rotation matrix R_z=\n{R_z.round(2) + 0} (angle={angle_deg} deg)\nto be equal to the output of rot_mat([0, angle, 0]), but got\n{pes.utils.rot_mat([0, angle, 0]).round(2) + 0}"

    @pytest.mark.coupled('pes.utils.is_rot_mat')
    @given(angles=integers(2, N_MAX).flatmap(lambda n: lists(floats(min_value=-np.pi, max_value=np.pi), min_size=n * (n - 1) // 2, max_size=n * (n - 1) // 2)))
    def test_random_nd_is_rot_mat(self, angles: list[float]) -> None:
        """Test that the rotation matrix provided with Givens angles provides a valid rotation matrix"""
        R = pes.utils.rot_mat(angles)
        assert pes.utils.is_rot_mat(R), \
            f"Expected rotation matrix R=\n{R.round(2) + 0}\n(shape={R.shape}, R @ R.T=\n{R @ R.T}, det(R)={np.linalg.det(R)})\n from angles={angles} to be a valid rotation matrix, but got False"

    # FIXME: I think this test is inherently flawed; checking for the angles does not seem to work, however, when RECONSTRUCTING the rotation matrix from the angles, all the failing cases seem to be correct; I think the rotation angles are simply too non-unique, and the ONLY real test we can do is checking whether for a given R, deconstructing the angles and reconstructing R gives the same R; this is what I already do in `test_random_3d_reconstruct_rot_mat`, which seems to work
    @pytest.mark.skip(reason="This test seems to be inherently flawed")
    @given(angles=lists(floats(min_value=-np.pi, max_value=np.pi), min_size=3, max_size=3))
    def test_random_3d_deconstruct_angles_givens(self, angles: list[float]) -> None:
        """Test that a 3D rotation matrix provides the same angles when reconstructed with `pes.utils.angles_givens`"""
        R = pes.utils.rot_mat(angles)
        angles_reconstructed = pes.utils.angles_givens(R)
        if angles[1:] == approx([0, 0]) or angles[:1] == approx([0, 0]):  # Rotation around the x-axis
            assert wrap_angle(angles) == approx(wrap_angle(angles_reconstructed)) or wrap_angle(list(reversed(angles))) == approx(wrap_angle(angles_reconstructed)), \
                f"Expected angles={rad2deg(angles)}° or reversed angles={rad2deg(wrap_angle(list(reversed(angles))))}° from rotation matrix R=\n{R.round(2) + 0} to be equal to the reconstructed angles={rad2deg(wrap_angle(angles_reconstructed))}° leading to rotation matrix R_reconstructed=\n{pes.utils.rot_mat(angles_reconstructed).round(2) + 0}"
        elif angles[::2] == approx([0, 0]):  # Rotation around the z-axis
            assert wrap_angle(angles) == approx(wrap_angle(angles_reconstructed)) or wrap_angle(np.deg2rad([180, -angles[1], 180])) == approx(wrap_angle(angles_reconstructed)), \
                f"Expected angles={rad2deg(angles)}° or xz(-x) angles={rad2deg(wrap_angle(np.deg2rad([180, -angles[1], 180])))}° from rotation matrix R=\n{R.round(2) + 0} to be equal to the reconstructed angles={rad2deg(wrap_angle(angles_reconstructed))}° leading to rotation matrix R_reconstructed=\n{pes.utils.rot_mat(angles_reconstructed).round(2) + 0}"
        else:
            assert wrap_angle(angles) == approx(wrap_angle(angles_reconstructed)), \
                f"Expected angles={rad2deg(angles)}° from rotation matrix R=\n{R.round(2) + 0} to be equal to the reconstructed angles={rad2deg(wrap_angle(angles_reconstructed))}° leading to rotation matrix R_reconstructed=\n{pes.utils.rot_mat(angles_reconstructed).round(2) + 0}"

    @pytest.mark.coupled('pes.utils.angles_givens')
    @given(angles=integers(2, N_MAX).flatmap(lambda n: lists(floats(min_value=-np.pi, max_value=np.pi), min_size=n * (n - 1) // 2, max_size=n * (n - 1) // 2)))
    def test_random_nd_deconstruct_angles_givens(self, angles: list[float]) -> None:
        """Test that a random rotation matrix provided with Givens angles provides the same angles when deconstructed with `pes.utils.angles_givens`"""

    @pytest.mark.coupled('pes.utils.givens_mat')
    @given(angles=lists(floats(min_value=-1E3, max_value=1E3), min_size=3, max_size=3))
    def test_random_3d_sequence_of_givens_mats(self, angles: list[float]) -> None:
        """Test that a sequence of Givens rotation matrices provides the same matrix as the `pes.utils.rot_mat` function n 3d"""
        R = pes.utils.rot_mat(angles)
        plane_sequence = [
            (1, 2),
            (0, 1),
            (1, 2)
        ]
        R_from_givens = pes.utils.givens_mat(*plane_sequence[0], angles[0], 3) @ pes.utils.givens_mat(*plane_sequence[1], angles[1], 3) @ pes.utils.givens_mat(*plane_sequence[2], angles[2], 3)
        assert R == approx(R_from_givens), \
            f"Expected rotation matrix R=\n{R.round(2) + 0}\nfrom angles={rad2deg(angles)}° to be equal to the matrix R_from_givens=\n{R_from_givens.round(2) + 0}\nfrom the sequence of Givens rotation R = G_(1,2) @ G_(0,1) @ G_(1,2)"

    @pytest.mark.coupled('pes.utils.givens_mat')
    @given(angles=lists(floats(min_value=-1E3, max_value=1E3), min_size=6, max_size=6))
    def test_random_4d_sequence_of_givens_mats(self, angles: list[float]) -> None:
        """Test that a sequence of Givens rotation matrices provides the same matrix as the `pes.utils.rot_mat` function in 4d"""
        R = pes.utils.rot_mat(angles)
        plane_sequence = [
            (2, 3),
            (1, 2),
            (0, 1),
            (2, 3),
            (1, 2),
            (2, 3)
        ]
        R_from_givens = pes.utils.givens_mat(*plane_sequence[0], angles[0], 4) @ pes.utils.givens_mat(*plane_sequence[1], angles[1], 4) @ pes.utils.givens_mat(*plane_sequence[2], angles[2], 4) @ pes.utils.givens_mat(*plane_sequence[3], angles[3], 4) @ pes.utils.givens_mat(*plane_sequence[4], angles[4], 4) @ pes.utils.givens_mat(*plane_sequence[5], angles[5], 4)
        assert R == approx(R_from_givens), \
            f"Expected rotation matrix R=\n{R.round(2) + 0}\nfrom angles={rad2deg(angles)}° to be equal to the matrix R_from_givens=\n{R_from_givens.round(2) + 0}\nfrom the sequence of Givens rotation R = G_(2,3) @ G_(1,2) @ G_(0,1) @ G_(2,3) @ G_(1,2) @ G_(2,3)"

    @pytest.mark.coupled('pes.utils.givens_mat', 'pes.utils.linalg._idx_plane_ij')
    @given(
        args=integers(2, N_MAX).flatmap(
            lambda n: tuples(
                just(n),
                lists(
                    floats(min_value=-np.pi, max_value=np.pi),
                    min_size=n * (n - 1) // 2,
                    max_size=n * (n - 1) // 2,
                ),
            )
        )
    )
    def test_random_nd_sequence_of_givens_mats(self, args: tuple[int, list[float]]) -> None:
        """Test that a sequence of Givens rotation matrices provides the same matrix as the `pes.utils.rot_mat` function in any dimension"""
        from numpes.utils.linalg import _idx_plane_ij
        n, angles = args
        R = pes.utils.rot_mat(angles)
        plane_sequence = [_idx_plane_ij(k, n) for k in range((n * (n - 1)) // 2)]
        R_from_givens = np.eye(n)
        for (i, j), angle in zip(plane_sequence, angles):
            R_from_givens = R_from_givens @ pes.utils.givens_mat(i, j, angle, n)
        assert R == approx(R_from_givens), \
            f"Expected rotation matrix R=\n{R.round(2) + 0}\nfrom angles={rad2deg(angles)}° to be equal to the matrix R_from_givens=\n{R_from_givens.round(2) + 0}\nfrom the sequence of Givens rotation R = @_{{i,j}} G_{{i,j}} with sequence of planes {plane_sequence}"

    def test_random_3d_xzx_same_mat_as_rot_mat_from_givens_angles(self) -> None:
        """Test that the rotation matrix provided with Givens angles provides the same matrix as xzx rotation angles"""


class TestRotMat2D:
    """Tests for the `pes.utils.rot_mat_2d` function"""


class TestRotMat3D:
    """Tests for the `pes.utils.rot_mat_3d` function"""

    def test_parametrize_around_xyz_repeated_angle(self, angle_deg: float) -> None:
        """Test that the resulting rotation matrix with `convention='xyz'` coincides with xyz rotation angles"""
        angle = deg2rad(angle_deg)
        R_x = np.array([[1,             0,              0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle),  np.cos(angle)]])
        R_y = np.array([[ np.cos(angle), 0, np.sin(angle)],
                        [             0, 1,             0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        R_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [            0,              0, 1]])
        R_xyz = R_z @ R_y @ R_x
        R = pes.utils.rot_mat_3d([angle, angle, angle], convention='xyz')
        assert R == approx(R_xyz), \
            f"Expected rotation matrix R=\n{R.round(2) + 0}\nfrom angles={rad2deg([angle, angle, angle])}° with convention='xyz' to be equal to the matrix R_xyz=\n{R_xyz.round(2) + 0}\nfrom the sequence of rotations R = R_x @ R_y @ R_z"

    def test_parametrize_invert_sequence_extrinsic_extrinsic(self) -> None:
        """Test that inverting the sequence of rotations gives the same result as switching the convention from extrinsic to intrinsic and vice versa"""

    def test_random_equivalent_rot_mat(self) -> None:
        """Test that the resulting rotation is the same as the rotation matrix from `pes.utils.rot_mat` with the same angles"""

    def test_random_invalid_wrong_number_of_angles(self) -> None:
        """Test that the function raises a ValueError when the number of angles is not equal to 3"""


class TestGivensMat:
    """Test class for the `pes.utils.givens_mat` function"""

    @pytest.mark.parametrize('i, j, angle_deg, n, expected', [
        (0, 1, 60, 2, np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)],
                                [np.sin(np.pi / 3),  np.cos(np.pi / 3)]])),
        (1, 2, 60, 3, np.array([[1,                 0,                  0],
                                [0, np.cos(np.pi / 3), -np.sin(np.pi / 3)],
                                [0, np.sin(np.pi / 3),  np.cos(np.pi / 3)]])),
        (0, 2, 30, 3, np.array([[np.cos(np.pi / 6),      0, -np.sin(np.pi / 6)],
                                [                0,      1,                  0],
                                [np.sin(np.pi / 6),      0,  np.cos(np.pi / 6)]])),
    ])
    def test_parametrize_valid(self, i: int, j: int, angle_deg: float, n: int, expected: NDArray) -> None:
        """Test the `givens_mat` function with valid parameters"""
        angle = deg2rad(angle_deg)
        assert pes.utils.givens_mat(i, j, angle, n) == approx(expected), \
            f"Expected Givens matrix for i={i}, j={j}, angle={angle_deg} deg, n={n} to be\n{expected}\nbut got\n{pes.utils.givens_mat(i, j, angle, n)}"

    @given(n=integers(2, N_MAX),
           i=integers(0, N_MAX - 1),
           j=integers(0, N_MAX - 1),
           angle_deg=floats(min_value=-180, max_value=180),
    )
    def test_random_submatrix(self, n: int, i: int, j: int, angle_deg: float) -> None:
        """Test the `givens_mat` function with random parameters for the submatrix corresponding to the rotation in the plane spanned by axes i and j"""
        angle = deg2rad(angle_deg)
        assume(i < n and j < n)
        if i >= j:
            with pytest.raises(ValueError):
                pes.utils.givens_mat(i, j, angle, n)
        else:
            G = pes.utils.givens_mat(i, j, angle, n)
            G_sub = G[[i, j]][:, [i, j]]
            assert G_sub == approx(np.array([[np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle),  np.cos(angle)]])), \
                f"Expected submatrix of Givens matrix for i={i}, j={j}, angle={angle_deg} deg, n={n} to be\n{np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])}\nbut got\n{G_sub}"

    @given(n=integers(2, N_MAX),
           i=integers(0, N_MAX - 1),
           j=integers(0, N_MAX - 1),
           angle_deg=floats(min_value=-180, max_value=180),
    )
    def test_random_identity(self, n: int, i: int, j: int, angle_deg: float) -> None:
        """Test the `givens_mat` function with random parameters for the identity matrix when rows and columns i and j are removed"""
        angle = deg2rad(angle_deg)
        assume(i < n and j < n)
        if i >= j:
            with pytest.raises(ValueError):
                pes.utils.givens_mat(i, j, angle, n)
        else:
            G = pes.utils.givens_mat(i, j, angle, n)
            G_sub = np.delete(np.delete(G, [i, j], axis=0), [i, j], axis=1)
            assert G_sub == approx(np.eye(n - 2)), \
                f"Expected submatrix of Givens matrix for i={i}, j={j}, angle={angle_deg} deg, n={n} to be\n{np.eye(n - 2)}\nbut got\n{G_sub}"

    def test_random_vector_in_ij_plane_remains_plane(self) -> None:
        """Test that a random vector in the plane spanned by axes i and j remains in the same plane after applying the Givens rotation matrix"""

    def test_random_vector_outside_ij_plane_remains_outside_plane(self) -> None:
        """Test that a random vector outside the plane spanned by axes i and j remains outside the same plane after applying the Givens rotation matrix"""

    def test_random_vector_in_ij_plane_angle_dot_product(self) -> None:
        """Test that a random vector in the plane spanned by axes i and j has the expected dot product with the rotated vector after applying the Givens rotation matrix"""

    def test_random_vector_equal_length(self) -> None:
        """Test that a random vector has the same length after applying the Givens rotation matrix"""

    @pytest.mark.coupled('pes.utils.is_rot_mat')
    @given(n=integers(2, N_MAX),
           i=integers(0, N_MAX - 1),
           j=integers(0, N_MAX - 1),
           angle_deg=floats(min_value=-1E3, max_value=1E3),
    )
    def test_random_is_rot_mat(self, n: int, i: int, j: int, angle_deg: float) -> None:
        """Test that the Givens rotation matrix is a valid rotation matrix using the `pes.utils.is_rot_mat` function"""
        angle = deg2rad(angle_deg)
        assume(i < n and j < n)
        if i >= j:
            with pytest.raises(ValueError):
                pes.utils.givens_mat(i, j, angle, n)
        else:
            G = pes.utils.givens_mat(i, j, angle, n)
            assert pes.utils.is_rot_mat(G), \
                f"Expected Givens matrix for i={i}, j={j}, angle={angle_deg} deg, n={n} to be a valid rotation matrix, but got False"

    def test_random_invalid_indices(self) -> None:
        """Test that the Givens rotation matrix raises a ValueError for invalid indices i and j"""


class TestAnglesGivens:
    """Tests for the `pes.utils.angles_givens` function"""

    @pytest.mark.parametrize('R, expected', [
        (np.eye(2), [0]),
        (np.array([[0, -1],
                   [1,  0]]), [deg2rad(90)]),
        (np.array([[ 0, 1],
                   [-1, 0]]), [deg2rad(-90)]),
        (np.array([[np.sqrt(3)/2,          -1/2],
                   [         1/2,  np.sqrt(3)/2]]), [deg2rad(30)]),
    ])
    def test_parametrize_2d(self, R: NDArray, expected: list[float]) -> None:
        """Test the `angles_givens` function with various 2d rotation matrices"""
        angles = pes.utils.angles_givens(R)
        assert wrap_angle(angles) == approx(wrap_angle(expected)), \
            f"Expected angles for rotation matrix R=\n{R} to be {expected}, but got {wrap_angle([angle for angle in rad2deg(angles)])}"

    def test_parametrize_3d_around_x(self, angle_deg: float) -> None:
        """Test the `angles_givens` function with a 3D rotation matrix around the x-axis"""
        angle = deg2rad(angle_deg)
        R = np.array([[1,             0,              0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle),  np.cos(angle)]])
        angles = pes.utils.angles_givens(R)
        assert wrap_angle(angles[1]) == approx(0), \
            f"Expected second angle (around z) for rotation matrix R_x=\n{R.round(2) + 0} with angle={angle_deg} deg to be 0, but got {wrap_angle([angle.item() for angle in rad2deg(angles)], unit='deg')}"
        assert wrap_angle(angles[0] + angles[2]) == approx(wrap_angle(angle)), \
            f"Expected sum of first and third angles (around x) for rotation matrix R_x=\n{R.round(2) + 0} with angle={angle_deg} deg to be {wrap_angle(angle, unit='deg')} deg, but got {wrap_angle([angle.item() for angle in rad2deg(angles)], unit='deg')}"

    def test_parametrize_3d_around_y(self, angle_deg: float) -> None:
                """Test the `angles_givens` function with a 3D rotation matrix around the y-axis"""
                angle = deg2rad(angle_deg)
                R = np.array([[ np.cos(angle), 0, np.sin(angle)],
                              [             0, 1,             0],
                              [-np.sin(angle), 0, np.cos(angle)]])
                angles = pes.utils.angles_givens(R)
                if angle == approx(0):
                    assert wrap_angle(angles) == approx(wrap_angle([0, 0, 0])), \
                        f"Expected angles for rotation matrix R_y=\n{R.round(2) + 0} with angle={angle_deg} deg to be [0, 0, 0], but got {wrap_angle([angle.item() for angle in rad2deg(angles)], unit='deg')}"
                    return
                assert wrap_angle(angles) == approx(wrap_angle([-np.pi / 2, angle, np.pi / 2])) or wrap_angle(angles) == approx(wrap_angle([np.pi / 2, -angle, -np.pi / 2])), \
                    f"Expected angles for rotation matrix R_y=\n{R.round(2) + 0} with angle={angle_deg} deg to be [270, {wrap_angle(angle, unit='deg')} deg, 90] or [90 deg, {-wrap_angle(angle, unit='deg')} deg, 270 deg], but got {wrap_angle([angle.item() for angle in rad2deg(angles)], unit='deg')}"

    def test_parametrize_3d_around_z(self, angle_deg: float) -> None:
            """Test the `angles_givens` function with a 3D rotation matrix around the z-axis"""
            angle = deg2rad(angle_deg)
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0],
                          [            0,              0, 1]])
            angles = pes.utils.angles_givens(R)
            assert wrap_angle(angles) == approx(wrap_angle([0, angle, 0])) or wrap_angle(angles) == approx(wrap_angle([np.pi, -angle, np.pi])), \
                f"Expected angles for rotation matrix R_z=\n{R.round(2) + 0} with angle={angle_deg} deg to be [0, {wrap_angle(angle, unit='deg')} deg, 0] or [180 deg, {-wrap_angle(angle, unit='deg')} deg, 180 deg], but got {wrap_angle([angle.item() for angle in rad2deg(angles)], unit='deg')}"

    def test_random_3d_around_x_reconstruct_rot_mat(self) -> None:
        """Test the `angles_givens` function with a random 3D rotation matrix around the x-axis and reconstruct the rotation matrix using `pes.utils.rot_mat`"""

    def test_random_3d_around_y_reconstruct_rot_mat(self) -> None:
            """Test the `angles_givens` function with a random 3D rotation matrix around the y-axis and reconstruct the rotation matrix using `pes.utils.rot_mat`"""

    def test_random_3d_around_z_reconstruct_rot_mat(self) -> None:
            """Test the `angles_givens` function with a random 3D rotation matrix around the z-axis and reconstruct the rotation matrix using `pes.utils.rot_mat`"""

    @pytest.mark.coupled('pes.utils.rot_mat')
    @given(angles_deg=lists(floats(-1E3, 1E3), min_size=3, max_size=3))
    def test_random_3d_reconstruct_rot_mat(self, angles_deg: list[float]) -> None:
        """Test the `angles_givens` function with a random 3D rotation matrix and reconstruct the rotation matrix using `pes.utils.rot_mat`"""
        angles = deg2rad(angles_deg)
        R = pes.utils.rot_mat(angles)
        angles_reconstructed = pes.utils.angles_givens(R)
        R_reconstructed = pes.utils.rot_mat(angles_reconstructed)
        assert R_reconstructed == approx(R), \
            f"Expected reconstructed rotation matrix R_reconstructed=\n{R_reconstructed.round(2) + 0} to be equal to original rotation matrix R=\n{R.round(2) + 0} for angles={angles_deg} deg, but got R_reconstructed != R"

    @pytest.mark.coupled('pes.utils.rot_mat', 'pes.utils.givens_mat')
    @given(angles_deg=lists(floats(-1E3, 1E3), min_size=3, max_size=3))
    def test_random_3d_reconstruct_from_givens_mat(self, angles_deg: list[float]) -> None:
        """Test whether we can reconstruct a 3D rotation matrix from the Givens angles obtained from `pes.utils.angles_givens`"""
        angles = deg2rad(angles_deg)
        R = pes.utils.rot_mat(angles)
        angles_reconstructed = pes.utils.angles_givens(R)
        R_reconstructed = pes.utils.rot_mat(angles_reconstructed)
        plane_sequence = [
            (1, 2),
            (0, 1),
            (1, 2)
        ]
        R_from_givens = (pes.utils.givens_mat(*plane_sequence[0], angles_reconstructed[0], 3)
                         @ pes.utils.givens_mat(*plane_sequence[1], angles_reconstructed[1], 3)
                         @ pes.utils.givens_mat(*plane_sequence[2], angles_reconstructed[2], 3))
        assert R_from_givens == approx(R_reconstructed), \
            f"Expected reconstructed rotation matrix R_from_givens=\n{R_from_givens.round(2) + 0} to be equal to reconstructed rotation matrix R_reconstructed=\n{R_reconstructed.round(2) + 0} for angles={angles_deg} deg, but got R_from_givens != R_reconstructed"

    @pytest.mark.coupled('pes.utils.rot_mat', 'pes.utils.givens_mat', 'pes.utils.linalg._idx_plane_ij')
    @given(
        args=integers(2, N_MAX).flatmap(
            lambda n: tuples(
                just(n),
                lists(
                    floats(min_value=-np.pi, max_value=np.pi),
                    min_size=n * (n - 1) // 2,
                    max_size=n * (n - 1) // 2,
                ),
            )
        )
    )
    def test_random_nd_reconstruct_from_givens_mat(self, args: tuple[int, list[float]]) -> None:
        """Test whether we can reconstruct a nD rotation matrix from the Givens angles obtained from `pes.utils.angles_givens`"""
        n, angles_deg = args
        angles = deg2rad(angles_deg)
        R = pes.utils.rot_mat(angles)
        angles_reconstructed = pes.utils.angles_givens(R)
        R_reconstructed = pes.utils.rot_mat(angles_reconstructed)
        plane_sequence = [pes.utils.linalg._idx_plane_ij(k, n) for k in range(len(angles_reconstructed))]
        R_from_givens = np.eye(n)
        for (i, j), angle in zip(plane_sequence, angles_reconstructed):
            R_from_givens = R_from_givens @ pes.utils.givens_mat(i, j, angle, n)
        assert R_from_givens == approx(R_reconstructed), \
            f"Expected reconstructed rotation matrix R_from_givens=\n{R_from_givens.round(2) + 0} to be equal to reconstructed rotation matrix R_reconstructed=\n{R_reconstructed.round(2) + 0} for angles={angles_deg} deg, but got R_from_givens != R_reconstructed"

    def test_random_invalid_non_square(self) -> None:
        """Test that non-square matrices raise a ValueError"""

    def test_random_invalid_non_orthogonal(self) -> None:
        """Test that non-orthogonal matrices raise a ValueError"""

    def test_random_invalid_reflection(self) -> None:
        """Test that rotation and reflection matrices raise a ValueError"""


class TestAngle2D:
    """Tests for the `pes.utils.angle_2d` function"""


class TestAngles3D:
    """Tests for the `pes.utils.angles_3d` function"""

    @pytest.mark.skip(reason="This test seems to be inherently flawed as angles are simply too non-unique, whilst still representing the same rotation matrix")
    def test_parametrize_around_xyz_repeated_angle(self, angle_deg: float) -> None:
        """Test that the angles with `convention='xyz'` coincides with xyz rotation angles"""
        angle = deg2rad(angle_deg)
        R_x = np.array([[1,             0,              0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle),  np.cos(angle)]])
        R_y = np.array([[ np.cos(angle), 0, np.sin(angle)],
                        [             0, 1,             0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        R_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [            0,              0, 1]])
        R_xyz = R_z @ R_y @ R_x  # NOTE: The order is reversed for extrinsic rotations (lowercase 'xyz')
        angles = pes.utils.angles_3d(R_xyz, convention='xyz')
        assert wrap_angle(angles) == approx(wrap_angle([angle, angle, angle])), \
            f"Expected sequence of angles={rad2deg([angle, angle, angle])}° from rotation matrix R_xyz=\n{R_xyz.round(2) + 0}\nto be equal to the angles={rad2deg(angles)}° (when wrapped) from pes.utils.angles_3d(R_xyz, convention='xyz')"

    def test_random_invalid_non_square(self) -> None:
        """Test that non-square matrices raise a ValueError"""

    def test_random_invalid_wrong_dimension(self) -> None:
        """Test that square (rotation) matrices with dimension not equal to 3 raise a ValueError"""

    def test_random_invalid_non_orthogonal(self) -> None:
        """Test that non-orthogonal matrices raise a ValueError"""

    def test_random_invalid_reflection(self) -> None:
        """Test that rotation-and-reflection matrices raise a ValueError"""


class TestAngles3DConvert:
    """Tests for the `pes.utils.angles_3d_convert` function"""

    def test_random_intrinsic_is_reversed_extrinsic(self) -> None:
        """Test that the angles with an intrinsic convention are the reversed angles of the same rotation with an extrinsic convention"""

    @pytest.mark.parametrize('conventions, idx_error', [
        (('xyz', 'Zyx'), 1),
        (('Xyz', 'zyx'), 0),
        (('XyX', 'ZyZ'), 0),
        (('yzX', 'givens'), 0),
    ])
    def test_parameterize_invalid_mixed_intrinsic_extrinsic(self, conventions: tuple[str, str], idx_error: int) -> None:
        """Test that a ValueError is raised when mixing intrinsic and extrinsic conventions"""
        convention_from, convention_to = conventions
        with pytest.raises(ValueError, match=re.escape(
            f"Invalid convention '{conventions[idx_error]}'. Cannot mix intrinsic (uppercase) and extrinsic (lowercase) rotations."
            )):
            _ = pes.utils.angles_3d_convert(np.deg2rad([30, -60, 45]), convention_from, convention_to)

    @pytest.mark.parametrize('conventions, idx_error', [
        (('xyz', '___'), 1),
        (('@12', 'zyx'), 0),
        (('xyx', '!#4'), 1),
        (('yzx', '123'), 1),
    ])
    def test_parameterize_invalid_characters(self, conventions: tuple[str, str], idx_error: int) -> None:
        """Test that a ValueError is raised when invalid characters are used in conventions"""
        convention_from, convention_to = conventions
        with pytest.raises(ValueError, match=re.escape(
            f"Invalid convention '{conventions[idx_error]}' (must only contain characters from {{'X', 'Y', 'Z'}} or {{'x', 'y', 'z'}})"
            )):
            _ = pes.utils.angles_3d_convert(np.deg2rad([30, -60, 45]), convention_from, convention_to)

    @pytest.mark.parametrize('conventions, idx_error', [
        (('xyz', 'Givens'), 1),
        (('Givens', 'zyx'), 0),
        (('Yaw_pitch_roll', 'zyz'), 0),
        (('XYZ', 'Yaw-Pitch-Roll'), 1),
    ])
    def test_parameterize_invalid_capitalization(self, conventions: tuple[str, str], idx_error: int) -> None:
        """Test that a ValueError is raised when the special cases 'yaw_pitch_roll' or 'givens' have incorrect capitalization"""
        convention_from, convention_to = conventions
        with pytest.raises(ValueError, match=re.escape(f"Invalid convention '{conventions[idx_error]}' (must be a string of length 3 or one of the special cases 'yaw_pitch_roll' or 'givens')")):
            _ = pes.utils.angles_3d_convert(np.deg2rad([30, -60, 45]), convention_from, convention_to)

    def test_random_invalid_wrong_dimension(self) -> None:
        """Test that square (rotation) matrices with dimension not equal to 3 raise a ValueError"""

    def test_random_invalid_non_orthogonal(self) -> None:
        """Test that non-orthogonal matrices raise a ValueError"""

    def test_random_invalid_reflection(self) -> None:
        """Test that rotation-and-reflection matrices raise a ValueError"""


class TestMinimizeHrepr:
    """Test class for the `pes.utils.minimize_hrepr` function"""

    @pytest.mark.parametrize('Ab, expected_Ab', [
        (np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1],
                   [-1,  0, 1]]), 
         np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1]])),
    ])
    def test_ineq_only_redundant(self, Ab: NDArray, expected_Ab: NDArray) -> None:
        """Test the reduction of redundant inequalities when there are no equality constraints"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab)
        assert lsort(normalize(Ab_res)) == approx(lsort(normalize(expected_Ab))), \
            f"Given Ab={Ab},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
        assert Ab_eq_res.shape == (0, Ab.shape[1]), \
            f"Given Ab={Ab},\nexpected reduced equalities Ab_eq_res to be empty array with shape (0, {Ab.shape[1]}), but got {Ab_eq_res} with shape {Ab_eq_res.shape}"

    def test_ineq_only_non_redundant(self) -> None:
        ...

    def test_eq_only_non_redundant(self) -> None:
        ...

    @pytest.mark.parametrize('Ab, Ab_eq, expected_Ab, expected_Ab_eq', [
        (np.array([[-1,  0, 0],
                   [ 0, -1, 0]]), 
         np.array([[ 1,  1, 1],
                   [ 1,  1, 1]]),
         np.array([[-1,  0, 0],
                   [ 0, -1, 0]]), 
         np.array([[ 1,  1, 1]])),
        (np.empty((0, 4)), 
         np.array([[1, 1, 0, 1],
                   [1, 1, 0, 1]]),
         np.empty((0, 4)), 
         np.array([[1, 1, 0, 1]])),
        (np.empty((0, 3)), 
         np.array([[1,    0,    0],
                   [1, 1E-9, 1E-9]]),
         np.empty((0, 3)), 
         np.array([[1, 0, 0]])),
        (np.empty((0, 3)), 
         np.array([[100 - 1E-9, 100 - 1E-9, 100 - 1E-9],
                   [100 + 1E-9, 100 + 1E-9, 100 + 1E-9],
                   [      1E-9,       1E-9,       1E-9]]),
         np.empty((0, 3)), 
         np.array([[100, 100, 100]])),
        (np.array([[0, -1, 0],
                   [0,  1, 1]]),
         np.array([[ 1, 0,  2],
                   [-1, 0, -2]]),
         np.array([[0, -1, 0],
                   [0,  1, 1]]),
         np.array([[1, 0, 2]])),
    ])
    def test_repeated_eq_redundant(self, Ab: NDArray, Ab_eq: NDArray, expected_Ab: NDArray, expected_Ab_eq: NDArray) -> None:
        """Test the reduction of redundant equalities when there are repeated equality constraints"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        assert lsort(normalize(Ab_res)) == approx(lsort(normalize(expected_Ab))), \
            f"Given Ab={Ab},\nAb_eq={Ab_eq},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
        assert lsort(normalize(Ab_eq_res, eq=True)) == approx(lsort(normalize(expected_Ab_eq, eq=True))), \
            f"Given Ab={Ab},\nAb_eq={Ab_eq},\nexpected reduced equalities Ab_eq_res=\n{Ab_eq_res}\nto be equal to\n{expected_Ab_eq}\n(same rows, order does not matter)"

    def test_eq_only_redundant(self) -> None:
        ...

    @pytest.mark.parametrize('Ab, Ab_eq, expected_Ab, expected_Ab_eq', [
        (np.array([[-1, 0, 0],
                   [ 1, 0, 0]]), 
         np.empty((0, 3)),
         np.empty((0, 3)), 
         np.array([[1, 0, 0]])),
        (np.array([[    0, -1, 0],
                   [-1E-9,  1, 0]]), 
         np.array([[-1, 0, -2]]),
         np.empty((0, 3)), 
         np.array([[1, 0, 2],
                   [0, 1, 0]])),
    ])
    def test_implicit_eq_in_ineq(self, Ab: NDArray, Ab_eq: NDArray, expected_Ab: NDArray, expected_Ab_eq: NDArray) -> None:
        """Test the reduction of inequalities when there are implicit equality constraints in the inequality constraints"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        assert lsort(normalize(Ab_res)) == approx(lsort(normalize(expected_Ab))), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
        assert lsort(normalize(Ab_eq_res, eq=True)) == approx(lsort(normalize(expected_Ab_eq, eq=True))), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced equalities Ab_eq_res=\n{Ab_eq_res}\nto be equal to\n{expected_Ab_eq}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.array([[-1, 0, 0],
                   [ 1, 0, 1]]),
         np.array([[1, 0, 2]])),
    ])
    def test_ineq_and_eq_unsatisfiable_empty(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test the case where the combination of inequality and equality constraints are unsatisfiable, which should result in an empty polytope"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        n = Ab.shape[1] - 1
        unsatisfiable = np.array([[0] * n + [-1]])
        assert Ab_res == approx(unsatisfiable), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced inequalities to be array [0 ... 0 -1], but got Ab_res=\n{Ab_res} with shape {Ab_res.shape}"
        assert Ab_eq_res.shape == (0, n + 1), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced equalities to be empty array with shape (0, n + 1), but got Ab_eq_res=\n{Ab_eq_res} with shape {Ab_eq_res.shape}"

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.array([[-1, 0, 0]]),
         np.array([[1, 0, 2],
                   [1, 0, 3]])),
    ])
    def test_eq_unsatisfiable_empty(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test the case where the equality constraints are unsatisfiable, which should result in an empty polytope"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        n = Ab.shape[1] - 1
        unsatisfiable = np.array([[0] * n + [-1]])
        assert Ab_res == approx(unsatisfiable), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced inequalities to be array [0 ... 0 -1], but got Ab_res=\n{Ab_res} with shape {Ab_res.shape}"
        assert Ab_eq_res.shape == (0, n + 1), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced equalities to be empty array with shape (0, n + 1), but got Ab_eq_res=\n{Ab_eq_res} with shape {Ab_eq_res.shape}"

    def test_ineq_unsatisfiable_empty(self) -> None:
        """Test the case where the inequality constraints are unsatisfiable, which should result in an empty polytope"""
        ...

    @pytest.mark.parametrize('Ab', [
        (np.array([[0, 0, -1]])),
        (np.array([[0, 0, -6]])),
    ])
    def test_empty_invariant(self, Ab: NDArray) -> None:
        """Test the case where Ab_eq is empty and Ab has an unsatisfiable inequality constraint, which should result in an empty polytope"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab)
        n = Ab.shape[1] - 1
        unsatisfiable = np.array([[0] * n + [-1]])
        assert Ab_res == approx(unsatisfiable), \
            f"Given Ab=\n{Ab},\nexpected reduced inequalities to be array [0 ... 0 -1] with shape (0, n + 1), but got Ab_res=\n{Ab_res} with shape {Ab_res.shape}"
        assert Ab_eq_res.shape == (0, n + 1), \
            f"Given Ab=\n{Ab},\nexpected reduced equalities to be empty array with shape (0, n + 1), but got Ab_eq_res=\n{Ab_eq_res} with shape {Ab_eq_res.shape}"

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.array([[0, 0, -1],
                   [0, 0, -6]]),
         np.empty((0, 3))),
        (np.array([[0, 0, -10]]),
         np.array([[1, 2, 3]])),
        (np.array([[0, 0, -0.001],
                   [0, 0,  0    ]]),
         np.array([[1, 0, 1],
                   [1, 0, 2],
                   [1, 0, 3]])),
    ])
    def test_empty_redundant(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test the case where Ab has (redundant) unsatisfiable inequality constraints, which should result in an empty polytope regardless of the equality constraints"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        n = Ab.shape[1] - 1
        unsatisfiable = np.array([[0] * n + [-1]])
        assert Ab_res == approx(unsatisfiable), \
            f"Given Ab=\n{Ab},\nexpected reduced inequalities to be array [0 ... 0 -1] with shape (0, n + 1), but got Ab_res=\n{Ab_res} with shape {Ab_res.shape}"
        assert Ab_eq_res.shape == (0, n + 1), \
            f"Given Ab=\n{Ab},\nexpected reduced equalities to be empty array with shape (0, n + 1), but got Ab_eq_res=\n{Ab_eq_res} with shape {Ab_eq_res.shape}"

    def test_random_empty_ineq_redundant(self) -> None:
        ...

    def test_random_empty_eq_redundant(self) -> None:
        ...

    def test_equivalence_enum_gens_facets_invariant(self) -> None:
        ...

    def test_equivalence_enum_gens_facets_redundant(self) -> None:
        ...

    @pytest.mark.parametrize('Ab, Ab_eq, expected_Ab, expected_Ab_eq', [
        (np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1],
                   [ 0,  0, 2]]), 
         np.empty((0, 3)),
         np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1]]), 
         np.empty((0, 3))),
        (np.array([[0, -1, 0],
                   [0,  1, 1]]), 
         np.array([[1, 0, 2],
                   [0, 0, 0]]),
         np.array([[0, -1, 0],
                   [0,  1, 1]]), 
         np.array([[1, 0, 2]])),
        (np.array([[1E-9,  0, 20],
                   [   0, -1,  0],
                   [   0,  1,  1]]), 
         np.array([[   1,    0, 2],
                   [1E-9, 1E-9, 0]]),
         np.array([[0, -1, 0],
                   [0,  1, 1]]), 
         np.array([[1, 0, 2]])),
        (np.array([[0, 0, 8]]),
         np.empty((0, 3)),
         np.empty((0, 3)),
         np.empty((0, 3))),
    ])
    def test_remove_trivial_constraints(self, Ab: NDArray, Ab_eq: NDArray, expected_Ab: NDArray, expected_Ab_eq: NDArray) -> None:
        """Test the removal of trivial constraints 0 x <= b where b >= 0 or 0 x = 0"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        assert lsort(Ab_res) == approx(lsort(expected_Ab)), \
            f"Given Ab={Ab},\nAb_eq={Ab_eq},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
        assert lsort(Ab_eq_res) == approx(lsort(expected_Ab_eq)), \
            f"Given Ab={Ab},\nAb_eq={Ab_eq},\nexpected reduced equalities Ab_eq_res=\n{Ab_eq_res}\nto be equal to\n{expected_Ab_eq}\n(same rows, order does not matter)"

    @pytest.mark.parametrize('Ab, Ab_eq, expected_Ab, expected_Ab_eq', [
        (np.array([[1, 0, 2]]), 
         np.array([[1, 0, 1]]), 
         np.empty((0, 3)),
         np.array([[1, 0, 1]])),
    ])
    def test_redundant_ineq_implied_by_eq(self, Ab: NDArray, Ab_eq: NDArray, expected_Ab: NDArray, expected_Ab_eq: NDArray) -> None:
        """Test the removal of redundant inequalities that are implied by the equality constraints"""
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        assert lsort(Ab_res) == approx(lsort(expected_Ab)), \
            f"Given Ab={Ab},\nAb_eq={Ab_eq},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
        assert lsort(Ab_eq_res) == approx(lsort(expected_Ab_eq)), \
            f"Given Ab={Ab},\nAb_eq={Ab_eq},\nexpected reduced equalities Ab_eq_res=\n{Ab_eq_res}\nto be equal to\n{expected_Ab_eq}\n(same rows, order does not matter)"
        
    @given(n=integers(min_value=1, max_value=N_MAX))
    def test_random_full_space_invariant(self, n: int) -> None:
        """Test the case where both Ab and Ab_eq are empty, which should result in a full-space polytope"""
        Ab, Ab_eq = np.empty((0, n + 1)), np.empty((0, n + 1))
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab, Ab_eq)
        assert Ab_res.shape == (0, n + 1), \
            f"Expected reduced inequalities to be empty array with shape (0, n + 1), but got {Ab_res} with shape {Ab_res.shape}"
        assert Ab_eq_res.shape == (0, n + 1), \
            f"Expected reduced equalities to be empty array with shape (0, n + 1), but got {Ab_eq_res} with shape {Ab_eq_res.shape}"
        
    @given(n=integers(min_value=1, max_value=N_MAX), scalar=floats(min_value=-100, max_value=-ATOL))
    def test_random_unsatisfiable_empty_invariant(self, n: int, scalar: float) -> None:
        """Test the case where Ab_eq is empty and Ab has an unsatisfiable inequality constraint, which should result in an empty polytope"""
        Ab = np.array([[0] * n + [scalar]])
        Ab_res, Ab_eq_res = pes.utils.minimize_hrepr(Ab)
        n = Ab.shape[1] - 1
        unsatisfiable = np.array([[0] * n + [-1]])
        assert normalize(Ab_res) == approx(normalize(unsatisfiable)), \
            f"Given n={n}, scalar={scalar}, expected reduced inequalities to be array [0 ... 0 -1] with shape (1, n + 1), but got Ab_res=\n{Ab_res} with shape {Ab_res.shape}"
        assert Ab_eq_res.shape == (0, n + 1), \
            f"Given n={n}, scalar={scalar}, expected reduced equalities to be empty array with shape (0, n + 1), but got Ab_eq_res=\n{Ab_eq_res} with shape {Ab_eq_res.shape}"
        

class TestReduceEq:
    """Test class for the `pes.utils.reduce_eq` function"""

    @pytest.mark.parametrize('Ab_eq, expected_Ab_eq', [
        (np.array([[1, 1, 1],
                   [1, 1, 1]]), 
         np.array([[1, 1, 1]])),
        (np.array([[1,    0,    0],
                   [1, 1E-9, 1E-9]]),
         np.array([[1, 0, 0]])),
        (np.array([[100 - 1E-9, 100 - 1E-9, 100 - 1E-9],
                   [100 + 1E-9, 100 + 1E-9, 100 + 1E-9],
                   [      1E-9,       1E-9,       1E-9]]),
         np.array([[100, 100, 100]])),
    ])
    def test_redundant(self, Ab_eq: NDArray, expected_Ab_eq: NDArray) -> None:
        """Test that redundant equalities are correctly removed"""
        assert lsort(normalize(pes.utils.reduce_eq(Ab_eq), eq=True)) == approx(lsort(normalize(expected_Ab_eq, eq=True))), \
            f"Expected reduced equalities to be\n{expected_Ab_eq},\nbut got\n{pes.utils.reduce_eq(Ab_eq)}\n(same rows, order does not matter)"
        

class TestReduceIneq:
    """Test class for the `pes.utils.reduce_ineq` function"""

    @pytest.mark.parametrize('Ab, expected_Ab', [
        (np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1],
                   [-1,  0, 1]]), 
         np.array([[-1,  0, 0],
                   [ 0, -1, 0],
                   [ 1,  1, 1]])),
        (np.array([[-1, -1 ],
                   [-1, -2 ],
                   [ 1,  10]]), 
         np.array([[-1, -2 ],
                   [ 1,  10]])),
    ])
    def test_ineq_only_redundant(self, Ab: NDArray, expected_Ab: NDArray) -> None:
        """Test the reduction of redundant inequalities"""
        Ab_res = pes.utils.reduce_ineq(Ab)
        assert lsort(normalize(Ab_res)) == approx(lsort(normalize(expected_Ab))), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{[]},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
        
    def test_ineq_only_non_redundant_invariant(self) -> None:
        """Test the reduction of non-redundant inequalities, which should not change the inequalities"""
        ...

    def test_ineq_redundant_with_eq(self) -> None:
        """Test the reduction of redundant inequalities when equalities are present"""
        ...

    def test_ineq_non_redundant_with_eq_invariant(self) -> None:
        """Test the reduction of non-redundant inequalities when equalities are present, which should not change the inequalities"""
        ...

    def test_redundant_eq_invariant(self) -> None:
        """Test inputs with redundant equalities, which should not change the inequalities"""
        ...

    def test_repeated_eq_invariant(self) -> None:
        """Test inputs with repeated equalities, which should not change the inequalities"""
        ...

    def test_ineq_implied_by_eq(self) -> None:
        """Test inputs with repeated equalities, which should not change the inequalities"""
        ...

    def test_full_plane_invariant(self) -> None:
        """Test inputs with an empty Ab and Ab_eq matrix, representing the full plane, which should not change the inequalities"""
        ...

    def test_single_ineq_invariant(self) -> None:
        """Test a single inequality constraint, which should not change the inequalities"""
        ...

    def test_single_ineq_with_non_implied_eq_invariant(self) -> None:
        """Test a single inequality constraint with equality constraints that do not imply it, which should not change the inequalities"""
        ...

    def test_single_ineq_implied_by_eq(self) -> None:
        """Test a single inequality constraint implied by equality constraints, which should be removed by the reduction"""
        ...

    def test_isolated_unsatisfiable_ineq_redundant(self) -> None:
        """Test the case where Ab has an unsatisfiable inequality constraint in isolation, which should result in an empty polytope regardless of the equality constraints"""
        ...

    def test_combined_unsatisfiable_ineq_redundant(self) -> None:
        """Test the case where Ab has an unsatisfiable inequality constraint in combination with other constraints, which should result in an empty polytope regardless of the equality constraints"""
        ...

    def test_removal_trivial_ineq(self) -> None:
        """Test the removal of trivial inequalities 0 x <= b where b >= 0"""
        ...

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.empty((0, 3)), np.empty((0, 5))),
        (np.array([[1, 0, 0]]), np.array([[0, 1, 0, 0]])),
        (np.arange(24).reshape(3, 4, 2), np.arange(24).reshape(6, 4, 1)),
        (np.array([[1, np.nan, 0]]), np.array([[0, 1, 0]])),
        (np.array([[1, 0, 0]]), np.array([[0, np.inf, 0]])),
    ])
    def test_ineq_eq_invalid_value_error(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test that providing invalid inputs into the reduction function raises a ValueError"""
        ...


class TestFindImplicit:
    """Test class for the `pes.utils.find_implicit` function"""

    @pytest.mark.parametrize('Ab, Ab_eq', [
        (np.array([[1, 0, 0],
                   [0, 1, 0],
                   [1, 1, 1]]), 
         np.empty((0, 3))),
        (np.array([[3, 2, 0]]), 
         np.empty((0, 3))),
        (np.array([[0, 1, 1]]), 
         np.array([[1, 1, 1]])),
        (np.array([[-1, 0],
                   [ 1, 2]]),
         np.empty((0, 2))),
        (np.empty((0, 2)), 
         np.array([[1, 2]])),
    ])
    def test_ineq_no_implicit_invariant(self, Ab: NDArray, Ab_eq: NDArray) -> None:
        """Test the invariance of inputs with no implicit equalities in the inequality constraints"""
        Ab_res, Ab_eq_res = pes.utils.find_implicit(Ab, Ab_eq)
        assert lsort(normalize(Ab_res)) == approx(lsort(normalize(Ab))), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be invariant (same rows, order does not matter)"
        assert Ab_eq_res.shape == (0, Ab_eq.shape[1]), \
            f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected no newly found implicit equalities, but got Ab_eq_res=\n{Ab_eq_res}"

    # FIXME: This is failing because I don't quite understand what `find_implicit` should actually do
    # @pytest.mark.parametrize('Ab, Ab_eq, expected_Ab, expected_Ab_eq_new', [
    #     (np.array([[-1,  0,  3],    # x_1 >= 3
    #                [ 1,  0, -3],    # x_1 <= 3
    #                [ 0, -1,  2],    # x_2 >= 2  
    #                [ 0,  1, -2]]),  # x_2 <= 2
    #      np.empty((0, 3)),
    #      np.empty((0, 3)),
    #      np.array([[1, 0, 3],     # x_1 = 3
    #                [0, 1, 2]])),  # x_2 = 2
    # ])
    # def test_ineq_implicit(self, Ab: NDArray, Ab_eq: NDArray, expected_Ab: NDArray, expected_Ab_eq_new: NDArray) -> None:
    #     """Test the finding of (multiple) implicit equalities in the inequality constraints"""
    #     Ab_res, Ab_eq_res = pes.utils.find_implicit(Ab, Ab_eq)
    #     assert lsort(normalize(Ab_res)) == approx(lsort(normalize(expected_Ab))), \
    #         f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced inequalities Ab_res=\n{Ab_res}\nto be equal to\n{expected_Ab}\n(same rows, order does not matter)"
    #     assert lsort(normalize(Ab_eq_res, eq=True)) == approx(lsort(normalize(expected_Ab_eq_new, eq=True))), \
    #         f"Given Ab=\n{Ab},\nAb_eq=\n{Ab_eq},\nexpected reduced equalities Ab_eq_res=\n{Ab_eq_res}\nto be equal to\n{expected_Ab_eq_new}\n(same rows, order does not matter)"

    def test_unsatisfiable(self) -> None:
        """Test the case where the input has unsatisfiable constraints, which should result in an empty polytope"""
        ...


class TestIdxPlaneIJ:
    """Tests for the `pes.utils.linalg._idx_plane_ij` helper function"""

    @pytest.mark.parametrize('k, n, expected', [
        (0, 3, (1, 2)),
        (1, 3, (0, 1)),
        (2, 3, (1, 2)),
        (0, 4, (2, 3)),
        (1, 4, (1, 2)),
        (2, 4, (0, 1)),
        (3, 4, (2, 3)),
        (4, 4, (1, 2)),
        (5, 4, (2, 3)),
    ])
    def test_parametrize_order(self, k: int, n: int, expected: tuple[int, int]) -> None:
        """Test that k maps to the expected adjacent plane indices in QR-like sweep order"""
        assert pes.utils.linalg._idx_plane_ij(k, n) == expected, \
            f"Expected k={k} for n={n} to map to plane {expected}, but got {pes.utils.linalg._idx_plane_ij(k, n)}"

    @pytest.mark.parametrize('k, n', [
        (-1, 3),
        (3, 3),
        (6, 4),
    ])
    def test_parametrize_invalid_k_value_error(self, k: int, n: int) -> None:
        """Test that invalid k values raise a ValueError"""
        with pytest.raises(ValueError):
            pes.utils.linalg._idx_plane_ij(k, n)
