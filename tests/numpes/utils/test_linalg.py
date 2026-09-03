"""Test functions for the linalg utility module"""

from typing import TYPE_CHECKING

import numpy as np
import numpes as pes
import pytest
from hypothesis import given
from hypothesis.strategies import integers, floats

from tests.conftest import ATOL, N_MAX
from tests.helpers import approx, lsort, normalize

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestMinimizeVrepr:
    """Tests for the `pes.utils.minimize_vrepr` function"""

    @pytest.mark.parametrize('verts, rays', [
        (np.empty((0, 2)),
         np.array([[0, 0]])),
        (np.empty((0, 3)),
         np.array([[0, 0, 0],
                   [0, 0, 0]])),
    ])
    def test_parametrize_no_verts_zero_ray_maps_to_zero_vertex(self, verts: NDArray, rays: NDArray):
        """Test whether when provided with only a zero rays, this is correctly mapped to no rays, and a zero vertex"""
        verts_res, rays_res = pes.utils.minimize_vrepr(verts, rays)
        assert rays_res.shape[0] == 0, \
            f"Given verts={verts} and rays={rays}, expected reduced rays to be empty array with shape (0, {verts.shape[1]}), but got rays_res=\n{rays_res} with shape {rays_res.shape}"
        assert rays_res.size == 0, \
            f"Given verts={verts} and rays={rays}, expected reduced rays to be empty array with size 0, but got rays_res=\n{rays_res} with size {rays_res.size}"
        assert verts_res == approx(np.zeros((1, rays.shape[1]))), \
            f"Given verts={verts} and rays={rays}, expected reduced verts to be equal to a zero vertex with shape (1, {rays.shape[1]}), but got verts_res=\n{verts_res}"

    @pytest.mark.parametrize('verts, rays', [
        (np.array([[1]]),
         np.array([[ 1],
                   [-2]])),
        (np.empty((0, 1)),
         np.array([[ 0.5],
                   [-0.5]])),
        (np.zeros((1, 2)),
         np.array([[ 1,  0],
                   [-1,  0],
                   [ 0,  1],
                   [ 0, -1]])),
    ])
    def test_parameterize_ambient_space(self, verts: NDArray, rays: NDArray) -> None:
        """Test the reduction of rays that span the ambient space, which should return a canonical representation"""
        verts_res, rays_res = pes.utils.minimize_vrepr(verts, rays)
        expected_rays = np.vstack((np.eye(rays.shape[1]), -np.ones(rays.shape[1])))
        assert verts_res.shape == (0, rays.shape[1]), \
            f"Given verts=\n{verts}\nand rays=\n{rays},\nexpected reduced verts to empty array with shape (0, {rays.shape[1]}), but got verts_res=\n{verts_res}\nwith shape {verts_res.shape}"
        assert verts_res.size == 0, \
            f"Given verts=\n{verts}\nand rays=\n{rays},\nexpected reduced verts to empty array with size 0, but got verts_res=\n{verts_res}\nwith size {verts_res.size}"
        assert lsort(rays_res) == approx(lsort(expected_rays)), \
            f"Given verts=\n{verts}\nand rays=\n{rays},\nexpected reduced rays to be equal to the canonical representation [I; -1] (expected_rays=\n{expected_rays}),\nbut got rays_res=\n{rays_res}"

    @pytest.mark.parametrize('rays', [
        np.array([[1, 0],
                  [0, 1]]),
        np.array([[1, 1],
                  [1, 1],
                  [1, 0]]),
        np.array([[0, 1, 0],
                  [0, 0, 1]]),
    ])
    def test_parameterize_non_pointed_add_zero_vertex(self, rays: NDArray) -> None:
        """Test the reduction of rays that are non-pointed, which should add a zero vertex to the representation"""
        verts = np.empty((0, rays.shape[1]))
        verts_res, _ = pes.utils.minimize_vrepr(verts, rays) 
        assert verts_res == approx(np.zeros((1, rays.shape[1]))), \
            f"Given non-pointed rays=\n{rays},\nexpected reduced verts to be equal to a zero vertex with shape (1, {rays.shape[1]}), but got verts_res=\n{verts_res}"


class TestReduceRays:
    """Tests for the `pes.utils.reduce_rays` function"""

    @pytest.mark.parametrize('rays', [
        np.array([[1, 0],
                  [0, 1]]),
        np.array([[1, 1],
                  [1, 0]]),
        np.array([[ 1, 0],
                  [-1, 0]]),
        np.array([[1, 2, 3],
                  [4, 5, 6]]),
    ])
    def test_parameterize_non_redundant(self, rays: NDArray) -> None:
        """Test the reduction of non-redundant rays, which should not change the rays"""
        rays_res = pes.utils.reduce_rays(rays)
        assert lsort(rays_res) == approx(lsort(rays)), \
            f"Given (non-redundant) rays=\n{rays},\nexpected reduced rays to be equal to the original rays, but got rays_res=\n{rays_res}"
    
    @pytest.mark.parametrize('n', [
        1,
        2,
        3,
        5,
        10,
    ])
    def test_parameterize_empty(self, n: int) -> None:
        """Test the reduction of an empty array of rays which should return an empty array"""
        rays = np.empty((0, n))
        rays_res = pes.utils.reduce_rays(rays)
        assert rays_res.shape == (0, n), \
            f"Given empty rays with shape (0, {n}), expected reduced rays to be empty array with shape (0, {n}), but got rays_res=\n{rays_res} with shape {rays_res.shape}"
        assert rays_res.size == 0, \
            f"Given empty rays with shape (0, {n}), expected reduced rays to be empty array with size 0, but got rays_res=\n{rays_res} with size {rays_res.size}"

    @pytest.mark.parametrize('rays, expected_rays', [
        (np.array([[1, 0],
                   [0, 1],
                   [0, 0]]),
         np.array([[1, 0],
                   [0, 1]])),
        (np.array([[1, 0],
                   [0, 0]]),
         np.array([[1, 0]])),
        (np.array([[ 0, 0],
                   [ 1, 0],
                   [-1, 0]]),
         np.array([[ 1, 0],
                   [-1, 0]])),
        (np.array([[1, 2, 3],
                   [0, 0, 0],
                   [4, 5, 6]]),
         np.array([[1, 2, 3],
                   [4, 5, 6]])),
        (np.array([[0, 0, 0]]),
         np.array([[0, 0, 0]])),  # NOTE: A zero rays should actually return a zero ray, as that case is handled by `pes.utils.minimize_vrepr`
        (np.array([[0],
                   [0]]),
         np.array([[0]])),
    ])
    def test_parameterize_zero_ray(self, rays: NDArray, expected_rays: NDArray) -> None:
        """Test the reduction of rays which contain a zero vector, which should be removed as it is redundant"""
        rays_res = pes.utils.reduce_rays(rays)
        assert lsort(rays_res) == approx(lsort(expected_rays)), \
            f"Given rays=\n{rays},\n with zero ray, expected reduced rays to be equal to expected_rays=\n{expected_rays},\nbut got rays_res=\n{rays_res}"

    def test_parameterize_ambient_space(self) -> None:
        """Test the reduction of rays that span the ambient space"""

    def test_parameterize_single_ray(self) -> None:
        """Test the reduction of a single ray, which should not change the ray"""


class TestReduceVerts:
    """Tests for the `pes.utils.reduce_verts` function"""

    @pytest.mark.parametrize('verts, expected_verts', [
        (np.array([[-1],
                   [ 0],
                   [ 1]]),
         np.array([[-1],
                   [ 1]])),
        (np.array([[  0,   0],
                   [  0,   1],
                   [  1,   0],
                   [0.5, 0.5]]),
         np.array([[0, 0],
                   [0, 1],
                   [1, 0]])),
        (np.array([[0, 0],
                   [0, 1],
                   [0, 1],
                   [1, 0],
                   [1, 0]]),
         np.array([[0, 0],
                   [0, 1],
                   [1, 0]])),
    ])
    def test_parameterize_redundant_no_rays(self, verts: NDArray, expected_verts: NDArray) -> None:
        """Test the reduction of redundant vertices when there are no rays"""
        rays = np.empty((0, verts.shape[1]))
        verts_res = pes.utils.reduce_verts(verts, rays)
        assert lsort(verts_res) == approx(lsort(expected_verts)), \
            f"Given verts=\n{verts},\n with no rays, expected reduced vertices to be equal to expected_verts=\n{expected_verts},\nbut got verts_res=\n{verts_res}"

    def test_parameterize_redundant_repeated_verts(self) -> None:
        """Test the reduction of redundant vertices when there are repeated vertices, which should remove the duplicates"""

    def test_parameterize_non_redundant_no_rays(self) -> None:
        """Test the reduction of non-redundant vertices when there are no rays, which should not change the vertices"""

    @pytest.mark.parametrize('verts, rays, expected_verts', [
        (np.array([[0, 0],
                   [1, 1]]),
         np.array([[1, 0],
                   [0, 1]]),
         np.array([[0, 0]])),
    ])
    def test_parameterize_redundant_with_rays(self, verts: NDArray, rays: NDArray, expected_verts: NDArray) -> None:
        """Test the reduction of redundant vertices when there are rays"""
        verts_res = pes.utils.reduce_verts(verts, rays)
        assert lsort(verts_res) == approx(lsort(expected_verts)), \
            f"Given verts=\n{verts},\n with rays=\n{rays},\nexpected reduced vertices to be equal to expected_verts=\n{expected_verts},\nbut got verts_res=\n{verts_res}"

    def test_parameterize_non_redundant_with_rays(self) -> None:
        """Test the reduction of non-redundant vertices when there are rays, which should not change the vertices"""

    def test_parameterize_empty(self) -> None:
        """Test the reduction of an empty array of vertices which should return an empty array"""

    def test_parameterize_single_vertex(self) -> None:
        """Test the reduction of a single vertex, which should not change the vertex"""

    def test_parameterize_ambient_space(self) -> None:
        """Test the reduction of vertices that span the ambient space, which should an empty array"""

    @pytest.mark.coupled('pes.utils.conv')
    def test_parameterize_no_rays_equivalent_conv(self) -> None:
        """Test the reduction of vertices with no rays and verify that the result is equivalent to the convex hull of the vertices"""


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
