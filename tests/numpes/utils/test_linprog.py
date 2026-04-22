"""Test functions for the linprog utility module"""

from typing import TYPE_CHECKING

import numpy as np
import numpes as pes
from numpes.utils.linprog import Status, OptimizationProgramResult
import pytest

from tests.helpers import approx

if TYPE_CHECKING:
    from typing import Optional, Literal

    from numpy.typing import NDArray


class TestStatus:
    """Test the Status enum used in the LinearProgramResult dataclass"""

    def test_values(self):
        """Test that all expected status values exist"""
        assert Status.OPTIMAL
        assert Status.UNBOUNDED
        assert Status.INFEASIBLE
        assert Status.NUMERICAL_ISSUES_ENCOUNTERED
        assert Status.ITERATION_LIMIT_REACHED
        assert Status.UNKNOWN


    def test_uniqueness(self):
        """Test that all enum values are unique"""
        values = [status.value for status in Status]
        assert len(values) == len(set(values))


    def test_comparison(self):
        """Test enum equality and inequality"""
        assert Status.OPTIMAL == Status.OPTIMAL
        assert Status.OPTIMAL != Status.UNBOUNDED
        assert Status.UNBOUNDED != Status.INFEASIBLE


class TestLinearProgramResult:
    """Test the LinearProgramResult dataclass"""

    def test_fields(self):
        """Test that the LinearProgramResult dataclass has the expected fields"""
        res = OptimizationProgramResult(success=True, value=0.0, x_star=np.array([1.0, 2.0]), status=Status.OPTIMAL)
        assert res.success == True
        assert res.value == 0.0
        assert res.x_star == approx(np.array([1.0, 2.0]))
        assert res.status == Status.OPTIMAL

    def test_with_different_statuses(self):
        """Test LinearProgramResult with each possible status"""
        for status in Status:
            result = OptimizationProgramResult(
                success=(status == Status.OPTIMAL),
                value=0.0,
                x_star=np.zeros(1),
                status=status
            )
            assert result.status == status


class TestSolveLP:
    """Test the `solve_lp` function"""

    @pytest.mark.parametrize('backend', ['auto', 'scipy', 'cvxpy', 'pulp'])
    @pytest.mark.parametrize('c, A, b, A_eq, b_eq, bounds, expected_res', [
        (np.array([-1, 4]), 
         np.array([[-3,  1],
                   [ 1,  2]]),
         np.array([6, 4]),
         None,
         None,
         [(None, None), (-3, None)],
         OptimizationProgramResult(
             success=True, 
             status=Status.OPTIMAL,
             value=-22, 
             x_star=np.array([10, -3]),
         )),  # FROM: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html  # nopep8
        (np.array([10, 15]), 
         np.array([[1, 0],
                   [0, 1]]),
         np.array([100, 80]),
         np.array([[1, 1]]),
         np.array([150]),
         [(0, None), (0, None)],
         OptimizationProgramResult(
             success=True, 
             status=Status.OPTIMAL,
             value=1750, 
             x_star=np.array([100, 50]),
         )),  # FROM: https://www.datacamp.com/tutorial/linear-programming  # nopep8
        (np.array([-30, -40]), 
         np.array([[2, 1],
                   [1, 2]]),
         np.array([100, 80]),
         None,
         None,
         [(0, None), (0, None)],
         OptimizationProgramResult(
             success=True, 
             status=Status.OPTIMAL,
             value=-2000, 
             x_star=np.array([40, 20]),
         )),  # FROM: https://www.datacamp.com/tutorial/linear-programming  # nopep8
        (np.array([-1, -2]), 
         np.array([[ 2,  1],
                   [-4,  5],
                   [ 1, -2]]),
         np.array([20, 10, 2]),
         np.array([[-1, 5]]),
         np.array([15]),
         [(0, np.inf), (0, np.inf)],
         OptimizationProgramResult(
             success=True, 
             status=Status.OPTIMAL,
             value=-16.8181, 
             x_star=np.array([7.7272, 4.54545]),
         )),  # FROM: https://realpython.com/linear-programming-python/  # nopep8
        (np.array([-20, -12, -40, -25]), 
         np.array([[1, 1, 1, 1],
                   [3, 2, 1, 0],
                   [0, 1, 2, 3]]),
         np.array([50, 100, 90]),
         None,
         None,
         [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)],
         OptimizationProgramResult(
             success=True, 
             status=Status.OPTIMAL,
             value=-1900, 
             x_star=np.array([5, 0, 45, 0]),
         )),  # FROM: https://realpython.com/linear-programming-python/  # nopep8
    ])
    def test_solve_feasible_bounded(self, backend: Literal['scipy', 'cvxpy', 'pulp'], c: NDArray, A: NDArray, b: NDArray, A_eq: Optional[NDArray], b_eq: Optional[NDArray], bounds: list[tuple[float | None, float | None]], expected_res: OptimizationProgramResult):
        """Test solving a simple feasible and bounded linear program"""
        with pes.algo_options(lp_backend=backend):
            res = pes.utils.solve_lp(c, A=A, b=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            assert res.success == expected_res.success
            assert res.status == expected_res.status  
            assert res.value == approx(expected_res.value)
            assert res.x_star == approx(expected_res.x_star)

    def test_solve_feasible_unbounded(self):
        ...

    def test_solve_infeasible(self):
        ...

    def test_numerical_ill_conditioned(self):
        ...

    def test_initial_guess(self):
        ...

    def test_value_error(self):
        ...

    def test_solver_not_installed(self):
        ...