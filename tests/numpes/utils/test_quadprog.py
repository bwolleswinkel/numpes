"""Test functions for the quadprog utility module"""

from typing import TYPE_CHECKING

import numpy as np
import numpes as pes
from numpes.utils.quadprog import _solve_qp_scipy
import pytest

from tests.helpers import approx

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray


FEASIBLE_CASES = [
    (np.array([[2, 0],
               [0, 2]]),  # Q
     np.array([-4, -6]),  # c
     np.array([[ 1,  1],
               [-1,  0],
               [ 0, -1]]),  # A
     np.array([3, 0, 0]),  # b
     None,  # A_eq
     None,  # b_eq
     None,  # bounds
     pes.utils.OptimizationProgramResult(success=True, 
                                         status=pes.utils.Status.OPTIMAL,
                                         value=-11,  # NOTE: There is a mistake on the webpage, the optimal value is incorrectly reported as -9
                                         x_star=np.array([1, 2]))),  # FROM: https://dilipkumar.medium.com/quadratic-programming-basics-0609ddfb1408  # nopep8
    (np.array([[2, 0],
               [0, 2]]),  # Q
     np.array([-6, -8]),  # c
     None,  # A
     None,  # b
     None,  # A_eq
     None,  # b_eq
     None,  # bounds
     pes.utils.OptimizationProgramResult(success=True, 
                                         status=pes.utils.Status.OPTIMAL,
                                         value=-25,  # NOTE: The optimal value is not provided in the paper
                                         x_star=np.array([3, 4]))),  # FROM: https://ecal.studentorg.berkeley.edu/files/ce191/CH02-QuadraticProgramming.pdf  # nopep8
    ((M := np.array([[ 1, 2, 0],
                     [-8, 3, 2],
                     [ 0, 1, 1]])).T @ M,  # Q
     -M.T @ (b := np.array([3, 2, 3])),  # c
     np.array([[ 1, 2,  1],
               [ 2, 0,  1],
               [-1, 2, -1]]),  # A
     np.array([3, 2, -2]),  # b
     None,  # A_eq
     None,  # b_eq
     None,  # bounds
     pes.utils.OptimizationProgramResult(success=True, 
                                         status=pes.utils.Status.OPTIMAL,
                                         value=-5.5921,  # NOTE: The optimal value is not provided in the paper
                                         x_star=np.array([0.12997347, -0.06498674, 1.74005305]))),  # FROM: https://scaron.info/blog/quadratic-programming-in-python.html  # nopep8
    (np.array([[2, 0], 
               [0, 8]]),  # Q
     np.array([0, -32]),  # c
     np.array([[ 1,  1],
               [-1,  2],
               [-1,  0],
               [ 0, -1],
               [ 0,  1]]),  # A
     np.array([7, 4, 0, 0, 4]),  # b
     None,  # A_eq
     None,  # b_eq
     None,  # bounds
     pes.utils.OptimizationProgramResult(success=True, 
                                         status=pes.utils.Status.OPTIMAL,
                                         value=8 - (c0 := 64),  # NOTE: The original problem defines (1/2)*x.T*H*x + c*x + c0, with c0 = 64
                                         x_star=np.array([2, 3]))),  # FROM: https://stackoverflow.com/questions/17009774/quadratic-program-qp-solver-that-only-depends-on-numpy-scipy  # nopep8
]


class SolveQPMixin:
    """Mixin for QP backend test cases"""

    solve_fn = None

    @pytest.mark.parametrize('Q, c, A, b, A_eq, b_eq, bounds, expected_res',
        FEASIBLE_CASES,
    )
    def test_feasible(self,
                      Q: NDArray,
                      c: Optional[NDArray],
                      A: Optional[NDArray],
                      b: Optional[NDArray],
                      A_eq: Optional[NDArray],
                      b_eq: Optional[NDArray],
                      bounds: None,
                      expected_res: pes.OptimizationProgramResult,
                      ) -> None:
        """Test solving a feasible and bounded linear program."""
        res = self.solve_fn(Q, c, A=A, b=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        assert res.success == expected_res.success
        assert res.status == expected_res.status
        assert res.value == approx(expected_res.value, atol=1E-4)
        assert res.x_star == approx(expected_res.x_star, atol=1E-4)


class TestSolveQPScipy(SolveQPMixin):
    """Test the `_solve_qp_scipy` function with the SciPy backend."""

    solve_fn = staticmethod(_solve_qp_scipy)


class TestSolveLP(SolveQPMixin):
    """Test the `pes.utils.solve_qp` function"""

    solve_fn = staticmethod(pes.utils.solve_qp)

    def test_solve_zero_decision_variables(self):
        """Test how all backends handle a linear program with zero decision variables"""

    def test_solve_feasible_unbounded(self):
        ...

    def test_solve_set_backend(self):
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

    def test_random_feasible(self) -> None:
        """Test solving a randomly generated quadratic program that is feasible"""
