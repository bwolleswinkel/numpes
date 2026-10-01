"""Module containing quadratic programming functionality"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from numpes._config import CFG
from numpes.utils.linprog import Status, OptimizationProgramResult

try:
    import cvxpy as cvx
    CVXPY_INSTALLED: bool = True
except ImportError:
    CVXPY_INSTALLED = False

if TYPE_CHECKING:
    from typing import Optional, Sequence, Literal

    from numpy.typing import NDArray


def _solve_qp_scipy(Q: NDArray,
                    c: Optional[NDArray] = None,
                    A: Optional[NDArray] = None,
                    b: Optional[NDArray] = None,
                    A_eq: Optional[NDArray] = None,
                    b_eq: Optional[NDArray] = None,
                    bounds: Optional[Sequence[tuple[float | None, float | None]]] = None,
                    x_0: Optional[NDArray] = None,
                    ) -> OptimizationProgramResult:
    """Solve a quadratic program using SciPy's `minimize` function and sequential least squares method 'SLSQP'"""

    def fun(x: NDArray) -> float:
        return (1 / 2) * x.T @ Q @ x + c.T @ x

    def jac(x: NDArray) -> NDArray:
        return x.T @ Q + c

    x_0 = np.zeros(Q.shape[0]) if x_0 is None else x_0
    constraints = []
    if A is not None:
        constraints.append({'type': 'ineq',
                            'fun': lambda x: b - A @ x,
                            'jac': lambda _: -A})
    if A_eq is not None:
        constraints.append({'type': 'eq',
                            'fun': lambda x: b_eq - A_eq @ x,
                            'jac': lambda _: -A_eq})
    res_sp = sp.optimize.minimize(fun,
                                  jac=jac,
                                  constraints=constraints,
                                  bounds=bounds,
                                  method='SLSQP',
                                  x0=x_0,
                                  options={'disp': False})
    status = {0: Status.OPTIMAL,
              1: Status.ITERATION_LIMIT_REACHED,
              2: Status.INFEASIBLE,
              3: Status.UNBOUNDED,
              4: Status.NUMERICAL_ISSUES_ENCOUNTERED,
              }.get(res_sp.status, Status.UNKNOWN)
    success = status in {Status.OPTIMAL, Status.UNBOUNDED}
    return OptimizationProgramResult(success=success,
                                     value=res_sp.fun if status == Status.OPTIMAL else None,
                                     x_star=res_sp.x if status == Status.OPTIMAL else None,
                                     status=status)


def solve_qp(Q: NDArray,
             c: Optional[NDArray] = None,
             A: Optional[NDArray] = None,
             b: Optional[NDArray] = None,
             A_eq: Optional[NDArray] = None,
             b_eq: Optional[NDArray] = None,
             bounds: Optional[Sequence[tuple[float | None, float | None]]] = None,
             x_0: Optional[NDArray] = None,
             backend: Optional[Literal['scipy', 'cvxpy']] = None,
             ) -> OptimizationProgramResult:
    """Solve a quadratic program in the form `min (1 / 2) * x.T @ Q @ x + c.T @ x` subject to `A @ x <= b`, `A_eq @ x = b_eq`,
    and `bounds` on `x`"""

    def _validate_inputs(Q: NDArray,
                         c: Optional[NDArray] = None,
                         A: Optional[NDArray] = None,
                         b: Optional[NDArray] = None,
                         A_eq: Optional[NDArray] = None,
                         b_eq: Optional[NDArray] = None,
                         bounds: Optional[Sequence[tuple[float | None, float | None]]] = None,
                         x_0: Optional[NDArray] = None,
                         ) -> None:
        """Validate the inputs to the quadratic program solver

        Raises
        ------
        ValueError
            If any of the inputs are invalid (e.g. wrong shape, inconsistent dimensions, etc.)
        """
        if Q.ndim != 2:
            raise ValueError("Quadratic matrix Q must be a 2-d matrix")
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Quadratic matrix Q must be square")
        if c is not None and c.ndim != 1:
            raise ValueError("Constant vector c must be 1-dimensional")
        if c is not None and c.size != Q.shape[0]:
            raise ValueError("Quadratic matrix Q and constant vector c must both have shape (n, n) and (n,), respectively")
        if A is None and b is not None or A is not None and b is None:
            raise ValueError("Inequality constraint matrix A and vector b must both be provided or both be None")
        if A is not None and A.ndim != 2:
            raise ValueError("Inequality constraint matrix A must be 2-dimensional")
        if b is not None and b.ndim != 1:
            raise ValueError("Inequality constraint vector b must be 1-dimensional")
        if A is not None and b is not None and A.shape[0] != b.shape[0]:
            raise ValueError("Number of rows in A must match length of b")
        if A is not None and A.shape[1] != Q.shape[0]:
            raise ValueError("Number of columns in A must match length of c")
        if A_eq is None and b_eq is not None or A_eq is not None and b_eq is None:
            raise ValueError("Equality constraint matrix A_eq and vector b_eq must both be provided or both be None")
        if A_eq is not None and A_eq.ndim != 2:
            raise ValueError("Equality constraint matrix A_eq must be 2-dimensional")
        if b_eq is not None and b_eq.ndim != 1:
            raise ValueError("Equality constraint vector b_eq must be 1-dimensional")
        if A_eq is not None and b_eq is not None and A_eq.shape[0] != b_eq.shape[0]:
            raise ValueError("Number of rows in A_eq must match length of b_eq")
        if A_eq is not None and A_eq.shape[1] != Q.shape[0]:
            raise ValueError("Number of columns in A_eq must match length of c")
        if bounds is not None and len(bounds) != Q.shape[0]:
            raise ValueError("Number of elements in bounds must match length of c")
        if x_0 is not None and x_0.shape[0] != Q.shape[0]:
            raise ValueError("Length of x_0 must match length of c")

    c = np.zeros(Q.shape[0]) if c is None else c
    _validate_inputs(Q, c, A, b, A_eq, b_eq, bounds, x_0)

    # FIXME: This is completely temporary, replace with `CFG.qp_backend`
    backend = 'scipy'

    match backend:
        case 'scipy':
            res = _solve_qp_scipy(Q, c, A, b, A_eq, b_eq, bounds, x_0)
        case 'cvxpy':
            raise NotImplementedError("Solving with cvxpy is currently not implemented")
        case _:
            raise ValueError(f"Unknown QP backend '{backend}'")
    return res
