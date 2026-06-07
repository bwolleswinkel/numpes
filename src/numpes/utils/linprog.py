"""Module containing linear programming functionality"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from numpes._config import CFG

try:
    import cvxpy as cvx
    CVXPY_INSTALLED: bool = True
except ImportError as _:
    CVXPY_INSTALLED = False
try:
    import pulp  # type: ignore[import-untyped]
    PULP_INSTALLED: bool = True
except ImportError as _:
    PULP_INSTALLED = False

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

class Status(Enum):
    """Status of a optimization program solver run"""
    OPTIMAL = auto()
    UNBOUNDED = auto()
    INFEASIBLE = auto()
    NUMERICAL_ISSUES_ENCOUNTERED = auto()
    ITERATION_LIMIT_REACHED = auto()
    UNKNOWN = auto()


# FIXME: Do we actually need this?
@dataclass
class OptimizationProgramResult:
    """Result of a linear program solver run"""
    success: bool
    status: Status
    value: float | None
    x_star: NDArray | None


def _solve_lp_scipy(
    c: NDArray,
    A: Optional[NDArray] = None,
    b: Optional[NDArray] = None,
    A_eq: Optional[NDArray] = None,
    b_eq: Optional[NDArray] = None,
    bounds: Optional[list[tuple[float | None, float | None]]] = None,
    x_0: Optional[NDArray] = None,
) -> OptimizationProgramResult:
    """Solve a linear program using SciPy's linprog function"""
    if bounds is None:
        # NOTE: Passing `None` so SciPy applies its default non-negativity constraints.
        # In our implementation, `None` represents no bounds
        bounds_sp: tuple[None, None] | list[tuple[float | None, float | None]] = (None, None)
    else:
        bounds_sp = bounds
    res_sp = sp.optimize.linprog(
        c=c,
        A_ub=A,
        b_ub=b,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_sp,
        method=CFG.scipy_method,
        x0=x_0,
    )
    status = {
        0: Status.OPTIMAL,
        1: Status.ITERATION_LIMIT_REACHED,
        2: Status.INFEASIBLE,
        3: Status.UNBOUNDED,
        4: Status.NUMERICAL_ISSUES_ENCOUNTERED,
    }.get(res_sp.status, Status.UNKNOWN)
    success = status in {Status.OPTIMAL, Status.UNBOUNDED}
    return OptimizationProgramResult(
        success=success,
        value=res_sp.fun if status == Status.OPTIMAL else None,
        x_star=res_sp.x if status == Status.OPTIMAL else None,
        status=status
    )


def _solve_lp_cvxpy(
    c: NDArray,
    A: Optional[NDArray] = None,
    b: Optional[NDArray] = None,
    A_eq: Optional[NDArray] = None,
    b_eq: Optional[NDArray] = None,
    bounds: Optional[list[tuple[float | None, float | None]]] = None,
    x_0: Optional[NDArray] = None,
) -> OptimizationProgramResult:
    """Solve a linear program using CVXPY"""

    if not CVXPY_INSTALLED:
        raise ImportError("The package 'cvxpy' is not installed. Please install it to use the CVXPY backend.")

    bounds_cvx: NDArray | None = None
    if bounds is not None:
        bounds_cvx = np.array([
            (elem[0] if elem[0] is not None else -np.inf,
             elem[1] if elem[1] is not None else np.inf)
             for elem in bounds
        ])
    x = cvx.Variable(c.size)
    cons_ineq = [A @ x <= b] if A is not None and A.size > 0 and b is not None else []
    cons_eq = [A_eq @ x == b_eq] if A_eq is not None and A_eq.size > 0 and b_eq is not None else []
    cons_bounds = [x >= bounds_cvx[:, 0], x <= bounds_cvx[:, 1]] if bounds_cvx is not None else []
    problem = cvx.Problem(cvx.Minimize(c.T @ x), cons_ineq + cons_eq + cons_bounds)
    if x_0 is not None:
        x.value = x_0
    problem.solve(
        solver=CFG.cvxpy_solver,
        warm_start=x_0 is not None,
    )
    success = problem.status in {cvx.OPTIMAL, cvx.UNBOUNDED} | (
        {cvx.OPTIMAL_INACCURATE} if CFG.optimize_success == 'optimal_inaccurate' else set())
    status = {
        cvx.OPTIMAL: Status.OPTIMAL,
        cvx.OPTIMAL_INACCURATE: Status.OPTIMAL,
        cvx.UNBOUNDED: Status.UNBOUNDED,
        cvx.INFEASIBLE: Status.INFEASIBLE,
        cvx.INFEASIBLE_INACCURATE: Status.NUMERICAL_ISSUES_ENCOUNTERED,
        cvx.UNBOUNDED_INACCURATE: Status.NUMERICAL_ISSUES_ENCOUNTERED,
        None: Status.ITERATION_LIMIT_REACHED,
    }.get(problem.status if problem.status != cvx.USER_LIMIT else None, Status.UNKNOWN)
    return OptimizationProgramResult(
        success=success,
        value=problem.value if status == Status.OPTIMAL else None,
        x_star=x.value if status == Status.OPTIMAL else None,
        status=status
    )


def _solve_lp_pulp(
    c: NDArray,
    A: Optional[NDArray] = None,
    b: Optional[NDArray] = None,
    A_eq: Optional[NDArray] = None,
    b_eq: Optional[NDArray] = None,
    bounds: Optional[list[tuple[float | None, float | None]]] = None,
    x_0: Optional[NDArray] = None,
) -> OptimizationProgramResult:
    """Solve a linear program using PULP"""

    if not PULP_INSTALLED:
        raise ImportError("The package 'pulp' is not installed. Please install it to use the PuLP backend.")

    n = c.size
    prob = pulp.LpProblem()
    if bounds is None:
        bounds_pulp: list[tuple[float | None, float | None]] = [(None, None)] * n
    else:
        bounds_pulp = bounds
    x = []
    for i in range(n):
        lower = bounds_pulp[i][0]
        upper = bounds_pulp[i][1]
        lb = lower if lower is not None and np.isfinite(lower) else None
        ub = upper if upper is not None and np.isfinite(upper) else None
        x_i = pulp.LpVariable(f"x_{i}", lowBound=lb, upBound=ub)
        x.append(x_i)
    objective = pulp.lpSum([c[i] * x[i] for i in range(n)])
    prob += objective

    if A is not None and b is not None:
        for i in range(A.shape[0]):
            cons = pulp.lpSum([A[i, j] * x[j] for j in range(n)])
            prob += cons <= b[i]

    if A_eq is not None and b_eq is not None:
        for i in range(A_eq.shape[0]):
            cons = pulp.lpSum([A_eq[i, j] * x[j] for j in range(n)])
            prob += cons == b_eq[i]

    if x_0 is not None:
        if x_0.shape[0] != n:
            raise ValueError("Length of x_0 must match length of c")
        for i, x_i in enumerate(x):
            x_i.setInitialValue(float(x_0[i]))

    solver = pulp.PULP_CBC_CMD(msg=0, warmStart=x_0 is not None)
    prob.solve(solver)
    success = prob.status in {pulp.LpStatusOptimal, pulp.LpStatusUnbounded}
    status = {
        pulp.LpStatusOptimal: Status.OPTIMAL,
        pulp.LpStatusUnbounded: Status.UNBOUNDED,
        pulp.LpStatusInfeasible: Status.INFEASIBLE,
        pulp.LpStatusNotSolved: Status.UNKNOWN,
    }.get(prob.status, Status.UNKNOWN)

    return OptimizationProgramResult(
        success=success,
        value=prob.objective.value() if status == Status.OPTIMAL else None,
        x_star=np.array([var.varValue for var in x]) if status == Status.OPTIMAL else None,
        status=status,
    )


def solve_lp(
        c: NDArray,
        A: Optional[NDArray] = None,
        b: Optional[NDArray] = None,
        A_eq: Optional[NDArray] = None,
        b_eq: Optional[NDArray] = None,
        bounds: Optional[list[tuple[float | None, float | None]]] = None,
        x_0: Optional[NDArray] = None,
    ) -> OptimizationProgramResult:
    """Solve a linear program in the form `min c.T @ x` subject to `A @ x <= b`, `A_eq @ x = b_eq`, 
    and `bounds` on `x`"""

    def _validate_inputs(
        c: NDArray,
        A: Optional[NDArray] = None,
        b: Optional[NDArray] = None,
        A_eq: Optional[NDArray] = None,
        b_eq: Optional[NDArray] = None,
        bounds: Optional[list[tuple[float | None, float | None]]] = None,
        x_0: Optional[NDArray] = None,
    ) -> None:
        """Validate the inputs to the linear program solver
        
        Raises
        ------
        ValueError
            If any of the inputs are invalid (e.g. wrong shape, inconsistent dimensions, etc.)
        """
        if c.ndim != 1:
            raise ValueError("Objective vector c must be 1-dimensional")
        if A is None and b is not None or A is not None and b is None:
            raise ValueError("Inequality constraint matrix A and vector b must both be provided or both be None")
        if A is not None and A.ndim != 2:
            raise ValueError("Inequality constraint matrix A must be 2-dimensional")
        if b is not None and b.ndim != 1:
            raise ValueError("Inequality constraint vector b must be 1-dimensional")
        if A is not None and b is not None and A.shape[0] != b.shape[0]:
            raise ValueError("Number of rows in A must match length of b")
        if A is not None and A.shape[1] != c.shape[0]:
            raise ValueError("Number of columns in A must match length of c")
        if A_eq is None and b_eq is not None or A_eq is not None and b_eq is None:
            raise ValueError("Equality constraint matrix A_eq and vector b_eq must both be provided or both be None")
        if A_eq is not None and A_eq.ndim != 2:
            raise ValueError("Equality constraint matrix A_eq must be 2-dimensional")
        if b_eq is not None and b_eq.ndim != 1:
            raise ValueError("Equality constraint vector b_eq must be 1-dimensional")
        if A_eq is not None and b_eq is not None and A_eq.shape[0] != b_eq.shape[0]:
            raise ValueError("Number of rows in A_eq must match length of b_eq")
        if A_eq is not None and A_eq.shape[1] != c.shape[0]:
            raise ValueError("Number of columns in A_eq must match length of c")
        if bounds is not None and len(bounds) != c.shape[0]:
            raise ValueError("Number of elements in bounds must match length of c")
        if x_0 is not None and x_0.shape[0] != c.shape[0]:
            raise ValueError("Length of x_0 must match length of c")

    _validate_inputs(c, A, b, A_eq, b_eq, bounds, x_0)
    if CFG.lp_backend == 'auto':
        backend = 'cvxpy' if CVXPY_INSTALLED else 'scipy'
    else:
        backend = CFG.lp_backend
    match backend:
        case 'scipy':
            res = _solve_lp_scipy(c, A, b, A_eq, b_eq, bounds, x_0)
        case 'cvxpy':
            res = _solve_lp_cvxpy(c, A, b, A_eq, b_eq, bounds, x_0)
        case 'pulp':
            res = _solve_lp_pulp(c, A, b, A_eq, b_eq, bounds, x_0)
        case _:
            raise ValueError(f"Unknown LP backend '{backend}'")
    return res
