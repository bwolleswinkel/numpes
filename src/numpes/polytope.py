"""Module for polytope and zonotope functionalities

Classes
-------
Polytope
    A class representing a convex polytope defined by either linear (in)equalities 
    or as the convex hull of a set of vertices and corresponding rays.

Functions
---------
poly
    A factory function for creating Polytope instances.
poly_from_verts
    A factory function for creating a Polytope instance from vertices and rays.
poly_from_ineq
    A factory function for creating a Polytope instance from linear (in)equalities.
poly_from_bounds
    A factory function for creating a Polytope instance from upper and lower bounds.
"""

from __future__ import annotations
from itertools import product as iterproduct
from typing import TYPE_CHECKING, overload

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, PolyCollection  # type: ignore[import-untyped]
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

from numpes._config import CFG, algo_options
from numpes._internal import multipledispatch, wraps
from numpes.exceptions import InvalidCombinationOfArguments, InvalidRepresentation
from numpes.utils import enum_facets, enum_gens, signed_angle

if TYPE_CHECKING:
    from typing import Any, Literal, Optional, Self

    from matplotlib.axes import Axes  # FIXME: Should we make this a lazy import/exclude import error if matplotlib is not installed?
    from numpy.typing import ArrayLike, NDArray


# TODO: Inherit from a common base class ConvexRegion
class Polytope:
    """Polytope represented in either V-representation (vertices) or H-representation (half-spaces).
    
    A polytope is the convex hull of a finite set of points in R^n (V-representation) or the bounded intersection 
    of a finite number of half-spaces (H-representation). This class provides methods to convert between these 
    representations and perform operations on polytopes.
    
    Attributes
    ----------
    A: NDArray[('m', 'n'), float]
        The matrix of shape (m, n) defining the m half-spaces in H-representation (Ax <= b).
    b: NDArray[('m',), float]
        The vector of shape (m,) defining the m half-spaces in H-representation (Ax <= b).
    verts: NDArray[('k', 'n'), float]
        The matrix of shape (k, n) defining the k vertices in V-representation.

    Methods
    -------
    mink_sum
        Compute the Minkowski sum of this polytope with another polytope.
    """

    # pylint: disable=unused-argument
    @multipledispatch
    def __init__(self,
                 *args: Any,
                 n: Optional[int] = None,
                 verts: Optional[NDArray | ArrayLike] = None,
                 rays: Optional[NDArray | ArrayLike] = None,
                 A: Optional[NDArray | ArrayLike] = None,
                 b: Optional[NDArray | ArrayLike] = None,
                 A_eq: Optional[NDArray | ArrayLike] = None,
                 b_eq: Optional[NDArray | ArrayLike] = None,
                 ) -> None:
        """Initialize a Polytope from vertices or half-spaces.

        Parameters
        ----------
        args: tuple[()] | tuple[NDArray] | tuple[NDArray, NDArray]
            Variable length positional arguments list. Must be of size 0, 1, or 2,
            according to the initialization method:
            - len(0) -> (): Initialize an empty polytope in R^n (requires `n`). Note that if the keywords `A` and `b` or
            `verts` are provided instead, they will cause a dispatch 
            to the appropriate constructor instead.
            - len(1) -> (verts,): A matrix of shape (k, n) representing k vertices in R^n (V-representation).
            - len(2) -> (A, b): A matrix of shape (m, n) and a vector of length (m,),
            respectively, representing m half-spaces in R^n (H-representation).
        n: int, optional
            Dimension of the ambient space. Required if initializing an empty polytope (zero positional arguments).
        verts: NDArray[("k", "n"), float], optional
            A matrix of shape (k, n) representing k vertices in R^n (V-representation). Required if 
            initializing from vertices (one positional argument).
        rays: NDArray[("k", "n"), float], optional
            Rays for unbounded polytopes.
        A: NDArray[("m", "n"), float], optional
            A matrix of shape (m, n) representing m half-spaces in R^n (H-representation). Required if
            initializing from half-spaces (two positional arguments).
        b: NDArray[("m",), float], optional
            A vector of shape (m,) representing m half-spaces in R^n (H-representation). Required if
            initializing from half-spaces (two positional arguments).
        A_eq: NDArray[("m_eq", "n"), float], optional
            Matrix of shape (m_eq, n) defining m_eq equality constraints in H-representation (Ax = b).
        b_eq: NDArray[("m_eq",), float], optional
            Vector of shape (m_eq,) defining m_eq equality constraints in H-representation (Ax = b).

        Raises
        ------
        InvalidCombinationOfArguments
            If the provided arguments do not match any of the expected patterns for initialization.
        TypeError
            If the types of the provided arguments are inconsistent with the expected types for initialization.
        ValueError
            If the provided ambient dimension `n` is not a positive integer.

        Examples
        --------
        Initialize a polytope from vertices (V-representation):
        >>> verts = np.array([[0, 0], [1, 0], [0, 1]])
        >>> poly = pes.poly(verts)
        >>> print(poly)
        Polytope with 3 vertices in R^2

        Initialize a polytope from half-spaces (H-representation):
        >>> A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> b = np.array([1, 1, 0, 0])
        >>> poly = pes.poly(A, b)
        >>> print(poly)
        Polytope defined by 4 half-spaces in R^2

        Initialize an empty polytope in R^2:
        >>> poly = pes.poly(n=2)
        >>> print(poly)
        Empty polytope in R^2
        """
        self._vrepr: tuple[NDArray, NDArray] | None = None
        self._hrepr: tuple[NDArray, NDArray] | None = None
        self._is_empty: bool | None = None
        self._is_degen: bool | None = None
        self._is_bounded: bool | None = None
        self._is_full_dim: bool | None = None
        self._is_pointed: bool | None = None
        self._is_singleton: bool | None = None
        self._dim: int | None = None
        self._vol: float | None = None
        self._chebcr: tuple[NDArray, float] | None = None

        # NOTE: This is the fallback method if no dispatchers match, and should raise an error
        kwargs = {key: value for key, value in {
            'n': n,
            'verts': verts,
            'rays': rays,
            'A': A,
            'b': b,
            'A_eq': A_eq,
            'b_eq': b_eq,
        }.items() if value is not None}
        if len(args) !=0 or len(kwargs) != 0:
            raise InvalidCombinationOfArguments("An invalid number or combination of arguments was provided," \
            f" received args={args}, kwargs={kwargs}. Please refer to the documentation for details on valid " \
            "combinations or arguments.")

    @__init__.register(len_args=0, len_kwargs='!=0', exclude_kwargs=['verts', 'A', 'b'])
    def _init_empty(self,
                    **kwargs: int,
                    ) -> None:
        """Initialize an empty polytope in R^n. Requires the keyword argument `n` for the ambient dimension.

        Parameters
        ----------
        n: int
            Dimension of the ambient space.

        Raises
        ------
        TypeError
            If the required keyword argument `n` is missing or if any of the forbidden keyword arguments are provided.
        """

        def _validate_inputs(kwargs: dict[str, int]) -> int:
            """Validate the inputs for empty polytope initialization

            Returns
            -------
            n : int
                The ambient dimension for the empty polytope
            
            Raises
            ------
            TypeError
                If the required keyword argument `n` is missing or if any of the
                forbidden keyword arguments are provided.
            ValueError
                If the provided ambient dimension `n` is not a positive integer.
            """
            if 'n' not in kwargs:
                raise InvalidCombinationOfArguments("Dimension 'n' must be provided for empty polytope initialization")
            if 'rays' in kwargs:
                raise InvalidCombinationOfArguments("Cannot provide 'rays' when initializing an empty polytope")
            if 'A_eq' in kwargs or 'b_eq' in kwargs:
                raise InvalidCombinationOfArguments("Cannot provide 'A_eq' or 'b_eq'" \
                " when initializing an empty polytope")
            n = kwargs['n']
            if not isinstance(n, int):
                raise TypeError(f"Dimension 'n' must be a positive integer, received {n} of type '{type(n).__name__}'")
            if n <= 0:
                raise ValueError(f"Dimension 'n' must be a positive integer, got n={n}")
            return n

        n = _validate_inputs(kwargs)

        self._vrepr = (np.empty((0, n)), np.empty((0, n)))
        self._hrepr = (np.array([[0] * n + [-1]]), np.empty((0, n + 1)))
        self._is_empty = True
        self._is_degen = True
        self._is_bounded = True
        self._is_full_dim = False
        self._is_pointed = True
        self._is_singleton = False
        self._dim = 0
        self._vol = 0
        self._chebcr = (np.full(n, np.nan), np.nan)

    @overload
    def _init_vrepr(self, *args: ArrayLike) -> None: ...
    @overload
    def _init_vrepr(self, *, verts: ArrayLike) -> None: ...

    @__init__.register(len_args=1)
    @__init__.register(len_args=0, include_kwargs=['verts'])
    def _init_vrepr(self,
                    *args: ArrayLike,
                    **kwargs: ArrayLike,
                    ) -> None:
        """Initialize a polytope from vertices (V-representation).

        Parameters
        ----------
        verts: NDArray[("k", "n"), float]
            A matrix of shape (k, n) representing k vertices in R^n (V-representation).
        rays: NDArray[("k", "n"), float], optional
            Rays for unbounded polytopes.

        Raises
        ------
        InvalidCombinationOfArguments
            If the required keyword argument `verts` is missing or if the provided arguments are inconsistent.
        """
        known_kwargs = {
            'verts',
            'rays',
            'n',
            'A',
            'b',
            'A_eq',
            'b_eq'
        }
        unknown_kwargs = {k: v for k, v in kwargs.items() if k not in known_kwargs}
        if unknown_kwargs:
            raise InvalidCombinationOfArguments("An invalid number or combination of arguments was provided,"
                f" received args={args}, kwargs={dict(kwargs)}. Please refer to the documentation for details on valid "
                "combinations or arguments.")
        if 'n' in kwargs:
            raise InvalidCombinationOfArguments("Cannot provide 'n' when initializing from vertices")
        if 'A' in kwargs or 'b' in kwargs:
            raise InvalidCombinationOfArguments("Cannot provide 'A' or 'b' when initializing from vertices")
        if 'A_eq' in kwargs or 'b_eq' in kwargs:
            raise InvalidCombinationOfArguments("Cannot provide 'A_eq' or 'b_eq' when initializing from vertices")
        if len(args) == 1:
            if isinstance(*args, int):
                raise TypeError("A single positional argument cannot be an integer (for an empty polytope " \
                "initialization). Please refer to the documentation for valid argument combinations.")
            (verts,) = args
        else:
            verts = kwargs['verts']
        verts = np.atleast_2d(verts)
        if verts.ndim != 2:
            raise ValueError(f"Vertices must be provided as a 2D array of shape (k, n)," \
                             f" but received an array of shape {verts.shape}")
        if np.isnan(verts).any():
            raise ValueError("Vertices 'verts' cannot contain NaN values")
        if 'rays' in kwargs:
            rays = kwargs['rays']
            rays = np.atleast_2d(rays)
            if rays.ndim != 2 or rays.shape[1] != verts.shape[1]:
                raise ValueError(f"Rays must be provided as a 2D array of shape (k_rays, " \
                                 f"n={verts.shape[1]}), but received an array of shape {rays.shape}")
            if np.isnan(rays).any():
                raise ValueError("Rays 'rays' cannot contain NaN values")
        else:
            rays = np.empty((0, verts.shape[1]))

        self.vrepr = (verts, rays)
        self._hrepr = None

    @overload
    def _init_hrepr(self, *args: ArrayLike) -> None: ...
    @overload
    def _init_hrepr(self, *, A: ArrayLike, b: ArrayLike) -> None: ...

    @__init__.register(len_args=2)
    @__init__.register(len_args=0, include_kwargs=['A', 'b'])
    def _init_hrepr(self,
                    *args: ArrayLike,
                    **kwargs: ArrayLike,
                    ) -> None:
        """Initialize a polytope from half-spaces (H-representation).

        Parameters
        ----------
        A: NDArray[("m", "n"), float]
            A matrix of shape (m, n) representing m half-spaces in R^n (H-representation).
        b: NDArray[("m",), float]
            A vector of shape (m,) representing m half-spaces in R^n (H-representation).
        A_eq: NDArray[("m_eq", "n"), float], optional
            Matrix of shape (m_eq, n) defining m_eq equality constraints in H-representation (Ax = b).
        b_eq: NDArray[("m_eq",), float], optional
            Vector of shape (m_eq,) defining m_eqp equality constraints in H-representation (Ax = b).

        Raises
        ------
        InvalidCombinationOfArguments
            If the required keyword arguments `A` and `b` are missing or if the provided arguments are inconsistent.
        """
        known_kwargs = {
            'verts',
            'rays',
            'n',
            'A',
            'b',
            'A_eq',
            'b_eq'
        }
        unknown_kwargs = {k: v for k, v in kwargs.items() if k not in known_kwargs}
        if unknown_kwargs:
            raise InvalidCombinationOfArguments("An invalid number or combination of arguments was provided,"
                f" received args={args}, kwargs={dict(kwargs)}. Please refer to the documentation for details on valid "
                "combinations or arguments.")
        if 'n' in kwargs:
            raise InvalidCombinationOfArguments("Cannot provide 'n' when initializing from half-spaces")
        if 'verts' in kwargs:
            raise InvalidCombinationOfArguments("Cannot provide 'verts' when initializing from half-spaces")
        if 'rays' in kwargs:
            raise InvalidCombinationOfArguments("Cannot provide 'rays' when initializing from half-spaces")
        if len(args) == 2:
            A, b = args
        else:
            A, b = kwargs['A'], kwargs['b']
        A, b = np.atleast_2d(A), np.atleast_1d(b)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.size:
            raise ValueError(f"A must be a matrix of size (m, n) and b must be a vector of size (m,)," \
                             f" but received A={A.shape}, b={b.shape}.")
        if np.isnan(A).any() or np.isnan(b).any():
            raise ValueError("Inequality matrices 'A' and 'b' cannot contain NaN values")
        if 'A_eq' in kwargs and 'b_eq' in kwargs:
            A_eq, b_eq = kwargs['A_eq'], kwargs['b_eq']
            A_eq, b_eq = np.atleast_2d(A_eq), np.atleast_1d(b_eq)
            if A_eq.ndim != 2 or b_eq.ndim != 1 or A_eq.shape[0] != b_eq.size or A_eq.shape[1] != A.shape[1]:
                raise ValueError(f"A_eq must be a matrix of shape (m_eq, n={A.shape[1]}) and b_eq must be a vector " \
                                 f"of size (m_eq,), but received shape A_eq={A_eq.shape}, b_eq={b_eq.shape}.")
            if np.isnan(A_eq).any() or np.isnan(b_eq).any():
                raise ValueError("Equality matrices 'A_eq' and 'b_eq' cannot contain NaN values")
        else:
            A_eq, b_eq = np.empty((0, A.shape[1])), np.empty((0,))

        self._vrepr = None
        self.hrepr = (np.column_stack((A, b)), np.column_stack((A_eq, b_eq)))

    def _init_ambient(self,
                      n: int
                      ) -> None:
        """Initialize a polytope covering R^n.

        Parameters
        ----------
        n: int
            Dimension of the ambient space.

        Raises
        ------
        TypeError
            If the argument `n` is of the wrong type.
        ValueError
            If the provided ambient dimension `n` is not a positive integer.
        """

        def _validate_inputs(n: int) -> None:
            """Validate the inputs for polytope initialization covering R^n.
            
            Raises
            ------
            TypeError
                If the argument `n` is of the wrong type.
            ValueError
                If the provided ambient dimension `n` is not a positive integer.
            """
            if not isinstance(n, int):
                raise TypeError(f"Dimension 'n' must be a positive integer, received {n} of type '{type(n).__name__}'")
            if n <= 0:
                raise ValueError(f"Dimension 'n' must be a positive integer, got n={n}")

        _validate_inputs(n)

        self._vrepr = (np.empty((0, n)), np.vstack((np.eye(n), -np.ones(n))))
        self._hrepr = (np.empty((0, n + 1)), np.empty((0, n + 1)))
        self._is_empty = False
        self._is_degen = True
        self._is_bounded = False
        self._is_full_dim = True
        self._is_pointed = False
        self._is_singleton = False
        self._dim = n
        self._vol = np.inf
        self._chebcr = (np.full(n, np.nan), np.inf)

    @property
    def vrepr(self) -> tuple[NDArray, NDArray]:
        """V-representation of the polytope as a tuple (verts, rays)"""
        if self._vrepr is None and CFG.on_poly_convert():
            verts, rays = enum_gens(self.Ab, self.Ab_eq)
            self._vrepr = (verts, rays)
        return self._vrepr

    @vrepr.setter
    def vrepr(self, value: tuple[NDArray, NDArray]) -> None:
        """Set the V-representation of the polytope as a tuple (verts, rays)"""
        self._vrepr = value
        match CFG.on_property_assign:
            case 'pass':
                pass
            case 'minimal':
                self.minimal(repr='vrepr')
            case _:
                raise ValueError(f"Unknown value '{CFG.on_property_assign}' for 'on_property_assign' config setting")

    @property
    def hrepr(self) -> tuple[NDArray, NDArray]:
        """H-representation of the polytope as a tuple (A, b)"""
        if self._hrepr is None:
            Ab, Ab_eq = enum_facets(self.verts, self.rays)
            self._hrepr = (Ab, Ab_eq)
        return self._hrepr

    @hrepr.setter
    def hrepr(self, value: tuple[NDArray, NDArray]) -> None:
        """Set the H-representation of the polytope as a tuple (A, b)"""
        self._hrepr = value
        match CFG.on_property_assign:
            case 'pass':
                pass
            case 'minimal':
                self.minimal(repr='hrepr')
            case _:
                raise ValueError(f"Unknown value '{CFG.on_property_assign}' for 'on_property_assign' config setting")

    @property
    def n(self) -> int:
        """Dimension of the ambient space in which the polytope is defined"""
        if self._vrepr is not None:
            return self.verts.shape[1]
        if self._hrepr is not None:
            return self.A.shape[1]
        raise InvalidRepresentation("Polytope is not properly initialized with either " \
        "V-representation or H-representation")

    @property
    def verts(self) -> NDArray:
        """Vertices of the polytope"""
        return self.vrepr[0]

    @property
    def k(self) -> int:
        """Number of vertices in the V-representation of the polytope"""
        return self.verts.shape[0]

    @property
    def rays(self) -> NDArray:
        """Rays of the polytope"""
        return self.vrepr[1]

    @property
    def k_rays(self) -> int:
        """Number of rays in the V-representation of the polytope"""
        return self.rays.shape[0]

    @property
    def Ab(self) -> NDArray:
        """Matrix Ab in the H-representation of the polytope (Abx <= 0)"""
        return self.hrepr[0]

    @property
    def A(self) -> NDArray:
        """Matrix A in the H-representation of the polytope (Ax <= b)"""
        return self.Ab[:, :-1]

    @property
    def b(self) -> NDArray:
        """Vector b in the H-representation of the polytope (Ax <= b)"""
        return self.Ab[:, -1]

    @property
    def m(self) -> int:
        """Number of half-spaces in the H-representation of the polytope"""
        return self.Ab.shape[0]

    @property
    def Ab_eq(self) -> NDArray:
        """Matrix Ab_eq in the H-representation of the polytope (Ab_eq x = 0)"""
        return self.hrepr[1]

    @property
    def A_eq(self) -> NDArray:
        """Matrix A_eq in the H-representation of the polytope (A_eq x = b_eq)"""
        return self.Ab_eq[:, :-1]

    @property
    def b_eq(self) -> NDArray:
        """Vector b_eq in the H-representation of the polytope (A_eq x = b_eq)"""
        return self.Ab_eq[:, -1]

    @property
    def m_eq(self) -> int:
        """Number of equality constraints in the H-representation of the polytope"""
        return self.Ab_eq.shape[0]
    
    @property
    def is_empty(self) -> bool:
        """Whether the polytope is empty (i.e., has no points)"""
        if self._is_empty is None:
            if self._vrepr is not None:
                self._is_empty = self.verts.size == 0 and self.rays.size == 0
            elif self._hrepr is not None:
                self._is_empty = (np.all(self.Ab == np.array([[0] * self.n + [-1]])) and 
                                  self.Ab_eq.size == 0)
            else:
                raise InvalidRepresentation("Polytope is not properly initialized with either " \
                "V-representation or H-representation")
        return self._is_empty
    
    @property
    def vol(self) -> float:
        """Volume of the polytope. Returns `np.inf` for unbounded polytopes and 0 for empty or lower-dimensional polytopes."""
        if self._vol is None:
            if self.is_empty or self.is_singleton or not self.is_full_dim:
                self._vol = 0
            elif not self.is_bounded:
                self._vol = np.inf
            else:
                raise NotImplementedError("Volume computation for full-dimensional bounded polytopes is not implemented yet")
        return self._vol
    
    @property
    def chebcr(self) -> tuple[NDArray, float]:
        """Chebyshev center and radius of the largest inscribed ball in the polytope"""
        if self._chebcr is None:
            if self.is_empty:
                self._chebcr = (np.full(self.n, np.nan), np.nan)
            else:
                raise NotImplementedError("Chebyshev center computation for non-empty polytopes is not implemented yet")
        return self._chebcr
    
    @property
    def chebc(self) -> NDArray:
        """Chebyshev center of the largest inscribed ball in the polytope. Returns NaN values if the center is ambiguous or not well-defined (e.g., for empty or unbounded polytopes)."""
        return self.chebcr[0]
    
    @property
    def chebr(self) -> float:
        """Chebyshev radius of the largest inscribed ball in the polytope. Returns NaN if the radius is not well-defined (e.g., for empty polytopes) and returns `np.inf` for unbounded polytopes."""
        return self.chebcr[1]

    @classmethod
    def from_bounds(cls, lb: ArrayLike, ub: ArrayLike) -> Polytope:
        """Create a polytope from upper and lower bounds on each coordinate.

        Parameters
        ----------
        lb : ArrayLike
            Lower bound of each coordinate. Use `-np.inf` or `float('-inf')` for an unbounded lower bound on a coordinate.
        ub : ArrayLike
            Upper bound of each coordinate. Use `np.inf` or `float('inf')` for an unbounded upper bound on a coordinate.

        Returns
        -------
        Polytope
            A polytope with V- and H-representations set analytically.
            Returns an empty polytope if the bounds are infeasible.

        Raises
        ------
        ValueError
            If bounds are not 1-D arrays of equal length, contain NaN values,
            or are otherwise malformed.
        """
        lower, upper = np.atleast_1d(lb), np.atleast_1d(ub)
        if lower.ndim != 1 or upper.ndim != 1 or lower.size != upper.size:
            raise ValueError(
                f"Lower and upper bounds must be 1D arrays of the same size, but received "
                f"lb={lower.shape}, ub={upper.shape}"
            )
        if np.isnan(lower).any() or np.isnan(upper).any():
            raise ValueError("Lower and upper bounds cannot contain NaN values")
        n = lower.size

        if (lower == np.inf).any() or (upper == -np.inf).any():
            return cls(n=n)
        if ((lower > upper) & ~np.isclose(lower, upper, rtol=CFG.rtol, atol=CFG.atol)).any():
            return cls(n=n)

        is_eq = np.isclose(lower, upper, rtol=CFG.rtol, atol=CFG.atol)
        lb_is_finite, ub_is_finite = np.isfinite(lower), np.isfinite(upper)
        is_bounded = ~is_eq & lb_is_finite & ub_is_finite
        lb_only = ~is_eq & lb_is_finite & ~ub_is_finite
        ub_only = ~is_eq & ~lb_is_finite & ub_is_finite
        is_unbounded = ~is_eq & ~lb_is_finite & ~ub_is_finite

        I = np.eye(n)
        ub_idx, lb_idx, eq_idx = (np.where(ub_is_finite & ~is_eq)[0],
                                  np.where(lb_is_finite & ~is_eq)[0],
                                  np.where(is_eq)[0])
        Ab = np.vstack((
            np.column_stack(( I[ub_idx],  upper[ub_idx])),
            np.column_stack((-I[lb_idx], -lower[lb_idx])),
        )) if ub_idx.size or lb_idx.size else np.empty((0, n + 1))
        Ab_eq = np.column_stack((I[eq_idx], lower[eq_idx])
                                ) if eq_idx.size else np.empty((0, n + 1))

        anchor = np.where(lb_is_finite, lower, np.where(ub_is_finite, upper, 0.0))
        verts = np.array(list(iterproduct(*[
            (anchor[i], upper[i]) if is_bounded[i] else (anchor[i],) for i in range(n)
        ])), dtype=float)
        rays = np.vstack((
             I[lb_only | is_unbounded],
            -I[ub_only | is_unbounded],
        )) if (lb_only | ub_only | is_unbounded).any() else np.empty((0, n))

        poly = cls()
        poly._vrepr = (verts, rays)
        poly._hrepr = (Ab, Ab_eq)
        poly._is_empty = False
        poly._is_singleton = bool(is_eq.all())
        poly._is_bounded = bool(~(lb_only | ub_only | is_unbounded).any())
        poly._is_degen = bool(is_eq.any() or not poly._is_bounded)
        poly._is_full_dim = bool(~is_eq.any())
        poly._is_pointed = bool(~is_unbounded.any())
        poly._dim = int(n - is_eq.sum())

        if is_eq.any():
            poly._vol = 0.0
        elif not poly._is_bounded:
            poly._vol = np.inf
        else:
            poly._vol = float(np.prod(upper - lower))

        if poly._is_singleton:
            poly._chebcr = (lower.copy(), 0.0)
        elif not poly._is_full_dim:
            bounds_is_finite = lb_is_finite & ub_is_finite
            midpoints = np.full(n, np.nan)
            midpoints[bounds_is_finite] = (lower[bounds_is_finite] + upper[bounds_is_finite]) / 2
            poly._chebcr = (np.where(is_eq, lower, midpoints), 0.0)
        else:
            bounds_is_finite = lb_is_finite & ub_is_finite
            midpoints = np.full(n, np.nan)
            midpoints[bounds_is_finite] = (lower[bounds_is_finite] + upper[bounds_is_finite]) / 2
            chebc = np.where(is_eq, lower, midpoints)
            chebr = float(np.min(np.where(bounds_is_finite & ~is_eq, (upper - lower) / 2, np.inf)))
            poly._chebcr = (chebc, chebr)

        return poly

    def minimal(self,
                repr: Literal['both', 'vrepr', 'hrepr'] = 'both',  # FIXME: Shadows built-in name 'repr(...)'
                in_place: bool = True,
                ) -> None | Self:
        """Return a minimal representation of the polytope by removing redundant vertices and facets"""
        raise NotImplementedError("The 'minimal' method is not yet implemented")

    # pylint: disable=too-many-branches,too-many-statements
    def plot(self,
             color: str | None = None,
             alpha: float = 0.5,
             plot_edges: bool = True,
             label_verts: list[str] | bool = False,
             label_facets: list[str] | bool = False,
             show: bool = True,
             ax: Optional[Axes] = None,
             ) -> Axes:
        """Plot a polytope"""

        def _plot_poly_2d(points: NDArray, ax: Axes, color: str, alpha: float, plot_edges: bool) -> None:
            if points.shape[0] < 3:
                raise ValueError("At least three points are required to plot a polytope in 2D")
            centroid = np.mean(points, axis=0)
            points_sorted = sorted(points, key=lambda p: signed_angle(points[0] - centroid, p - centroid))
            ax.add_collection(PolyCollection([points_sorted],
                                             facecolor=mpl.colors.to_rgba(color, alpha=alpha),
                                             edgecolor=(mpl.colors.to_rgba(color, alpha=1)
                                                        if plot_edges
                                                        else None)))
            ax.autoscale_view()

        def _plot_facet_3d(points: NDArray, ax: Axes, color: str, alpha: float, plot_edges: bool) -> None:
            # NOTE: Assumes all points are coplanar
            if points.shape[0] < 3:
                raise ValueError("At least three points are required to define a facet in 3D")
            centroid = np.mean(points, axis=0)
            look = np.cross(points[1] - points[0], points[2] - points[0])
            points_sorted = sorted(points, key=lambda p: signed_angle(points[0] - centroid, p - centroid, look=look))
            ax.add_collection3d(Poly3DCollection([np.array(points_sorted)],  # type: ignore[attr-defined]
                                                 facecolor=mpl.colors.to_rgba(color, alpha=alpha),
                                                 edgecolor=(mpl.colors.to_rgba(color, alpha=1)
                                                            if plot_edges
                                                            else None)))

        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib is required for plotting." \
            " Please install it with 'pip install matplotlib' and try again.")

        if ax is None:
            if self.n == 1:
                raise NotImplementedError("Plotting is not yet implemented for 1D polytopes")
            if self.n == 2:
                fig, ax = plt.subplots()
            elif self.n == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                raise ValueError(f"Plotting is only supported for n-d polytopes with n <= 3, received n = {self.n}")
        else:
            fig = None

        if color is None:
            # pylint: disable=protected-access
            color = ax._get_lines.get_next_color()  # type: ignore[union-attr, attr-defined]

        # TODO: Also implement the logic when `self` is lower-dimensional, so when it is a single plane, or a line.
        # TODO: Also add a degeneracy check for plotting
        match self.n:
            case 1:
                raise NotImplementedError("Plotting is not yet implemented for 1D polytopes")
            case 2:
                if ax.name == '3d':
                    raise ValueError("The dimension of the polytope" \
                    " does not match the dimension of the provided axes 'ax'")
                _plot_poly_2d(self.verts, ax, color, alpha, plot_edges=plot_edges)
                if label_facets:
                    for idx in range(self.m):
                        verts_facet = self.verts[np.isclose(self.A[idx, :] @ self.verts.T,
                                                            self.b[idx],
                                                            rtol=CFG.rtol,
                                                            atol=CFG.atol), :]
                        label = label_facets[idx] if isinstance(label_facets, list) else fr"${idx}$"
                        ax.text(*np.mean(verts_facet, axis=0), label, color='black')  # type: ignore[call-arg]
            case 3:
                if ax.name != '3d':
                    raise ValueError("The dimension of the polytope" \
                    " does not match the dimension of the provided axes 'ax'")
                for idx in range(self.m):
                    verts_facet = self.verts[np.isclose(self.A[idx, :] @ self.verts.T,
                                                        self.b[idx],
                                                        rtol=CFG.rtol,
                                                        atol=CFG.atol), :]
                    _plot_facet_3d(verts_facet, ax, color, alpha, plot_edges=plot_edges)
                    if label_facets:
                        label = label_facets[idx] if isinstance(label_facets, list) else fr"${idx}$"
                        ax.text(*np.mean(verts_facet, axis=0), label, color='black')  # type: ignore[call-arg]
            case _:
                raise ValueError(f"Plotting is only supported for n-d polytopes with n <= 3, received n = {self.n}")

        if label_verts:
            for idx in range(self.k):
                label = label_verts[idx] if isinstance(label_verts, list) else fr"${idx}$"
                if self.n == 2:
                    ax.text(self.verts[idx, 0],
                            self.verts[idx, 1],
                            label,
                            color='black')
                elif self.n == 3:
                    ax.text(self.verts[idx, 0],
                            self.verts[idx, 1],
                            self.verts[idx, 2],
                            label,  # type: ignore[call-arg,arg-type]
                            color='black')

        if show:
            plt.show()

        return ax


@wraps(Polytope.__init__)  # pylint: disable=protected-access
def poly(*args: NDArray,
         n: Optional[int] = None,
         verts: Optional[NDArray | ArrayLike] = None,
         rays: Optional[NDArray | ArrayLike] = None,
         A: Optional[NDArray | ArrayLike] = None,
         b: Optional[NDArray | ArrayLike] = None,
         A_eq: Optional[NDArray | ArrayLike] = None,
         b_eq: Optional[NDArray | ArrayLike] = None,) -> Polytope:
    """Wrapper function for `Polytope.__init__` to create a polytope"""
    kwargs = {key: value for key, value in {
        'n': n,
        'verts': verts,
        'rays': rays,
        'A': A,
        'b': b,
        'A_eq': A_eq,
        'b_eq': b_eq,
    }.items() if value is not None}
    if len(args) == 0 and len(kwargs) == 0:
        raise InvalidCombinationOfArguments("No (keyword) arguments provided for polytope initialization. Please refer to the documentation for valid argument combinations.")
    return Polytope(*args, **kwargs)


@wraps(Polytope._init_empty)  # pylint: disable=protected-access
def poly_empty(n: int) -> Polytope:
    """Wrapper function for `Polytope._init_empty` to create an empty polytope"""
    return Polytope(n=n)


@wraps(Polytope._init_vrepr)  # pylint: disable=protected-access
def poly_from_verts(verts: NDArray, rays: Optional[NDArray]) -> Polytope:
    """Wrapper function for `Polytope._init_vrepr` to create a polytope from vertices, and optionally rays"""
    return Polytope(verts=verts, rays=rays)


@wraps(Polytope._init_hrepr)  # pylint: disable=protected-access
def poly_from_ineq(A: NDArray, b: NDArray, A_eq: Optional[NDArray] = None, b_eq: Optional[NDArray] = None) -> Polytope:
    """Wrapper function for `Polytope._init_hrepr` to create a polytope from inequalities, and optionally equalities"""
    return Polytope(A=A, b=b, A_eq=A_eq, b_eq=b_eq)


@wraps(Polytope._init_ambient)  # pylint: disable=protected-access
def poly_ambient(n: int) -> Polytope:
    """Wrapper function for `Polytope._init_ambient` to create a polytope covering R^n"""
    poly = Polytope()
    poly._init_ambient(n)  # pylint: disable=protected-access
    return poly


@wraps(Polytope.from_bounds)
def poly_from_bounds(lb: ArrayLike, ub: ArrayLike) -> Polytope:
    """Wrapper function for `Polytope.from_bounds` to create a polytope from lower and upper bounds"""
    return Polytope.from_bounds(lb, ub)
