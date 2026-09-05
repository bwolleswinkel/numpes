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

import re
from copy import copy
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

from numpes._config import CFG
from numpes._internal import multipledispatch, wraps
from numpes._internal.printing import format_as_set, pad
from numpes.exceptions import DimensionError, InvalidCombinationOfArgumentsError, InvalidOperationError, InvalidRepresentationError, ConversionError
from numpes.utils import conv, enum_facets, enum_gens, is_sing, is_square, minimize_hrepr, minimize_vrepr, signed_angle

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
    A : NDArray[('m', 'n'), float]
        The matrix of shape (m, n) defining the m half-spaces in H-representation (Ax <= b).
    b : NDArray[('m',), float]
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
                 verts: Optional[ArrayLike] = None,
                 rays: Optional[ArrayLike] = None,
                 A: Optional[ArrayLike] = None,
                 b: Optional[ArrayLike] = None,
                 A_eq: Optional[ArrayLike] = None,
                 b_eq: Optional[ArrayLike] = None,
                 ) -> None:
        """Initialize a Polytope from vertices or half-spaces.

        Parameters
        ----------
        args : tuple[()] | tuple[ArrayLike] | tuple[ArrayLike, ArrayLike]
            Variable length positional arguments list. Must be of size 0, 1, or 2,
            according to the initialization method:
            - len(0) -> (): Initialize an empty polytope in R^n (requires `n`). Note that if the keywords `A` and `b` or
            `verts` are provided instead, they will cause a dispatch 
            to the appropriate constructor instead.
            - len(1) -> (verts,): A matrix of shape (k, n) representing k vertices in R^n (V-representation).
            - len(2) -> (A, b): A matrix of shape (m, n) and a vector of length (m,),
            respectively, representing m half-spaces in R^n (H-representation).
        n : int, optional
            Dimension of the ambient space. Required if initializing an empty polytope (zero positional arguments).
        verts : NDArray[("k", "n"), float], optional
            A matrix of shape (k, n) representing k vertices in R^n (V-representation). Required if 
            initializing from vertices (one positional argument).
        rays : NDArray[("k", "n"), float], optional
            Rays for unbounded polytopes.
        A : NDArray[("m", "n"), float], optional
            A matrix of shape (m, n) representing m half-spaces in R^n (H-representation). Required if
            initializing from half-spaces (two positional arguments).
        b : NDArray[("m",), float], optional
            A vector of shape (m,) representing m half-spaces in R^n (H-representation). Required if
            initializing from half-spaces (two positional arguments).
        A_eq : NDArray[("m_eq", "n"), float], optional
            Matrix of shape (m_eq, n) defining m_eq equality constraints in H-representation (Ax = b).
        b_eq : NDArray[("m_eq",), float], optional
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
        self._diam: float | None = None
        self._width: float | None = None  # FIXME: We should create a method `width`, and call this`min_width`
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
            raise InvalidCombinationOfArgumentsError("An invalid number or combination of arguments was provided," \
            f" received args={args}, kwargs={kwargs}. Please refer to the documentation for details on valid " \
            "combinations or arguments.")

    @__init__.register(len_args=0, len_kwargs='!=0', exclude_kwargs=['verts', 'A', 'b'])
    def _init_empty(self,
                    **kwargs: int,
                    ) -> None:
        """Initialize an empty polytope in R^n. Requires the keyword argument `n` for the ambient dimension.

        Parameters
        ----------
        n : int
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
                raise InvalidCombinationOfArgumentsError("Dimension 'n' must be provided for empty polytope initialization")
            if 'rays' in kwargs:
                raise InvalidCombinationOfArgumentsError("Cannot provide 'rays' when initializing an empty polytope")
            if 'A_eq' in kwargs or 'b_eq' in kwargs:
                raise InvalidCombinationOfArgumentsError("Cannot provide 'A_eq' or 'b_eq'" \
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
        self._diam = np.nan
        self._width = 0
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
        verts : NDArray[("k", "n"), float]
            A matrix of shape (k, n) representing k vertices in R^n (V-representation).
        rays : NDArray[("k", "n"), float], optional
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
            raise InvalidCombinationOfArgumentsError("An invalid number or combination of arguments was provided,"
                f" received args={args}, kwargs={dict(kwargs)}. Please refer to the documentation for details on valid "
                "combinations or arguments.")
        if 'n' in kwargs:
            raise InvalidCombinationOfArgumentsError("Cannot provide 'n' when initializing from vertices")
        if 'A' in kwargs or 'b' in kwargs:
            raise InvalidCombinationOfArgumentsError("Cannot provide 'A' or 'b' when initializing from vertices")
        if 'A_eq' in kwargs or 'b_eq' in kwargs:
            raise InvalidCombinationOfArgumentsError("Cannot provide 'A_eq' or 'b_eq' when initializing from vertices")
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
        self._is_empty = None
        self._is_degen = None
        self._is_bounded = None
        self._is_full_dim = None
        self._is_pointed = None
        self._is_singleton = None
        self._dim = None
        self._vol = None
        self._diam = None
        self._width = None
        self._chebcr = None

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
        A : NDArray[("m", "n"), float]
            A matrix of shape (m, n) representing m half-spaces in R^n (H-representation).
        b : NDArray[("m",), float]
            A vector of shape (m,) representing m half-spaces in R^n (H-representation).
        A_eq : NDArray[("m_eq", "n"), float], optional
            Matrix of shape (m_eq, n) defining m_eq equality constraints in H-representation (Ax = b).
        b_eq : NDArray[("m_eq",), float], optional
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
            raise InvalidCombinationOfArgumentsError("An invalid number or combination of arguments was provided,"
                f" received args={args}, kwargs={dict(kwargs)}. Please refer to the documentation for details on valid "
                "combinations or arguments.")
        if 'n' in kwargs:
            raise InvalidCombinationOfArgumentsError("Cannot provide 'n' when initializing from half-spaces")
        if 'verts' in kwargs:
            raise InvalidCombinationOfArgumentsError("Cannot provide 'verts' when initializing from half-spaces")
        if 'rays' in kwargs:
            raise InvalidCombinationOfArgumentsError("Cannot provide 'rays' when initializing from half-spaces")
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
        self._is_empty = None
        self._is_degen = None
        self._is_bounded = None
        self._is_full_dim = None
        self._is_pointed = None
        self._is_singleton = None
        self._dim = None
        self._vol = None
        self._diam = None
        self._width = None
        self._chebcr = None

    def _init_ambient(self,
                      n: int
                      ) -> None:
        """Initialize a polytope covering R^n.

        Parameters
        ----------
        n : int
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
        self._diam = np.nan
        self._width = np.inf
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
                self.minimal(which_repr='vrepr')
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
                self.minimal(which_repr='hrepr')
            case _:
                raise ValueError(f"Unknown value '{CFG.on_property_assign}' for 'on_property_assign' config setting")

    @property
    def n(self) -> int:
        """Dimension of the ambient space"""
        if self._vrepr is not None:
            return self.verts.shape[1]
        if self._hrepr is not None:
            return self.A.shape[1]
        raise InvalidRepresentationError("Polytope is not properly initialized with either " \
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
        """Matrix Ab in the H-representation of the polytope (Ab x <= 0, where Ab = [A | b])"""
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
        """Matrix Ab_eq in the H-representation of the polytope (Ab_eq x = 0, where Ab_eq = [A_eq | b_eq])"""
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
        """Check whether the polytope is empty (i.e., has no points)"""
        if self._is_empty is None:
            if self._vrepr is not None:
                self._is_empty = self.verts.size == 0 and self.rays.size == 0
            elif self._hrepr is not None:
                if np.all(self.Ab == np.array([[0] * self.n + [-1]])).item() and self.Ab_eq.size == 0:
                    self._is_empty = True
                else:
                    raise NotImplementedError("This feature is not yer implemented")
            else:
                raise InvalidRepresentationError("Polytope is not properly initialized with either " \
                                                 "V-representation or H-representation")
        return self._is_empty

    @property
    def is_degen(self) -> bool:
        """Check whether the polytope is degenerate"""
        if self._is_degen is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._is_degen

    @property
    def is_bounded(self) -> bool:
        """Check whether the polytope is bounded"""
        if self._is_bounded is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._is_bounded

    @property
    def is_full_dim(self) -> bool:
        """Check whether the polytope is full-dimensional"""
        if self._is_full_dim is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._is_full_dim

    @property
    def is_pointed(self) -> bool:
        """Check whether the polytope is pointed (i.e., contains at least one vertex)"""
        if self._is_pointed is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._is_pointed

    @property
    def is_singleton(self) -> bool:
        """Check whether the polytope is a singleton (i.e., contains a single point)"""
        if self._is_singleton is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._is_singleton

    @property
    def dim(self) -> int:
        """Dimension of the polytope"""
        if self._dim is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._dim

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
    def diam(self) -> float:
        """Geometric diameter of the polytope. For the combinatorial diameter, see method `comb_diam`."""
        if self._diam is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._diam

    @property
    def width(self) -> float:
        """Width of the polytope (i.e., smallest distance between two parallel supporting hyperplanes that enclose the polytope)"""
        if self._width is None:
            raise NotImplementedError("This property is not yet implemented")
        return self._width

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

        I = np.eye(n)  # pylint: disable=invalid-name # noqa: E741
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

        polytope = cls()
        polytope._vrepr = (verts, rays)
        polytope._hrepr = (Ab, Ab_eq)
        polytope._is_empty = False
        polytope._is_singleton = bool(is_eq.all())
        polytope._is_bounded = bool(~(lb_only | ub_only | is_unbounded).any())
        polytope._is_degen = bool(is_eq.any() or not polytope._is_bounded)
        polytope._is_full_dim = bool(~is_eq.any())
        polytope._is_pointed = bool(~is_unbounded.any())
        polytope._dim = int(n - is_eq.sum())

        if is_eq.any():
            polytope._vol = 0.0
        elif not polytope._is_bounded:
            polytope._vol = np.inf
        else:
            polytope._vol = float(np.prod(upper - lower))

        if polytope._is_singleton:
            polytope._chebcr = (lower.copy(), 0.0)
        elif not polytope._is_full_dim:
            bounds_is_finite = lb_is_finite & ub_is_finite
            midpoints = np.full(n, np.nan)
            midpoints[bounds_is_finite] = (lower[bounds_is_finite] + upper[bounds_is_finite]) / 2
            polytope._chebcr = (np.where(is_eq, lower, midpoints), 0.0)
        else:
            bounds_is_finite = lb_is_finite & ub_is_finite
            midpoints = np.full(n, np.nan)
            midpoints[bounds_is_finite] = (lower[bounds_is_finite] + upper[bounds_is_finite]) / 2
            chebc = np.where(is_eq, lower, midpoints)
            chebr = float(np.min(np.where(bounds_is_finite & ~is_eq, (upper - lower) / 2, np.inf)))
            polytope._chebcr = (chebc, chebr)

        return polytope

    @classmethod
    def from_point(cls, point: ArrayLike) -> Polytope:
        """Create a singleton polytope containing the given point.

        Parameters
        ----------
        point : ArrayLike
            A point/vector in R^n.

        Returns
        -------
        Polytope
            A polytope that is a singleton containing the given point.

        Raises
        ------
        ValueError
            If the input point is not a 1D array or contains NaN or infinite values

        Examples
        --------
        >>> poly = pes.poly_from_point([1, 2, 3])
        >>> print(poly)
        Singleton polytope in R^3
        [[1. 0. 0.]  |    [[1.]
         [0. 1. 0.]  x ==  [2.]
         [0. 0. 1.]] |     [3.]]
        """
        point = np.atleast_1d(point)
        if point.ndim != 1:
            raise ValueError(f"Point must be a 1D array, but received an array of shape {point.shape}")
        if not np.isfinite(point).all() or np.isnan(point).any():
            raise ValueError(f"Point vector cannot contain inf or NaN values, received point={point}")
        n = point.size

        polytope = cls(point)
        polytope._hrepr = (np.empty((0, n + 1)),
                           np.column_stack((np.eye(n), point)))
        polytope._is_empty = False
        polytope._is_singleton = True
        polytope._is_bounded = True
        polytope._is_degen = True
        polytope._is_full_dim = False
        polytope._is_pointed = True
        polytope._dim = 0
        polytope._vol = 0
        polytope._diam = 0
        polytope._width = 0
        polytope._chebcr = (point.copy(), 0)

        return polytope

    __array_ufunc__ = None  # Disable NumPy ufuncs for Polytope objects to trigger fallback to dunder methods

    def __deepcopy__(self, memo: dict[int, Any]) -> Polytope:
        """Invoked when `copy.deepcopy` is called on the object"""
        return self.copy(deepcopy=True, memo=memo)

    def __matmul__(self, M: NDArray) -> Polytope:
        """Invoked when right-hand matrix multiplication is performed (i.e., `self @ M`)"""
        raise InvalidOperationError("Right-hand matrix multiplication is not defined for polytopes. Only left-hand matrix multiplication is allowed by reversing the order of operands (i.e., use 'M @ poly' instead of 'poly @ M').")

    def __rmatmul__(self, M: NDArray) -> Polytope:
        """Invoked when left-hand matrix multiplication is performed (i.e., `M @ self`)"""
        return self.mat_mul(M, in_place=False)

    def __str__(self) -> str:
        """Description of the polytope in either V-represenation or H-representation"""
        header = self._str_header()
        if self._hrepr is not None:
            return header + "\n" + self._str_hrepr()
        if self._vrepr is not None:
            return header + "\n" + self._str_vrepr()
        raise InvalidRepresentationError("Polytope is not properly initialized with either " \
                                         "V-representation or H-representation")

    # [untested/unverified]
    def _str_header(self) -> str:
        # NOTE: These methods are not yet implemented
        # if self.is_empty:
        #     return f"Empty polytope in R^{self.n}"
        # if self.is_singleton:
        #     return f"Singleton polytope in R^{self.n}"
        # if self.is_lower_dim:
        #     return f"Lower dimensional polytope in R^{self.n}"
        # if not self.is_bounded:
        #     return f"Unbounded polytope in R^{self.n}"
        # if self.is_full_space:
        #     return f"Full space polytope in R^{self.n}"
        return f"Polytope in R^{self.n}"

    # [untested/unverified]
    def _str_vrepr(self) -> str:
        """"Description of the polytope in V-represntation"""
        try:
            verts, rays = self.vrepr
        except ConversionError as e:
            raise ConversionError(f"Converting the polytope to V-representation for printing failed: {e}")
        verts += np.zeros_like(verts)
        rays += np.zeros_like(rays)
        edgeitems = np.get_printoptions()['edgeitems']
        if verts.size != 0:
            with np.printoptions(threshold=0):
                verts_lines = format_as_set([str(np.atleast_2d(vert).T) for vert in verts], edgeitems).splitlines()
            nlines = len(verts_lines)
            idx_text = nlines // 2
            conv_lines = ["     " if idx != idx_text else "conv " for idx in range(nlines)]
            comb_verts = "\n".join(["".join(line) for line in zip(conv_lines, verts_lines)])
        else:
            comb_verts = None
        if rays.size != 0:
            with np.printoptions(threshold=0):
                rays_lines = format_as_set([str(np.atleast_2d(ray).T) for ray in rays], edgeitems).splitlines()
            if comb_verts is None:
                nlines = len(rays_lines)
                idx_text = nlines // 2
            nonneg_lines = ["       " if idx != idx_text else "nonneg " for idx in range(nlines)]
            comb_rays = "\n".join(["".join(line) for line in zip(nonneg_lines, rays_lines)])
        else:
            comb_rays = None
        if comb_verts is not None and comb_rays is not None:
            comb =  "\n".join(["".join(line) for line in zip(comb_verts.splitlines(),
                                                             ["   " if idx != idx_text else " + " for idx in range(nlines)],
                                                             comb_rays.splitlines())])
        elif comb_verts is not None:
            comb = comb_verts
        elif comb_rays is not None:
            comb = comb_rays
        else:  # This must be an empty polytope
            comb = "conv {/}"
        return comb

    # [untested/unverified]
    # pylint: disable=invalid-name
    def _str_hrepr(self) -> str:
        """"Description of the polytope in H-represntation"""
        try:
            A, b, A_eq, b_eq = self.A, self.b, self.A_eq, self.b_eq
        except ImportError as e:
            raise ConversionError(f"Converting the polytope to H-representation for printing failed: {e}")
        A += np.zeros_like(A)
        b += np.zeros_like(b)
        A_eq += np.zeros_like(A_eq)
        b_eq += np.zeros_like(b_eq)
        if A.size != 0:
            with np.printoptions(threshold=0):
                A_as_str = str(A).splitlines()
            nlines = len(A_as_str)
            idx_text = nlines - (1 if nlines <= 2 else 2)
            A_lines = [pad(line, len(A_as_str[-1])) for line in A_as_str]
            x_lines = [" |    " if idx != idx_text else " x <= " for idx in range(nlines)]
            with np.printoptions(threshold=0):
                b_lines = str(np.atleast_2d(b).T).splitlines()
            comb_Ab = "\n".join(["".join(line) for line in zip(A_lines, x_lines, b_lines)])
        else:
            comb_Ab = None
        if A_eq.size != 0:
            with np.printoptions(threshold=0):
                A_eq_as_str = str(A_eq).splitlines()
            nlines_eq = len(A_eq_as_str)
            idx_text_eq = nlines_eq - (1 if nlines_eq <= 2 else 2)
            A_eq_lines = [pad(line, len(A_eq_as_str[-1])) for line in A_eq_as_str]
            x_eq_lines = [" |    " if idx != idx_text_eq else " x == " for idx in range(nlines_eq)]
            with np.printoptions(threshold=0):
                b_eq_lines = str(np.atleast_2d(b_eq).T).splitlines()
            comb_Ab_eq = "\n".join(["".join(line) for line in zip(A_eq_lines, x_eq_lines, b_eq_lines)])
        else:
            comb_Ab_eq = None
        if comb_Ab is not None and comb_Ab_eq is not None:
            if nlines == 1:
                comb_Ab = comb_Ab.replace("[[", " [" if nlines_eq != 1 else "[").replace("]]", "] " if nlines_eq != 1 else "]")
                if comb_Ab[-1] == " ":
                    comb_Ab = comb_Ab[:-1]
            if nlines_eq == 1:
                comb_Ab_eq = comb_Ab_eq.replace("[[", " [" if nlines != 1 else "[").replace("]]", "] " if nlines != 1 else "]")
            comb = "\n".join(["".join(line) for line in zip(comb_Ab.splitlines(), ['' if idx != (nlines - 1) else ',' for idx in range(nlines)])]) + "\nand\n" + comb_Ab_eq
        elif comb_Ab is not None:
            if nlines == 1:
                comb_Ab = comb_Ab.replace("[[", "[").replace("]]", "]")
            comb = comb_Ab
        elif comb_Ab_eq is not None:
            if nlines_eq == 1:
                comb_Ab_eq = comb_Ab_eq.replace("[[", "[").replace("]]", "]")
            comb = comb_Ab_eq
        else:  # This must be the entire ambient space
            comb = "No constraints on x"
        return comb

    def __repr__(self) -> str:
        """Return a representation of the polytopes attributes"""
        attrs = ", ".join(f"{key}={value}" for key, value in self._repr_items())
        return f"{self.__class__.__name__}({attrs})"

    def _repr_items(self) -> list[tuple[str, str]]:
        """Return (attribute, formatted-value) pairs used by repr formatting"""

        def _format_repr_value(value: Any) -> str:
            if isinstance(value, np.ndarray):
                return f"NDArray[shape={value.shape}, dtype={value.dtype}]"
            if isinstance(value, tuple):
                inner = ", ".join(_format_repr_value(item) for item in value)
                if len(value) == 1:
                    inner += ","
                return f"({inner})"
            return repr(value)

        return [(key, _format_repr_value(value)) for key, value in self.__dict__.items()]

    def __format__(self, format_spec: str) -> str:
        """Format specifier for f-strings (i.e., `f"{poly:format_spec}"`)"""
        if format_spec.startswith('r'):
            if len(format_spec) != 1:
                raise ValueError(f"Debug specifier 'r' must be used in isolation, got '{format_spec}'")
            attrs = ",\n".join(f"    {key}={value}" for key, value in self._repr_items())
            return f"{self.__class__.__name__}(\n{attrs}\n)"

        pattern = r'^(i)?([hv]{1,2})?(\.\d*[feE])?$'
        match = re.match(pattern, format_spec)
        if not match:
            raise ValueError(f"Unknown format code '{format_spec}' for object of type '{self.__class__.__name__}'")

        tag_i, modes, num_part, str_prec, char_type = match.groups()
        modes = modes or ""

        precision = None
        suppress = None
        if num_part:
            if tag_i and not modes:
                raise ValueError("Summary 'i' does not take numeric formatting")
            precision = int(str_prec) if str_prec else np.get_printoptions()['precision']
            suppress = char_type == 'f'

        res = []
        if tag_i or not format_spec:  # Default to summary if no format spec is provided
            res.append(f"{self.__class__.__name__} in R^{self.n}")

        for char in modes:
            match char:
                case 'h':
                    if precision is not None:
                        with np.printoptions(precision=precision, suppress=suppress):
                            res.append(self._str_hrepr())
                    else:
                        res.append(self._str_hrepr())
                case 'v':
                    if precision is not None:
                        with np.printoptions(precision=precision, suppress=suppress):
                            res.append(self._str_vrepr())
                    else:
                        res.append(self._str_vrepr())
                case _:
                    raise ValueError(f"Unknown format code '{char}' in format spec '{format_spec}' for object of type '{self.__class__.__name__}'")
        return "\n".join(res)

    # [untested/unverified]
    # pylint: disable=protected-access
    def copy(self,
             deepcopy: bool = True,
             memo: Optional[dict[int, Any]] = None,
             ) -> Polytope:
        """Return a (deep)copy of the polytope. 

        Parameters
        ----------
        deepcopy : bool, default=True
            If True, a deep copy of the polytope is returned (totally isolated from the original polytope). If False, a shallow copy is returned.
        memo : dict[int, Any], optional
            A dictionary of objects already copied during the current copying pass, used by `copy.deepcopy` to avoid infinite recursion when copying objects with circular references. If None, a new empty dictionary is created.

        Returns
        -------
        Polytope
            A (deep)copy of the polytope

        Warnings
        --------
        If `deepcopy` is set to False, the returned polytope will share references to the same underlying data as the original polytope. Modifications to the NumPy arrays (`verts`, `rays`, `Ab`, `Ab_eq`, and `chebc`) in either polytope will affect both polytopes.
        """
        if not deepcopy:
            return copy(self)

        memo = {} if memo is None else memo
        if id(self) in memo:
            return memo[id(self)]

        obj = copy(self)
        memo[id(self)] = obj

        if self._vrepr is not None:
            obj._vrepr = (self.verts.copy(), self.rays.copy())
        if self._hrepr is not None:
            obj._hrepr = (self.Ab.copy(), self.Ab_eq.copy())
        if self._chebcr is not None:
            obj._chebcr = (self.chebc.copy(), self.chebr)

        return obj

    # [untested/unverified]
    # pylint: disable=protected-access
    def mat_mul(self,
                M: NDArray,
                calc_chebcr: bool = False,
                in_place: bool = True,
                ) -> Polytope | Self:
        """Matrix multiplication with a matrix `M`.
        
        Parameters
        ----------
        M : NDArray
            A matrix to multiply the polytope by
        calc_chebcr : bool, default=False
            Whether to recalculate the Chebyshev center and radius after the multiplication by solving an LP
        in_place : bool, default=True
            If True, the polytope is modified in place

        Returns
        -------
        Polytope
            The resulting polytope after the matrix multiplication

        Raises
        ------
        TypeError
            If `M` is not a NumPy array
        ValueError
            If `M` is not a valid matrix
        DimensionError
            If `M` has incompatible dimensions for multiplication with the polytope
        """
        if not isinstance(M, np.ndarray):
            raise TypeError(f"Input 'M' must be a NumPy array, but received an object of type '{type(M).__name__}'")
        if not np.isfinite(M).all() or np.isnan(M).any():
            raise ValueError("Input 'M' must not contains NaN or inf values")
        if not M.ndim == 2:
            raise ValueError(f"Input matrix 'M' must be 2-dimensional, recieved shape={M.shape}")
        if M.shape[1] != self.n:
            raise DimensionError(f"Input matrix 'M' must be of size (m, {self.n}), received shape={M.shape}")

        vrepr, hrepr = self._vrepr, self._hrepr
        if vrepr is None and hrepr is None:
            raise InvalidRepresentationError("Polytope is not properly initialized with either " \
                                             "V-representation or H-representation")
        obj = self if in_place else self.copy()
        M_is_sing = is_sing(M)
        if vrepr is not None or M_is_sing:
            # FIXME: Should we use the `self._vrepr` instead?
            obj.vrepr = (obj.verts @ M.T, obj.rays @ M.T)
        if hrepr is not None:
            if M_is_sing:
                obj._hrepr = None
            else:
                obj.hrepr = (np.column_stack((obj.A @ np.linalg.inv(M), obj.b)),
                             np.column_stack((obj.A_eq @ np.linalg.inv(M), obj.b_eq)))

        if obj._dim is not None:
            obj._dim = obj._dim if not M_is_sing else None  # FIXME: Can we do better here?
        if obj._vol is not None:
            obj._vol = obj._vol * np.linalg.det(M) if not M_is_sing else (0 if is_square(M) else None)
        if obj._diam is not None:
            obj._dim = None  # FIXME: Maybe we can do better for `not is_square`?
        if obj._width is not None:
            obj._width = None if not M_is_sing else 0
        if obj._chebcr is not None:
            obj._chebcr = None  # FIXME: Apprantly, we can do better with solving an LP

        return obj

    def minimal(self,
                which_repr: Literal['both', 'vrepr', 'hrepr'] = 'both',
                in_place: bool = True,
                ) -> Polytope | Self:
        """Return a minimal representation of the polytope by removing redundant vertices and facets"""
        obj = self if in_place else self.copy()
        if which_repr in {'vrepr', 'both'}:
            if obj.rays.size > 0:
                obj._vrepr = minimize_vrepr(obj.verts, obj.rays)
            else:
                obj._vrepr = (conv(obj.verts), obj.rays)
        if which_repr in {'hrepr', 'both'}:
            obj._hrepr = minimize_hrepr(obj.Ab, obj.Ab_eq)
        return obj

    # pylint: disable=too-many-branches,too-many-statements
    def plot(self,
             color: str | None = None,
             alpha: float = 0.5,
             plot_edges: bool = True,
             annotate_verts: list[str] | bool = False,
             annotate_facets: list[str] | bool = False,
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
                if annotate_facets:
                    for idx in range(self.m):
                        verts_facet = self.verts[np.isclose(self.A[idx, :] @ self.verts.T,
                                                            self.b[idx],
                                                            rtol=CFG.rtol,
                                                            atol=CFG.atol), :]
                        label = annotate_facets[idx] if isinstance(annotate_facets, list) else fr"${idx}$"
                        ax.text(*np.mean(verts_facet, axis=0), label, color='black')  # type: ignore[call-arg]
                if CFG.aspect == 'equal':
                    ax.set_aspect('equal', adjustable='box')
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
                    if annotate_facets:
                        label = annotate_facets[idx] if isinstance(annotate_facets, list) else fr"${idx}$"
                        ax.text(*np.mean(verts_facet, axis=0), label, color='black')  # type: ignore[call-arg]
                if CFG.aspect == 'equal':
                    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])  # type: ignore[arg-type]
            case _:
                raise ValueError(f"Plotting is only supported for n-d polytopes with n <= 3, received n = {self.n}")

        if annotate_verts:
            for idx in range(self.k):
                label = annotate_verts[idx] if isinstance(annotate_verts, list) else fr"${idx}$"
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
def poly(*args: Any,
         n: Optional[int] = None,
         verts: Optional[ArrayLike] = None,
         rays: Optional[ArrayLike] = None,
         A: Optional[ArrayLike] = None,
         b: Optional[ArrayLike] = None,
         A_eq: Optional[ArrayLike] = None,
         b_eq: Optional[ArrayLike] = None,
         ) -> Polytope:
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
        raise InvalidCombinationOfArgumentsError("No (keyword) arguments provided for polytope initialization. Please refer to the documentation for valid argument combinations.")
    return Polytope(*args, **kwargs)


@wraps(Polytope._init_empty)  # pylint: disable=protected-access
def poly_empty(n: int) -> Polytope:
    """Wrapper function for `Polytope._init_empty` to create an empty polytope"""
    return Polytope(n=n)


@wraps(Polytope._init_vrepr)  # pylint: disable=protected-access
def poly_from_verts(verts: ArrayLike, rays: Optional[ArrayLike]) -> Polytope:
    """Wrapper function for `Polytope._init_vrepr` to create a polytope from vertices, and optionally rays"""
    return Polytope(verts=verts, rays=rays)


@wraps(Polytope._init_hrepr)  # pylint: disable=protected-access
def poly_from_ineq(A: ArrayLike, b: ArrayLike, A_eq: Optional[ArrayLike] = None, b_eq: Optional[ArrayLike] = None) -> Polytope:
    """Wrapper function for `Polytope._init_hrepr` to create a polytope from inequalities, and optionally equalities"""
    return Polytope(A=A, b=b, A_eq=A_eq, b_eq=b_eq)


@wraps(Polytope._init_ambient)  # pylint: disable=protected-access
def poly_ambient(n: int) -> Polytope:
    """Wrapper function for `Polytope._init_ambient` to create a polytope covering R^n"""
    polytope = Polytope()
    polytope._init_ambient(n)  # pylint: disable=protected-access
    return polytope


@wraps(Polytope.from_bounds)
def poly_from_bounds(lb: ArrayLike, ub: ArrayLike) -> Polytope:
    """Wrapper function for `Polytope.from_bounds` to create a polytope from lower and upper bounds"""
    return Polytope.from_bounds(lb, ub)


@wraps(Polytope.from_point)
def poly_from_point(point: ArrayLike) -> Polytope:
    """Wrapper function for `Polytope.from_point` to create a singleton polytope given a point/vector"""
    return Polytope.from_point(point)


# pylint: disable=protected-access
def poly_from_name(name: Literal['triangle',
                                 'square',
                                 'pentagon',
                                 'hexagon',
                                 'heptagon',
                                 'octagon',
                                 'tetrahedron',
                                 'simplex',
                                 'cube',
                                 'octahedron',
                                 'dodecahedron',
                                 'icosahedron',
                                 'house',
                                 'pyramid'],
                   ) -> Polytope:
    """Create a polytope from a libary based on a provided name.
    
    Parameters
    ----------
    name: str
        Name of the polytope in the libary to be created. Options are:
        - 'house' (2D)
        - 'pyramid' (3D)
    
    Returns
    -------
    poly : Polytope
        The resulting polytope in both V-representaion and H-representation
        
    Raises
    ------
    ValueError 
        If the provided `name` is not recognized
    """
    match name:
        case 'triangle':
            raise NotImplementedError("This shape is not yet implemented")
        case 'square':
            raise NotImplementedError("This shape is not yet implemented")
        case 'pentagon':
            raise NotImplementedError("This shape is not yet implemented")
        case 'hexagon':
            raise NotImplementedError("This shape is not yet implemented")
        case 'heptagon':
            raise NotImplementedError("This shape is not yet implemented")
        case 'octagon':
            raise NotImplementedError("This shape is not yet implemented")
        case 'tetrahedron':
            raise NotImplementedError("This shape is not yet implemented")
        case 'simplex':
            raise NotImplementedError("This shape is not yet implemented")
        case 'cube':
            raise NotImplementedError("This shape is not yet implemented")
        case 'octahedron':
            raise NotImplementedError("This shape is not yet implemented")
        case 'dodecahedron':
            raise NotImplementedError("This shape is not yet implemented")
        case 'icosahedron':
            raise NotImplementedError("This shape is not yet implemented")
        case 'house':
            n = 2
            verts = np.array([[  0,   0],
                              [  0,   1],
                              [  1,   0],
                              [  1,   1],
                              [0.5, 1.5]])
            A = np.array([[              0.0,             -1.0],  # Bottom: y >= 0
                          [             -1.0,              0.0],  # Left:   x >= 0
                          [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],  # Top-Left roof slant
                          [              1.0,              0.0],  # Right:  x <= 1
                          [ 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])  # Top-Right roof slant
            b = np.array([0, 0, 1 / np.sqrt(2), 1, np.sqrt(2)])
            is_empty = False
            is_degen = False
            is_bounded = True
            is_full_dim = True
            is_pointed = True
            is_singleton = False
            dim = 2
            vol = 1.25
            diam = float(np.linalg.norm(verts[0] - verts[4], ord=2))
            width = 1
            chebcr = (np.array([0.5, 0.5]), 0.5)
        case 'pyramid':
            n = 3
            verts = np.array([[  0,   0,  0],
                              [  0,   1,  0],
                              [  1,   0,  0],
                              [  1,   1,  0],
                              [0.5, 0.5,  1]])
            A = np.array([[ 0,  0, -1],  # Base: z >= 0
                          [-2,  0,  1],  # Left slant: -2x + z <= 0
                          [ 0, -2,  1],  # Front slant: -2y + z <= 0
                          [ 2,  0,  1],  # Right slant: 2x + z <= 2
                          [ 0,  2,  1]   # Back slant: 2y + z <= 2
            ])
            b = np.array([0, 0, 0, 2, 2])
            is_empty = False
            is_degen = False
            is_bounded = True
            is_full_dim = True
            is_pointed = True
            is_singleton = False
            dim = 3
            vol = 1 / 3
            diam = float(np.linalg.norm(verts[0] - verts[3], ord=2))
            width = 1
            chebcr = (np.array([0.5, 0.5, 0.25]), 0.75)
        case _:
            raise ValueError(f"Unrecognized name '{name}'")

    polytope = poly(verts)
    polytope._hrepr = (np.column_stack((A, b)), np.empty((0, n + 1)))
    polytope._is_empty = is_empty
    polytope._is_degen = is_degen
    polytope._is_bounded = is_bounded
    polytope._is_full_dim = is_full_dim
    polytope._is_pointed = is_pointed
    polytope._is_singleton = is_singleton
    polytope._dim = dim
    polytope._vol = vol
    polytope._diam = diam
    polytope._width = width
    polytope._chebcr = chebcr

    return polytope
