"""Module for ellipsoid functionality"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, cast

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Ellipse
except ImportError as _:
    pass

from numpes._config import CFG
from numpes._internal.common import get_axes_color
from numpes._internal.multipledispatch import multipledispatch
from numpes._internal.printing import format_spec_to_opts, pad, repr_items, sym_replace
from numpes._internal.wraps import wraps
from numpes.exceptions import InvalidRepresentationError, InvalidCombinationOfArgumentsError
from numpes.utils.linalg import angles_givens, is_posdef, is_rot_mat

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Optional, Self

    from matplotlib.axes import Axes  # FIXME: Should we make this a lazy import/exclude import error if matplotlib is not installed?
    from matplotlib.typing import ColorType
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
    from numpy.typing import ArrayLike, NDArray

    from numpes._internal.axes import Axes1D


# TODO: Inherit from a common base class ConvexRegion
class Ellipsoid:
    """Ellipsoid in R^n represented by a rotation matrix `R` and a set of radii `radii`.
    
    An non-degenerate ellipsoid is a convex region satisfying a quadratic inequality. This class 
    provides methods to perform operations and transformations on ellipsoids.
    
    Attributes
    ----------
    c : NDArray
        Center of the ellipsoid 
    Q : NDArray or None
        Positive (semi)-definite matrix defining the quadratic inequality 
        `(x - self.c).T @ self.Q @ (x - self.c)`.
    radii : NDArray
        Radii of the semi-principal axis of the ellipsoid
    R : NDArray
        Rotation matrix defining the orientation of the ellipsoid
    n : int
        Dimension of the ambient space
    vol : float
        Volume of the ellipsoid
    cov : NDArray or None
        Covariance matrix of the ellipsoid

    Methods
    -------
    proj
        Projects the ellipsoid to a subspace or a affine subset    
    """

    @multipledispatch
    def __init__(self,
                 *args: tuple[list[float], ArrayLike] | ArrayLike,
                 n: Optional[int] = None,
                 Q: Optional[ArrayLike] = None,
                 radii: Optional[list[float]] = None,
                 R: Optional[ArrayLike] = None,
                 c: Optional[ArrayLike] = None,
                 ) -> None:
        self._rrepr: tuple[list[float], NDArray] | None = None
        self._Q: NDArray = None
        self.c: NDArray = np.empty(0)
        self._angles: list[float] | None = None
        self._dim: int | None = None
        self._vol: float | None = None

        # NOTE: This is the fallback method if no dispatchers match, and should raise an error
        kwargs = {key: value for key, value in {
            'n': n,
            'Q': Q,
            'radii': radii,
            'R': R,
            'c': c,
        }.items() if value is not None}
        if len(args) !=0 or len(kwargs) != 0:
            raise InvalidCombinationOfArgumentsError("An invalid number or combination of arguments " \
                                                    f"was provided, received args={args}, kwarg={kwargs}. " \
                                                     "Please refer to the documentation for details on valid " \
                                                     "combinations or arguments.")

    @__init__.register(len_args=0, len_kwargs='!=0', exclude_kwargs=['Q', 'radii', 'R'])
    def _init_empty(self,
                    *,
                    n: int,
                    c: Optional[ArrayLike] = None,
                    ) -> None:
        """Initialize an empty ellipsoid"""
        if not isinstance(n, int):
            raise TypeError(f"Dimension 'n' must be a positive integer, received {n} of type '{type(n).__name__}'")
        if n <= 0:
            raise ValueError(f"Dimension 'n' must be a positive integer, got n={n}")
        if c is not None:
            raise InvalidCombinationOfArgumentsError("Center 'c' cannot be provided when " \
                                                    f"initializing an empty ellipsoid, received c={c}")
        self._rrepr = ([float('nan') for _ in range(n)], np.full((n, n), np.nan))
        self._Q = np.diag([np.inf for _ in range(n)])
        self.c = np.full(n, np.nan)
        self._angles = [float('nan') for _ in range(n)]
        self._dim = 0
        self._vol = 0

    @__init__.register(len_args=1)
    @__init__.register(len_args=0, include_kwargs=['Q'], exclude_kwargs=['n'])
    def _init_quad(self,
                   Q: ArrayLike,
                   c: Optional[ArrayLike] = None,
                   ) -> NDArray:
        """Initialize the ellipsoid from a quadratic matrix `Q`."""
        Q = np.atleast_2d(Q)
        if not Q.ndim == 2:
            raise ValueError(f"Quadratic matrix 'Q' must be a two-dimensional array, received {Q.shape}")
        if not Q.shape[0] == Q.shape[1]:
            raise ValueError(f"Quadratic matrix 'Q' much be a square matrix of size (n, n), received {Q.shape}")
        if not is_posdef(Q, semi_def=True):
            raise ValueError(f"Quadratic matrix 'Q' much be positive (semi)-definite, received matrix with eigenvalues {np.linalg.eigvals(Q)}")
        if c is None:
            c = np.zeros(Q.shape[0], dtype=Q.dtype)
        else:
            c = np.atleast_1d(c)
            if not c.ndim == 1:
                raise ValueError(f"Center 'c' must be a one-dimensional array, received {c.shape}")
            if not c.size == Q.shape[0]:
                raise ValueError(f"Center 'c' must be a vector of size (n,), n={Q.shape[0]}, received {c.shape}")
        self._rrepr = None
        self.Q = Q
        self.c = c
        self._angles: list[float] | None = None
        self._dim: int | None = None
        self._vol: float | None = None

    @__init__.register(len_args=2)
    @__init__.register(len_args=0, include_kwargs=['radii'], exclude_kwargs=['n'])
    @__init__.register(len_args=0, include_kwargs=['radii', 'R'], exclude_kwargs=['n'])
    def _init_rrepr(self,
                    radii: list[float],
                    R: Optional[NDArray] = None,
                    c: Optional[NDArray] = None,
                    ) -> None:
        radii_arr = np.atleast_1d(radii)
        if radii_arr.ndim != 1:
            raise ValueError(f"Radii 'radii' must be a one-dimensional list of floats or array-like, received {radii}")
        radii = radii_arr.tolist()
        R = (np.atleast_2d(R)
             if R is not None
             else np.eye(len(radii)))
        if R.ndim != 2:
            raise ValueError(f"Rotation matrix 'R' must be a two-dimensional array, received {R.shape}")
        if R.shape[0] != R.shape[1]:
            raise ValueError(f"Rotation matrix 'R' much be a square matrix of size (n, n), received {R.shape}")
        if len(radii) != R.shape[0]:
            raise ValueError(f"Number of 'radii' must equal shape of 'R', received radii of length {len(radii)} and rotation matrix of shape {R.shape}")
        if not (R.size == 1 and np.allclose(R, 1)) and not is_rot_mat(R):
            raise ValueError(f"Rotation matrix 'R' must be a valid rotation matrix, received R @ R.T={R @ R.T}, np.linalg.det(R)={np.linalg.det(R)}")
        if any([radius < 0 for radius in radii]):
            raise ValueError(f"Radii must be strictly non-negative, received non-negative radius of {radii_arr[np.argwhere(radii < 0).min()]} at index {np.argwhere(radii_arr < 0).min()}")
        if c is None:
            c = np.zeros(len(radii), dtype=radii_arr.dtype)
        else:
            c = np.atleast_1d(c)
            if not c.ndim == 1:
                raise ValueError(f"Center 'c' must be a one-dimensional array, received {c.shape}")
            if not c.size == len(radii):
                raise ValueError(f"Center 'c' must be a vector of size (n,), n={len(radii)}, received {c.shape}")
        self.rrepr = (radii, R)
        self._Q = None
        self.c = c
        self._angles: list[float] | None = None
        self._dim: int | None = None
        self._vol: float | None = None

    @property
    def rrepr(self) -> tuple[list[float], NDArray]:
        """R-representation of the ellipsoid"""
        if self._rrepr is None:
            if self._Q is None:
                raise InvalidRepresentationError(f"The ellipsoid contains neither an " \
                                                  "R-representation nor a quadratic matrix Q, " \
                                                  "implying it is in an invalid state")
            eigvals, R = np.linalg.eigh(self.Q)
            with np.errstate(divide='ignore'):
                radii = 1 / np.sqrt(np.maximum(eigvals, 0))
            for i in range(R.shape[1] - 1):
                if R[i, i] < 0:
                    R[:, i] *= -1
            if np.linalg.det(R) < 0:
                R[:, -1] *= -1  # Ensure R is a proper rotation matrix with det(R) = 1
            self._rrepr = (radii, R)
        return self._rrepr

    @rrepr.setter
    def rrepr(self, value: tuple[list[float], NDArray]) -> None:
        """Set the R-representation of the polytope as a tuple (radii, R)"""
        self._rrepr = value
        match CFG.on_property_assign:
            case 'pass':
                pass
            case 'minimal':  # FIXME: I don't think minimal here is nice; better is 'reduce', or even 'canon' (for canonical)
                self.minimal()
            case _:
                raise ValueError(f"Unknown value '{CFG.on_property_assign}' for 'on_property_assign' config setting")

    @property
    def Q(self) -> NDArray:
        """Quadratic matrix of the ellipsoid"""
        if self._Q is None:
            if self._rrepr is None:
                raise InvalidRepresentationError(f"The ellipsoid contains neither an R-representation nor a quadratic matrix Q, implying it is in an invalid state")
            if not np.isclose(self.radii, 0, rtol=CFG.rtol, atol=CFG.atol).any() and np.isfinite(self.radii).all():
                self.Q = self.R @ np.diag(1 / np.square(self.radii)) @ self.R.T
            else:
                self.Q = np.full((self.n, self.n), np.nan)
        return self._Q

    @Q.setter
    def Q(self, value: NDArray) -> None:
        """Set the quadratic matrix of the ellipsoid of shape (n, n)"""
        self._Q = value
        match CFG.on_property_assign:
            case 'pass' | 'minimal':
                pass
            case _:
                raise ValueError(f"Unknown value '{CFG.on_property_assign}' for 'on_property_assign' config setting")

    @property
    def n(self) -> int:
        """Dimension of the ambient space"""
        if self._rrepr is not None:
            return len(self.radii)
        if self._Q is not None:
            return self.Q.shape[0]
        raise InvalidRepresentationError(f"The ellipsoid contains neither an R-representation " \
                                          "nor a quadratic matrix Q, implying it is in an invalid state")

    @property
    def radii(self) -> list[float]:
        """Radii of the ellipsoid"""
        return self.rrepr[0]

    @property
    def R(self) -> NDArray:
        """Rotation matrix of the ellipsoid"""
        return self.rrepr[1]

    @property
    def angles(self) -> list[float]:
        """Rotation angles of the ellipsoid"""
        if self._angles is None:
            self._angles = angles_givens(self.R)
        return self._angles

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Invoked when `copy.deepcopy` is called on the object"""
        return self.copy(deepcopy=True, memo=memo)

    # [untested/unverified]
    def __str__(self) -> str:
        """Description of the ellipsoid"""
        header = self._str_header()
        str_quad = self._str_quad()
        return header + "\n" + str_quad

    # [untested/unverified]
    def _str_header(self) -> str:
        # NOTE: These methods are not yet implemented
        # if self.is_empty:
        #     return f"Empty ellipsoid in R^{self.n}"
        # if self.is_singleton:
        #     return f"Singleton ellipsoid in R^{self.n}"
        # if self.is_lower_dim:
        #     return f"Lower dimensional ellipsoid in R^{self.n}"
        # if not self.is_bounded:
        #     return f"Unbounded ellipsoid in R^{self.n}"
        # if self.is_full_space:
        #     return f"Full space ellipsoid in R^{self.n}"
        return f"Ellipsoid in R^{self.n}"

    # [untested/unverified]
    # pylint: disable=invalid-name
    def _str_quad(self, to_dtype: Optional[Literal['float', 'int']] = None) -> str:
        """Quadratic description of the ellipsoid"""
        if to_dtype is None:
            c, Q = self.c, self.Q
        elif to_dtype in {'int', 'float'}:
            c, Q = self.c.astype(dtype := int if to_dtype == 'int' else float), self.Q.astype(dtype)
        else:
            raise ValueError(f"Unrecognized value '{to_dtype}' for 'to_dtype'")
        c_as_str, Q_as_str = str(np.atleast_2d(c + np.zeros_like(c)).T), str(Q + np.zeros_like(Q))  # Add array of zeros to avoid `-0.` in print output
        c_lines, Q_lines = c_as_str.splitlines(), Q_as_str.splitlines()
        nlines = len(Q_lines)
        try:
            idx_trunc = Q_lines.index(' ...')
            c_lines = c_lines[:idx_trunc] + [' ...'] + c_lines[-idx_trunc:]  # NOTE: This assumes the number of edgeitems above and below is always identical
        except ValueError as _:
            idx_trunc = None
        idx_text = nlines - (1
                            if (nlines <= 2 or (nlines == 3 and idx_trunc is not None))
                            else 2)

        c_text = ['     ' if idx != idx_text else ', c: ' for idx in range(nlines)]
        c_vals = c_lines
        Q_text = ['   ' if idx != idx_text else 'Q: ' for idx in range(nlines)]
        Q_sym_lines = sym_replace(Q_as_str).splitlines()
        Q_vals = [pad(line, max(map(len, Q_sym_lines))) for line in Q_sym_lines]
        comb = '\n'.join([''.join(line) for line in zip(Q_text, Q_vals, c_text, c_vals)])
        if self.n == 1:
            comb = comb.replace('[[', '[').replace(']]', ']')

        return comb

    # [untested/unverified]
    def __repr__(self) -> str:
        """Return a representation of the ellipsoid attributes"""
        attrs = ", ".join(f"{key}={value}" for key, value in repr_items(self))
        return f"{self.__class__.__name__}({attrs})"

    # [untested/unverified]
    def __format__(self, format_spec: str) -> str:
        """Format the printed description of the ellipsoid based on a format specifier"""
        if format_spec == '':
                    return str(self)
        
        which_debug, which_repr, to_dtype, edgeitems, formatter, sign = format_spec_to_opts(format_spec)

        if which_debug == 'r':
            return repr(self)
        if which_debug == '#':
            attrs = ",\n    ".join(f"{key}={value}" for key, value in repr_items(self, compact_ndarray=True))
            return f"{self.__class__.__name__}(\n    {attrs},\n)"

        comb = ""
        if which_debug == 'i':
            comb += self._str_header()
        if which_repr is None:
            return comb

        with np.printoptions(threshold=0 if edgeitems is not None else None,
                             edgeitems=edgeitems,
                             formatter=cast('Any', formatter),
                             sign=sign,
                             ):
                str_quad = self._str_quad(to_dtype=to_dtype)
        if 'E' in format_spec:
            str_quad = str_quad.replace('e', 'E')
        comb += ("" if len(comb) == 0 else "\n") + str_quad
        
        return comb

    # [untested/unverified]
    # pylint: disable=protected-access
    def copy(self,
             deepcopy: bool = True,
             memo: Optional[dict[int, Any]] = None,
             ) -> Self:
        """Return a (deep)copy of the ellipsoid. 

        Parameters
        ----------
        deepcopy : bool, default=True
            If True, a deep copy of the ellipsoid is returned (totally isolated from the original ellipsoid). If False, a shallow copy is returned.
        memo : dict[int, Any], optional
            A dictionary of objects already copied during the current copying pass, used by `copy.deepcopy` to avoid infinite recursion when copying objects with circular references. If None, a new empty dictionary is created.

        Returns
        -------
        Ellipsoid
            A (deep)copy of the ellipsoid

        Warnings
        --------
        If `deepcopy` is set to False, the returned ellipsoid will share references to the same underlying data as the original ellipsoid. Modifications to the NumPy arrays or lists (`radii`, `R`, `Q`, `c`, and `angles`) in either ellipsoid will affect both ellipsoids.
        """
        if not deepcopy:
            return copy(self)

        memo = {} if memo is None else memo
        if id(self) in memo:
            return memo[id(self)]

        obj = copy(self)
        memo[id(self)] = obj

        if self._rrepr is not None:
            obj._rrepr = (list(self.radii), self.R.copy())
        if self._Q is not None:
            obj._Q = self.Q.copy()
        if self._angles is not None:
            obj._angles = list(self.angles)
        obj.c = self.c.copy()

        return obj

    def minimal(self,
                in_place: bool = True,
                ) -> Self:
        """Return a minimal representation of the ellipsoid by sorting the radii and rotation matrix in descending order"""
        obj = self if in_place else self.copy()
        idx_sort = np.argsort(obj.radii)[::-1]
        R_sorted = obj.R[:, idx_sort]
        if not np.isnan(R_sorted).any() and np.linalg.det(R_sorted) < 0:
            R_sorted[:, -1] *= -1
        obj._rrepr = ([obj.radii[idx] for idx in idx_sort], R_sorted)
        return obj

    def plot(self,
             color: Optional[ColorType] = None,
             alpha: float = 0.5,
             linewidth: Optional[float] = None,
             linestyle: str = '-',
             plot_edges: bool = True,
             plot_radii: bool = False,
             label: Optional[str] = None,
             show: bool = True,
             ax: Optional[Axes1D | Axes | Axes3D] = None,
             ) -> Axes1D | Axes | Axes3D:
        """Plot the ellipsoid. Only available when `self.n` ∈ {1, 2, 3}. Matplotlib must be installed.
                
        Parameters
        ----------
        color : ColorType, optional
            Color of the ellipsoid. If not provided, the next color-in-line (as determined by Matplotlib) is automatically selected. Note that `ColorType` is an alias for options such as named colors (e.g., `blue`) or RGB(A) tuples `(r, g, b, a)`.
        alpha : float, default=0.5
            Transparency of the ellipsoid
        linewidth : float, optional
            Linewidth of the edge when `plot_edges=True`. If `None`, the default linewidth will be used.
        linestyle : str, default='-'
            Linestyle of the edge when `plot_edges=True`. The default linestyle '-' is a solid line.
        plot_edges : bool, default=True
            Whether to plot the edges (equal to the boundary in 2d) of the ellipsoid
        plot_radii : bool, default=False
            Whether to plot the radii of the ellipsoid
        label : str, optional
            Label shown in the legend. If provided, a legend is automatically added to `ax`.
        show : bool, default=True
            Whether to show the ellipsoid using `plt.show()`
        ax : Axes1D, Axes, or Axes3D, optional
            An pre-defined axes object on which to plot (for plotting multiple convex regions)
        
        Returns
        -------
        ax : Axes1D, Axes, or Axes3D
            Axes object on which the ellipsoid is plotted

        Raises
        ------
        ImportError
            When Matplotlib is not installed
        ValueError
            When `self.n` ∉ {1, 2, 3} or when the provided `ax` object does not match `self.n`

        See also
        --------
        plot_radii : Plot the radii of the ellipsoid.

        Examples
        --------
        >>> Q = [[ 1  , -1/4],
        ...      [-1/4,    2]]
        >>> ellps = pes.ellps(Q)
        >>> ellps.plot()  # doctest: +SKIP
        .. image:: # FIXME
        """
        display_name = f"{self.__class__.__name__.lower()}"
        match self.n:
            case 1:
                ax, color = get_axes_color(ax, color, 1, display_name=display_name)
                ax.plot(edges := self.c + self.radii[0] * np.array([-1, 1]),
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        label=label)
                if plot_edges:
                    ax.scatter(edges, color=color)
                if label is not None:
                    ax.legend()
            case 2:
                ax, color = get_axes_color(ax, color, 2, display_name=display_name)
                ax.add_patch(Ellipse(xy=(self.c[0], self.c[1]),
                                     width=2 * self.radii[0],
                                     height=2 * self.radii[1],
                                     angle=np.rad2deg(self.angles).item(),
                                     facecolor=to_rgba(color, alpha=alpha),
                                     edgecolor=(to_rgba(color, alpha=1)
                                                if plot_edges
                                                else None),
                                     linewidth=linewidth,
                                     linestyle=linestyle,
                                     label=label,
                                     ))
                ax.autoscale_view()
                if label is not None:
                    ax.legend()
                if CFG.plot_aspect == 'equal':
                    ax.set_aspect('equal', adjustable='box')
            case 3:
                ax, color = get_axes_color(ax, color, 3, display_name=display_name)
                num_points = 100  # FIXME: Maybe make this a CFG setting?
                u, v = (np.linspace(0, 2 * np.pi, num_points),
                        np.linspace(0,     np.pi, num_points))
                sphere = np.array([self.radii[0] * np.outer(np.cos(u), np.sin(v)),
                                   self.radii[1] * np.outer(np.sin(u), np.sin(v)),
                                   self.radii[2] * np.outer(np.ones_like(u), np.cos(v))])
                xx, yy, zz = [(self.R @ sphere.reshape(3, -1)).reshape(3, *sphere.shape[1:])[i] + \
                               self.c[i] for i in range(3)]
                ax.plot_surface(xx, yy, zz,  # type: ignore[union-attr]
                                rstride=4, cstride=4,
                                color=color,
                                edgecolor=None if not plot_edges else color,
                                alpha=alpha,
                                label=label,
                                )
                if label is not None:
                    ax.legend()
                if CFG.plot_aspect == 'equal':
                    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])  # type: ignore[arg-type]
            case _:
                raise RuntimeError(f"Received n = {self.n} which is invalid, and should have been caught by previous code. Please report this bug.")

        if plot_radii:
            self.plot_radii(color=color, show=False, ax=ax)
        if show:
            plt.show()

        return ax

    def plot_radii(self,
                   color: Optional[ColorType] = None,
                   annotate: list[str] | bool = False,
                   label: Optional[str] = None,
                   show: bool = True,
                   ax: Optional[Axes1D | Axes | Axes3D] = None,
                   ) -> Axes1D | Axes | Axes3D:
        """Plot the radii of the ellipsoid. Only available when `self.n` ∈ {1, 2, 3}. Matplotlib must be installed.
                
        Parameters
        ----------
        color : ColorType, optional
            Color of the radii. If not provided, the next color-in-line (as determined by Matplotlib) is automatically selected. Note that `ColorType` is an alias for options such as named colors (e.g., `blue`) or RGB(A) tuples `(r, g, b, a)`.
        annotate : list[str] or bool, default=False
            Whether to annotate the radii. If `True`, an incremental annotation 0, 1, ... will be used. A custom list of annotations can be provided.
        label : str, optional
            Label shown in the legend. If provided, a legend is automatically added to `ax`.
        show : bool, default=True
            Whether to show the radii using `plt.show()`
        ax : Axes1D, Axes, or Axes3D, optional
            An pre-defined axes object on which to plot (for plotting multiple convex regions)
        
        Returns
        -------
        ax : Axes1D, Axes, or Axes3D
            Axes object on which the radii are plotted

        Raises
        ------
        ImportError
            When Matplotlib is not installed
        ValueError
            When `self.n` ∉ {1, 2, 3} or when the provided `ax` object does not match `self.n`

        See also
        --------
        plot : Plot the ellipsoid.

        Examples
        --------
        >>> Q = [[ 1  , -1/4],
        ...      [-1/4,    2]]
        >>> ellps = pes.ellps(Q)
        >>> ellps.plot_radii()  # doctest: +SKIP
        .. image:: # FIXME
        """
        ax, color = get_axes_color(ax, color, self.n, display_name=f"{self.__class__.__name__.lower()}")

        for idx, (vec, radius) in enumerate(zip(self.R.T, self.radii)):
            ax.plot(*(elem for elem in zip(self.c, self.c + (vec * radius))),
                    color=color,
                    label=label)
            if annotate:
                ax.text(*(self.c + vec * (radius / 2)), str(idx) if isinstance(annotate, bool) else annotate[idx])

        if label is not None:
            ax.legend()
        if show:
            plt.show()

        return ax


@wraps(Ellipsoid.__init__)
def ellps(*args: tuple[list[float], ArrayLike] | ArrayLike,
          n: Optional[int] = None,
          Q: Optional[ArrayLike] = None,
          radii: Optional[list[float]] = None,
          R: Optional[ArrayLike] = None,
          c: Optional[ArrayLike] = None,
          ) -> Ellipsoid:
    """Wrapper function for `Ellipsoid.__init__` to create an ellipsoid"""
    kwargs = {key: value for key, value in {
        'n': n,
        'Q': Q,
        'radii': radii,
        'R': R,
        'c': c,
    }.items() if value is not None}
    return Ellipsoid(*args, **{key: value for key, value in kwargs.items() if value is not None})


@wraps(Ellipsoid._init_quad)
def ellps_from_quad(Q: ArrayLike,
                    c: Optional[ArrayLike] = None,
                    ) -> Ellipsoid:
    """Wrapper function for `Ellipsoid.from_quad` to create an ellipsoid from a quadratic matrix"""
    return Ellipsoid(Q, c=c)


@wraps(Ellipsoid._init_rrepr)
def ellps_from_radii(radii: list[float],
                     R: Optional[ArrayLike] = None,
                     c: Optional[ArrayLike] = None,
                     ) -> Ellipsoid:
    """Construct an ellipsoid from a rotation matrix `R` and a set of radii `radii`"""
    return Ellipsoid(radii, R, c=c)


@wraps(Ellipsoid._init_empty)
def ellps_empty(n: int) -> Ellipsoid:
    """Construct an empty ellipsoid in R^n"""
    return Ellipsoid(n=n)
