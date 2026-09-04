"""Module for ellipsoid functionality"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

from numpes._config import CFG
from numpes._internal.printing import pad, sym_replace
from numpes._internal.wraps import wraps
from numpes.utils.linalg import angles_givens, is_posdef

if TYPE_CHECKING:
    from typing import Literal, Optional

    from matplotlib.axes import Axes  # FIXME: Should we make this a lazy import/exclude import error if matplotlib is not installed?
    from numpy.typing import ArrayLike, NDArray


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

    def __init__(self,
                 R: ArrayLike,
                 radii: ArrayLike,
                 Q: Optional[ArrayLike] = None,
                 c: Optional[ArrayLike] = None,
                 ) -> None:
        R = np.atleast_2d(R)
        radii = np.atleast_1d(radii)
        if not R.ndim == 2:
            raise ValueError(f"Rotation matrix 'R' must be a two-dimensional array, received {R.shape}")
        if not R.shape[0] == R.shape[1]:
            raise ValueError(f"Rotation matrix 'R' much be a square matrix of size (n, n), received {R.shape}")
        if not radii.ndim == 1:
            raise ValueError(f"Radii 'radii' must be a one-dimensional array-like, received {radii.shape}")
        if not radii.size == R.shape[0]:
            raise ValueError(f"Number of 'radii' must equal shape of 'R', received radii of size {radii.size} and rotation matrix of shape {R.shape}")
        if (radii < 0).any():
            raise ValueError(f"Radii must be strictly non-negative, received non-negative radius of {radii[np.argwhere(radii < 0).min()]} at index {np.argwhere(radii < 0).min()}")
        if np.isnan(radii).any():
            raise ValueError(f"Radii cannot be NaN value, received {radii}")
        self.radii: NDArray = radii
        self.R: NDArray = R  # FIXME: Here, upon assignment, I should do the reordering of R based on the order or radii
        # FIXME: I should implement this instead | Or should I do both representations? Tough decision...
        self._rrepr: tuple[NDArray, NDArray] | None = (R, radii)
        if Q is not None:
            Q = np.atleast_2d(Q)
            if not Q.ndim == 2:
                raise ValueError(f"Quadratic matrix 'Q' must be a two-dimensional array, received {Q.shape}")
            if not Q.shape[0] == Q.shape[1]:
                raise ValueError(f"Quadratic matrix 'Q' much be a square matrix of size (n, n), received {Q.shape}")
            if not Q.shape[0] == self.n:
                raise ValueError(f"Quadratic matrix 'Q' must have size (n, n), n={self.n}, received {Q.shape}")
            self.Q: NDArray = Q
        else:
            if not np.isclose(radii, 0, rtol=CFG.rtol, atol=CFG.atol).any() and np.isfinite(radii).all():
                self.Q = self.R @ np.diag(self.radii) @ self.R.T
            else:
                self.Q = np.full((self.n, self.n), np.nan)
        if c is None:
            self.c: NDArray = np.zeros(self.n)
        else:
            c = np.atleast_1d(c)
            if not c.ndim == 1:
                raise ValueError(f"Center 'c' must be a one-dimensional array, received {c.shape}")
            if not c.size == self.n:
                raise ValueError(f"Center 'c' must be a vector of size (n,), n={self.n}, received {c.shape}")
            self.c = c
        self.cov: NDArray | None = None
        self.chi: float | None = None
        # FIXME: Maybe all of these can just be properties that are computed on demand, as they are not expensive to compute
        self._is_empty: bool | None = np.isnan(radii).all()
        self._is_ambient: bool | None = np.isinf(radii).all()
        self._is_degen: bool | None = (np.isclose(radii, 0, atol=CFG.atol).any()
                                       or np.isinf(radii).any())
        self._is_bounded: bool | None = np.isfinite(radii).all()
        self._is_full_dim: bool | None = not np.isclose(radii, 0, atol=CFG.atol).any()
        self._is_singleton: bool | None = np.isclose(radii, 0, atol=CFG.atol).all()
        self._angles: list[float] | None = None
        self._dim: int | None = None
        self._vol: float | None = None

    @property
    def n(self) -> int:
        """Dimension of the ambient space"""
        return self.radii.size

    @property
    def angles(self) -> list[float]:
        """Rotation angles of the ellipsoid"""
        if self._angles is None:
            self._angles = angles_givens(self.R)
        return self._angles

    @classmethod
    def from_quad(cls,
                  Q: ArrayLike,
                  c: Optional[ArrayLike] = None,
                  ) -> Ellipsoid:
        """Construct an ellipsoid from a quadratic matrix `Q` and optionally a center `c`"""
        Q = np.atleast_2d(Q)
        if not Q.ndim == 2:
            raise ValueError(f"Quadratic matrix 'Q' must be a two-dimensional array, received {Q.shape}")
        if not Q.shape[0] == Q.shape[1]:
            raise ValueError(f"Quadratic matrix 'Q' much be a square matrix of size (n, n), received {Q.shape}")
        if c is None:
            c = np.zeros(Q.shape[0])
        else:
            c = np.atleast_1d(c)
            if not c.ndim == 1:
                raise ValueError(f"Center 'c' must be a one-dimensional array, received {c.shape}")
            if not c.size == Q.shape[0]:
                raise ValueError(f"Center 'c' must be a vector of size (n,), n={Q.shape[0]}, received {c.shape}")
        if not is_posdef(Q, semi_def=True):
            raise ValueError("Quadratic matrix 'Q' must be positive (semi)-definite")
        eigvals, R = np.linalg.eigh(Q)
        with np.errstate(divide='ignore'):
            radii = 1 / np.sqrt(np.maximum(eigvals, 0))
        # FROM: Gemini 3.1 Pro | 2026/08/22[untested/unverified]
        # Flip columns sign to map the Givens angles to [0, π)
        for i in range(R.shape[1] - 1):
            if R[i, i] < 0:
                R[:, i] *= -1
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1  # Ensure R is a proper rotation matrix with det(R) = 1
        return cls(R, radii, Q, c)

    @classmethod
    def from_cov(cls,
                 cov: ArrayLike,
                 a: float,
                 mu: Optional[ArrayLike] = None,
                 mode: Literal['chi', 'nstd', 'abs'] = 'chi',
                 ) -> Ellipsoid:
        """Construct an ellipsoid from a covariance matrix `cov`, factor `a`, and optionally a mean `mu`"""
        cov = np.atleast_2d(cov)
        if mu is None:
            mu = np.zeros(cov.shape[0])
        else:
            mu = np.atleast_1d(mu)
            if not mu.ndim == 1:
                raise ValueError(f"Mean 'mu' must be a one-dimensional array, received {mu.shape}")
            if not mu.size == cov.shape[0]:
                raise ValueError(f"Mean 'mu' must be a vector of size (n,), n={cov.shape[0]}, received {mu.shape}")
        if not cov.ndim == 2:
            raise ValueError(f"Covariance matrix 'cov' must be a two-dimensional array, received {cov.shape}")
        raise NotImplementedError(...)

    # [untested/unverified]
    # pylint: disable=invalid-name
    def __str__(self) -> str:
        """Descriptive representation of the ellipsoid"""
        header = self._str_header()
        with np.printoptions(threshold=0):
            c_as_str, Q_as_str = str(np.atleast_2d(self.c).T), str(self.Q)
        c_lines, Q_lines = c_as_str.splitlines(), Q_as_str.splitlines()
        nlines = len(Q_lines)
        try:
            idx_trunc = Q_lines.index(' ...')
            # NOTE: This assumes the number of edgeitems above and below is always identical
            c_lines = c_lines[:idx_trunc] + [' ...'] + c_lines[-idx_trunc:]
        except ValueError as _:
            idx_trunc = None
        idx_text = nlines - (1
                            if (nlines <= 2 or (nlines == 3 and idx_trunc is not None))
                            else 2)

        c_text = ['   ' if idx != idx_text else 'c: ' for idx in range(nlines)]
        c_vals = [pad(line, max(len(line) for line in c_lines)) for line in c_lines]
        Q_text = ['     ' if idx != idx_text else ', Q: ' for idx in range(nlines)]
        Q_vals = sym_replace(Q_as_str).splitlines()
        comb = '\n'.join([''.join(line) for line in zip(c_text, c_vals, Q_text, Q_vals)])
        if self.n == 1:
            comb = comb.replace('[[', '[').replace(']]', ']')

        comb = header + "\n" + comb
        return comb

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

    # untested/unverified
    def plot(self,
             num_points: int = 100,
             color: Optional[str] = None,
             alpha: float = 0.5,
             plot_edges: bool = True,
             plot_radii: bool = False,
             label: Optional[str] = None,
             show: bool = True,
             ax: Optional[Axes | Axes3D] = None,
             ) -> Axes | Axes3D:
        """Plot the ellipsoid"""

        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib is required for plotting. " \
                              "Please install it with 'pip install matplotlib' and try again.")

        if ax is None:
            if self.n == 1:
                raise NotImplementedError("Plotting is not yet implemented for 1D ellipsoids")
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

        match self.n:
            case 1:
                raise NotImplementedError("Plotting is not yet implemented for 1D ellipsoids")
            case 2:
                if ax.name == '3d':
                    raise ValueError("The dimension of the ellipsoid" \
                                     " does not match the dimension of the provided axes 'ax'")
                ax.add_patch(Ellipse(xy=(self.c[0], self.c[1]),
                                     width=2 * self.radii[0],
                                     height=2 * self.radii[1],
                                     angle=np.rad2deg(self.angles).item(),
                                     facecolor=mpl.colors.to_rgba(color, alpha=alpha),
                                     edgecolor=(mpl.colors.to_rgba(color, alpha=1)
                                                if plot_edges
                                                else None),
                                     label=label,
                                     ))
                ax.autoscale_view()
            case 3:
                if ax.name != '3d':
                    raise ValueError(f"The dimension of the ellipsoid n={self.n} " \
                                      "does not match the dimension of the provided axes 'ax'")
                u, v = (np.linspace(0, 2 * np.pi, num_points),
                        np.linspace(0,     np.pi, num_points))
                sphere = np.array([self.radii[0] * np.outer(np.cos(u), np.sin(v)),
                                   self.radii[1] * np.outer(np.sin(u), np.sin(v)),
                                   self.radii[2] * np.outer(np.ones_like(u), np.cos(v))])
                xx, yy, zz = [(self.R @ sphere.reshape(3, -1)).reshape(3, *sphere.shape[1:])[i] + \
                               self.c[i] for i in range(3)]
                ax.plot_surface(xx,
                                yy,
                                zz,
                                rstride=4,
                                cstride=4,
                                color=color,
                                edgecolor=None if not plot_edges else color,
                                alpha=alpha,
                                label=label,
                                )
            case _:
                raise ValueError(f"Plotting is only supported for n-d ellipsoid with n <= 3, received n = {self.n}")

        if plot_radii:
            self.plot_radii(color=color, show=False, ax=ax)

        if show:
            plt.show()
        return ax

    # [untested/unverified]
    def plot_radii(self,
                   color: Optional[str] = None,
                   label: Optional[str] = None,
                   annotate: bool | list[str] = True,
                   show: bool = True,
                   ax: Optional[Axes | Axes3D] = None,
                   ) -> Axes | Axes3D:
        """Plot the radii of the ellipse"""
        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib is required for plotting." \
                                " Please install it with 'pip install matplotlib' and try again.")

        if ax is None:
            if self.n == 1:
                raise NotImplementedError("Plotting is not yet implemented for 1D ellipsoids")
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

        if self.n > 3:
            raise ValueError(f"Plotting is only supported for n-d ellipsoid with n <= 3, received n = {self.n}")
        for idx, (vec, radius) in enumerate(zip(self.R.T, self.radii)):
            ax.plot(*(elem for elem in zip(self.c, self.c + (vec * radius))), color=color, label=label)
            if annotate:
                ax.text(*(self.c + vec * (radius / 2)), str(idx) if isinstance(annotate, bool) else annotate[idx])

        if show:
            plt.show()
        return ax


@wraps(Ellipsoid.from_quad)
def ellps(Q: ArrayLike, c: Optional[ArrayLike] = None,) -> Ellipsoid:
    """Wrapper function for `Ellipsoid.from_quad` to create an ellipsoid from a quadratic matrix"""
    ellpsoid = Ellipsoid.from_quad(Q, c)
    return ellpsoid


def ellps_empty(n: int) -> Ellipsoid:
    """Construct an empty ellipsoid in R^n"""
    R = np.empty((n, n))
    radii = np.empty(n)
    ellpsoid = Ellipsoid(R, radii)
    return ellpsoid


@wraps(Ellipsoid.__init__)
def ellps_from_radii(radii: ArrayLike, R: Optional[ArrayLike] = None, c: Optional[ArrayLike] = None) -> Ellipsoid:
    """Construct an ellipsoid from a rotation matrix `R` and a set of radii `radii`"""
    if R is None:
        # FIXME: This is not correct! This should create an R matrix based on the sorted radii (I think), such that the major axis is always the largest. Oh no this is correct, this 'magic' should be done by the property assignment inside ellps
        R = np.eye(np.atleast_1d(radii).size)
    ellpsoid = Ellipsoid(R, radii, c=c)
    return ellpsoid
