"""Module for ellipsoid functionality"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from numpes._config import CFG
from numpes._internal.wraps import wraps
from numpes._internal.printing import sym_replace, pad
from numpes.utils.linalg import is_posdef

if TYPE_CHECKING:
    from typing import Literal, Optional

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
        self.R: NDArray = R
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
            if not np.isclose(radii, 0, rtol=CFG.rtol, atol=CFG.atol) or np.isfinite(radii).all():
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
        self._dim: int | None = None
        self._vol: float | None = None

    @property
    def n(self) -> int:
        """Dimension of the ambient space"""
        return self.radii.size

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
        U, sing, _ = np.linalg.svd(Q)
        R = U
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1  # Ensure R is a proper rotation matrix with det(R) = 1
        with np.errstate(divide='ignore'):
            radii = 1 / np.sqrt(sing)
        ellipsoid = cls(R, radii, Q, c)  # FIXME: Naming this `ellps` gives a shadowing warning, but I don't know if that is actually problematic?
        return ellipsoid

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
        c_vals = [pad(line, max([len(line) for line in c_lines])) for line in c_lines]
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


@wraps(Ellipsoid.from_quad)
def ellps(Q: ArrayLike, c: Optional[ArrayLike] = None,) -> Ellipsoid:
    ellps = Ellipsoid.from_quad(Q, c)
    return ellps


def ellps_empty(n: int) -> Ellipsoid:
    """Construct an empty ellipsoid in R^n"""
    R = np.empty((n, n))
    radii = np.empty(n)
    ellps = Ellipsoid(R, radii)
    return ellps


@wraps(Ellipsoid.__init__)
def ellps_from_radii(R: NDArray, radii: NDArray) -> Ellipsoid:
    """Construct an ellipsoid from a rotation matrix `R` and a set of radii `radii`"""
    ellps = Ellipsoid(R, radii)
    return ellps
