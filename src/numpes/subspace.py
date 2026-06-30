"""Module for subspace functionality

Classes
-------
Subspace
    Class representing a subspace by means of its basis vectors
AffineSubset
    Class representing a affine subset, which is a linear translation of a subspace by some offset vector
QuotientSpace
    Class representing a quotient space, which is a vector space of affine subsets forming a equivelance relation between similar bases
    
    
Functions
---------
subs
    Factory function for creating a Subspace given a basis
subs_empty
    Factory function for creating an empty Subspace
subs_full
    Factory function for creating a full Subspace
subs_from_orth
    Factory function for creating a Subspace given a normal (orthogonal) vector
subs_from_card
    Factory function for creating a Subspace given a list of cardinal axis to be included
aff_subs
    Factory function for creating a AffineSubset given a basis and an offset vector
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from numpes._config import CFG
from numpes._internal import wraps

if TYPE_CHECKING:
    from typing import Optional, Self

    from numpy.typing import ArrayLike, NDArray


# TODO: Inherit from a common base class ConvexRegion
class Subspace:
    """Subspace represented by its basis vectors.
    
    A (linear) subspace is the collection of all vectors that are the linear combination of a set of
    basis vectors. A subspace cannot be empty and always includes the zero vector: if this is the only
    element, it is refferred to as the trivial subspace.
    
    Attributes
    ----------
    basis: NDArray[shape=(d, n)]
        Matrix of shape `(d, n)` defining basis vectors as rows which span the subspace
    n: int
        Dimension of the ambient space in which the subspace is embedded
    dim: int
        Dimension of the subspace. Note that `dim <= n`.

    Methods
    -------
    to_aff
        Convert the subspace to a AffineSubset
    """

    def __init__(self,
                 basis: Optional[ArrayLike] = None,
                 n: Optional[int] = None,
                 ) -> None:
        r"""Initialize a Subspace from a set of basis vectors.

        Parameters
        ----------
        basis: ArrayLike[shape=(d, n)]
            Matrix of shape `(d, n)` defining `d` basis vectors of dimension `n`

        Raises
        ------
        TypeError
            If the types of the provided arguments are inconsistent with the expected types for initialization
        ValueError
            If the entries of `basis` contain invalid numerical values such as NaN of inf

        Examples
        --------
        Initialize a subspace from basis vectors.

        >>> basis = [[1, 0,  0],
        ...          [0, 1, -1]]
        >>> subs = pes.subs(basis)
        >>> print(subs)
        Subspace with 2 basis vectors in R^3
             /[[1]  [[ 0] \
        span < [0] , [ 1] >
             \ [0]]  [-1]]/

        Initialize a trivial subspace in R^n.

        >>> subs = pes.subs(n=5)
        >>> print(subs)
        Trivial subspace in R^5
        /[[0] \
        | [0] |
        < [0] >
        | [0] |
        \ [0]]/
        """
        self._basis: NDArray = None
        self._is_minimal: bool | None = None  # FIXME: Should this be 'minimal'? 'reduced'? '(non)-redundant'?  'echelon'? 'canonical'?
        self._is_trivial: bool | None = None
        self._is_bounded: bool | None = None
        self._is_lower_dim: bool | None = None
        self._is_full_dim: bool | None = None
        self._is_singleton: bool | None = None
        self._dim: int | None = None
        self._vol: float | None = None

        # FIXME: Temporary fix
        if basis is None:
            raise NotImplementedError("Providing a None basis (empty initializtion) is not yet implemented")
        else:
            self._init_basis(basis)

    def _init_basis(self,
                    basis: ArrayLike,
                    ) -> None:
        """Initialize a subspace given a set of basis vectors"""
        basis = np.atleast_2d(basis)
        if basis.ndim != 2:
            raise ValueError(f"Basis vectors must be provided as a 2D array of shape (d, n)," \
                             f" but received an array of shape {basis.shape}")
        if np.isnan(basis).any() or not np.isfinite(basis).all():
            raise ValueError("Vertices 'basis' cannot contain NaN of inf values")

        self.basis = basis

    @property
    def basis(self) -> NDArray:
        """Basis vectors of the subspace"""
        return self._basis
    
    @basis.setter
    def basis(self, value: NDArray) -> None:
        """Set the basis vectors of the subspace.
        
        Note
        ----
        By default, the config option `pes.get_config()['on_property_assign'] == 'reduce'`,
        which means linearly dependent rows of the representation will be removed on assignment.
        """
        self._basis = value
        match CFG.on_property_assign:
            case 'pass':
                pass
            case 'minimal':
                self.minimal()
            case _:
                raise ValueError(f"Unknown value '{CFG.on_property_assign}' for 'on_property_assign' config setting")

    @property
    def n(self) -> int:
        """Dimension of the ambient space"""
        return self.basis.shape[1]
    
    @property
    def d(self) -> int:
        """Number of basis vectors of the subspace. Note that if the subspace is not reduced, `d >= dim`."""
        return self.basis.shape[0]
    
    @property
    def is_trivial(self) -> bool:
        """Check whether the subspace is trivial (meaning it only contains the zero vector)"""
        return self.basis.size == 0
    
    def minimal(self,
                in_place: bool = True,
                ) -> Subspace:  # FIXME: Should this not always return a instance of Subspace? Either self, or a new instance?
        """Compute a minimal representation of the subspace by removing linearly dependent basis vectors"""
        # basis = span(self._basis)
        # if in_place:
        #     self._basis = basis
        #     return self
        # else:
        #     return Subspace(basis)


# FIXME: I need to change `wraps` such that only the docstring gets copied, but not the signature
@wraps(Subspace.__init__)  # pylint: disable=protected-access
def subs(basis: Optional[ArrayLike] = None,
         /,
         n: Optional[int] = None,
         ) -> Subspace:
    """Wrapper function for `Subspace.__init__` to create a subspace"""
    kwargs = {key: value for key, value in {
        'n': n,
        'basis': basis,
    }.items() if value is not None}
    return Subspace(**kwargs)
