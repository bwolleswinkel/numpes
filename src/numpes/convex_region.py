"""Module containing the ConvexRegion class, which represents convex regions in a vector space (more specifically, Euclidean space) and provides methods for checking if points are contained within the region"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class ConvexRegion(ABC):
    """Base class for all convex regions. A convex region in R^n is defined as a collection U = {x ∈ R^n} for which x, y in U implies x * λ + y * (λ - 1) in U, for all λ ∈ [0, 1].
    
    Properties
    ----------
    n : int
        Dimension of the ambient space
    dim : int
        Dimension of the convex region
    vol : float
        Volume of the convex region

    Methods
    -------
    __bool__
        Check if the convex region is empty
    __contains__
        Check if a point x is contained in the convex region

    Notes
    -----
    This class has no way of enforcing any inheriting class to actually define a convex region. As such, this is merely expected to be implemented as such.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Dimension of the ambient space"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the convex region"""

    @property
    @abstractmethod
    def vol(self) -> float:
        """Volume of the convex region"""

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if the convex region is empty"""

    @abstractmethod
    def __contains__(self, other: NDArray) -> bool:
        """Check if a point x is contained in the convex region"""
