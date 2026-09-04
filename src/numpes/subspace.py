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
from copy import copy

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

from numpes._config import CFG
from numpes._internal import wraps
from numpes._internal.printing import format_as_set, repr_items
from numpes.exceptions import InvalidRepresentationError
from numpes.utils.linalg import span
from numpes.utils.plot import add_1d_subplot, plot_box, plot_line, plot_plane, plot_vector

if TYPE_CHECKING:
    from typing import Self, Literal, Optional, Any

    from matplotlib.axes import Axes
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

    # [untested/unverified]
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
        self._basis: NDArray = np.empty((0, 0))
        self._is_minimal: bool | None = None  # FIXME: Should this be 'minimal'? 'reduced'? '(non)-redundant'?  'echelon'? 'canonical'?
        self._is_trivial: bool | None = None
        self._is_bounded: bool | None = None
        self._is_lower_dim: bool | None = None
        self._is_full_dim: bool | None = None
        self._is_singleton: bool | None = None
        self._dim: int | None = None
        self._vol: float | None = None

        if basis is None:
            if n is None:
                raise ValueError("If no basis is provided, the keyword argument 'n' must be provided")
            if not isinstance(n, int):
                raise ValueError(f"Ambient dimension 'n' must be of type integer, but received '{n}'")
            if n <= 0:
                raise ValueError(f"Ambient dimension 'n' must be positive integer, but received '{n}'")
            basis = np.empty((0, n))
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
            raise ValueError("Vertices 'basis' cannot contain NaN or inf values")

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
    def dim(self) -> int:
        """Dimension of the subspace"""
        if self._dim is None:
            self._dim = np.linalg.matrix_rank(self.basis, CFG.atol)
        return self._dim

    # [untested/unverified]
    @property
    def is_trivial(self) -> bool:
        """Check whether the subspace is trivial (meaning it only contains the zero vector)"""
        if self._is_trivial is None:
            self._is_trivial = (np.linalg.matrix_rank(self.basis, tol=CFG.atol) == 0).item()
        return self._is_trivial

    # [untested/unverified]
    @property
    def perp(self) -> Subspace:
        """"Returns the subspace orthogonal to the currecnt subspace"""
        basis_ortho = sp.linalg.null_space(self.basis).T
        return Subspace(basis_ortho)

    def __str__(self) -> str:
        """Descriptive representation of the subspace"""
        header = self._str_header()
        with np.printoptions(threshold=0):
            str_basis = self._str_basis()
        return header + "\n" + str_basis

    # [untested/unverified]
    def _str_header(self) -> str:
        # NOTE: These methods are not yet implemented
        if self.is_trivial:
            return f"Trivial subspace in R^{self.n}"
        # if self.is_full_space:
        #     return f"Full space subspace in R^{self.n}"
        return f"Subspace in R^{self.n}"

    # [untested/unverified]
    def _str_basis(self, to_dtype: Optional[Literal['float', 'int']] = None) -> str:
        """"Descriptive representation of the basis of the polytope"""
        edgeitems = np.get_printoptions()['edgeitems']
        match to_dtype:
            case None:
                basis = self.basis
            case 'float':
                basis = self.basis.astype(float)
            case 'int':
                basis = self.basis.astype(int)
            case _:
                raise ValueError(f"Unrecognized value '{to_dtype}' for 'to_dtype'")
        if basis.size != 0:
            if basis.dtype == float:  # Add array of zeros to avoid `-0.` in print output
                basis_lines = format_as_set([str(np.atleast_2d(base).T + np.zeros((self.n, 1))) for base in basis], edgeitems).splitlines()
            else:
                basis_lines = format_as_set([str(np.atleast_2d(base).T) for base in basis], edgeitems).splitlines()
            nlines = len(basis_lines)
            idx_text = nlines // 2
            span_lines = ["     " if idx != idx_text else "span " for idx in range(nlines)]
            comb = "\n".join(["".join(line) for line in zip(span_lines, basis_lines)])
        else:  # This must be a trivial subspace
            dtype = int if to_dtype in {None, 'int'} else float
            comb = format_as_set([str(np.zeros((self.n, 1), dtype=dtype))], edgeitems)
        return comb

    def __repr__(self) -> str:
        """Return a representation of the subspace attributes"""
        attrs = ", ".join(f"{key}={value}" for key, value in repr_items(self))
        return f"{self.__class__.__name__}({attrs})"

    def __format__(self, format_spec: str) -> str:
        """Format the printed description of the subspace based on a format specifier"""
        token = format_spec
        kwargs, comb = {'threshold': 0}, ""

        if token == '':
            return str(self)
        if token == 'r':
            return repr(self)
        if token == '#':
            attrs = ",\n    ".join(f"{key}={value}" for key, value in repr_items(self, compact_ndarray=True))
            return f"{self.__class__.__name__}(\n    {attrs},\n)"
        if 'r' in token or '#' in token:
            raise ValueError(f"Format specifiers 'r' and '#' do not except any additional symbols, received '{format_spec}'")
        if token.startswith('i'):
            comb += self._str_header()
            token = token[1:]
        if len(token) == 0:
            return comb
        if token[0] in {' ', '+', '-'}:
            kwargs['sign'] = token[0]
            token = token[1:]
        if token and token[0] in {'f', 'e', 'E'}:
            sign = kwargs.setdefault('sign', ' ')
            float_format = 'f' if token[0] == 'f' else 'e'
            kwargs['formatter'] = {
                'float': lambda value, sign=sign, float_format=float_format: f'{value:{sign}{float_format}}',
            }
            kwargs['to_dtype'] = 'float'
            token = token[1:]
        if token.startswith('.'):
            digits, idx = '', 1
            while idx < len(token) and token[idx].isdigit():
                digits += token[idx]
                idx += 1
            if not digits:
                raise ValueError(f"Invalid format '{format_spec}': '.' character must be preceded by at least one digit, received '{token}'")
            token = token[idx:]
            if token and token[0] in {'f', 'e', 'E'}:
                char = token[0]
                kwargs.setdefault('sign', ' ')
                if char == 'f' and digits == '0':
                    kwargs['formatter'] = {
                        'float': lambda value: f"{value:{kwargs.get('sign', '')}.0f}.",
                    }
                else:
                    kwargs['formatter'] = {
                        'float': lambda value: f"{value:{kwargs.get('sign', '')}.{int(digits)}{char}}",
                    }
                kwargs['to_dtype'] = 'float'
                token = token[1:]
            elif token and token[0] == 'd':
                raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' cannot be followed by 'd' as integer type does not take precision")
            elif not token:
                raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' must be followed by 'f', 'e', or 'E', but was not followed by any character")
            else:
                raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' must be followed by 'f', 'e', or 'E', received '{token[0]}'")
        elif token.startswith('d'):
            kwargs['to_dtype'] = 'int'
            token = token[1:]
        if token.startswith('~'):
            digits = token[1:]
            if not digits.isdigit():
                raise ValueError(f"Invalid format '{format_spec}': edgeitems modifier '~*' must be the final element and followed by a positive integer, received '{token}'")
            kwargs['edgeitems'] = int(digits)
            token = ''
        if token != '':
            raise ValueError(f"Invalid format '{format_spec}': trailing junk characters '{token}'")

        to_dtype = kwargs.get('to_dtype', None)
        kwargs.pop('to_dtype', None)
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        with np.printoptions(**kwargs):
            str_basis = self._str_basis(to_dtype=to_dtype)
            if 'E' in format_spec:
                str_basis = str_basis.replace('e', 'E')
            comb += ("" if len(comb) == 0 else "\n") + str_basis

        return comb

    # [untested/unverified]
    def __iter__(self) -> NDArray:
        return iter(self.basis)

    # [untested/unverified]
    # pylint: disable=protected-access
    def copy(self,
             deepcopy: bool = True,
             memo: Optional[dict[int, Any]] = None,
             ) -> Subspace:
        """Return a (deep)copy of the subspace. 

        Parameters
        ----------
        deepcopy : bool, default=True
            If True, a deep copy of the subspace is returned (totally isolated from the original polytope). If False, a shallow copy is returned.
        memo : dict[int, Any], optional
            A dictionary of objects already copied during the current copying pass, used by `copy.deepcopy` to avoid infinite recursion when copying objects with circular references. If None, a new empty dictionary is created.

        Returns
        -------
        Subspace
            A (deep)copy of the subspace

        Warnings
        --------
        If `deepcopy` is set to False, the returned subspace will share references to the same underlying data as the original subspace. Modifications to the NumPy array `basis` in either subspace will affect both subspace.
        """
        if not deepcopy:
            return copy(self)

        memo = {} if memo is None else memo
        if id(self) in memo:
            return memo[id(self)]

        obj = copy(self)
        memo[id(self)] = obj

        obj._basis = self._basis.copy()

        return obj

    # [untested/unverified]
    def minimal(self,
                in_place: bool = True,
                ) -> Subspace | Self:
        """Compute a minimal representation of the subspace by removing linearly dependent basis vectors"""
        obj = self if in_place else self.copy()
        obj._basis = span(self._basis.T).T
        return obj

    # [untested/unverified]
    def plot(self,
             color: Optional[str] = None,
             alpha: float = 0.5,
             label: Optional[str] = None,
             plot_basis: bool = False,
             show: bool = True,
             ax: Optional[Axes] = None,
             ) -> Axes:
        """Plot the subspace"""
        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib is required for plotting." \
            " Please install it with 'pip install matplotlib' and try again.")

        if ax is None:
            if self.n == 1:
                fig = plt.figure()
                ax = add_1d_subplot(fig)
            elif self.n == 2:
                fig, ax = plt.subplots()
            elif self.n == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                raise ValueError(f"Plotting is only supported for n-d polytopes with n <= 3, received n={self.n}")
        else:
            fig = None

        if color is None:
            # pylint: disable=protected-access
            color = ax._get_lines.get_next_color()  # type: ignore[union-attr, attr-defined]

        if self.n > 3:
            raise ValueError(f"Plotting is only supported for n-d subspaces with n <= 3, received n={self.n}")

        match self.dim:
            case 0:
                ax.plot(*[0 for _ in range(self.n)], 'o', color=color, label=label)
            case 1:
                line = plot_line(ax, self.basis[0, :], color=color, alpha=alpha)
                if label is not None:
                    line.set_label(label)
            case 2:
                if self.n == 1:
                    raise InvalidRepresentationError(f"Expected dimension 'n' to be smaller or equal to 'dim', but recieved n={self.n}, dim={self.dim}, indicating the attributes of this subspace are in an invalid state")
                plane = plot_plane(ax, self.perp.basis.squeeze() if self.n == 3 else None, color=color, alpha=alpha)
                if label is not None:
                    plane.set_label(label)
            case 3:
                if self.n <= 2:
                    raise InvalidRepresentationError(f"Expected dimension 'n' to be smaller or equal to 'dim', but recieved n={self.n}, dim={self.dim}, indicating the attributes of this subspace are in an invalid state")
                box = plot_box(ax, color=color, alpha=alpha)
                if label is not None:
                    box.set_label(label)
            case _:
                raise InvalidRepresentationError(f"Expected dimension 'n' to be smaller or equal to 'dim', but recieved n={self.n}, dim={self.dim}, indicating the attributes of this subspace are in an invalid state")

        for idx in range(self.n):
            lims = getattr(ax, f'get_{['x', 'y', 'z'][idx]}lim')()
            getattr(ax, f'set_{['x', 'y', 'z'][idx]}lim')(min(lims[0], -1), max(lims[1], 1))

        if plot_basis:
            self.plot_basis(color=color, show=False, ax=ax)

        if show:
            plt.show()

        return ax

    # [untested/unverified]
    def plot_basis(self,
                   color: str | None = None,
                   label: Optional[list[str]] = None,
                   show: bool = True,
                   ax: Optional[Axes] = None,
                   ) -> Axes:
        """Plot the basis of the subspace"""
        if not MATPLOTLIB_INSTALLED:
            raise ImportError("Matplotlib is required for plotting." \
            " Please install it with 'pip install matplotlib' and try again.")

        if ax is None:
            if self.n == 1:
                fig = plt.figure()
                ax = add_1d_subplot(fig)
            elif self.n == 2:
                fig, ax = plt.subplots()
            elif self.n == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                raise ValueError(f"Plotting is only supported for n-d polytopes with n <= 3, received n={self.n}")
        else:
            fig = None

        if color is None:
            # pylint: disable=protected-access
            color = ax._get_lines.get_next_color()  # type: ignore[union-attr, attr-defined]

        if self.n > 3:
            raise ValueError(f"Plotting is only supported for n-d subspaces with n <= 3, received n={self.n}")

        for idx, basis_vector in enumerate(self):
            plot_vector(ax, basis_vector, color=color, label=label if idx == 0 else None)

        if show:
            plt.show()

        return ax


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
