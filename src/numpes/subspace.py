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
from numpes._internal.printing import format_as_set
from numpes.utils.linalg import span

if TYPE_CHECKING:
    from typing import Optional, Any, Literal

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
    def is_trivial(self) -> bool:
        """Check whether the subspace is trivial (meaning it only contains the zero vector)"""
        return self.basis.size == 0
    
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
            basis_lines = format_as_set([str(np.atleast_2d(base).T) for base in basis], edgeitems).splitlines()
            nlines = len(basis_lines)
            idx_text = nlines // 2
            span_lines = ["     " if idx != idx_text else "span " for idx in range(nlines)]
            comb = "\n".join(["".join(line) for line in zip(span_lines, basis_lines)])
        else:  # This must be a trivial subspace
            comb = format_as_set([str(np.zeros((self.n, 1)))], edgeitems)
        return comb

    def __repr__(self) -> str:
        """Return a representation of the subspace attributes"""
        attrs = ", ".join(f"{key}={value}" for key, value in self._repr_items())
        return f"{self.__class__.__name__}({attrs})"

    def _repr_items(self,
                    pretty_ndarray: bool = False,
                    ) -> list[tuple[str, str]]:
        """Return (attribute, formatted-value) pairs used by repr formatting"""

        def fmt_repr_value(value: Any, pretty_ndarray: bool) -> str:
            if isinstance(value, np.ndarray):
                if not pretty_ndarray:
                    return " ".join([elem.strip() for elem in repr(value).splitlines()])
                return f"NDArray[shape={value.shape}, dtype={value.dtype}]"
            if isinstance(value, tuple):
                inner = ", ".join(fmt_repr_value(item, pretty_ndarray) for item in value)
                if len(value) == 1:
                    inner += ","
                return f"({inner})"
            return repr(value)

        return [(key, fmt_repr_value(value, pretty_ndarray)) for key, value in self.__dict__.items()]

    def __format__(self, format_spec: str) -> str:
        """Format the printed description of the subspace based on a format specifier"""
        token = format_spec
        kwargs, comb = {'threshold': 0}, ""

        if token == '':
            return str(self)
        if token == '?':
            return repr(self)
        if token == '#':
            attrs = ",\n    ".join(f"{key}={value}" for key, value in self._repr_items(pretty_ndarray=True))
            return f"{self.__class__.__name__}(\n    {attrs},\n)"
        if '?' in token or '#' in token:
            raise ValueError(f"Format specifiers '?' and '#' do not except any additional symbols, recieved '{format_spec}'")
        if token.startswith('!'):
            comb += self._str_header()
            token = token[1:]
        if len(token) == 0:
            return comb
        if token and token[0] in {' ', '+', '-'}:
            kwargs['sign'] = token[0]
            token = token[1:]
        if token[0] in {'f', 'e', 'E'}:
            char = token[0]  # FIXME: Why do I need this workaround? Because simply putting 'token[0]' leads to a index-out-of-bounds'error?
            kwargs['formatter'] = {'float': lambda x: f'{x:{kwargs.get('sign', '')}{('f' if char == 'f' else 'e')}}'}  # FIXME: Does NOT seem to work
            kwargs['to_dtype'] = 'float'
            token = token[1:]
        if token.startswith('.'):
            digits, idx = '', 1
            while idx < len(token) and token[idx].isdigit():
                digits += token[idx]
                idx += 1
            if not digits:
                raise ValueError(f"Invalid format '{format_spec}': '.' character must be followed by at least one digit, recieved '{token}'")
            token = token[idx:]
            if token and token[0] in {'f', 'e', 'E'}:
                char = token[0]
                kwargs['formatter'] = {'float': lambda x: f'{x:{kwargs.get('sign', '')}.{int(digits)}{('f' if char == 'f' else 'e')}}'}  # FIXME: Does NOT seem to work, seems to be completely ignored?
                kwargs['to_dtype'] = 'float'
                token = token[1:]
            elif token and token[0] == 'd':
                raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' cannot be followed by 'd' as integer type does not take precision")
            elif not token:
                raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' must be followed by 'f', 'e', or 'E', but was not followed by any character")
            else:
                raise ValueError(f"Invalid format '{format_spec}': precision format specifier '.*' must be followed by 'f', 'e', or 'E', recieved '{token[0]}'")
        elif token.startswith('d'):
            kwargs['to_dtype'] = 'int'
            token = token[1:]
        if token.startswith('~'):
            digits = token[1:]
            if not digits.isdigit():
                raise ValueError(f"Invalid format '{format_spec}': edgeitems modifier '~*' must be the final element and followed by a positive integer, recieved '{token}'")
            kwargs['edgeitems'] = int(digits)
            token = ''
        if token != '':
            raise ValueError(f"Invalid format '{format_spec}': trailing junk characters '{token}'")

        to_dtype = kwargs.get('to_dtype', None)
        kwargs.pop('to_dtype', None)
        with np.printoptions(**kwargs):
            str_basis = self._str_basis(to_dtype=to_dtype)
            if 'E' in format_spec:
                str_basis = str_basis.replace('e', 'E')
            comb += ("" if len(comb) == 0 else "\n") + str_basis

        return comb
    
    # [untested/unverified]
    def minimal(self,
                in_place: bool = True,
                ) -> Subspace:  # FIXME: Should this not always return a instance of Subspace? Either self, or a new instance?
        """Compute a minimal representation of the subspace by removing linearly dependent basis vectors"""
        basis = span(self._basis.T).T
        if in_place:
            self._basis = basis
            return self
        else:
            return Subspace(basis)


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
