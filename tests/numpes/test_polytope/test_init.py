"""Tests for the constructor of the Polytope class"""

from typing import TYPE_CHECKING
import re

import numpy as np
import numpes as pes
from numpes import InvalidCombinationOfArguments
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from tests.conftest import N_MAX

if TYPE_CHECKING:
    from typing import EllipsisType
    from numpy.typing import NDArray


@pytest.mark.parametrize('args, kwargs', [
    ((), {'n': 1}),
    ((), {'n': 2}),
    ((), {'n': 3}),
    ((), {'n': 5}),
    ((), {'n': 10}),
    ((), {'n': 20_000})
])
def test_polytope_init_empty_valid(args: tuple[()], kwargs: dict[str, int]):
    _ = pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((), {'n': -1}),
    ((), {'n': 0}),
])
def test_polytope_init_empty_value_error(args: tuple[()], kwargs: dict[str, int]):
    with pytest.raises(ValueError, match=re.escape(
        f"Dimension 'n' must be a positive integer, got n={kwargs['n']}")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((), {'n': 0.5}),
    ((), {'n': np.inf}),
    ((), {'n': np.nan}),
    ((), {'n': '3'}),
    ((), {'n': ...})
])
def test_polytope_init_empty_type_error(args: tuple[()], kwargs: dict[str, float | int]):
    with pytest.raises(TypeError, match=re.escape(
        f"Dimension 'n' must be a positive integer, received {kwargs['n']} of type '{type(kwargs['n']).__name__}'")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((2,), {})
])
def test_polytope_init_empty_dispatch_init_vrepr_type_error(args: tuple[()] | tuple[int], kwargs: dict[str, float | int]):
    with pytest.raises(TypeError, match=re.escape(
        "A single positional argument cannot be an integer (for an empty polytope initialization). Please refer to the documentation for valid argument combinations.")):
        pes.Polytope(*args, **kwargs)


def test_polytope_init_empty_manual_verts():
    ...


def test_polytope_init_empty_manual_facets():
    ...


@pytest.mark.parametrize('args, kwargs, expected_msg', [
    ((), {},
     "Dimension 'n' must be provided for empty polytope initialization"),
    ((), {'n': 2, 'verts': ...},
     "Cannot provide 'n' when initializing from vertices"),  # NOTE: Should dispatch to `_init_vrepr`
    ((), {'n': 2, 'rays': ...},
     "Cannot provide 'rays' when initializing an empty polytope"),
    ((), {'n': 2, 'A': ...},
     "An invalid number or combination of arguments was provided, received args=(), kwargs={'n': 2, 'A': Ellipsis}. Please refer to the documentation for details on valid combinations or arguments."),  # NOTE: Should 'dispatch' to `__init__` (default method if no dispatchers match)
    ((), {'n': 2, 'b': ...},
     "An invalid number or combination of arguments was provided, received args=(), kwargs={'n': 2, 'b': Ellipsis}. Please refer to the documentation for details on valid combinations or arguments."),  # NOTE: Should 'dispatch' to `__init__` (default method if no dispatchers match)
    ((), {'n': 2, 'A_eq': ...},
     "Cannot provide 'A_eq' or 'b_eq' when initializing an empty polytope"),
    ((), {'n': 2, 'b_eq': ...},
     "Cannot provide 'A_eq' or 'b_eq' when initializing an empty polytope")
])
def test_polytope_init_empty_invalid_combination(args: tuple[()], kwargs: dict[str, int | EllipsisType], expected_msg: str):
    with pytest.raises(InvalidCombinationOfArguments, match=re.escape(expected_msg)):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((np.array([0]),), {}),
    ((np.array([[0]]),), {}),
    ((np.array([[0], 
                [1]]),), {}),
    ((np.array([[0  ], 
                [0.5], 
                [1  ]]),), {}),
    ((np.array([[ 3, 3]]),), {}),
    ((np.array([[-1, 2], 
                [-3, 4],
                [-5, 6]]),), {}),
    ((np.array([[ 0.5,   0.001], 
                [-0.2, 1/3    ],
                [   0,   0    ]]),), {}),
    ((np.array([1, 2, 3]),), {}),
    ((np.array([[1, 2, 3]]),), {}),
    ((np.array([[ 0,  0,  0],
                [-1,  0,  1], 
                [ 1,  1,  1], 
                [ 0,  1,  0],
                [-1, -1, -1]]),), {}),
    (tuple(), {'verts': np.array([1, 0.1, 0.01, 0.001])}),
    (tuple(), {'verts': np.array([[0, 1],
                                  [1, 0]])}),  # NOTE: Should dispatch to `_init_vrepr`
    ([[1, 2, 3, 4]], {}),  # NOTE: `[1, 2, 3, 4]` will be converted to a NumPy array due to `verts = np.atleast_2d(verts)` in `_init_vrepr`
    ((np.array([1, 0]),), {'rays': np.array([[1, 0]])}),
])
def test_polytope_init_vrepr_valid(args: tuple[NDArray], kwargs: dict[str, NDArray]):
    _ = pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((2,), {})
])
def test_polytope_init_vrepr_type_error(args: tuple[int], kwargs: dict[str, int]):
    with pytest.raises(TypeError, match=re.escape(
        "A single positional argument cannot be an integer (for an empty polytope initialization). Please refer to the documentation for valid argument combinations.")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((np.ones((2, 2, 2)),), {})
])
def test_polytope_init_vrepr_value_error(args: tuple[NDArray], kwargs: dict[str, NDArray]):
    with pytest.raises(ValueError, match=re.escape(
        f"Vertices must be provided as a 2D array of shape (k, n), but received an array of shape {args[0].shape}")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((np.array([[0, 1],
                [1, 0],
                [0, 0]]),), {'rays': np.array([[1, 0, 0]])})
])
def test_polytope_init_vrepr_value_error_rays(args: tuple[NDArray], kwargs: dict[str, NDArray]):
    with pytest.raises(ValueError, match=re.escape(
        f"Rays must be provided as a 2D array of shape (k_rays, n={args[0].shape[1]}), but received an array of shape {kwargs['rays'].shape}")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, expected_msg', [
    ((2,), {'n': 2},
     "Cannot provide 'n' when initializing from vertices"),
    ((np.array([[0.1], 
                [0.5]]),), {'n': ...},
     "Cannot provide 'n' when initializing from vertices"),
    ((np.array([[-1, 2], 
                [-3, 4],
                [-5, 6]]),), {'A': ...},
     "Cannot provide 'A' or 'b' when initializing from vertices"),
    ((np.array([[ 1.1, 1/2], 
                [ 3  , 4.0],
                [-5  , 0  ]]),), {'b': ...},
     "Cannot provide 'A' or 'b' when initializing from vertices"),
    ((np.zeros((4, 4)),), {'A_eq': ..., 'b_eq': ...},
     "Cannot provide 'A_eq' or 'b_eq' when initializing from vertices")
])
def test_polytope_init_vrepr_invalid_combination(args: tuple[NDArray], kwargs: dict[str, NDArray | EllipsisType], expected_msg: str):
    with pytest.raises(InvalidCombinationOfArguments, match=expected_msg):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    (((np.array([[0, 1],
                 [1, 0]]),
       np.array([0,
                 1]))), {}),
    (((np.zeros((4, 4)), np.zeros(4)), {})),
    (((np.array([[0.01, 4.54],
                 [2.11, 0.90],
                 [3.21, 0.03]])),
       np.array([ 1,
                  0,
                 -1])), {}),
    (([1, 2], 3), {}),
    (([[1, 2],
       [1, 0],
       [0, 1]],
      [-3,
        1,
        1]), {}),
])
def test_polytope_init_hrepr_valid(args: tuple[NDArray, NDArray], kwargs: dict[str, NDArray]):
    _ = pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    (..., ...)
])
@pytest.mark.skip(reason="'init_hrepr' is currently not raising any TypeErrors.")
def test_polytope_init_hrepr_type_error(args, kwargs):
    with pytest.raises(TypeError):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((np.ones((4, 3)), np.zeros(3)), {})
])
def test_polytope_init_hrepr_value_error(args: tuple[NDArray, NDArray], kwargs: dict[str, NDArray]):
    with pytest.raises(ValueError, match=re.escape(
        f"A must be a matrix of size (m, n) and b must be a vector of size (m,), but received A={args[0].shape}, b={args[1].shape}.")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    ((np.ones((4, 3)), np.zeros(4)), {'A_eq': np.empty((0, 4)), 'b_eq': np.empty((0,))})
])
def test_polytope_init_hrepr_value_error_eq(args: tuple[NDArray, NDArray], kwargs: dict[str, NDArray]):
    with pytest.raises(ValueError, match=re.escape(
        f"A_eq must be a matrix of shape (m_eq, n={args[0].shape[1]}) and b_eq must be a vector of size (m_eq,), " \
            f"but received shape A_eq={kwargs['A_eq'].shape}, b_eq={kwargs['b_eq'].shape}.")):
        pes.Polytope(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, expected_msg', [
    ((..., ...), {'verts': ...},
     "Cannot provide 'verts' when initializing from half-spaces"),
])
def test_polytope_init_hrepr_invalid_combination(args: tuple[EllipsisType, EllipsisType], kwargs: dict[str, EllipsisType], expected_msg: str):
    with pytest.raises(InvalidCombinationOfArguments, match=expected_msg):
        pes.Polytope(*args, **kwargs)


class TestInitAmbient:
    """Tests for the method `Polytope._init_ambient`"""

    @pytest.mark.parametrize('n', [
        1,
        2,
        3,
        5,
        10,
        20_000,
    ])
    def parametrize_valid(self, n: int):
        _ = pes.poly_ambient(n)

    @pytest.mark.parametrize('n', [
        -1,
        0,
    ])
    def test_parametrize_value_error(self, n: int):
        with pytest.raises(ValueError, match=re.escape(
            f"Dimension 'n' must be a positive integer, got n={n}")):
            pes.poly_ambient(n)

    @pytest.mark.parametrize('n', [
        0.5,
        np.inf,
        np.nan,
        '3',
        ...,
    ])
    def test_parametrize_type_error(self, n: int):
        with pytest.raises(TypeError, match=re.escape(
            f"Dimension 'n' must be a positive integer, received {n} of type '{type(n).__name__}'")):
            pes.poly_ambient(n)

    @pytest.mark.skip(reason="The operator == is not yet implemented for the Polytope class")
    @given(n=integers(min_value=1, max_value=N_MAX))
    def test_random_manual_gens_equivalent(self, n: int):
        verts, rays = np.empty((0, n)), np.vstack((np.eye(n), -np.eye(n)))
        poly_gens = pes.poly(verts, rays=rays)
        poly_ambient = pes.poly_ambient(n)
        assert poly_gens == poly_ambient, \
            f"Expected the polytope initialized with the generators of the ambient polytope to be equal to the ambient polytope, but got {poly_gens} and {poly_ambient}."

    def test_manual_from_facets(self):
        ...

    @given(n=integers(min_value=1, max_value=N_MAX))
    @pytest.mark.parametrize('attr_name, attr_val', [
        ('_is_empty', False),
        ('_is_degen', True),
        ('_is_bounded', False),
        ('_is_full_dim', True),
        ('_is_pointed', False),
        ('_is_singleton', False),
        ('_dim', ...),
        ('_vol', np.inf),
        ('_chebcr', ...),
    ])
    def test_random_attribute_values(self, n: int, attr_name: str, attr_val: bool | float | int):
        poly = pes.poly_ambient(n)
        if attr_name == '_dim':
            attr_val = n
        elif attr_name == '_chebcr':
            attr_val = (np.full(n + 1, np.nan), np.inf)
        if attr_name == '_chebcr':
            assert (
                np.array_equal(getattr(poly, attr_name)[0], attr_val[0], equal_nan=True) and
                getattr(poly, attr_name)[1] == attr_val[1]
                ), \
                f"Expected attribute '{attr_name}' to be {attr_val} for the ambient polytope in dimension {n}, but got {getattr(poly, attr_name)}."
        else:
            assert getattr(poly, attr_name) == attr_val, \
                f"Expected attribute '{attr_name}' to be {attr_val} for the ambient polytope in dimension {n}, but got {getattr(poly, attr_name)}."
        
    @pytest.mark.skip(reason="These properties are not implemented yet")
    @given(n=integers(min_value=1, max_value=N_MAX))
    @pytest.mark.parametrize('attr_name, attr_val', [
        ('is_empty', True),
        ('is_degen', True),
        ('is_bounded', False),
        ('is_full_dim', True),
        ('is_pointed', True),
        ('is_singleton', True),
        ('dim', ...),
        ('vol', np.inf),
        ('chebc', ...),
        ('chebr', ...),
        ('n', ...),
    ])
    def test_random_property(self):
        ...

    @given(n=integers(min_value=1, max_value=N_MAX))
    def test_random_gens(self, n: int):
        poly = pes.poly_ambient(n)
        assert poly.verts.shape == (0, n), \
            f"Expected the ambient polytope in dimension {n} to have 0 vertices in ambient dimension {n}, but got {poly.verts.shape}."
        assert np.array_equal(poly.rays, np.vstack((np.eye(n), -np.ones(n)))), \
            f"Expected the ambient polytope in dimension {n} to have rays equal to the standard basis vectors and [-1, -1, ..., -1], but got rays with shape {poly.rays.shape} and values\n{poly.rays}."

    @given(n=integers(min_value=1, max_value=N_MAX))
    def test_random_facets(self, n: int):
        poly = pes.poly_ambient(n)
        assert poly.Ab.shape == (0, n + 1), \
            f"Expected the ambient polytope in dimension {n} to have 0 inequalities in ambient dimension {n}, but got {poly.Ab.shape}."
        assert poly.Ab_eq.shape == (0, n + 1), \
            f"Expected the ambient polytope in dimension {n} to have 0 equalities in ambient dimension {n}, but got {poly.Ab_eq.shape}."
