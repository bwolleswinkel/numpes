"""Module containing the global configurations for the package, such as global constants and settings"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

try:
    import cvxpy as cvx
    CVXPY_INSTALLED: bool = True
except ImportError as _:
    CVXPY_INSTALLED = False

from numpes._internal.wraps import wraps
from numpes.exceptions import ConversionError

if TYPE_CHECKING:
    from typing import Any, Final, Generator, Literal, Optional


@dataclass
class ConfigSchema:
    """Schema for all attributes of the config file"""
    rtol: float = 1E-5
    atol: float = 1E-8

    on_poly_convert: Literal['pass', 'warning', 'error'] = 'pass'
    on_property_assign: Literal['pass', 'minimal'] = 'minimal'
    on_hash_degen: Literal['error', 'unsafe'] = 'error'

    lp_backend: Literal['auto', 'scipy', 'cvxpy', 'pulp'] = 'auto'  # NOTE: 'auto' will try CVXPY is installed, otherwise will fall back to SciPy
    sdp_backend: Literal['cvxpy'] = 'cvxpy'
    scipy_method: Literal['highs', 'highs-ds', 'highs-ipm'] | str = 'highs'
    cvxpy_solver: Any | None = cvx.HIGHS if CVXPY_INSTALLED else None
    optimize_success: Literal['optimal', 'optimal_inaccurate'] = 'optimal'  # NOTE: 'optimal_inaccurate' treats solutions that are optimal, or optimal but inaccurate, as successful

    verbose: int = 0
    plot_aspect: Literal['auto', 'equal'] = 'auto'
    sym_char: str = '*'
    format_spec_poly: str | None = None
    format_spec_ellps: str | None = None
    format_spec_subs: str | None = None


@final
class GlobalConfig:
    """Class representing the global configuration for the package, containing global constants and settings"""

    def __init__(self) -> None:
        self.__dict__['_data'] = ConfigSchema()
        self.__dict__['_locked'] = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)

    def __setattr__(self, name: str, value: float | bool | str) -> None:
        if name == '_locked':
            raise AttributeError("This attribute should not be changed externally")
        if getattr(self, '_locked', True):
            raise AttributeError("Use set_algo_options() or set_display_options() to modify global settings")
        super().__setattr__(name, value)

    def _update(self, **kwargs) -> None:
        """Internal bridge to update the locked dict"""
        for key in kwargs:
            if not hasattr(self._data, key):
                raise KeyError(f"Unknown config key '{key}'")
        self.__dict__['_locked'] = False
        try:
            for key, value in kwargs.items():
                setattr(self._data, key, value)
        finally:
            self.__dict__['_locked'] = True

    def _reset(self) -> None:
        """Reset all settings to their default values"""
        self._update(**ConfigSchema().__dict__)

    def on_poly_convert_(self) -> Literal[True]:
        """Method that is called when a polytope is converted from one representation to another.
        
        Returns
        -------
        Literal[True]
            If the conversion is allowed to proceed
            
        Raises
        ------
        UserWarning
            If the conversion is allowed but a warning should be issued
        ConversionError
            If the conversion is not allowed and an error should be raised
        ValueError
            If the config value is invalid
        """
        match self._data.on_poly_convert:
            case 'pass':
                return True
            case 'warning':
                # TODO: Implement stack trace
                warnings.warn("Conversion between polytope representations", UserWarning, stacklevel=3)
                return True
            case 'error':
                # TODO: Implement stack trace
                raise ConversionError("The value of 'CFG.on_poly_convert' is set to 'error', so conversion between polytope representations is not allowed")
            case _:
                raise ValueError(f"Unknown value '{self._data.on_poly_convert}' for 'on_poly_convert' config setting")


# NOTE: This is a global instance that needs to be initialized once here
CFG: Final[GlobalConfig] = GlobalConfig()


# pylint: disable=unused-argument
def set_algo_options(*,
                     rtol: Optional[float] = None,
                     atol: Optional[float] = None,
                     on_poly_convert: Optional[Literal['pass', 'warning', 'error']] = None,
                     on_property_assign: Optional[Literal['pass', 'minimal']] = None,
                     on_hash_degen: Optional[Literal['error', 'unsafe']] = None,
                     lp_backend: Optional[Literal['auto', 'scipy', 'cvxpy', 'pulp']] = None,
                     sdp_backend: Optional[Literal['cvxpy']] = None,
                     scipy_method: Optional[Literal['highs', 'highs-ds', 'highs-ipm'] | str] = None,
                     cvxpy_solver: Optional[Any] = None,
                     optimize_success: Optional[Literal['optimal', 'optimal_inaccurate']] = None,
                     **extra_kwargs: dict[str, Any],
                     ) -> None:
    """Set the global configuration for the package that is used internally by the algorithms. Provide all settings as keyword arguments.

    Parameters
    ----------
    rtol : float, default=1E-5
        Relative tolerance for numerical comparisons
    atol : float, default=1E-8
        Absolute tolerance for numerical comparisons
    on_poly_convert : {'pass', 'warning', 'error'}, default='pass'
        Behavior when converting polytope representations
    on_property_assign: {'pass', 'minimal'}, default='pass'
        Behavior when assigning properties
    on_hash_degen: {'error', 'unsafe'}, default='error'
        Behavior when hashing degenerate objects
    lp_backend : {'auto', 'scipy', 'cvxpy', 'pulp'}, default='auto'
        Linear programming backend
    sdp_backend: {'cvxpy'}, default='cvxpy'
        Semi-definite programming backend
    scipy_method : {'highs', 'highs-ds', 'highs-ipm'} or str, default='highs'
        SciPy optimization method
    cvxpy_solver: Any, default=None
        The solver to use for CVXPY
    optimize_success : {'optimal', 'optimal_inaccurate'}, default='optimal'
        What constitutes a successful optimization
    extra_kwargs: {}
        Additional keyword arguments (should be empty) used to catch invalid config keys and printing options
    
    Raises
    ------
    KeyError
        If an unknown configuration key or printing option is provided

    Examples
    --------
    >>> pes.set_algo_options(atol=0.01)
    ... # The absolute tolerance is set to 0.01 globally
    ... print(pes.get_config('atol'))
    0.01

    Multiple settings can be updated at once.

    >>> pes.set_algo_options(on_property_assign='minimal', lp_backend='pulp')
    ... # From now on, assignment of properties will call 'self.minimal()' and linear programs will be solved using PuLP
    """
    values = locals()
    extra_kwargs = values.pop('extra_kwargs')
    _invalid_algo_keys(extra_kwargs, prefix='set_')
    kwargs = {k: v for k, v in values.items() if v is not None}
    CFG._update(**kwargs)  # pylint: disable=protected-access


@contextmanager
def algo_options(*,
                 rtol: Optional[float] = None,
                 atol: Optional[float] = None,
                 on_poly_convert: Optional[Literal['pass', 'warning', 'error']] = None,
                 on_property_assign: Optional[Literal['pass', 'minimal']] = None,
                 on_hash_degen: Optional[Literal['error', 'unsafe']] = None,
                 lp_backend: Optional[Literal['auto', 'scipy', 'cvxpy', 'pulp']] = None,
                 sdp_backend: Optional[Literal['cvxpy']] = None,
                 scipy_method: Optional[Literal['highs', 'highs-ds', 'highs-ipm'] | str] = None,
                 cvxpy_solver: Optional[Any] = None,
                 optimize_success: Optional[Literal['optimal', 'optimal_inaccurate']] = None,
                 **extra_kwargs: dict[str, Any],
                 ) -> Generator[None, None, None]:
    """Context-manager version of `set_algo_options`; settings are reset to their original values after the context is exited. See `set_algo_options` for parameter descriptions.

    Raises
    ------
    KeyError
        If an unknown configuration key or printing option is provided

    Examples
    --------
    >>> with pes.algo_options(atol=0.01):
    ...     # Within this context, the absolute tolerance is set to 0.01
    ...     print(pes.get_config('atol'))
    0.01
    ... # After this context, the absolute tolerance is reset to its original value of 1E-8
    ... print(pes.get_config('atol'))
    1E-08

    Multiple settings can be updated at once.

    >>> with pes.algo_options(on_property_assign='minimal', lp_backend='pulp'):
    ...     # Within this context, assignment of properties will call 'self.minimal()' and linear programs will be solved using PuLP
    """
    values = locals()
    extra_kwargs = values.pop('extra_kwargs')
    _invalid_algo_keys(extra_kwargs, prefix='')
    kwargs = {k: v for k, v in values.items() if v is not None}

    old_values = {k: getattr(CFG._data, k) for k in kwargs}  # pylint: disable=protected-access
    CFG._update(**kwargs)  # pylint: disable=protected-access
    try:
        yield
    finally:
        CFG._update(**old_values)  # pylint: disable=protected-access


def set_display_options(*,
                        verbose: Optional[int] = None,
                        plot_aspect: Optional[Literal['auto', 'equal']] = None,
                        sym_char: Optional[str] = None,
                        format_spec_poly: Optional[str | None] = None,
                        format_spec_ellps: Optional[str | None] = None,
                        format_spec_subs: Optional[str | None] = None,
                        **extra_kwargs: dict[str, Any],
                        ) -> None:
    """Set the global display configuration for the package. Provide all settings as keyword arguments.

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level for console output
    plot_aspect : {'auto', 'equal'}, default='auto'
        Aspect ratio used when plotting
    sym_char : str, default='*'
        Character used to replace the (upper-triangular) symmetric part of a matrix
    format_spec_poly : str or None, default=None
        Format spec used when printing polytopes
    format_spec_ellps : str or None, default=None
        Format spec used when printing ellipsoids
    format_spec_subs : str or None, default=None
        Format spec used when printing subspaces
    extra_kwargs: {}
        Additional keyword arguments (should be empty) used to catch invalid config keys and algo options

    Raises
    ------
    KeyError
        If an unknown configuration key or algo option is provided

    Examples
    --------
    >>> pes.set_display_options(sym_char='#')
    ... # The symbolic character is set to '#' globally
    ... print(pes.get_config('sym_char'))
    #
    """
    values = locals()
    extra_kwargs = values.pop('extra_kwargs')
    _invalid_display_keys(extra_kwargs, prefix='set_')
    kwargs = {k: v for k, v in values.items() if v is not None}
    CFG._update(**kwargs)  # pylint: disable=protected-access


@contextmanager
def display_options(*,
                    verbose: Optional[int] = None,
                    plot_aspect: Optional[Literal['auto', 'equal']] = None,
                    sym_char: Optional[str] = None,
                    format_spec_poly: Optional[str | None] = None,
                    format_spec_ellps: Optional[str | None] = None,
                    format_spec_subs: Optional[str | None] = None,
                    **extra_kwargs: dict[str, Any],
                    ) -> Generator[None, None, None]:
    """Context-manager version of `set_display_options`; settings are reset to their original values after the context is exited. See `set_display_options` for parameter descriptions.

    Raises
    ------
    KeyError
        If an unknown configuration key or algo option is provided

    Examples
    --------
    >>> with pes.display_options(sym_char='#'):
    ...     # Within this context, the symbolic character is set to '#'
    ...     print(pes.get_config('sym_char'))
    #
    ... # After this context, the symbolic character is reset to its original value of '*'
    ... print(pes.get_config('sym_char'))
    *
    """
    values = locals()
    extra_kwargs = values.pop('extra_kwargs')
    _invalid_display_keys(extra_kwargs, prefix='')
    kwargs = {k: v for k, v in values.items() if v is not None}

    old_values = {k: getattr(CFG._data, k) for k in kwargs}  # pylint: disable=protected-access
    CFG._update(**kwargs)  # pylint: disable=protected-access
    try:
        yield
    finally:
        CFG._update(**old_values)  # pylint: disable=protected-access


# FIXME: Do we actually want this? Maybe this should actually return a frozendict...
def get_config(key: Literal['atol',
                            'rtol',
                            'on_poly_convert',
                            'on_property_assign',
                            'on_hash_degen',
                            'lp_backend',
                            'sdp_backend',
                            'scipy_method',
                            'cvxpy_solver',
                            'optimize_success',
                            'verbose',
                            'plot_aspect',
                            'sym_char',
                            'format_spec_poly',
                            'format_spec_ellps',
                            'format_spec_subs',
                            ]) -> int | float | str | None:
    """Get the current configuration of the global config. Valid keys are:
    
    - atol -> `float`
    - rtol -> `float`
    - on_poly_convert -> `str`
    - on_property_assign -> `str`
    - on_hash_degen -> `str`
    - lp_backend -> `str`
    - sdp_backend -> `str`
    - scipy_method -> `str`
    - cvxpy_solver -> `None | cvxpy.Solver`
    - optimize_success -> `str`
    - verbose -> `int`
    - plot_aspect -> `str`
    - sym_char -> `str`
    - format_spec_poly -> `None | str`
    - format_spec_ellps -> `None | str`
    - format_spec_subs -> `None | str`
    """
    return getattr(CFG._data, key)  # pylint: disable=protected-access


@wraps(GlobalConfig._reset)  # pylint: disable=protected-access
def reset_config() -> None:
    """Reset the global config to its default values"""
    CFG._reset()  # pylint: disable=protected-access


def _invalid_algo_keys(extra_kwargs: dict[str, Any], prefix: str = '') -> None:
    """Helper function to check for invalid config keys and display options"""

    display_keys = frozenset([
        'verbose',
        'plot_aspect',
        'sym_char',
        'format_spec_poly',
        'format_spec_ellps',
        'format_spec_subs',
    ])

    for key in extra_kwargs:
        if key in display_keys:
            raise KeyError(f"'{key}' is a display option, use {prefix}display_options() instead")
        if not hasattr(CFG._data, key):  # pylint: disable=protected-access
            raise KeyError(f"Unknown config key '{key}'")


def _invalid_display_keys(extra_kwargs: dict[str, Any], prefix: str = '') -> None:
    """Helper function to check for invalid config keys and algo options"""

    algo_keys = frozenset([
        'rtol',
        'atol',
        'on_poly_convert',
        'on_property_assign',
        'on_hash_degen',
        'lp_backend',
        'sdp_backend',
        'scipy_method',
        'cvxpy_solver',
        'optimize_success',
    ])

    for key in extra_kwargs:
        if key in algo_keys:
            raise KeyError(f"'{key}' is a algo option, use {prefix}algo_options() instead")
        if not hasattr(CFG._data, key):  # pylint: disable=protected-access
            raise KeyError(f"Unknown config key '{key}'")
