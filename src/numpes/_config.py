"""Module containing the global configurations for the package, such as global constants and settings"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import cvxpy as cvx
    CVXPY_INSTALLED: bool = True
except ImportError as _:
    CVXPY_INSTALLED = False

from ._internal.wraps import wraps
from .exceptions import ConversionError

if TYPE_CHECKING:
    from typing import Any, Generator, Literal, Optional, ContextManager


@dataclass
class ConfigSchema:
    """Schema for all attributes of the config file"""
    rtol: float = 1E-5
    atol: float = 1E-8

    on_poly_convert: Literal['pass', 'warning', 'error'] = 'pass'
    on_property_assign: Literal['pass', 'minimal'] = 'pass'  # FIXME: Placeholder, should be 'minimal' once implemented
    on_hash_degen: Literal['error', 'unsafe'] = 'error'

    lp_backend: Literal['auto', 'scipy', 'cvxpy', 'pulp'] = 'auto'  # NOTE: 'auto' will try CVXPY is installed, otherwise will fall back to SciPy
    sdp_backend: Literal['cvxpy'] = 'cvxpy'
    scipy_method: Literal['highs', 'highs-ds', 'highs-ipm'] | str = 'highs'
    cvxpy_solver: Any | None = cvx.CLARABEL if CVXPY_INSTALLED else None
    optimize_success: Literal['optimal', 'optimal_inaccurate'] = 'optimal'  # NOTE: 'optimal_inaccurate' treats solutions that are optimal, or optimal but inaccurate, as successful

    verbose: int = 0
    print_num_verts: int = 4
    print_format_poly: str = '...'
    print_format_ellps: str = '...'
    print_format_subs: str = '...'


class GlobalConfig:
    """Class representing the global configuration for the package, containing global constants and settings"""

    def __init__(self) -> None:
        self.__dict__['_data'] = ConfigSchema()
        self.__dict__['_locked'] = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)

    def __setattr__(self, name: str, value: float | bool | str) -> None:
        if getattr(self, '_locked', True):
            raise AttributeError("Use set_algo_options() or set_print_options() to modify global settings")
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

    @contextmanager
    def _algo_options(self,
                     *,
                     rtol: Optional[float] = None,
                     atol: Optional[float] = None,
                     on_poly_convert: Optional[Literal['pass', 'warning', 'error']] = None,
                     lp_backend: Optional[Literal['auto', 'scipy', 'cvxpy', 'pulp']] = None,
                     scipy_method: Optional[Literal['highs', 'highs-ds', 'highs-ipm'] | str] = None,
                     optimize_success: Optional[Literal['optimal', 'optimal_inaccurate']] = None,
                     on_property_assign: Optional[Literal['pass', 'minimal']] = None,
                     on_hash_degen: Optional[Literal['unsafe', 'error']] = None,
                     sdp_backend: Optional[Literal['cvxpy']] = None,
                     verbose: Optional[int] = None,
                     **extra_kwargs: dict[str, Any],
                     ) -> Generator[None, None, None]:
        """Context manager to temporarily update algorithm settings within a context. 
        The settings will be reset to their original values after the context is exited.
        All settings should be provided as keyword arguments.

        Parameters
        ----------
        rtol : float, default=1E-5
            Relative tolerance for numerical comparisons
        atol : float, default=1E-8
            Absolute tolerance for numerical comparisons
        on_poly_convert : {'pass', 'warning', 'error'}, default='pass'
            Behavior when converting polytope representations
        lp_backend : {'auto', 'scipy', 'cvxpy', 'pulp'}, default='auto'
            Linear programming backend
        scipy_method : {'highs', 'highs-ds', 'highs-ipm'} or str, default='highs'
            SciPy optimization method
        optimize_success : {'optimal', 'optimal_inaccurate'}, default='optimal'
            What constitutes a successful optimization
        on_property_assign: {'pass', 'minimal'}, default='minimal'
            Behavior when assigning properties
        on_hash_degen: {'error', 'unsafe'}, default='error'
            Behavior when hashing degenerate objects
        sdp_backend: {'cvxpy'}, default='cvxpy'
            Semi-definite programming backend
        verbose: int, default=0
            Verbosity level for output
        extra_kwargs: {}
            Additional keyword arguments (should be empty) used to catch invalid config keys and printing options
        
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
        params = {
            'rtol': rtol,
            'atol': atol,
            'on_poly_convert': on_poly_convert,
            'lp_backend': lp_backend,
            'scipy_method': scipy_method,
            'optimize_success': optimize_success,
            'on_property_assign': on_property_assign,
            'on_hash_degen': on_hash_degen,
            'sdp_backend': sdp_backend,
            'verbose': verbose,
        }
        kwargs = {k: v for k, v in params.items() if v is not None}

        _invalid_keys(extra_kwargs)

        old_values = {k: getattr(self._data, k) for k in kwargs.keys()}
        self._update(**kwargs)
        try:
            yield
        finally:
            self._update(**old_values)

    def on_poly_convert(self) -> Literal[True]:
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
CFG: GlobalConfig = GlobalConfig()


# FIXME: Do we actually want this?
def get_config(key: str) -> int | float | str:
    """Get the current configuration of the global config"""
    return getattr(CFG._data, key)  # pylint: disable=protected-access


def set_algo_options(*,
                     rtol: Optional[float] = None,
                     atol: Optional[float] = None,
                     on_poly_convert: Optional[Literal['pass', 'warning', 'error']] = None,
                     lp_backend: Optional[Literal['auto', 'scipy', 'cvxpy', 'pulp']] = None,
                     scipy_method: Optional[Literal['highs', 'highs-ds', 'highs-ipm'] | str] = None,
                     optimize_success: Optional[Literal['optimal', 'optimal_inaccurate']] = None,
                     on_property_assign: Optional[Literal['pass', 'minimal']] = None,
                     on_hash_degen: Optional[Literal['error', 'unsafe']] = None,
                     sdp_backend: Optional[Literal['cvxpy']] = None,
                     verbose: Optional[int] = None,
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
    lp_backend : {'auto', 'scipy', 'cvxpy', 'pulp'}, default='auto'
        Linear programming backend
    scipy_method : {'highs', 'highs-ds', 'highs-ipm'} or str, default='highs'
        SciPy optimization method
    optimize_success : {'optimal', 'optimal_inaccurate'}, default='optimal'
        What constitutes a successful optimization
    on_property_assign: {'pass', 'minimal'}, default='pass'
        Behavior when assigning properties
    on_hash_degen: {'error', 'unsafe'}, default='error'
        Behavior when hashing degenerate objects
    sdp_backend: {'cvxpy'}, default='cvxpy'
        Semi-definite programming backend
    verbose: int, default=0
        Verbosity level for output
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
    params = {
        'rtol': rtol,
        'atol': atol,
        'on_poly_convert': on_poly_convert,
        'lp_backend': lp_backend,
        'scipy_method': scipy_method,
        'optimize_success': optimize_success,
        'on_property_assign': on_property_assign,
        'on_hash_degen': on_hash_degen,
        'sdp_backend': sdp_backend,
        'verbose': verbose,
    }
    kwargs = {k: v for k, v in params.items() if v is not None}
    _invalid_keys(extra_kwargs)
    CFG._update(**kwargs)  # pylint: disable=protected-access


@wraps(GlobalConfig._algo_options)  # pylint: disable=protected-access
def algo_options(*,
                 rtol: Optional[float] = None,
                 atol: Optional[float] = None,
                 on_poly_convert: Optional[Literal['pass', 'warning', 'error']] = None,
                 lp_backend: Optional[Literal['auto', 'scipy', 'cvxpy', 'pulp']] = None,
                 scipy_method: Optional[Literal['highs', 'highs-ds', 'highs-ipm'] | str] = None,
                 optimize_success: Optional[Literal['optimal', 'optimal_inaccurate']] = None,
                 on_property_assign: Optional[Literal['pass', 'minimal']] = None,
                 on_hash_degen: Optional[Literal['error', 'unsafe']] = None,
                 sdp_backend: Optional[Literal['cvxpy']] = None,
                 verbose: Optional[int] = None,
                 **extra_kwargs: dict[str, Any]) -> ContextManager[None]:
    """Wrapper for the contextmanager of the global config"""
    return CFG._algo_options(rtol=rtol,  # pylint: disable=protected-access
                             atol=atol,
                             on_poly_convert=on_poly_convert,
                             lp_backend=lp_backend,
                             scipy_method=scipy_method,
                             optimize_success=optimize_success,
                             on_property_assign=on_property_assign,
                             on_hash_degen=on_hash_degen,
                             sdp_backend=sdp_backend,
                             verbose=verbose,
                             **extra_kwargs,
                             )


def _invalid_keys(extra_kwargs: dict[str, Any]) -> None:
    """Helper function to check for invalid config keys and printing options"""
    print_options = {
        'print_num_verts',
        'print_format_poly',
        'print_format_ellps',
        'print_format_subs'
    }
    for key in extra_kwargs:
        if key in print_options:
            raise KeyError(f"'{key}' is a printing option, use set_print_options() instead")
        if not hasattr(CFG._data, key):  # pylint: disable=protected-access
            raise KeyError(f"Unknown config key '{key}'")


@wraps(GlobalConfig._reset)  # pylint: disable=protected-access
def reset_config() -> None:
    """Reset the global config to its default values"""
    CFG._reset()  # pylint: disable=protected-access
