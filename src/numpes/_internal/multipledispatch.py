"""Module for multiple dispatch functionality used in the Polytope class"""

import re
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union


_LEN_SPEC_PATTERN = re.compile(r'^(<=|>=|!=|<|>)(\d+)$')


# FROM: GitHub Copilot, Claude Sonnet 4 | 2026/01/15[untested/unverified]
def _check_len_spec(spec: "Union[int, str]", n: int, param_name: str = 'len') -> bool:
    """Check if n satisfies a length specification.

    Parameters
    ----------
    spec : int or str
        If int, checks exact equality. If str, must match one of the patterns
        '<X', '<=X', '>X', '>=X', '!=X' where X is a non-negative integer.
    n : int
        The length value to check.
    param_name : str
        Name of the parameter, used in error messages.

    Returns
    -------
    bool
        Whether n satisfies the specification.

    """
    if isinstance(spec, int):
        return n == spec
    match = _LEN_SPEC_PATTERN.match(spec.strip())
    if match is None:
        raise ValueError(
            f"Invalid {param_name} pattern: {spec!r}. "
            "Expected '<X', '<=X', '>X', '>=X', or '!=X' where X is a non-negative integer."
        )
    op, x = match.group(1), int(match.group(2))
    if op == '<':
        return n < x
    if op == '<=':
        return n <= x
    if op == '>':
        return n > x
    if op == '>=':
        return n >= x
    return n != x  # op == '!='


class DispatcherFunction(Protocol):
    """Protocol describing a function with dispatch capabilities"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def register(self,
                 len_args: Optional[Union[int, str]] = None,
                 exclude_kwargs: Optional[list] = None,
                 include_kwargs: Optional[list] = None,
                 kwargs_values: Optional[dict] = None,
                 len_kwargs: Optional[Union[int, str]] = None,
                 ) -> Callable[[Callable], Callable]:
        """Method to register a new dispatch function with specific conditions"""


# FROM: GitHub Copilot, Claude Sonnet 4 | 2026/01/15[untested/unverified]
def multipledispatch(func: Callable[[Any], Any]) -> DispatcherFunction:
    """Decorator that enables multiple dispatch functionality for a function or method.
    
    The decorated function `func` can then use dispatch decorators, such as 
    `@func.args(len=n)` to dispatch based on the number of positional arguments.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function or method to be decorated for multiple dispatching.

    Returns
    -------
    DispatcherFunction
        A wrapper function that handles multiple dispatching based on argument length.

    Examples
    --------
    >>> @multipledispatch
    ... def example_func(*args):
    ...     return "default"
    ...
    ... @example_func.args(len=1)
    ... def _(arg1):
    ...     return f"one arg: {arg1}"
    ...
    ... @example_func.args(len=2)
    ... def _(arg1, arg2):
    ...     return f"two args: {arg1}, {arg2}"
    """
    # Create a dictionary to store dispatchers
    dispatchers: dict[int | tuple[int, tuple[str, ...], tuple[str, ...], tuple[tuple[str, Any], ...]], Callable] = {}

    def dispatcher_wrapper(*args, **kwargs):
        # For methods, the first argument is self
        if args:
            # Get the actual arguments (excluding self for methods)
            if hasattr(args[0], func.__name__):
                # This is a method call
                instance = args[0]
                actual_args = args[1:]
            else:
                # This is a function call
                instance = None
                actual_args = args
        else:
            instance = None
            actual_args = args

        # Check all registered dispatchers for a match
        for key, dispatcher_func in dispatchers.items():
            if _match_dispatch_condition(key, actual_args, kwargs):
                if instance is not None:
                    return dispatcher_func(instance, *actual_args, **kwargs)
                return dispatcher_func(*actual_args, **kwargs)

        # Fallback to original function
        return func(*args, **kwargs)

    def _match_dispatch_condition(condition, args, kwargs):
        """Check if the current call matches a dispatch condition"""
        if isinstance(condition, int):
            # Simple length-based dispatch (exact int, no other criteria)
            return len(args) == condition
        if isinstance(condition, tuple):
            # Complex condition with multiple criteria
            arg_len_spec, exclude_kwargs, include_kwargs, kwargs_values, len_kwargs_spec = condition

            # Check positional-argument length condition
            if arg_len_spec is not None and not _check_len_spec(arg_len_spec, len(args), 'len_args'):
                return False

            # Check kwargs exclusions
            if exclude_kwargs and any(key in kwargs for key in exclude_kwargs):
                return False

            # Check kwargs inclusions
            if include_kwargs and any(key not in kwargs for key in include_kwargs):
                return False

            # Check kwargs values
            if kwargs_values:
                for key, expected_value in kwargs_values.items():
                    if kwargs.get(key) != expected_value:
                        return False

            # Check number of keyword arguments
            if len_kwargs_spec is not None and not _check_len_spec(len_kwargs_spec, len(kwargs), 'len_kwargs'):
                return False

            return True

        return False

    def register(len_args: Optional["Union[int, str]"] = None,
                 exclude_kwargs: Optional[list] = None,
                 include_kwargs: Optional[list] = None,
                 kwargs_values: Optional[dict] = None,
                 len_kwargs: Optional["Union[int, str]"] = None,
                 ) -> Callable[[Callable], Callable]:
        """Create a dispatcher that matches based on various conditions

        Parameters
        ----------
        len_args : int or str, optional
            Condition on the number of positional arguments. An int checks for
            exact equality. A string must match one of the patterns '<X',
            '<=X', '>X', '>=X', '!=X' where X is a non-negative integer.
        exclude_kwargs : list, optional
            List of kwarg keys that must NOT be present
        include_kwargs : list, optional
            List of kwarg keys that must be present
        kwargs_values : dict, optional
            Dictionary of kwarg key-value pairs that must match exactly
        len_kwargs : int or str, optional
            Condition on the number of keyword arguments. An int checks for
            exact equality. A string must match one of the patterns '<X',
            '<=X', '>X', '>=X', '!=X' where X is a non-negative integer.
        """
        def decorator(dispatch_func):
            if (isinstance(len_args, int)
                    and exclude_kwargs is None
                    and include_kwargs is None
                    and kwargs_values is None
                    and len_kwargs is None):
                # Simple case - exact integer arg count, no other criteria
                condition = len_args
            else:
                # Complex case - build condition as a hashable tuple
                condition = (
                    len_args,
                    tuple(exclude_kwargs) if exclude_kwargs else (),
                    tuple(include_kwargs) if include_kwargs else (),
                    tuple(sorted(kwargs_values.items())) if kwargs_values else (),
                    len_kwargs,
                )

            dispatchers[condition] = dispatch_func
            return dispatch_func
        return decorator

    # Attach the args method to the wrapper
    dispatcher_wrapper.register = register  # type: ignore[attr-defined]

    # Copy function attributes
    dispatcher_wrapper.__name__ = func.__name__
    dispatcher_wrapper.__qualname__ = func.__qualname__
    dispatcher_wrapper.__doc__ = func.__doc__
    dispatcher_wrapper.__module__ = func.__module__

    return cast(DispatcherFunction, dispatcher_wrapper)
