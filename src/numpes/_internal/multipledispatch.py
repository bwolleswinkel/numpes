"""Module for multiple dispatch functionality used in the Polytope class"""

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from typing import Any, Callable, Optional


class DispatcherFunction(Protocol):
    """Protocol describing a function with dispatch capabilities"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def register(self,
                 len_args: Optional[int] = None,
                 exclude_kwargs: Optional[list] = None,
                 include_kwargs: Optional[list] = None,
                 kwargs_values: Optional[dict] = None,
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
            # Simple length-based dispatch
            return len(args) == condition
        if isinstance(condition, tuple):
            # Complex condition with multiple criteria
            arg_len, exclude_kwargs, include_kwargs, kwargs_values = condition

            # Check length condition
            if arg_len != -1 and len(args) != arg_len:
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

            return True

        return False

    def register(len_args: Optional[int] = None,
                 exclude_kwargs: Optional[list] = None,
                 include_kwargs: Optional[list] = None,
                 kwargs_values: Optional[dict] = None,
                 ) -> Callable[[Callable], Callable]:
        """Create a dispatcher that matches based on various conditions
        
        Parameters
        ----------
        len_args : int, optional
            Required number of positional arguments
        exclude_kwargs : list, optional
            List of kwarg keys that must NOT be present
        include_kwargs : list, optional
            List of kwarg keys that must be present
        kwargs_values : dict, optional
            Dictionary of kwarg key-value pairs that must match exactly
        """
        def decorator(dispatch_func):
            if len_args is not None and exclude_kwargs is None and include_kwargs is None and kwargs_values is None:
                # Simple case - just length-based dispatch
                condition = len_args
            else:
                # Complex case - build condition as a hashable tuple
                condition = (
                    len_args if len_args is not None else -1,
                    tuple(exclude_kwargs) if exclude_kwargs else (),
                    tuple(include_kwargs) if include_kwargs else (),
                    tuple(sorted(kwargs_values.items())) if kwargs_values else ()
                )

            dispatchers[condition] = dispatch_func
            return dispatch_func
        return decorator

    # Attach the args method to the wrapper
    dispatcher_wrapper.register = register  # type: ignore[attr-defined]

    # Copy function attributes
    dispatcher_wrapper.__name__ = func.__name__
    dispatcher_wrapper.__doc__ = func.__doc__
    dispatcher_wrapper.__module__ = func.__module__

    return cast(DispatcherFunction, dispatcher_wrapper)
