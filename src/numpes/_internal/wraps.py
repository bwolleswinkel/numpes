"""Contains a modified version of `functools.wraps` that also correctly resolves signatures and 
docstrings of decorated functions, by adapting the `tightwraps` package by Tinche, which is licensed 
under the MIT License (© Tinche, 2024)

We have vendored the 'src/tightwraps/__init__.py' file in agreement with the announcement message on tag v24.1.0"""

import sys
from functools import partial
from functools import wraps as functools_wraps
from inspect import Signature, _empty
from types import GetSetDescriptorType, ModuleType
from typing import Any, Callable, Dict, ParamSpec, TypeVar, Tuple, cast

Annotations = Dict[str, Any]
Globals = Dict[str, Any]
Locals = Dict[str, Any]
GetAnnotationsResults = Tuple[Annotations, Globals, Locals]


def eval_if_necessary(source: Any, globals_dict: Globals, locals_dict: Locals) -> Any:
    """Evaluate the annotation if it's a string, otherwise return it as is"""
    if not isinstance(source, str):
        return source
    # FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/20[untested/unverified]
    try:
        return eval(source, globals_dict, locals_dict)
    except NameError:
        return source


def get_annotations(obj: Callable[..., Any]) -> GetAnnotationsResults:
    """Get the evaluated annotations of a callable, along with the globals and locals used for evaluation"""
    obj_globals: Any
    obj_locals: Any
    unwrap: Any

    if isinstance(obj, type):
        obj_dict = getattr(obj, "__dict__", None)
        if obj_dict and hasattr(obj_dict, "get"):
            ann = obj_dict.get("__annotations__", None)
            if isinstance(ann, GetSetDescriptorType):
                ann = None
        else:
            ann = None

        obj_globals = None
        module_name = getattr(obj, "__module__", None)

        if module_name:
            module = sys.modules.get(module_name, None)

            if module:
                obj_globals = getattr(module, "__dict__", None)

        obj_locals = dict(vars(obj))
        unwrap = obj

    elif isinstance(obj, ModuleType):
        ann = getattr(obj, "__annotations__", None)
        obj_globals = getattr(obj, "__dict__", None)
        obj_locals = None
        unwrap = None

    elif callable(obj):
        ann = getattr(obj, "__annotations__", None)
        obj_globals = getattr(obj, "__globals__", None)
        obj_locals = None
        unwrap = obj

    else:
        raise TypeError(f"{obj!r} is not a module, class, or callable.")

    if ann is None:
        return cast(GetAnnotationsResults, ({}, obj_globals, obj_locals))

    if not isinstance(ann, dict):
        raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

    if not ann:
        return cast(GetAnnotationsResults, ({}, obj_globals, obj_locals))

    if unwrap is not None:
        while True:
            if hasattr(unwrap, "__wrapped__"):
                unwrap = unwrap.__wrapped__
                continue
            if isinstance(unwrap, partial):
                unwrap = unwrap.func
                continue
            break
        if hasattr(unwrap, "__globals__"):
            obj_globals = unwrap.__globals__

    return_value = {
        key: eval_if_necessary(value, obj_globals, obj_locals)
        for key, value in cast(Dict[str, Any], ann).items()
    }

    return cast(GetAnnotationsResults, (return_value, obj_globals, obj_locals))


P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


def _get_resolved_signature(fn: Callable[..., Any]) -> Signature:
    signature = Signature.from_callable(fn)
    evaluated_annotations, fn_globals, fn_locals = get_annotations(fn)
    parameters = [
        parameter.replace(annotation=evaluated_annotations.get(name, _empty))
        for name, parameter in signature.parameters.items()
    ]
    new_return_annotation = eval_if_necessary(
        signature.return_annotation, fn_globals, fn_locals
    )
    signature = signature.replace(parameters=parameters, return_annotation=new_return_annotation)
    return signature


def wraps(wrapped: Callable[P, Any]) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """Apply `functools.wraps`"""

    def wrapper(fn: Callable[..., R]) -> Callable[P, R]:
        wrapper_return = _get_resolved_signature(fn).return_annotation
        orig_qualname = fn.__qualname__
        res = functools_wraps(wrapped)(fn)
        res.__qualname__ = orig_qualname

        orig_sig = _get_resolved_signature(wrapped)

        if orig_sig.return_annotation != wrapper_return:
            new_sig = orig_sig.replace(return_annotation=wrapper_return)
            cast(Any, res).__signature__ = new_sig

        return cast(Callable[P, R], res)

    return wrapper
