"""Module for snippets of code that are common across several modules to avoid repeated code"""

from __future__ import annotations
from typing import TYPE_CHECKING

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

from numpes.utils import add_1d_subplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType


def get_axes_color(ax: Axes,
                   color: ColorType | None,
                   n: int,
                   text_err: str = "object",
                   ) -> tuple[Axes, ColorType]:

    if not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib is required for plotting. " \
                          "Please install it with 'pip install matplotlib' and try again.")

    if ax is None:
        match n:
            case 1:
                ax = add_1d_subplot(plt.figure())
            case 2:
                _, ax = plt.subplots()
            case 3:
                ax = plt.figure().add_subplot(111, projection='3d')
            case _:
                raise ValueError(f"Plotting is only supported for an n-d {text_err} with n <= 3, received n = {n}")
    
    if color is None:
        # pylint: disable=protected-access
        color = ax._get_lines.get_next_color()  # type: ignore[union-attr, attr-defined]

    return ax, color
