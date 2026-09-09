"""Module for snippets of code that are common across several modules to avoid repeated code"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

from numpes.utils.axes import Axes1D
from numpes.utils.plot import add_1d_subplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType


@overload
def get_axes_color(ax: Axes1D | Axes | Axes3D | None,  # type: ignore[overload-overlap]
                   color: ColorType | None,
                   n: Literal[1],
                   display_name: str = "object",
                   ) -> tuple[Axes1D, ColorType]: ...


@overload
def get_axes_color(ax: Axes1D | Axes | Axes3D | None,
                   color: ColorType | None,
                   n: Literal[2],
                   display_name: str = "object",
                   ) -> tuple[Axes, ColorType]: ...


@overload
def get_axes_color(ax: Axes1D | Axes | Axes3D | None,
                   color: ColorType | None,
                   n: Literal[3],
                   display_name: str = "object",
                   ) -> tuple[Axes3D, ColorType]: ...


@overload
def get_axes_color(ax: Axes1D | Axes | Axes3D | None,
                   color: ColorType | None,
                   n: int,
                   display_name: str = "object",
                   ) -> tuple[Axes1D | Axes | Axes3D, ColorType]: ...


def get_axes_color(ax: Axes1D | Axes | Axes3D | None,
                   color: ColorType | None,
                   n: int,
                   display_name: str = "object",
                   ) -> tuple[Axes1D | Axes | Axes3D, ColorType]:
    """Get the correct axes object and color based on provided arguments.
    
    Parameters
    ----------
    ax : Axes or None
        Matplotlib Axes object if provided or else None. If None, a new axes is created of appropriated dimension.
    color : ColorType or None
        Color to be used in plotting or else None. If None, the next-in-line color (as determined by Matplotlib) is chosen.
    n : int
        Dimension of the object/convex region
    display_name : str, default="object"
        Text to be plotted in the error message when `n` ∉ {1, 2, 3}, displayed as "Plotting is only supported for an n-d {`display_name`} with n <= 3, received n = {`n`}"

    Returns
    -------
    ax : Axes
        Matplotlib Axes object
    color : ColorType
        Color to be used in plotting
    
    Raises
    ------
    ImportError
        When Matplotlib is not installed
    ValueError
        When `n` ∉ {1, 2, 3} or when the provided `ax` object does not match `n`
    """

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
                raise ValueError(f"Plotting is only supported for an n-d {display_name} with n <= 3, received n = {n}")
    else:
        if isinstance(ax, Axes1D) and n != 1:
            raise ValueError(f"The provided axes 'ax' is 1d, but the {display_name} is {n}d")
        if (not isinstance(ax, Axes1D) and not isinstance(ax, Axes3D)) and n != 2:
            raise ValueError(f"The provided axes 'ax' is 2d, but the {display_name} is {n}d")
        if isinstance(ax, Axes3D) and n != 3:
            raise ValueError(f"The provided axes 'ax' is 3d, but the {display_name} is {n}d")
    if color is None:
        # pylint: disable=protected-access
        color = ax._get_lines.get_next_color()  # type: ignore[union-attr, attr-defined]

    return ax, color
