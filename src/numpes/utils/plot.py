"""Functionality for overriding callback functionality on Matplotlib axes objects"""

from typing import TYPE_CHECKING

import numpy as np

try:
    from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection  # type: ignore[import-untyped]
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    MATPLOTLIB_INSTALLED: bool = True
except ImportError as _:
    MATPLOTLIB_INSTALLED = False

if TYPE_CHECKING:
    from typing import Optional

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike


# [untested/unverified]
# FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/12[unverified/untested]
# FIXME: We need to add the 'bidirectional' functionality
def plot_line(ax: Axes, line: ArrayLike, point: Optional[ArrayLike] = None, bidirectional: bool = True, **kwargs) -> None:
    """Plot an indefinite line in 2D or 3D, defined by a point and a direction vector"""

    def update_line():
        # Get axis limits
        limits = [ax.get_xlim(), ax.get_ylim()] + ([ax.get_zlim()] if ndim == 3 else [])
        
        # Calculate t-values for intersections
        t_values = []
        coords = []
        
        for i, (lim, d, c) in enumerate(zip(limits, line, center)):
            if abs(d) > 1e-10:
                t_left, t_right = (lim[0] - c) / d, (lim[1] - c) / d
                t_values.extend([t_left, t_right])
            else:
                # Line parallel to this axis - check if within bounds
                if c < lim[0] or c > lim[1]:
                    # Hide line if outside bounds in any parallel direction
                    coords = [[] for _ in range(ndim)]
                    break
        
        if coords == [] and t_values:  # Line is visible
            if ndim == 2:
                # 2D case with extension
                t_min, t_max = min(t_values), max(t_values)
                t_range = t_max - t_min
                t_min -= t_range * 0.1
                t_max += t_range * 0.1
                coords = [center + t * line for t in [t_min, t_max]]
                line_obj.set_xdata([c[0] for c in coords])
                line_obj.set_ydata([c[1] for c in coords])
            else:
                # 3D case with clipping
                dir_checks = [(limits[i], line[i], center[i]) for i in range(3)]
                t_bounds = []
                
                for lim, d, c in dir_checks:
                    if abs(d) > 1e-10:
                        t1, t2 = (lim[0] - c) / d, (lim[1] - c) / d
                        t_bounds.append([min(t1, t2), max(t1, t2)])
                    else:
                        t_bounds.append([-1e10, 1e10])
                
                t_enter = max(tb[0] for tb in t_bounds)
                t_exit = min(tb[1] for tb in t_bounds)
                
                if t_enter <= t_exit:
                    coords = [center + t * line for t in [t_enter, t_exit]]
                    line_obj.set_data_3d([c[0] for c in coords], [c[1] for c in coords], [c[2] for c in coords])
                else:
                    line_obj.set_data_3d([], [], [])
        else:
            # Handle parallel case or empty coords
            if ndim == 2 and not coords:
                # 2D parallel line case
                range_max = max(abs(lim[1] - lim[0]) for lim in limits)
                coords = [center + t * line for t in [-range_max, range_max]]
                line_obj.set_xdata([c[0] for c in coords])
                line_obj.set_ydata([c[1] for c in coords])
            elif ndim == 3:
                line_obj.set_data_3d([], [], [])
    
    line = np.array(line, dtype=float)
    if np.allclose(line, 0):
        raise ValueError("Vector cannot be zero")
    
    ndim = len(line)
    
    if point is None:
        point = np.zeros(ndim)
    point = np.array(point, dtype=float)
    if len(point) != ndim:
        raise ValueError(f"Point must be {ndim}D for {ndim}D line")
    
    # Calculate center point
    center = point + line / 2
    
    # Create line object and setup callbacks
    if ndim == 1:
        raise NotImplementedError("1D lines are not supported yet")
    elif ndim == 2:
        line_obj, = ax.plot([], [], **kwargs)
        callbacks = ['xlim_changed', 'ylim_changed']
    elif ndim == 3:
        line_obj, = ax.plot([], [], [], **kwargs)
        callbacks = ['xlim_changed', 'ylim_changed', 'zlim_changed']
    else:
        raise ValueError("Line vector must be 2D or 3D")
    
    for callback in callbacks:
        ax.callbacks.connect(callback, lambda _: update_line())
    
    update_line()


# [untested/unverified]
# FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/12[unverified/untested]
def plot_plane(ax: Axes, plane_normal: ArrayLike, point: ArrayLike = None, **kwargs) -> None:
    """Plot an indefinite plane in 3D, defined by a normal vector and a point"""
    
    def update_plane():
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        
        # Find intersections with cube edges
        intersections = []
        edges = [
            ([xlim[0], xlim[1]], ylim[0], zlim[0]), ([xlim[0], xlim[1]], ylim[1], zlim[0]),
            ([xlim[0], xlim[1]], ylim[0], zlim[1]), ([xlim[0], xlim[1]], ylim[1], zlim[1]),
            (xlim[0], [ylim[0], ylim[1]], zlim[0]), (xlim[1], [ylim[0], ylim[1]], zlim[0]),
            (xlim[0], [ylim[0], ylim[1]], zlim[1]), (xlim[1], [ylim[0], ylim[1]], zlim[1]),
            (xlim[0], ylim[0], [zlim[0], zlim[1]]), (xlim[0], ylim[1], [zlim[0], zlim[1]]),
            (xlim[1], ylim[0], [zlim[0], zlim[1]]), (xlim[1], ylim[1], [zlim[0], zlim[1]]),
        ]
        
        for edge in edges:
            if isinstance(edge[0], list):
                y_val, z_val = edge[1], edge[2]
                if abs(normal[0]) > 1e-10:
                    x_intersect = (d - normal[1]*y_val - normal[2]*z_val) / normal[0]
                    if xlim[0] <= x_intersect <= xlim[1]:
                        intersections.append([x_intersect, y_val, z_val])
            elif isinstance(edge[1], list):
                x_val, z_val = edge[0], edge[2]
                if abs(normal[1]) > 1e-10:
                    y_intersect = (d - normal[0]*x_val - normal[2]*z_val) / normal[1]
                    if ylim[0] <= y_intersect <= ylim[1]:
                        intersections.append([x_val, y_intersect, z_val])
            else:
                x_val, y_val = edge[0], edge[1]
                if abs(normal[2]) > 1e-10:
                    z_intersect = (d - normal[0]*x_val - normal[1]*y_val) / normal[2]
                    if zlim[0] <= z_intersect <= zlim[1]:
                        intersections.append([x_val, y_val, z_intersect])
        
        if intersections:
            intersections = np.unique(np.array(intersections), axis=0)
            if len(intersections) >= 3:
                center = np.mean(intersections, axis=0)
                u = intersections[1] - intersections[0]
                v = np.cross(normal, u)
                if np.linalg.norm(v) > 1e-10:
                    v = v / np.linalg.norm(v)
                    u = u / np.linalg.norm(u)
                    coords_2d = [(np.dot(p - center, u), np.dot(p - center, v)) for p in intersections]
                    angles = [np.arctan2(y, x) for x, y in coords_2d]
                    ordered_points = intersections[np.argsort(angles)]
                    poly_collection.set_verts([ordered_points])
                else:
                    poly_collection.set_verts([intersections])
            else:
                poly_collection.set_verts([])
        else:
            poly_collection.set_verts([])

    # FIXME: This is temporary code
    if not ax.name in {'1d', '3d'} and plane_normal is None:
        # Create polygon collection
        poly_collection = PolyCollection([], linewidths=0, **kwargs)
        ax.add_collection(poly_collection)
        # Set the lambda function to call
        def update_plane_full():
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            edges = [
                (xlim[0], ylim[0]),
                (xlim[0], ylim[1]),
                (xlim[1], ylim[1]),
                (xlim[1], ylim[0]),
            ]
            poly_collection.set_verts([edges])
        # Add a callback
        for callback in ['xlim_changed', 'ylim_changed']:
            ax.callbacks.connect(callback, lambda _: update_plane_full())
        # Call the function for the first time
        update_plane_full()
        # Return
        return
    
    plane_normal = np.array(plane_normal, dtype=float)
    if np.allclose(plane_normal, 0):
        raise ValueError("Normal vector cannot be zero")
    
    if len(plane_normal) != 3:
        raise ValueError("Plane normal must be 3D")
    
    if point is None:
        point = np.zeros(3)
    point = np.array(point, dtype=float)
    if len(point) != 3:
        raise ValueError("Point must be 3D for 3D plane")
    
    # Normalize normal vector and calculate plane constant
    normal = plane_normal / np.linalg.norm(plane_normal)
    d = np.dot(normal, point)
    
    # Create polygon collection
    poly_collection = Poly3DCollection([], linewidths=0, **kwargs)
    ax.add_collection3d(poly_collection)
    
    # Set up callbacks
    for callback in ['xlim_changed', 'ylim_changed', 'zlim_changed']:
        ax.callbacks.connect(callback, lambda _: update_plane())
    
    update_plane()


# [untested/unverified]
# FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/12[unverified/untested]
def plot_box(ax: Axes, **kwargs) -> None:
    """Plot an indefinite box in 3D, centered at the origin with edges from -inf to inf"""
    
    def update_box():
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        corners = np.array([[xlim[0], ylim[0], zlim[0]],
                            [xlim[1], ylim[0], zlim[0]],
                            [xlim[1], ylim[1], zlim[0]],
                            [xlim[0], ylim[1], zlim[0]],
                            [xlim[0], ylim[0], zlim[1]],
                            [xlim[1], ylim[0], zlim[1]],
                            [xlim[1], ylim[1], zlim[1]],
                            [xlim[0], ylim[1], zlim[1]]])
        
        faces = [[corners[j] for j in [0, 1, 2, 3]],
                 [corners[j] for j in [4, 5, 6, 7]],
                 [corners[j] for j in [0, 1, 5, 4]],
                 [corners[j] for j in [2, 3, 7, 6]],
                 [corners[j] for j in [1, 2, 6, 5]],
                 [corners[j] for j in [4, 7, 3, 0]]]
        
        box_collection.set_verts(faces)
    
    box_collection = Poly3DCollection([], linewidths=0, **kwargs)
    ax.add_collection3d(box_collection)
    
    for callback in ['xlim_changed', 'ylim_changed', 'zlim_changed']:
        ax.callbacks.connect(callback, lambda _: update_box())
    
    update_box()


# [untested/unverified]
# TODO: For vectors, also add the callback mechanism such that they are properly displayed in 3d plots
def plot_vector(ax: Axes, vector: ArrayLike, point: Optional[ArrayLike] = None, **kwargs) -> None:
    """Plot a vector as an arrow"""

    class FancyArrowPatch3D(FancyArrowPatch):
        """Extension of the FancyArrowPatch to work with 3d axis"""

        def __init__(self, point, vector, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = tuple(elem for elem in zip(point, vector))

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs) if hasattr(self, 'axes') else 0

    vector = np.atleast_1d(vector)
    if vector.ndim != 1:
        raise ValueError(f"Vector must be 1-dimensional, recieved shape {vector.shape}")
    if vector.size > 3:
        raise ValueError(f"Vector must be 1, 2, or three-dimensional, recieved size {vector.size}")
    if vector.size == 2 and ax.name == '3d':
        raise ValueError(f"Provided vector is 2d, but axes object is 3d")
    if vector.size == 3 and ax.name != '3d':
        raise ValueError(f"Provided vector is 3d, but axes object is 2d")

    if point is None:
        point = np.zeros(vector.size)
    else:
        point = np.atleast_1d(point)
        if point.ndim != 1:
            raise ValueError(f"Point must be 1-dimensional, recieved shape {point.shape}")
        if point.size != vector.size:
            raise ValueError(f"Point and vector must have the same size, but recieved size {point.size} and {vector.size} respectively")

    if vector.size == 2:
        arrow = FancyArrowPatch(point,
                                vector,
                                mutation_scale=20,
                                lw=2,
                                arrowstyle="-|>",
                                color=kwargs.get('color', None),
                                )
    elif vector.size == 3:
        arrow = FancyArrowPatch3D(point,
                                  vector,
                                  mutation_scale=20,
                                  lw=2,
                                  arrowstyle="-|>",
                                  color=kwargs.get('color', None),
                                  )

    ax_lims = [ax.get_xlim(), ax.get_ylim()] + ([ax.get_zlim()] if vector.size == 3 else []) 
    for idx, ax_lim in enumerate(ax_lims):
        # FIXME: This 1.1 should not be hardcoded
        exec(f'ax.set_{['x', 'y', 'z'][idx]}lim(1.1 * min([ax_lim[0], vector[idx], point[idx]]), 1.1 * max([ax_lim[1], vector[idx], point[idx]]))')

    ax.add_patch(arrow)