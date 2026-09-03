"""Script containing archetypes for polytopes used in tests"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PolytopeData:
    name: str
    n: int
    verts: np.ndarray
    rays: np.ndarray
    A: np.ndarray
    b: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    k: int
    k_rays: int
    m: int
    m_eq: int
    is_degen: bool
    is_bounded: bool
    is_full_dim: bool
    is_singleton: bool
    is_empty: bool
    is_minimal: bool
    vol: float


# =============
# 1D ARCHETYPES
# =============

UNIT_LINE_SEGMENT_1D = PolytopeData(
    name="unit_line_segment_1d",
    n=1,
    verts=np.array([[0],
                    [1]]),
    rays=np.empty((0, 1)),
    A=np.array([[ 1],
                [-1]]),
    b=np.array([1, 
                0]),
    A_eq=np.empty((0, 1)),
    b_eq=np.empty((0,)),
    k=2,
    k_rays=0,
    m=2,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=1
)

# NOTE: This is a translation of the unit line segment by one unit
OFFSET_LINE_SEGMENT_1D = PolytopeData(
    name="offset_line_segment_1d",
    n=1,
    verts=np.array([[1],
                    [2]]),
    rays=np.empty((0, 1)),
    A=np.array([[ 1],
                [-1]]),
    b=np.array([ 2,
                -1]),
    A_eq=np.empty((0, 1)),
    b_eq=np.empty((0,)),
    k=2,
    k_rays=0,
    m=2,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=1
)

# NOTE: This is a scaled version of the unit line segment, scaled by a factor of 2
SCALED_LINE_SEGMENT_1D = PolytopeData(
    name="scaled_line_segment_1d",
    n=1,
    verts=np.array([[0],
                    [2]]),
    rays=np.empty((0, 1)),
    A=np.array([[ 1],
                [-1]]),
    b=np.array([2,
                0]),
    A_eq=np.empty((0, 1)),
    b_eq=np.empty((0,)),
    k=2,
    k_rays=0,
    m=2,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=2
)

# NOTE: Degenerate, unbounded
LINE_SEGMENT_UNBOUNDED_1D = PolytopeData(
    name="line_segment_unbounded_1d",
    n=1,
    verts=np.array([[0]]),
    rays=np.array([[1]]),
    A=np.array([[-1]]),
    b=np.array([0]),
    A_eq=np.empty((0, 1)),
    b_eq=np.empty((0,)),
    k=1,
    k_rays=1,
    m=2,
    m_eq=0,
    is_degen=True,
    is_bounded=False,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=np.inf
)

# NOTE: Degenerate, unbounded
LINE_UNBOUNDED_1D = PolytopeData(
    name="line_unbounded_1d",
    n=1,
    verts=np.empty((0, 1)),
    rays=np.array([[ 1],
                   [-1]]),
    A=np.empty((0, 1)),
    b=np.empty((0,)),
    A_eq=np.empty((0, 1)),
    b_eq=np.empty((0,)),
    k=0,
    k_rays=2,
    m=0,
    m_eq=0,
    is_degen=True,
    is_bounded=False,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=np.inf
)

# NOTE: Degenerate, single point
SINGLE_POINT_1D = PolytopeData(
    name="single_point_1d",
    n=1,
    verts=np.array([[0]]),
    rays=np.empty((0, 1)),
    A=np.empty((0, 1)),
    b=np.empty((0,)),
    A_eq=np.array([[1]]),
    b_eq=np.array([0]),
    k=1,
    k_rays=0,
    m=0,
    m_eq=1,
    is_degen=True,
    is_bounded=True,
    is_full_dim=False,
    is_singleton=True,
    is_empty=False,
    is_minimal=True,
    vol=0
)

# NOTE: Degenerate, single point
OFFSET_SINGLE_POINT_1D = PolytopeData(
    name="offset_single_point_1d",
    n=1,
    verts=np.array([[1]]),
    rays=np.empty((0, 1)),
    A=np.empty((0, 1)),
    b=np.empty((0,)),
    A_eq=np.array([[1]]),
    b_eq=np.array([1]),
    k=1,
    k_rays=0,
    m=0,
    m_eq=1,
    is_degen=True,
    is_bounded=True,
    is_full_dim=False,
    is_singleton=True,
    is_empty=False,
    is_minimal=True,
    vol=0
)

# NOTE: Degenerate, empty
EMPTY_1D = PolytopeData(
    name="empty_1d",
    n=1,
    verts=np.empty((0, 1)),
    rays=np.empty((0, 1)),
    A=np.array([[0]]),
    b=np.array([-1]),
    A_eq=np.empty((0, 1)),
    b_eq=np.empty((0,)),
    k=0,
    k_rays=0,
    m=1,
    m_eq=0,
    is_degen=True,
    is_bounded=True,
    is_full_dim=False,
    is_singleton=False,
    is_empty=True,
    is_minimal=True,
    vol=0
)

# =============
# 2D ARCHETYPES
# =============

UNIT_SQUARE_2D = PolytopeData(
    name="unit_square_2d",
    n=2,
    verts=np.array([[0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1]]),
    rays=np.empty((0, 2)),
    A=np.array([[ 1,  0],
                [ 0,  1],
                [-1,  0],
                [ 0, -1]]),
    b=np.array([1,
                1,
                0,
                0]),
    A_eq=np.empty((0, 2)),
    b_eq=np.empty((0,)),
    k=4,
    k_rays=0,
    m=4,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=1
)

CENTERED_SQUARE_2D = PolytopeData(
    name="centered_square_2d",
    n=2,
    verts=np.array([[ 1,  1],
                    [ 1, -1],
                    [-1, -1],
                    [-1,  1]]),
    rays=np.empty((0, 2)),
    A=np.array([[ 1,  0],
                [ 0,  1],
                [-1,  0],
                [ 0, -1]]),
    b=np.array([1,
                1,
                1,
                1]),
    A_eq=np.empty((0, 2)),
    b_eq=np.empty((0,)),
    k=4,
    k_rays=0,
    m=4,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=4
)

SLIM_BEAM_2D = PolytopeData(
    name="slim_beam_2d",
    n=2,
    verts=np.array([[  0, 0],
                    [100, 0],
                    [100, 1],
                    [  0, 1]]),
    rays=np.empty((0, 2)),
    A=np.array([[ 1,  0],
                [ 0,  1],
                [-1,  0],
                [ 0, -1]]),
    b=np.array([100,
                  1,
                  0,
                  0]),
    A_eq=np.empty((0, 2)),
    b_eq=np.empty((0,)),
    k=4,
    k_rays=0,
    m=4,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=4
)

TRIANGLE_2D = PolytopeData(
    name="triangle_2d",  # FIXME: Rename to 'simplex', triangle is ambiguous
    n=2,
    verts=np.array([[0, 0],
                    [1, 0],
                    [0, 1]]),
    rays=np.empty((0, 2)),
    A=np.array([[-1,  0],
                [ 0, -1],
                [ 1,  1]]),
    b=np.array([0,
                0,
                1]),
    A_eq=np.empty((0, 2)),
    b_eq=np.empty((0,)),
    k=3,
    k_rays=0,
    m=3,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=0.5
)

HOUSE_2D = PolytopeData(
    name="house_2d",
    n=2,
    verts=np.array([[0  , 0  ],
                    [1  , 0  ],
                    [0  , 1  ],
                    [1  , 1  ],
                    [0.5, 1.5]]),
    rays=np.empty((0, 2)),
    A=np.array([[-1,  0],
                [ 1,  0],
                [ 0, -1],
                [-1,  1],
                [ 1,  1]]),
    b=np.array([0,
                1,
                0,
                1,
                2]),
    A_eq=np.empty((0, 2)),
    b_eq=np.empty((0,)),
    k=5,
    k_rays=0,
    m=5,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=1.25
)

# =============
# 3D ARCHETYPES
# =============

UNIT_CUBE_2D = PolytopeData(
    name="unit_cube_3d",
    n=3,
    verts=np.array([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0, 1, 1],
                    [1, 1, 1]]),
    rays=np.empty((0, 3)),
    A=np.array([[ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0,  1],
                [ 0,  0, -1]]),
    b=np.array([1,
                1,
                0,
                0,
                1,
                0]),
    A_eq=np.empty((0, 3)),
    b_eq=np.empty((0,)),
    k=6,
    k_rays=0,
    m=6,
    m_eq=0,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
    vol=1
)

_POLYTOPES_ALL: list[PolytopeData] = [v for v in globals().values() if isinstance(v, PolytopeData)]

_DIMS = [1, 2, 3]

_KEYWORD_FILTERS: dict[str, tuple[str, bool]] = {
    'degen':     ('is_degen',    True),
    'nondegen':  ('is_degen',    False),
    'bounded':   ('is_bounded',  True),
    'unbounded': ('is_bounded',  False),
    'full_dim':  ('is_full_dim', True),
    'lower_dim': ('is_full_dim', False),
    'empty':     ('is_empty',    True),
    'nonempty':  ('is_empty',    False),
    'minimal':   ('is_minimal',  True),
    'redundant': ('is_minimal',  False)
}


# FROM: GitHub Copilot Claude Opus 4.6 | 26/03/19
def _build_registry(
    polytopes: list[PolytopeData],
    dims: list[int],
    filters: dict[str, tuple[str, bool]],
) -> dict[str, dict[int, list[PolytopeData]]]:
    """Build a registry of polytopes filtered by keyword and dimension"""
    registry: dict[str, dict[int, list[PolytopeData]]] = {
        'all': {dim: [p for p in polytopes if p.n == dim] for dim in dims}
    }
    for keyword, (field, value) in filters.items():
        registry[keyword] = {
            dim: [p for p in polytopes if p.n == dim and getattr(p, field) == value]
            for dim in dims
        }
    return registry


POLYTOPES_ARCHETYPES_REGISTRY = _build_registry(_POLYTOPES_ALL, _DIMS, _KEYWORD_FILTERS)