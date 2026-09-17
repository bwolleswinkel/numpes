"""Timing benchmarks for the convex hull operation"""

import numpy as np
import pytest
import numpes as pes


class TimeConv:
    """Timing benchmarks for the `pes.utils.conv` functions"""

    @pytest.mark.parametrize('n', [
        1,
        2,
        3,
        4,
        5,
        6,
    ])
    @pytest.mark.parametrize('num_type', [
        'simplex',
        'cube_like',
        'many',
    ])
    def time_random_vertices(self, n: int, num_type: str) -> None:
        """Time the creation of a list of numbers"""
        match num_type:
            case 'simplex':
                num_verts = lambda n: n + 1
            case 'cube_like':
                num_verts = lambda n: 2 ** (n + 1)
            case 'many':
                num_verts = lambda n: 5 ** (n + 1)
        verts = np.random.normal(size=(num_verts(n), n))
        _ = pes.utils.conv(verts)
