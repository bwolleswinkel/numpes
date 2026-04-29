"""Script containing fixtures used in testing, such as resetting globals and constructing test classes from archetypes and generators"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import settings

import numpes as pes
from tests.data.archetypes_polytope import _POLYTOPES_ALL, POLYTOPES_ARCHETYPES_REGISTRY, PolytopeData
from tests.data.generators_polytope import centered_hypercube, cross_polytope, simplex, unit_hypercube

if TYPE_CHECKING:
    from typing import Callable, Literal

    from pytest import Metafunc

    from numpes import Polytope

RTOL: float = 1E-5
ATOL: float = 1E-8
N_MAX: int = 10
MAX_EXAMPLES: int = 100

settings.register_profile('default', max_examples=MAX_EXAMPLES, deadline=500)
settings.load_profile('default')


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Hook that generates test cases from archetypes and automatically parametrizes tests that use the `polytope_data` fixture"""

    for fixture_name in metafunc.fixturenames:
        match fixture_name.split('_'):
            case ['poly', 'arch', p_type, repr, n]:
                try:
                    n = int(n[:-1])
                except (IndexError, ValueError):
                    raise RuntimeError(f"Error in fixture name '{fixture_name}'")
                category = POLYTOPES_ARCHETYPES_REGISTRY.get(p_type, {})
                data_list = category.get(n, [])
                polys = [(poly_init_safe(data, repr), data) for data in data_list]
                ids = [data.name for data in data_list]
                metafunc.parametrize(fixture_name, polys, ids=ids)
            case ['poly', 'arch', p_type, n]:
                category = POLYTOPES_ARCHETYPES_REGISTRY.get(p_type, {})
                if n == 'all':
                    data_list = [data for category in POLYTOPES_ARCHETYPES_REGISTRY.values() for dim_list in category.values() for data in dim_list]
                else:
                    try:
                        n = int(n[:-1])
                        data_list = category.get(n, [])
                    except (IndexError, ValueError):
                        raise RuntimeError(f"Error in fixture name '{fixture_name}'")
                polys = [(poly_init_safe(data, 'both'), data) for data in data_list]
                ids = [data.name for data in data_list]
                metafunc.parametrize(fixture_name, polys, ids=ids)
            case ['poly', 'arch', 'all']:
                data_list = [data for category in POLYTOPES_ARCHETYPES_REGISTRY.values() for dim_list in category.values() for data in dim_list]
                polys = [(poly_init_safe(data, 'both'), data) for data in data_list]
                ids = [data.name for data in data_list]
                metafunc.parametrize(fixture_name, polys, ids=ids)
            case _:
                # NOTE: This assumes that the fixture is not meant to be parametrized with archetype data, and is defined somewhere else
                pass


def poly_init_safe(poly_data: PolytopeData, repr: Literal['vrepr', 'hrepr', 'both']) -> Polytope:
    try:
        match repr:
            case 'vrepr':
                poly = pes.Polytope(poly_data.verts, rays=poly_data.rays)
            case 'hrepr':
                poly = pes.Polytope(poly_data.A, poly_data.b, A_eq=poly_data.A_eq, b_eq=poly_data.b_eq)
            case 'both':
                poly = pes.Polytope(n=poly_data.n)
                poly._vrepr = (poly_data.verts, poly_data.rays)
                poly._hrepr = (np.column_stack((poly_data.A, poly_data.b)), np.column_stack((poly_data.A_eq, poly_data.b_eq)))
            case _:
                raise ValueError(f"Unknown representation type '{repr}' specified (must be one of 'both', 'vrepr', or 'hrepr')")     
        return poly
    except Exception as e:
        raise RuntimeError(f"Failed to construct polytope from polytope '{poly_data.name}'") from e


@pytest.fixture
def poly_gen_factory() -> Callable[[Literal['unit_hypercube', 'centered_hypercube', 'simplex', 'cross_polytope'], int, Literal['vrepr', 'hrepr', 'both']], tuple[Polytope, PolytopeData]]:
    """Factory fixture for generating polytopes dynamically"""
    
    def _make_polytope(generator_name: Literal['unit_hypercube', 'centered_hypercube', 'simplex', 'cross_polytope'], n: int, repr: Literal['vrepr', 'hrepr', 'both'] = 'both') -> tuple[Polytope, PolytopeData]:
        generators = {
            'simplex': simplex,
            'unit_hypercube': unit_hypercube,
            'centered_hypercube': centered_hypercube,
            'cross_polytope': cross_polytope
        }
        if generator_name not in generators:
            raise ValueError(f"Unknown generator '{generator_name}'")
        generator = generators[generator_name]
        name, verts, (A, b), vol = generator(n)
        poly_data = PolytopeData(
            name=name,
            n=n,
            verts=verts,
            rays=np.empty((0, n)),
            A=A,
            b=b,
            A_eq=np.empty((0, n)),
            b_eq=np.empty((0,)),
            k=0,
            k_rays=0,
            m=0,
            m_eq=0,
            is_degen=False,
            is_bounded=False,
            is_full_dim=True,
            is_singleton=False,
            is_empty=False,
            is_minimal=True,
            vol=vol
        )
        poly = poly_init_safe(poly_data, repr)
        return poly, poly_data
    
    return _make_polytope


# FROM: GitHub Copilot Claude Sonnet 4 | 26/04/14[untested/unverified]
def _create_polytope_fixture(poly_data: PolytopeData) -> Callable[[], Polytope]:
    """Create a fixture function for a specific polytope"""
    def fixture_func() -> Polytope:
        return poly_init_safe(poly_data, 'both')
    return fixture_func


# FROM: GitHub Copilot Claude Sonnet 4 | 26/04/14[untested/unverified]
for poly_data in _POLYTOPES_ALL:
    fixture_name = f"poly_arch_{poly_data.name}"
    fixture_func = _create_polytope_fixture(poly_data)
    fixture_func.__doc__ = f"Archetype polytope: {poly_data.name} ({poly_data.n}D)"
    globals()[fixture_name] = pytest.fixture(fixture_func)


@pytest.fixture(autouse=True)
def reset_config() -> None:
    """Reset the global configuration before each test to ensure that tests are independent and do not interfere with each other through global state"""
    pes.reset_config()