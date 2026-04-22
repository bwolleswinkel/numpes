"""Test function for the config module"""

import re

import numpes as pes
from numpes._config import CFG
import pytest

try:
    import cvxpy as cvx
    CVXPY_INSTALLED: bool = True
except ImportError as _:
    CVXPY_INSTALLED = False


config_params = ('key, expected', [
    ('rtol', 1E-5),
    ('atol', 1E-8),
    ('on_poly_convert', 'pass'),
    ('lp_backend', 'auto'),
    ('scipy_method', 'highs'),
    ('optimize_success', 'optimal'),
])


@pytest.mark.parametrize(*config_params)
def test_config_check_default_values(key, expected):
    assert pes.get_config(key) == expected, \
        f"Expected default value of {key} to be {expected}, but got {pes.get_config(key)}"


@pytest.mark.parametrize('key, value', [
    ('unknown_key', 10),
    ('backend_lp', ...),
    ('tol', 1E4),
    ('on_ellps_convert', 'error'),
])
def test_config_set_algo_invalid_unknown(key, value):
    with pytest.raises(KeyError, match=re.escape(f"Unknown config key '{key}'")):
        pes.set_algo_options(**{key: value})


@pytest.mark.parametrize('key, value', [
    ('print_num_verts', 10),
    ('print_format_poly', ...),
    ('print_format_ellps', None),
    ('print_format_subs', False),
])
def test_config_set_algo_invalid_print_option(key, value):
    with pytest.raises(KeyError, match=re.escape(
        f"'{key}' is a printing option, use set_print_options() instead")):
        pes.set_algo_options(**{key: value})


def test_config_set_atol():
    pes.set_algo_options(atol=0.01)
    assert CFG.atol == 0.01, \
        f"Expected atol to be set to 0.01, but got {CFG.atol}"


def test_config_set_atol_context():
    with pes.algo_options(atol=0.01):
        assert CFG.atol == 0.01, \
            f"Expected atol to be 0.01 within context, but got {CFG.atol}"
    assert CFG.atol == 1E-8, \
        f"Expected atol to be reset to 1E-8, but got {CFG.atol}"


def test_config_set_rtol():
    pes.set_algo_options(rtol=0.01)
    assert CFG.rtol == 0.01, f"Expected rtol to be set to 0.01, but got {CFG.rtol}"


def test_config_set_rtol_context():
    with pes.algo_options(rtol=0.01):
        assert CFG.rtol == 0.01, f"Expected rtol to be 0.01 within context, but got {CFG.rtol}"
    assert CFG.rtol == 1E-5, f"Expected rtol to be reset to 1E-5, but got {CFG.rtol}"


@pytest.mark.parametrize(*config_params)
def test_config_reset(key, expected):
    pes.set_algo_options(**{key: None})
    pes.reset_config()
    assert pes.get_config(key) == expected, \
        f"Expected {key} to be reset to {expected}, but got {pes.get_config(key)}"


def test_config_on_poly_convert_pass():
    with pes.algo_options(on_poly_convert='pass'):
        assert CFG.on_poly_convert() == True, \
            f"Expected on_poly_convert to return True within context, but got {CFG.on_poly_convert()}"


def test_config_on_poly_convert_warning():
    with pes.algo_options(on_poly_convert='warning'):
        with pytest.warns(UserWarning, match="Conversion between polytope representations"):
            assert CFG.on_poly_convert() == True, \
                f"Expected on_poly_convert to return True within context, but got {CFG.on_poly_convert()}"


def test_config_on_poly_convert_error():
    with pes.algo_options(on_poly_convert='error'):
        with pytest.raises(pes.ConversionError, match="The value of 'CFG.on_poly_convert' is set to 'error', so conversion between polytope representations is not allowed"):
            CFG.on_poly_convert()


def test_config_on_poly_convert_invalid_value():
    with pes.algo_options(on_poly_convert='invalid_value'):
        with pytest.raises(ValueError, match="Unknown value 'invalid_value' for 'on_poly_convert' config setting"):
            CFG.on_poly_convert()