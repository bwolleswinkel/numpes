"""Test function for the config module"""

import re
from typing import Any

from dataclasses import fields
import numpes as pes
from numpes._config import CFG, ConfigSchema, GlobalConfig
import pytest

try:
    import cvxpy as cvx
    CVXPY_INSTALLED: bool = True
except ImportError as _:
    CVXPY_INSTALLED = False

expected_algo_attr = {
    'rtol',
    'atol',
    'on_poly_convert',
    'on_property_assign',
    'on_hash_degen',
    'lp_backend',
    'sdp_backend',
    'scipy_method',
    'cvxpy_solver',
    'optimize_success',
}

expected_display_attr = {
    'verbose',
    'plot_aspect',
    'sym_char',
    'format_spec_poly',
    'format_spec_ellps',
    'format_spec_subs',
}

expected_attr = expected_algo_attr | expected_display_attr

class TestConfigSchema:
    """Test for the `numpes._config.ConfigSchema` dataclass"""

    def test_attr_no_unexpected(self) -> None:
        """Test that the ConfigSchema does not contain any attributes beyond the ones listed"""
        found_attr = {field.name for field in fields(ConfigSchema)}
        assert found_attr - expected_attr == set(), \
            f"Unexpected attributes found in ConfigSchema: {found_attr - expected_attr}"

    def test_attr_no_missing(self) -> None:
        """Test that the ConfigSchema contains all of the attributes listed"""
        found_attr = {field.name for field in fields(ConfigSchema)}
        assert expected_attr - found_attr == set(), \
            f"Expected attributes missing from ConfigSchema: {expected_attr - found_attr}"

    expected_default_values = {
        'rtol': 1E-5,
        'atol': 1E-8,
        'on_poly_convert': 'pass',
        'on_property_assign': 'minimal',
        'on_hash_degen': 'error',
        'lp_backend': 'auto',
        'sdp_backend': 'cvxpy',
        'scipy_method': 'highs',
        # 'cvxpy_solver': 'cvxpy',
        'optimize_success': 'optimal',
        'verbose': 0,
        'plot_aspect': 'auto',
        'sym_char': '*',
        'format_spec_poly': None,
        'format_spec_ellps': None,
        'format_spec_subs': None,
    }

    def test_attr_default_values(self) -> None:
        """Test that all attributes have the expected default values"""
        config_schema = ConfigSchema()
        for key, value in self.expected_default_values.items():
            assert getattr(config_schema, key) == value, \
                f"Expected default value of '{value}' for attribute '{key}', but received {getattr(config_schema, key)}"


class TestGlobalConfig:
    """Test for the global `numpes._config.GlobalConfig` class"""

    def test_init(self) -> None:
        """Test that initializing raises no error"""
        _ = GlobalConfig()

    @pytest.mark.parametrize("arg", [
        0,
        None,
        ...,
    ])
    def test_parameterize_init_invalid(self, arg: Any) -> None:
        """Test that providing any argument raises a TypeError"""
        with pytest.raises(TypeError):
            _ = GlobalConfig(arg)

    @pytest.mark.parametrize("name", list(expected_attr))
    def test_getattr(self, name: str) -> None:
        """Test that the __getattr__ works for getting attributes"""
        global_config = GlobalConfig()
        _ = getattr(global_config, name)

    @pytest.mark.parametrize("name", list(expected_attr))
    def test_setattr_raises_attribute_error(self, name: str) -> None:
        """Test that setting any of the attributes raises a """
        global_config = GlobalConfig()
        with pytest.raises(AttributeError, match=re.escape(
                "Use set_algo_options() or set_display_options() to modify global settings"
            )):
            setattr(global_config, name, ...)

    def test_set_locked_attribute_error(self) -> None:
        """Test that setting `_locked` raises another AttributeError"""
        global_config = GlobalConfig()
        with pytest.raises(AttributeError, match=re.escape(
                "This attribute should not be changed externally"
            )):
            global_config._locked = False

    def test_on_poly_convert_always_true(self) -> None:
        """Tests whether calling `CFG.on_poly_convert()` always returns true"""
        assert CFG.on_poly_convert_(), \
            f"Expected on_poly_convert to always return True within context, but got {CFG.on_poly_convert_()}"

    @pytest.mark.coupled('CFG', 'pes.algo_options', 'pes.ConversionError')
    def test_on_poly_convert_to_error_raises_conversion_error(self) -> None:
        """Tests whether calling `CFG.on_poly_convert()` in a contextmanager raises a ConversionError"""
        with pes.algo_options(on_poly_convert='error'):
            with pytest.raises(pes.ConversionError, match=re.escape(
                    "The value of 'CFG.on_poly_convert' is set to 'error', so conversion between polytope representations is not allowed"
                )):
                CFG.on_poly_convert_()

class TestSetAlgoOptions:
    """Tests for the `pes.set_algo_options` function"""

    @pytest.mark.parametrize('key, value', [
        ('unknown_key', 10),
        ('backend_lp', ...),
        ('tol', 1E4),
        ('on_ellps_convert', 'error'),
        ('...', None),
    ])
    def test_invalid_unknown_key_raises_key_error(self, key: str, value: Any):
        """Test that an unknown key raises a KeyError"""
        with pytest.raises(KeyError, match=re.escape(
                f"Unknown config key '{key}'"
            )):
            pes.set_algo_options(**{key: value})

    @pytest.mark.parametrize('key', [
        'verbose',
        'plot_aspect',
        'sym_char',
        'format_spec_poly',
        'format_spec_ellps',
        'format_spec_subs',
    ])
    def test_invalid_display_key_raise_key_error(self, key: str):
        """Test that a display key raises a KeyError"""
        with pytest.raises(KeyError, match=re.escape(
                f"'{key}' is a display option, use set_display_options() instead"
            )):
            pes.set_algo_options(**{key: ...})

    @pytest.mark.coupled('CFG')
    @pytest.mark.parametrize('key, value', [
        ('rtol', 1E-2),
        ('atol', 0.01),
        ('on_poly_convert', 'error'),
        ('optimize_success', 'optimal_inaccurate'),
    ])
    def test_parameterize_change_value(self, key: str, value: Any) -> None:
        """Test that changing a values using updates CFG"""
        pes.set_algo_options(**{key: value})
        assert getattr(CFG, key) == value, \
            f"Expected key '{key}' to be equal to '{value}', but received {getattr(CFG, key)}"


class TestAlgoOptions:
    """Tests for the `pes.algo_options` context manager"""

    @pytest.mark.coupled('CFG')
    @pytest.mark.parametrize('key, value, default', [
        ('rtol', 1E-2, 1E-5),
        ('atol', 0.01, 1E-8),
        ('on_poly_convert', 'error', 'pass'),
        ('optimize_success', 'optimal_inaccurate', 'optimal'),
    ])
    def test_parameterize_change_value(self, key: str, value: Any, default: Any) -> None:
        """Test that changing a values within the context updates CFG, and afterwards reverts to the old value"""
        with pes.algo_options(**{key: value}):
            assert getattr(CFG, key) == value, \
                f"Expected key '{key}' to be equal to '{value}' within the context, but received {getattr(CFG, key)}"
        assert getattr(CFG, key) == default, \
            f"Expected key '{key}' to be equal to the default value '{default}' after the context, but received {getattr(CFG, key)}"


class TestSetDisplayOptions:
    """Tests for the `pes.set_display_options` function"""


class TestDisplayOptions:
    """Tests for the `pes.display_options` context manager"""


class TestGetConfig:
    """Tests for the `pes.get_config` function"""


class TestResetConfig:
    """Tests for the `pes.reset_config` function"""


class TestInvalidAlgoKeys:
    """Tests for the `numpes._config._invalid_algo_keys` function"""


class TestInvalidDisplayKeys:
    """Tests for the `numpes._config._invalid_display_keys` function"""
