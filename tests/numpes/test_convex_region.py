"""Tests for the convex region abstract base class"""

import re

import numpes as pes
import pytest


class TestConvexRegion:
    """Tests for the `pes.ConvexRegion` abstract base class"""


    class ValidSubclass(pes.ConvexRegion):
        """Valid dummy subclass of pes.ConvexRegion"""

        @property
        def n(self):
            return 0

        @property
        def dim(self):
            return 0

        @property
        def vol(self):
            return 0.0

        def __bool__(self):
            return True

        def __contains__(self, _other) -> bool:
            return True


    def test_instantiate_valid_subclass(self) -> None:
        """Test that instantiating a valid subclass raises no error"""
        _ = self.ValidSubclass()

    @pytest.mark.parametrize('method', [
        'n',
        'dim',
        'vol',
        '__bool__',
        '__contains__',
    ])
    def test_parameterize_instantiate_invalid_subclass(self, method: str) -> None:
        """Test that omitting one required member prevents instantiation"""
        namespace = {name: value
                     for name, value in vars(self.ValidSubclass).items()
                     if name != method}
        InvalidSubclass = type("InvalidSubclass",
                               (pes.ConvexRegion,),
                               namespace)
        with pytest.raises(TypeError, match=re.escape(
            f"Can't instantiate abstract class InvalidSubclass without an implementation for abstract method '{method}'"
            )):
            _ = InvalidSubclass()

    @pytest.mark.parametrize('property', [
        'n',
        'dim',
        'vol',
    ])
    def test_properties_valid_subclass_read_only(self, property: str) -> None:
        """Test that for a minimal valid subclass all properties are read-only"""
        obj = self.ValidSubclass()
        with pytest.raises(AttributeError, match=re.escape(
            f"property '{property}' of 'TestConvexRegion.ValidSubclass' object has no setter"
        )):
            setattr(obj, property, ...)

    def test_bool_method_valid_subclass(self) -> None:
        """Test that the `__bool__` method for a minimal valid subclass raise no errors"""
        obj = self.ValidSubclass()
        _ = bool(obj)
        if obj:
            pass
        assert obj, \
            f"Expressions like `assert obj` or `if obj` should trigger the `__bool__` method which should be implemented (and set to True for ValidSubclass), but it evaluated to False"

    def test_contains_method_valid_subclass(self) -> None:
        """Test that the `__contains__` method for a minimal valid subclass raise no errors"""
        obj = self.ValidSubclass()
        if ... in obj:
            pass
        assert ... in obj, \
            f"Expressions like `... in obj` should trigger the `__contains__` method which should be implemented (and set to True for ValidSubclass), but it evaluated to False"

    def test_contains_invalid_method_definition(self) -> None:
        """Test that the `__contains__` method incorrectly implemented (no other argument) raise an error"""
        namespace = {name: value
                     if name != '__contains__'
                     else lambda _: True
                     for name, value in vars(self.ValidSubclass).items()}
        InvalidSubclass = type("InvalidSubclass",
                               (pes.ConvexRegion,),
                               namespace)
        obj = InvalidSubclass()
        with pytest.raises(TypeError, match=\
            r".* takes 1 positional argument but 2 were given"
            ):
            if ... in obj:
                pass
