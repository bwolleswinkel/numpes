"""Tests for the polytope module"""

import numpy as np
import pytest

import numpes as pes

@pytest.mark.parametrize('number, expected', [
    (2, True),
    (3, True),
    (4, False),
    (17, True),
    (18, False),
    (19, True),
    (20, False),
    (1, False),
    (0, False),
    (-5, False),
])

def test_is_prime(number, expected):
    assert pes.is_prime(number) == expected, f"is_prime({number}) should be {expected}"