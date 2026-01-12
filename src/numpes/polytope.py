"""Module for polytope and zonotope functionalities"""

import numpy as np


def hello_world() -> None:
    """A simple function that prints 'Hello, NumPES!'"""
    print("Hello, NumPES!")


def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True