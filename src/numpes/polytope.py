"""Module for polytope and zonotope functionalities

Classes
-------
Polytope
    A class representing a convex polytope defined by either linear (in)equalities or as the convex hull of a set of vertices and corresponding rays.
Zonotope
    A class representing a zonotope defined by a center and a set of generator vectors.

Functions
---------
poly
    A factory function for creating Polytope instances.
poly_from_verts
    A factory function for creating a Polytope instance from vertices and rays.
poly_from_ineq
    A factory function for creating a Polytope instance from linear (in)equalities.
poly_from_bounds
    A factory function for creating a Polytope instance from upper and lower bounds.
hello_world
    A simple function that prints 'Hello, NumPES!'.
is_prime
    Check if a number is prime.

# TODO: Update this docstring

"""

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