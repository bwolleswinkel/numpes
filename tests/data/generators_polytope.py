"""Script containing generators for several known polytopes used in tests"""

import math
from itertools import combinations

import numpy as np


def unit_hypercube(n: int) -> tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Generator for the unit hypercube in n dimensions"""
    verts = np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)
    A, b = np.vstack((np.eye(n), -np.eye(n))), np.hstack((np.ones(n), np.zeros(n)))
    vol = 1
    return f"unit_hypercube_{n}d", verts, (A, b), vol


def centered_hypercube(n: int) -> tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Generator for a centered hypercube in n dimensions"""
    verts = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
    A, b = np.vstack((np.eye(n), -np.eye(n))), np.hstack((np.ones(n), np.ones(n)))
    vol = 2 ** n
    return f"centered_hypercube_{n}d", verts, (A, b), vol


def simplex(n: int) -> tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Generator for the simplex in n dimensions"""
    verts = np.vstack([np.zeros((1, n)), np.eye(n)])
    A, b = np.vstack((np.ones((1, n)), -np.eye(n))), np.hstack((np.ones(1), np.zeros(n)))
    vol = 1 / math.factorial(n)
    return f"simplex_{n}d", verts, (A, b), vol


def cross_polytope(n: int) -> tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Generator for the cross-polytope in n dimensions"""
    verts = np.vstack((np.eye(n), -np.eye(n)))
    A, b = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n), np.ones(2 ** n)
    vol = 2 ** n / math.factorial(n)
    return f"cross_polytope_{n}d", verts, (A, b), vol


def cyclic_polytope(n: int, d: int) -> tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray], None]:
    """Generator for the cyclic polytope in n dimensions with n + 1 vertices"""

    def gale_facets(n: int, d: int, verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Helper function to compute the Gale evenness condition for cyclic polytopes"""
        # FROM: Google Gemini 3 | 26/03/17[untested/unverified]
        all_indices = list(range(n))
        A_rows, b_vals = [], []
        for combo in combinations(all_indices, d):
            others = [i for i in all_indices if i not in combo]
            is_facet = True
            for i in range(len(others) - 1):
                start, end = others[i], others[i + 1]
                count_between = len([c for c in combo if start < c < end])
                if count_between % 2 != 0:
                    is_facet = False
                    break
            if is_facet:
                roots = np.arange(1, n + 1)[list(combo)]
                poly_coeffs = np.poly(roots)
                a, b = poly_coeffs[:-1][::-1], -poly_coeffs[-1]
                test_v = verts[others[0]]
                val = np.dot(a, test_v)
                if val > b and not np.isclose(val, b):
                    a, b = -a, -b
                A_rows.append(a)
                b_vals.append(b)
        return np.array(A_rows), np.array(b_vals)

    verts = np.power.outer(np.arange(1, n + 1), np.arange(1, d + 1))
    A, b = gale_facets(n, d, verts)
    vol = None
    return f"cyclic_polytope_{n}d", verts, (A, b), vol