.. rst-class:: monospace-title

============
pes.Polytope
============

Initialize a Polytope from vertices or half-spaces.

Parameters
----------
``args`` : ``tuple[()] | tuple[ArrayLike] | tuple[ArrayLike, ArrayLike]``
    Variable-length positional arguments. Use zero, one, or two arguments to
    initialize an empty polytope, a V-representation, or an H-representation.
``n`` : ``int``, optional
    Dimension of the ambient space. Required for an empty polytope.
``verts`` : ``NDArray[('k', 'n'), float]``, optional
    Vertices for a V-representation.
``rays`` : ``NDArray[('k', 'n'), float]``, optional
    Rays for an unbounded polytope.
``A`` : ``NDArray[('m', 'n'), float]``, optional
    Matrix defining the half-spaces in an H-representation.
``b`` : ``NDArray[('m',), float]``, optional
    Vector defining the half-spaces in an H-representation.
``A_eq`` : ``NDArray[('m_eq', 'n'), float]``, optional
    Matrix defining equality constraints in an H-representation.
``b_eq`` : ``NDArray[('m_eq',), float]``, optional
    Vector defining equality constraints in an H-representation.

Raises
------
:exc:`numpes.InvalidCombinationOfArgumentsError`
    If the provided arguments do not match an expected initialization pattern.
:exc:`TypeError`
    If an argument has an invalid type.
:exc:`ValueError`
    If ``n`` is not a positive integer.

Examples
--------
.. code-block:: pycon

   >>> verts = [[0, 0], [1, 0], [0, 1]]
   >>> poly = pes.poly(verts)
   >>> print(poly)
   Polytope in R^2
        /[[0]  [[1]  [[0] \\
   conv \\ [0]], [0]], [1]]/

.. code-block:: pycon

   >>> A = [[1, 0], [0, 1], [-1, 0], [0, -1]]
   >>> b = [1, 1, 0, 0]
   >>> poly = pes.poly(A, b)
   >>> print(poly)
   Polytope in R^2
   [[ 1  0]  |    [[1]
    [ 0  1]  |     [1]
    [-1  0]  x <=  [0]
    [ 0 -1]] |     [0]]

.. code-block:: pycon

   >>> poly = pes.poly(n=2)
   >>> print(poly)
   Polytope in R^2
   [0 0] x <= [-1]

.. toctree::
    :hidden:
    :caption: Classmethods

    pes_polytope/pes_polytope_from_ambient
    pes_polytope/pes_polytope_from_bounds
    pes_polytope/pes_polytope_from_point

.. toctree::
    :hidden:
    :caption: Methods

    pes_polytope/pes_polytope_plot

.. toctree::
    :hidden:
    :caption: Properties
 
    pes_polytope/pes_polytope_vrepr
    pes_polytope/pes_polytope_hrepr
    pes_polytope/pes_polytope_n
    pes_polytope/pes_polytope_verts
    pes_polytope/pes_polytope_rays
    pes_polytope/pes_polytope_k
    pes_polytope/pes_polytope_k_rays
    pes_polytope/pes_polytope_Ab
    pes_polytope/pes_polytope_A
    pes_polytope/pes_polytope_b
    pes_polytope/pes_polytope_Ab_eq
    pes_polytope/pes_polytope_A_eq
    pes_polytope/pes_polytope_b_eq
    pes_polytope/pes_polytope_m
    pes_polytope/pes_polytope_m_eq
    pes_polytope/pes_polytope_is_empty
    pes_polytope/pes_polytope_is_singleton
    pes_polytope/pes_polytope_is_ambient
    pes_polytope/pes_polytope_is_degen
    pes_polytope/pes_polytope_is_bounded
    pes_polytope/pes_polytope_is_lower_dim
    pes_polytope/pes_polytope_is_pointed
    pes_polytope/pes_polytope_dim
    pes_polytope/pes_polytope_vol
    pes_polytope/pes_polytope_diam
    pes_polytope/pes_polytope_width
    pes_polytope/pes_polytope_chebcr
    pes_polytope/pes_polytope_chebc
    pes_polytope/pes_polytope_chebr
