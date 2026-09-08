.. rst-class:: monospace-title

======================================
pes.InvalidCombinationOfArgumentsError
======================================

An invalid combination of arguments is provided to a function, method, or constructor.

Examples
--------
>>> try:
...     pes.poly([[1,  2],
...               [0, -1]], n=2)
... except pes.InvalidCombinationOfArgumentsError as e:
...     print(f"Caught {type(e).__name__}: {e}")
Caught InvalidCombinationOfArgumentsError: Cannot provide 'n' when initializing from vertices
