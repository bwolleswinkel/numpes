## Package structure

Below is the current structure of the package.

```
numpes
├─ ...
│
├─ src/numpes
│  │
│  ├─ _internal
│  │  ├─ __init__.py
│  │  ├─ multipledispatch.py  # For multipledispatch method decorator for polytopes
│  │  └─ wraps.py  # For better wrapper
│  │
│  ├─ control
│  │  ├─ __init__.py
│  │  ├─ geometric.py  # For max_cond_inv(...), min_ouput_null(...), ...
│  │  └─ mpc.py  # For mrpi(...), mpis(...), ...
│  │
│  ├─ utils
│  │  ├─ __init__.py
│  │  ├─ linalg.py  # For is_pos_def(...), rot_mat(...), ...
│  │  ├─ spatial.py  # For conv(...), enum_verts(...), ...
│  │  ├─ linprog.py  # For solving linear programs (which have a predictable format)
│  │  └─ timeout.py  # For timeout context manager (needed for rejection-sampling)
│  │
│  ├─ __init__.py
│  ├─ _config.py  # For global config settings
│  ├─ convex_region.py
│  ├─ ellipsoid.py
│  ├─ exceptions.py  # For DimensionsError, InternalInconsistencyError, InfeasibleProblemError, ....
│  ├─ polytope.py
│  ├─ py.typed
│  ├─ shared.py  # For operations applicable to several objects
│  └─ subspace.py
│
...
```