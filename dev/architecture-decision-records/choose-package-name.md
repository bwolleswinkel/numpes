# Choosing a name for the Python package

## Status

Accepted

## Context

A name for the package needed to be decided on.

Several alternatives were proposed:
- `numpes` Numerical operations/emulation on/of polytopes, ellipsoids, and subspaces | Abbreviation: `pes` or `ps`
- `geopes` Geometric operations on polytopes, ellipsoids, and subspaces | Abbreviation: `gp`
- `pypes` Python polytopes, ellipsoids, and subspaces | Abbreviation: `pes` or `ps`

## Decision

The name `numpes`, with spelling "NumPES", was deemed most fitting and illuminative. Furthermore, it was decided to use the short form

``` py
import numpes as pes
```

## Consequences

The name NumPES is final and fixed. This is because, whilst the repository on GitHub can in essence be replaced, the PyPI name is reserved. 