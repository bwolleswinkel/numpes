"""This is an example of how to create, manipulate, and visualize a 2D polytope using the Polytope class"""

import numpy as np
import numpes as pes

# Define the vertices of a 2D polytope
verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

# Create a Polytope object from the vertices
poly_org = pes.poly(verts)

# Plot the original polytope
poly_org.plot(show=True)

# Perform a linear transformation on the polytope
A = np.array([[2, -1], [2, 1]])

poly_mapped = A @ poly_org

# Plot the transformed polytope
poly_mapped.plot(show=True)

# Compute the intersection of the original and transformed polytopes
poly_isect = poly_org & poly_mapped

# Plot the intersection polytope
poly_isect.plot(show=True)

# Compute and print the area of the original, transformed, and intersection polytopes
area_org, area_mapped, area_isect = poly_org.vol, poly_mapped.vol, poly_isect.vol

# Print the polytopes and their areas
print(f"Original Polytope:\n{poly_org}\nArea: {area_org}\n")
print(f"Transformed Polytope:\n{poly_mapped}\nArea: {area_mapped}\n")
print(f"Intersection Polytope:\n{poly_isect}\nArea: {area_isect}\n")
