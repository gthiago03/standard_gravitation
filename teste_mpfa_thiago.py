# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:45:42 2022

@author: gthia
"""

import numpy as np
import porepy as pp

# Create grid
n = 5
g = pp.CartGrid([n,n])
g.compute_geometry()

# Define boundary type
dirich = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
bound = pp.BoundaryCondition(g, dirich, 'dir')

# Create permeability matrix
k = np.ones(g.num_cells)
perm = pp.SecondOrderTensor(k)

top_faces = np.ravel(np.argwhere(g.face_centers[1] > n - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))

p_b = np.zeros(g.num_faces)
p_b[top_faces] = -1 * g.face_areas[top_faces]
p_b[bot_faces] = 0

mpfa_solver = pp.Mpfa("flow")

f = np.zeros(g.num_cells)

specified_parameters = {"second_order_tensor": perm, "source": f, "bc": bound, "bc_values": p_b}
data = pp.initialize_default_data(g, {}, "flow", specified_parameters)

mpfa_solver.discretize(g, data)
A, b = mpfa_solver.assemble_matrix_rhs(g, data)

p_class = np.linalg.solve(A.A, b)

pp.plot_grid(g, cell_value=p_class, figsize=(15, 12))

flux = data[pp.DISCRETIZATION_MATRICES]["flow"]["flux"]
bound_flux = data[pp.DISCRETIZATION_MATRICES]["flow"]["bound_flux"]
print("flux matrix shape: {}".format(flux.shape))
print("bound_flux matrix shape: {} ".format(bound_flux.shape))

div = pp.fvutils.scalar_divergence(g)
A = div * flux
b = f - div * bound_flux * p_b

p = np.linalg.solve(A.A, b)

assert np.allclose(p, p_class)

pp.plot_grid(g, cell_value=p, figsize=(15, 12))

F = flux * p + bound_flux * p_b

neumann_faces = np.argwhere(bound.is_neu)

assert np.allclose(np.abs(p_b[neumann_faces]), np.abs(F[neumann_faces]))

F_n = F * g.face_normals
pp.plot_grid(g, vector_value=F_n, figsize=(15, 12))

# We now test the solution (mostly for debugging purposes)
p_exact = g.cell_centers[1]
assert np.allclose(p, p_exact)