import numpy as np
import porepy as pp

nx = np.array([3, 2])
g = pp.CartGrid(nx)

phys_dims = np.array([10, 10])
g_2 = pp.CartGrid(nx, phys_dims)

g.num_cells

g.num_faces

g.num_nodes

# And finally dimension
g.dim

g.nodes

g_2.nodes

g.compute_geometry()
print(g.cell_centers)

g_2.compute_geometry()
print(g_2.cell_centers)
g_2.nodes = g_2.nodes + np.random.random((g_2.nodes.shape))
g_2.compute_geometry()
print(g_2.cell_centers)

#matplotlib inline
pp.plot_grid(g, figsize=(15,12))

cell_id = np.arange(g.num_cells)
pp.plot_grid(g, cell_value=cell_id, info='c', alpha=0.5, figsize=(15,12))

e = pp.Exporter(g, 'grid')
e.write_vtk()

nodes = g.nodes[:2]
nodes[1, 5:7] = np.array([1.2, 0.8])
g = pp.TriangleGrid(nodes)
g.compute_geometry()
pp.plot_grid(g, figsize=(15,12))
