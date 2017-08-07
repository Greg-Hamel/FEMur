from FEMur import *

a = Mesh2D('Base_mesh_tria3.msh', [0.0]) # Import du Mesh a partir de GMSH
# a = Mesh2D('Base_mesh_quad.msh', [0.0]) # Import du Mesh a partir de GMSH
a.mesh() # Creation du Mesh interne
a.solve_elements() # Resolution des elements

# Definition de l'environnement
dirichlet = [0, 1]
a.set_environment(20.0, 50, 0.01, 75.0, dirichlet, 170.0)
a.solve_omega() # Resolution des matrices et vecteurs globaux

a.plot_results()
