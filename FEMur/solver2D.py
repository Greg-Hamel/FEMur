from FEMur import *
import sys
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
from math import ceil


class Solver(object):
    '''
    2-dimensional solver top class.

    Provides common initialization  to all child solver classes.
    '''
    def __init__(self, meshfile, analysis_type):
        self.meshfile = meshfile
        self.analysis_type = analysis_type
        self.get_mesh()

        self.D = None
        self.gbl_stiff = None
        self.gbl_load = None
        self.gbl_omega = None
        self.dirichlet_applied = False

    def weakform(self):
        '''
        Prints weak form used by the solver to approximate the results.
        '''

    def get_mesh(self):
        '''
        Call Mesh class to create the mesh.
        '''

        try:
            a = self.meshfile
        except AttributeError:
            print('A mesh file has not been provided.')
            sys.exit(1)
        self.mesh = Mesh2D(self.meshfile, self.analysis_type)
        self.mesh.mesh()

    def solve(self):
        '''
        Solves the equation system.

        Solves for O in:
            [K]{O}={F}


        where K is the global stiffness matrix (or conductivitity)
        O is the displacement (or temperature)
        F is the applied load (or [...])
        '''
        if self.gbl_stiff is None or self.gbl_load is None:
            self.assemble_stiff_load()
        if self.dirichlet_applied is False:
            self.update_stiff_load_dirichlet()

        print('\n# SOLVING FOR OMEGA #\n')

        new_stiff = sy.matrix2numpy(self.gbl_stiff, dtype=float)
        new_load = sy.matrix2numpy(self.gbl_load, dtype=float)

        new_omega =  np.linalg.solve(new_stiff, new_load)

        self.gbl_omega = new_omega
        print(self.gbl_omega)

    def plot_results(self):

        if self.gbl_omega is None:  # Check if the system has been solved
            self.solve()

        x = np.zeros(self.mesh.num_nodes)
        y = np.zeros(self.mesh.num_nodes)
        z = self.gbl_omega.T

        for i in self.mesh.nodes.keys():
            x[i] = self.mesh.nodes[i].x
            y[i] = self.mesh.nodes[i].y

        xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        trias = np.zeros((self.mesh.num_elem, 3))
        for i in self.mesh.elements.keys():
            if self.mesh.elements[i].num_nodes < 6:
                pass
            else:
                for j in range(3):
                    trias[i, j] = self.mesh.elements[i].nodes[j].index

        rbf = sc.interpolate.Rbf(x, y, z, function='linear')
        zi = rbf(xi, yi)

        plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  cmap=plt.get_cmap('plasma'))
        plt.scatter(x, y, c=z, cmap=plt.get_cmap('plasma'))
        plt.colorbar()
        plt.triplot(x, y, trias, 'o-', ms=3, lw=1.0, color='black')
        plt.show()


class SteadyHeatSolver(Solver):
    '''
    2-dimensional steady state heat transfer solver.
    '''
    def __init__(self, meshfile):
        Solver.__init__(self, meshfile, "SSHeat")

    def assemble_stiff_load(self):
        '''
        Assemble the Global Stiffness Matrix and Global Load Vector based on
        elements.

        Only affects nodes pertaining to shell elements. It will overwrite nodes
        that were defined under the first assembler.
        '''

        try:
            a = self.mesh.elements[0].nodes[0].x
        except AttributeError:
            self.get_mesh()

        self.gbl_stiff = sy.zeros(self.mesh.num_nodes)
        self.gbl_load = sy.zeros(self.mesh.num_nodes, 1)
        print('\n# STARTING ASSEMBLY #\n')

        if self.mesh.calculated is None:
            self.mesh.solve_elements()

        for i in self.mesh.elements.keys():
            key = int(i)
            # To do: Create a method to provide all elements with the
            # surrounding conditions.
            self.mesh.elements[i].D = self.D
            self.mesh.elements[i].h = self.h
            self.mesh.elements[i].e = self.e
            self.mesh.elements[i].t_ext = self.t_ext

            is_point = isinstance(self.mesh.elements[i], Point1)  # Is it a Point1

            if not is_point:
                print(f"Calculating Element({key})'s Stiffness Matrix")
                # self.elements[i].solve_heat_stiff()
                print(f"Calculating Element({key})'s Load Vector")
                self.mesh.elements[i].get_C()
                # self.elements[i].solve_heat_load()

                print(f"Applying Element({key})' Globally")

                nodes_indexes = []
                for j in self.mesh.elements[i].nodes.keys():
                    # Create a list with all the element nodes indexes
                    nodes_indexes.append(self.mesh.elements[i].nodes[j].index)

                for j in range(self.mesh.elements[i].num_nodes):
                    # Assemble Stiffness matrix
                    for k in range(self.mesh.elements[i].num_nodes):
                        self.gbl_stiff[nodes_indexes[j], nodes_indexes[k]] += (
                            self.mesh.elements[i].K_e[j, k]
                            )

                    # Assemble Load vector
                    self.gbl_load[nodes_indexes[j]] += (
                        self.mesh.elements[i].F_e[j]
                        )

        return None

    def set_environment(self, t_ext, h, e, dirichlet, dirichlet_nodes, k_x,
                        k_y=None, k_xy=None):
        '''
        Provide the environment variable to the mesh

        'T_ext' the temperature of the surounding air.
        'h' the convection factors.
        'e' the thickness of the shell.

        [D] with its diffusion factors (K) will be as follows:

        [D] = [k_x  k_xy]
              [k_xy  k_y]
        '''

        print('Applying Environment')
        self.t_ext = t_ext
        self.h = h
        self.e = e
        self.dirichlet = dirichlet
        self.dirichlet_nodes = dirichlet_nodes  # table with nodes affected.

        if k_y is None:
            k_y = k_x
        if k_xy is None:
            k_xy = 0

        self.D = sy.Matrix([[k_x, k_xy], [k_xy, k_y]])
        print('Environment Applied')

    def update_stiff_load_dirichlet(self):
        '''
        Impose the 'impose' value on all nodes corresponding to value of x or y
        provided.

        This will clear the row and column associated with all nodes,
        effectively cancelling all neighboring nodes from having an impact on the dirichlet nodes.
        '''

        if self.gbl_stiff is None or self.gbl_load is None:
            self.assemble_stiff_load()

        new_gbl_stiff = self.gbl_stiff
        new_gbl_load = self.gbl_load

        print('\n# IMPOSING DIRICHLET #\n')

        for i in self.dirichlet_nodes:
            print(f"Imposing Dirichlet on Node({i}).")
            new_gbl_load -= (new_gbl_stiff[:, self.mesh.nodes[i].index]
                               * self.dirichlet)

            for j in range(self.mesh.num_nodes):
                new_gbl_stiff[self.mesh.nodes[i].index, j] = 0
                new_gbl_stiff[j, self.mesh.nodes[i].index] = 0
            new_gbl_stiff[self.mesh.nodes[i].index, self.mesh.nodes[i].index] = 1
            new_gbl_load[self.mesh.nodes[i].index] = self.dirichlet


        self.gbl_stiff = new_gbl_stiff
        self.gbl_load = new_gbl_load

        self.dirichlet_applied = True

        return None


class SteadyStructureSolver(Solver):
    '''
    2-dimensional steady state structure solver.
    '''
    def __init__(self, meshfile):
        Solver.__init__(self, meshfile, "SSMech")
        self.dof = 2

    def assemble_stiff_load(self):
        '''
        Assemble the Global Stiffness Matrix and Global Load Vector based on
        elements.

        Only affects nodes pertaining to shell elements. It will overwrite nodes
        that were defined under the first assembler.
        '''

        try:
            a = self.mesh.elements[0].nodes[0].x
        except AttributeError:
            self.get_mesh()

        self.gbl_stiff = sy.zeros(self.mesh.num_nodes * self.dof)
        self.gbl_load = sy.zeros(self.mesh.num_nodes * self.dof, 1)

        print('\n# STARTING ASSEMBLY #\n')

        if self.mesh.calculated is None:
            self.mesh.solve_elements()

        for i in self.mesh.elements.keys():
            key = int(i)
            # To do: Create a method to provide all elements with the
            # surrounding conditions.

            is_point = isinstance(self.mesh.elements[i], Point1)  # Is it a Point1

            if not is_point:
                print(f"Calculating Element({key})'s Stiffness Matrix")
                # self.elements[i].solve_heat_stiff()
                print(f"Calculating Element({key})'s Load Vector")
                self.mesh.elements[i].get_C()
                # self.elements[i].solve_heat_load()

                print(f"Applying Element({key})' Globally")

                nodes_indexes = []
                for j in self.mesh.elements[i].nodes.keys():
                    # Create a list with all the element nodes indexes
                    nodes_indexes.append(self.mesh.elements[i].nodes[j].index)

                for j in range(self.mesh.elements[i].num_nodes):
                    # Assemble Stiffness matrix
                    for k in range(self.mesh.elements[i].num_nodes):
                        self.gbl_stiff[nodes_indexes[j], nodes_indexes[k]] += (
                            self.mesh.elements[i].K_e[j, k]
                            )

                    # Assemble Load vector
                    self.gbl_load[nodes_indexes[j]] += (
                        self.mesh.elements[i].F_e[j]
                        )

        return None

    def set_environment(self, E, nu, e, P, a, assumption):
        '''
        Provide the environment variable to the mesh.
        !! This is for a Cylinder with internal pressure only !!

        'E'             Young's Modulus (N/m^2)
        'nu'            Poisson coefficient (0 < nu < 0.5)
        'e'             Body thickness (m)
        'P'             Pressure inside cylinder (N/m^2)
        'a'             Internal Diameter (m)
        'assumption'    Plain Stress or Plain Strain ('PStress' | 'PStrain')
        '''

        print('Applying Environment')
        self.E = # coding=utf-8
        self.nu = nu
        self.e = e
        self.pressure = P
        self.a = a

        print('Environment Applied')

        traction_nodes = []
        dirichlet_x_nodes = []
        dirichlet_y_nodes = []

        print('Finding all constrained nodes')

        for i in self.mesh.nodes.keys():
            x = self.mesh.nodes[i].x
            y = self.mesh.nodes[i].y


            if round((x ** 2 + y ** 2) ** 0.5, 2) == a:
                # Find all nodes which correspond to the interior of the
                # cylinder.
                traction_nodes.append(self.mesh.nodes[i].index)

            if x == 0:
                # Find all nodes which are on the y axis (blocked in
                # x-direction)
                dirichlet_x_nodes.append(self.mesh.nodes[i].index)

            if y == 0:
                # Find all nodes which are on the x axis (blocked in
                # y-direction)
                dirichlet_y_nodes.append(self.mesh.nodes[i].index)

        if len(traction_nodes) == 0:
            print('WARNING: No node has been found for the traction criteria')

        if len(dirichlet_x_nodes) == 0:
            print('WARNING: No node has been found for the Dirichlet Y-axis')

        if len(dirichlet_y_nodes) == 0:
            print('WARNING: No node has been found for the Dirichlet X-axis')

        print('Constrained nodes evaluated.')

        print('Defining D Matrix based on assumption')

        if assumption == 'PStress':
            self.D = ((self.E / (1 - self.nu ** 2))
                 * sy.Matrix([[1, self.nu, 0],
                              [self.nu, 1, 0],
                              [0, 0, (1 - self.nu)/ 2]]))

        elif assumption == 'PStrain':
            self.D = ((self.E / (1 + self.nu) * (1 - (2 * self.nu)))
                 * sy.Matrix([[1 - self.nu, self.nu, 0],
                              [self.nu, 1 - self.nu, 0],
                              [0, 0, (1 - (2 * self.nu))/ 2]]))

        else:
            raise ValueError('Assumption provided is unknown.')

        print('D Matrix defined as', assumption)

    def update_stiff_load_dirichlet(self):
        '''
        Impose the 'impose' value on all nodes corresponding to value of x or y
        provided.

        This will clear the row and column associated with all nodes,
        effectively cancelling all neighboring nodes from having an impact on the dirichlet nodes.
        '''

        if self.gbl_stiff is None or self.gbl_load is None:
            self.assemble_stiff_load()

        new_gbl_stiff = self.gbl_stiff
        new_gbl_load = self.gbl_load

        print('\n# IMPOSING DIRICHLET #\n')

        for i in self.dirichlet_nodes:
            print(f"Imposing Dirichlet on Node({i}).")
            new_gbl_load -= (new_gbl_stiff[:, self.mesh.nodes[i].index]
                               * self.dirichlet)

            for j in range(self.mesh.num_nodes):
                new_gbl_stiff[self.mesh.nodes[i].index, j] = 0
                new_gbl_stiff[j, self.mesh.nodes[i].index] = 0
            new_gbl_stiff[self.mesh.nodes[i].index, self.mesh.nodes[i].index] = 1
            new_gbl_load[self.mesh.nodes[i].index] = self.dirichlet


        self.gbl_stiff = new_gbl_stiff
        self.gbl_load = new_gbl_load

        self.dirichlet_applied = True

        return None
