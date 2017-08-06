from FEMur import *
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
from math import ceil


class Mesh2D(Mesh):
    """
    2 Dimensional Mesh definition based on GMSH 3.0 ASCII Mesh Format.
    This class will define import GMSH ASCII Mesh.
    """

    def __init__(self, file_name, conditions):
        self.file_name = file_name
        self.d = sy.Matrix(conditions)
        self.calculated = False #  Solving has not been completed yet.
        self.cauchy_applied = False
        self.dirichlet_applied = False

        # Define used variable as None in order to check for their definition
        # later on.
        self.nodes = None
        self.elements = None
        self.nodal_distance = None
        self.Le_container = None
        self.de_container = None
        self.D = None
        self.gbl_stiff = None
        self.gbl_load = None
        self.gbl_omega = None

    def __str__(self):
        if self.elements is None:
            raise ValueError('No mesh has been created yet.')
        else:
            output = ''
            for i in self.elements.keys():
                output = output + str(self.elements[i]) + '\n'

            return output

    def get_nodes_files(self):
        '''
        Import nodes from a GMSH generated '.msh' file.

        Nodes in a '.msh' are delimited by two strings
        '$Nodes' is the start of the nodes range
        '$EndNodes' is the end of the nodes range
        All nodes are listed between these two strings as described in
        http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format
        '''

        self.nodes = {}
        gmshFile = open(self.file_name, 'r')
        record_node = False
        for line in gmshFile:  # Reads the document line by line
            line = line.strip('\n')
            if line == '$EndNodes': # Break when end of Nodes Range is found
                break
            elif record_node: # Add current node to Nodes dict
                node = {}
                if len(line.split(' ')) == 1:
                    pass
                else:
                    node['num'], node['x'], node['y'], node['z'] = line.split(' ')
                    self.nodes[int(node['num']) - 1] = Node2D(float(node['x']),
                                                     float(node['y']),
                                                     int(node['num']) - 1)
            elif line == '$Nodes':  # Set Record_node Flag
                record_node = True
            else:
                pass

        self.num_nodes = len(self.nodes)
        return None


    def get_elements_files(self):
        '''
        Import elements from a GMSH generated '.msh' file.

        Elements in a '.msh' are delimited by two strings
        '$Elements' is the start of the elements range
        '$EndElements' is the end of the elements range
        All elements are listed between these two strings as described in
        http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format
        '''

        self.elements = {}
        gmshFile = open(self.file_name, 'r')
        record_node = False
        for line in gmshFile:  # Reads the document line by line
            line = line.strip('\n')
            if line == '$EndElements':  # Break when end of Nodes Range is found
                break
            elif record_node:  # Add current node to Nodes dict
                element = {}
                node_table = []
                line = line.split(' ')
                if len(line) == 1:
                    pass
                else:
                    element['num'] = int(line[0]) - 1  # number '1' -> index '0'
                    element['type'] = int(line[1])
                    element['tag_number'] = int(line[2])

                    for i in range(int(line[2])):
                        # tags definition
                        tag_list = ['phys_elem_number', 'geo_elem_number']
                        element[tag_list[i]] = int(line[3+i])

                    node_index_start = 3 + element['tag_number']

                    for i in range(len(line) - node_index_start):
                        # node_table creation
                        element[i] = int(line[node_index_start + i]) - 1
                        node_table.append(self.nodes[element[i]])

                    if element['type'] == 1:
                        # 2-node Line Element
                        self.elements[element['num']] = Line2(node_table,
                                                              element['num'])
                    elif element['type'] == 2:
                        # 3-node Triangle Element
                        self.elements[element['num']] = Tria3(node_table,
                                                              element['num'])
                    elif element['type'] == 3:
                        # 4-node Quad Element
                        self.elements[element['num']] = Quad4(node_table,
                                                              element['num'])
                    elif element['type'] == 8:
                        # 3-node 2nd-order Line Element
                        self.elements[element['num']] = Line3(node_table,
                                                              element['num'])
                    elif element['type'] == 9:
                        # 6-node 2nd-order Triangle Element
                        self.elements[element['num']] = Tria6(node_table,
                                                              element['num'])
                    elif element['type'] == 10:
                        # 9-node 2nd-order Quad Element
                        self.elements[element['num']] = Quad9(node_table,
                                                              element['num'])
                    elif element['type'] == 15:
                        # 1-node Point Element
                        self.elements[element['num']] = Point1(node_table,
                                                               element['num'])
                    elif element['type'] == 16:
                        # 8-node 2nd-order Quad Element
                        self.elements[element['num']] = Quad8(node_table,
                                                              element['num'])
                    elif element['type'] >= 1 or element['type'] <= 31:
                        e_type = element['type']
                        raise ValueError(f'Unsupported element type. {e_type} is'
                                          ' not supported by FEMur yet. See the'
                                          ' documentation for a list of all'
                                          ' available element types.')
                    elif element['type'] < 1 or element['type'] > 31:
                        e_type = element['type']
                        raise ValueError(f'Unexpected element type. {e_type} is'
                                          ' within the 1 - 31 range of nodes used'
                                          ' by GMSH.')
                    else:
                        e_type = element['type']
                        raise ValueError(f'Unexpected element type. {e_type} is'
                                          ' not an integer.')

            elif line == '$Elements':  # Set Record_node Flag
                record_node = True
            else:
                pass

        self.num_elem = len(self.elements)
        return None

    def mesh(self):
        print('Importing nodes from file.')
        self.get_nodes_files()
        print('Nodes import complete.')
        print('Importing elements from file.')
        self.get_elements_files()
        print('Elements import complete.')

    def show_nodes(self):
        if self.nodes is None:
            raise ValueError('Nodes have not been assigned yet. Please import'
                             ' nodes using Node2D.get_nodes_files()')
        elif type(self.nodes) == dict:
            print(len(self.nodes), "nodes in total.")
            for i in self.nodes.keys():
                print(self.nodes[i])

    def show_elements(self):
        if self.elements is None:
            raise ValueError('Elements have not been assigned yet. Please'
                             ' import elements using'
                             ' Node2D.get_elements_files()')
        elif type(self.elements) == dict:
            print(len(self.elements), "nodes in total.")
            for i in self.elements.keys():
                print(self.elements[i])

    def get_Le_container(self):
        '''
        Creates an class-specific Le_container dictionary.

        Le_container contains as many matrices as there are elements.
        Le_container keys are the elements index (or number)

        Creates matrices like the following for each element.

        [1 0 0 0 0 0]    [0 0 0 1 0 0]
        [0 1 0 0 0 0] or [1 0 0 0 0 0] depending on the element's relationship
        [0 0 1 0 0 0]    [0 0 1 0 0 0] with its nodes

        These matrics identify what nodes the element is acting upon. The size
        of the L_e matrix is <Number of nodes in element> x <number ofnodes in
        total>

        This is useful when determining the outside forces applied onto a
        structure for example. The d_e equal the dot-product of the L_e
        matrix and the d matrix like:

        [d_e] = [L_e] * [d]
        '''

        if self.elements is None:
            raise ValueError('Elements have not been assigned yet. Please'
                             ' import elements using'
                             ' Node2D.get_elements_files()')
        Le = {}
        for i in self.elements.keys():
            Le[i] = sy.zeros(self.elements[i].num_nodes, len(self.nodes))
            for j in self.elements[i].nodes.keys():
                Le[i][int(j), self.elements[i].nodes[j].index] = 1

        self.Le_container = Le

    def get_de_container(self):
        '''
        Get the dictionary which contains all d_e matrices

        '''

        de = {}
        if self.Le_container is None:
            self.get_Le_container()

        for i in self.elements.keys():
            de[i] = self.Le_container[i] * self.d

        self.de = de

    def solve_elements(self):
        '''
        Add Description !!!
        '''

        for i in self.elements.keys():
            key = int(i)
            self.elements[i]

            print(f"Calculating Element({key})'s shape functions")
            self.elements[i].get_Ne_ref()

            validation = self.elements[i].validate_Ne_ref()
            print(f'Validation of shape function is: {validation}')

            self.calculated = True

    def assemble_stiff_load(self):
        '''
        Assemble the Global Stiffness Matrix and Global Load Vector based on
        elements.

        Only affects nodes pertaining to shell elements. It will overwrite nodes
        that were defined under the first assembler.
        '''

        self.gbl_stiff = sy.zeros(self.num_nodes)
        self.gbl_load = sy.zeros(self.num_nodes, 1)
        print('\n# STARTING ASSEMBLY #\n')

        for i in self.elements.keys():
            key = int(i)
            # To do: Create a method to provide all elements with the
            # surrounding conditions.
            self.elements[i].D = self.D
            self.elements[i].h = self.h
            self.elements[i].e = self.e
            self.elements[i].t_ext = self.t_ext

            is_point = isinstance(self.elements[i], Point1)  # Is it a Point1

            if not is_point:
                print(f"Calculating Element({key})'s Stiffness Matrix")
                # self.elements[i].solve_heat_stiff()
                print(f"Calculating Element({key})'s Load Vector")
                self.elements[i].get_C()
                # self.elements[i].solve_heat_load()

                print(f"Applying Element({key})'s Stiffness Matrix and Load Vector"
                  f"to the global Stiffness Matrix and Global Load Vector")

                nodes_indexes = []
                for j in self.elements[i].nodes.keys():
                    # Create a list with all the element nodes indexes
                    nodes_indexes.append(self.elements[i].nodes[j].index)

                for j in range(self.elements[i].num_nodes):
                    # Assemble Stiffness matrix
                    for k in range(self.elements[i].num_nodes):
                        self.gbl_stiff[nodes_indexes[j], nodes_indexes[k]] += (
                            self.elements[i].K_e[j, k]
                            )

                    # Assemble Load vector
                    self.gbl_load[nodes_indexes[j]] += (
                        self.elements[i].F_e[j]
                        )

        return None

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
            new_gbl_load -= (new_gbl_stiff[:, self.nodes[i].index]
                               * self.dirichlet)

            for j in range(self.num_nodes):
                new_gbl_stiff[self.nodes[i].index, j] = 0
                new_gbl_stiff[j, self.nodes[i].index] = 0
            new_gbl_stiff[self.nodes[i].index, self.nodes[i].index] = 1
            new_gbl_load[self.nodes[i].index] = self.dirichlet


        self.gbl_stiff = new_gbl_stiff
        self.gbl_load = new_gbl_load

        self.dirichlet_applied = True

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


    def solve_omega(self):
        if self.gbl_stiff is None or self.gbl_load is None:
            self.assemble_stiff_load()
        if self.dirichlet_applied is False:
            self.update_stiff_load_dirichlet()

        sy.pprint(self.elements[4].K_e)
        sy.pprint(self.gbl_stiff)
        sy.pprint(self.gbl_load)
        print('\n# SOLVING FOR OMEGA #\n')

        new_stiff = sy.matrix2numpy(self.gbl_stiff, dtype=float)
        new_load = sy.matrix2numpy(self.gbl_load, dtype=float)

        new_omega =  np.linalg.solve(new_stiff, new_load)

        self.gbl_omega = new_omega
        print(self.gbl_omega)

    def plot_results(self):
        if self.gbl_omega is None:
            self.solve_omega()

        x = np.zeros(self.num_nodes)
        y = np.zeros(self.num_nodes)
        z = self.gbl_omega.T
        for i in self.nodes.keys():
            x[i] = self.nodes[i].x
            y[i] = self.nodes[i].y

        xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        trias = np.zeros((self.num_elem, 3))
        for i in self.elements.keys():
            if self.elements[i].num_nodes < 6:
                pass
            else:
                for j in range(3):
                    trias[i, j] = self.elements[i].nodes[j].index

        rbf = sc.interpolate.Rbf(x, y, z, function='linear')
        zi = rbf(xi, yi)

        plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  cmap=plt.get_cmap('plasma'))
        plt.scatter(x, y, c=z, cmap=plt.get_cmap('plasma'))
        plt.colorbar()
        plt.triplot(x, y, trias, 'o-', ms=3, lw=1.0, color='black')
        plt.show()
