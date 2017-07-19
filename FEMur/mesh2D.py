from FEMur import *
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
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
                    self.nodes[node['num']] = Node2D(float(node['x']),
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
                        node_number = str(i)
                        element[node_number] = line[node_index_start + i]
                        node_table.append(self.nodes[element[node_number]])

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

            is_point = isinstance(self.elements[i], Point1)  # Is it a Point1
            is_line2 = isinstance(self.elements[i], Line2)  # Is it a Line2
            is_line3 = isinstance(self.elements[i], Line3)  # Is it a Line3

            if not is_point and not is_line2 and not is_line3:
                # If something else than a point or a line.
                print(f"Calculating Element({key})'s shape functions"
                      f" derivatives")
                self.elements[i].get_Be()

            self.calculated = True

    def assemble_stiff_load(self):
        '''
        Add Description !!!
        '''

        self.gbl_stiff = sy.zeros(self.num_nodes)
        self.gbl_load = sy.zeros(self.num_nodes, 1)
        self.gbl_omega = sy.zeros(self.num_nodes, 1)
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
            is_line2 = isinstance(self.elements[i], Line2)  # Is it a Line2
            is_line3 = isinstance(self.elements[i], Line3)  # Is it a Line3

            if not is_point:
                print(f"Calculating Element({key})'s Stiffness Matrix")
                self.elements[i].solve_heat_stiff()
                print(f"Calculating Element({key})'s Load Vector")
                self.elements[i].solve_heat_load()

            print(f"Applying Element({key})'s Stiffness Matrix and Load Vector"
                  f"to the global Stiffness Matrix and Global Load Vector")
            if not is_point and not is_line2 and not is_line3:
                # If something else than a point or a line.
                nodes_indexes = []
                for j in self.elements[i].nodes.keys():
                    # Create a list with all the element nodes indexes
                    nodes_indexes.append(self.elements[i].nodes[j].index)

                for j in range(self.elements[i].num_nodes):
                    # Assemble Stiffness matrix
                    for k in range(self.elements[i].num_nodes):
                        self.gbl_stiff[nodes_indexes[j], nodes_indexes[k]] = (
                            self.gbl_stiff[nodes_indexes[j], nodes_indexes[k]]
                            + self.elements[i].K_e[j, k]
                            )

                    # Assemble Load vector
                    self.gbl_load[nodes_indexes[j]] = (
                        self.gbl_load[nodes_indexes[j]]
                        + self.elements[i].F_e[j]
                        )

        return None

    def update_stiff_load_cauchy(self):
        '''
        Add Description !!!

        Only affects nodes pertaining to line elements. It will overwrite nodes
        that were defined under the first assembler.
        '''
        if self.gbl_stiff is None:
            self.assemble_stiff_load()

        new_gbl_stiff = self.gbl_stiff
        new_gbl_load = self.gbl_load
        print('\n# UPDATING LINE ELEMENTS WITH CAUCHY #\n')

        for i in self.elements.keys():
            # Find all nodes related to the cauchy conditions and
            # reset them to zero.
            key = int(i)


            is_line2 = isinstance(self.elements[i], Line2)  # Is it a Line2
            is_line3 = isinstance(self.elements[i], Line3)  # Is it a Line3

            if is_line2 or is_line3:
                print(f"Resetting Element({key})'s Nodes for Cauchy")
                # If a line.
                nodes_indexes = []
                for j in self.elements[i].nodes.keys():
                    # Create a list with all the element nodes indexes
                    nodes_indexes.append(self.elements[i].nodes[j].index)

                for j in nodes_indexes:
                    for k in nodes_indexes:  # Reset line nodes in stiffness
                        new_gbl_stiff[j, k] = 0

                    new_gbl_load[j] = 0  # Reset line nodes in load vector

        for i in self.elements.keys():
            key = int(i)

            is_line2 = isinstance(self.elements[i], Line2)  # Is it a Line2
            is_line3 = isinstance(self.elements[i], Line3)  # Is it a Line3

            if is_line2 or is_line3:
                # If a line.
                print(f"Applying Cauchy Condition to Element({key})'s Nodes.")
                nodes_indexes = []
                for j in self.elements[i].nodes.keys():
                    # Create a list with all the element nodes indexes
                    nodes_indexes.append(self.elements[i].nodes[j].index)

                for j in range(self.elements[i].num_nodes):
                    # Update lines nodes with cauchy condition in stiffness
                    for k in range(self.elements[i].num_nodes):
                        new_gbl_stiff[nodes_indexes[j], nodes_indexes[k]] = (
                            new_gbl_stiff[nodes_indexes[j], nodes_indexes[k]]
                            + self.elements[i].K_e[j, k]
                            )

                    new_gbl_load[nodes_indexes[j]] = (
                        new_gbl_load[nodes_indexes[j]]
                        + self.elements[i].F_e[j]
                        )

        self.cauchy_applied = True
        self.gbl_stiff = new_gbl_stiff
        self.gbl_load = new_gbl_load

        return None

    def update_stiff_load_dirichlet(self, impose, x=None, y=None):
        '''
        Impose the 'impose' value on all nodes corresponding to value of x or y
        provided.

        This will clear the row and column associated with all nodes,
        effectively cancelling all neighboring nodes from having an impact on the dirichlet nodes.
        '''
        if x == y and x is None:
            raise ValueError('No coordinates were provided.')
        elif x is not None and y is not None:
            if x < 0 or y < 0:
                raise ValueError('One or more coordinate(s) is negative.')

        if self.gbl_stiff is None or self.gbl_load is None:
            self.assemble_stiff_load()
        if self.cauchy_applied is False:
            self.update_stiff_load_cauchy()

        new_gbl_stiff = self.gbl_stiff
        new_gbl_load = self.gbl_load
        new_gbl_omega = self.gbl_omega

        print('\n# IMPOSING DIRICHLET #\n')

        if x is not None:
            for i in self.nodes.keys():
                if self.nodes[i].x == x:
                    key = int(i)
                    print(f"Imposing Dirichlet on Node({key}).")
                    new_gbl_stiff.row_op(self.nodes[i].index, lambda i, j: 0)
                    new_gbl_stiff.col_op(self.nodes[i].index, lambda i, j: 0)
                    new_gbl_stiff[self.nodes[i].index, self.nodes[i].index] = 1

                    new_gbl_load[self.nodes[i].index] = 0

                    new_gbl_omega[self.nodes[i].index] = impose

        elif y is not None:
            for i in self.nodes.keys():
                if self.nodes[i].y == y:
                    key = int(i)
                    print(f"Imposing Dirichlet on Node({key}).")
                    new_gbl_stiff.row_op(self.nodes[i].index, lambda i, j: 0)
                    new_gbl_stiff.col_op(self.nodes[i].index, lambda i, j: 0)
                    new_gbl_stiff[self.nodes[i].index, self.nodes[i].index] = 1

                    new_gbl_load[self.nodes[i].index] = 0

                    new_gbl_omega[self.nodes[i].index] = impose

        else:  # x and y and not None
            for i in self.nodes.keys():
                if self.nodes[i].y == y and self.nodes[i].x == x:
                    key = int(i)
                    print(f"Imposing Dirichlet on Node({key}).")
                    new_gbl_stiff.row_op(self.nodes[i].index, lambda i, j: 0)
                    new_gbl_stiff.col_op(self.nodes[i].index, lambda i, j: 0)
                    new_gbl_stiff[self.nodes[i].index, self.nodes[i].index] = 1

                    new_gbl_load[self.nodes[i].index] = 0

                    new_gbl_omega[self.nodes[i].index] = impose

        self.gbl_stiff = new_gbl_stiff
        self.gbl_load = new_gbl_load
        self.gbl_omega = new_gbl_omega

        self.dirichlet_applied = True

        return None

    def set_environment(self, t_ext, h, e, dirichlet, k_x, k_y=None, k_xy=None,
                        d_x=None, d_y=None):
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
        self.dir_x = d_x
        self.dir_y = d_y

        if k_y is None:
            k_y = k_x
        if k_xy is None:
            k_xy = 0

        self.D = sy.Matrix([[k_x, k_xy], [k_xy, k_y]])


    def solve_omega(self):
        if self.gbl_stiff is None or self.gbl_load is None:
            self.assemble_stiff_load()
        if self.cauchy_applied is False:
            self.update_stiff_load_cauchy()
        if self.dirichlet_applied is False:
            self.update_stiff_load_dirichlet(self.dirichlet,
                                             self.dir_x,
                                             self.dir_y)

        print('ok')
        sy.pprint(self.gbl_stiff)

        print('\n# SOLVING FOR OMEGA #\n')

        new_omega = self.gbl_omega

        new_omega = (self.gbl_stiff ** -1) * gbl_load

        self.gbl_omega = new_omega

    def resolve(self):
        pass
