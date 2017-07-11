from FEMur import *
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from math import ceil
from scipy.special import p_roots


class Mesh2D(Mesh):
    """
    2 Dimensional Mesh definition based on GMSH 3.0 ASCII Mesh Format.
    This class will define import GMSH ASCII Mesh.
    """

    def __init__(self, file_name, conditions):
        self.file_name = file_name
        self.d = sy.Matrix(conditions)
        self.calculated = False #  Solving has not been completed yet.

        # Define used variable as None in order to check for their definition
        # later on.
        self.nodes = None
        self.elements = None
        self.nodal_distance = None
        self.Le_container = None
        self.de_container = None

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

                    if element['type'] == 1:  # 2-node Line Element
                        self.elements[element['num']] = Line2(node_table,
                                                              element['num'])
                    elif element['type'] == 2:  # 3-node Triangle Element
                        self.elements[element['num']] = Tria3(node_table,
                                                               element['num'])
                    elif element['type'] == 3:  # 4-node Quad Element
                        self.elements[element['num']] = Quad4(node_table,
                                                               element['num'])
                    elif element['type'] == 8:  # 3-node 2nd-order Line Element
                        self.elements[element['num']] = Line3(node_table,
                                                              element['num'])
                    elif element['type'] == 9:  # 6-node 2nd-order Triangle Element
                        self.elements[element['num']] = Tria6(node_table,
                                                               element['num'])
                    elif element['type'] == 10:  # 9-node 2nd-order Quad Element
                        self.elements[element['num']] = Quad9(node_table,
                                                               element['num'])
                    elif element['type'] == 15:  # 1-node Point Element
                        self.elements[element['num']] = Point1(node_table,
                                                               element['num'])
                    elif element['type'] == 16:  # 8-node 2nd-order Quad Element
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

    def set_D_matrix(k_x, k_y=None, k_xy=None):
        '''
        Provide [D] with its diffusion factors (K)

        [D] = [k_x  k_xy]
              [k_xy  k_y]
        '''
        if k_y is None:
            k_y = k_x
        if k_xy is None:
            k_xy = k_x

        self.D = sy.Matrix([[k_x, k_xy], [k_xy, k_y]])

    def solve_elements(self):
        # Solve all current elements (shape functions, approximation, etc)

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

    def solve_stiff_int(self):
        self.stiff_int = sy.zeros(self.num_nodes)
