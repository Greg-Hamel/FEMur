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

    def __init__(self, file_name):
        self.file_name = file_name
        self.nodes = None
        self.elements = None
        self.nodal_distance = None

    def get_nodes_files(self):
        self.nodes = {}
        gmshFile = open(self.file_name, 'r')
        record_node = False
        for line in gmshFile:  # Reads the document line by line
            line = line.strip('\n')
            if line == '$EndNodes': # Break when end of Nodes Range is found
                break
            elif record_node: # Add current node to Nodes dict
                node = {}
                node['num'], node['x'], node['y'], node['z'] = line.split(' ')
                self.nodes[node['num']] = Node2D(float(node['x']),
                                                 float(node['y']),
                                                 int(node['num']))
            elif line == '$Nodes':  # Set Record_node Flag
                record_node = True
            else:
                pass

        return None

    def get_elements_files(self):
        self.elements = {}
        gmshFile = open(self.file_name, 'r')
        record_node = False
        for line in gmshFile:  # Reads the document line by line
            line = line.strip('\n')
            if line == '$EndElements': # Break when end of Nodes Range is found
                break
            elif record_node: # Add current node to Nodes dict
                element = {}
                node_table = []
                line = line.split(' ')

                element['num'] = int(line[0])
                element['type'] = int(line[1])
                element['tag_number'] = int(line[2])

                for i in range(line[2]):
                    tag_list = ['phys_elem_number', 'geo_elem_number']
                    element[tag_list[i]] = int(line[3+i])

                for i in range(element['phys_elem_number']):
                    node_number = 'node' + str(i)
                    element[node_number] = int(line(3+element['tag_number']+i))
                    node_table.append(element[node_number])

                if element['type'] == 1:  # 2-node Line Element
                    self.elements[element['num']] = Line2(node_table,
                                                          node['num'])
                elif element['type'] == 2:  # 3-node Triangle Element
                    self.elements[element['num']] = Tria3(node_table,
                                                           node['num'])
                elif element['type'] == 3:  # 4-node Quad Element
                    self.elements[element['num']] = Quad4(node_table,
                                                           node['num'])
                elif element['type'] == 8:  # 3-node 2nd-order Line Element
                    self.elements[element['num']] = Line3(node_table,
                                                          node['num'])
                elif element['type'] == 9:  # 6-node 2nd-order Triangle Element
                    self.elements[element['num']] = Tria6(node_table,
                                                           node['num'])
                elif element['type'] == 10:  # 9-node 2nd-order Quad Element
                    self.elements[element['num']] = Quad9(node_table,
                                                           node['num'])
                elif element['type'] == 15:  # 1-node Point Element
                    self.elements[element['num']] = Point(node_table,
                                                          node['num'])
                elif element['type'] == 16:  # 8-node 2nd-order Quad Element
                    self.elements[element['num']] = Quad8(node_table,
                                                           node['num'])
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
            raise ValueError('Nodes have not been assigned yet. Please create'
                             ' nodes using Node2D.get_ref_nodes()')
        elif type(self.nodes) == dict:
            print(len(self.nodes), "nodes in total.")
            for i in self.nodes.keys():
                print(self.nodes[i])
