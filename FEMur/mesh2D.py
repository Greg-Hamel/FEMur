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
        self.nodal_distance = None

        self.nodes_start = None
        self.nodes_end = None

        self.elements_start = None
        self.elements_end = None

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
                self.nodes[node['num']] = Node2D(node['x'], node['y'],
                                                 node['num'])
            elif line == '$Nodes':  # Set Record_node Flag
                record_node = True
            else:
                pass

        return None

    def show_nodes(self):
        if self.nodes is None:
            raise ValueError('Nodes have not been assigned yet. Please create'
                             ' nodes using Node2D.get_ref_nodes()')
        elif type(self.nodes) == dict:
            print(len(self.nodes), "nodes in total.")
            for i in self.nodes.keys():
                print(self.nodes[i])
