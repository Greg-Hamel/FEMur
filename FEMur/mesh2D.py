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
        self.nodes = None
        self.nodal_distance = None

        self.nodes_start = None
        self.nodes_end = None

        self.elements_start = None
        self.elements_end = None

    def find_nodes_range(self):
        gmshFile = open(file_name, 'r')
        record_node = False
        for line in gmshFile:
            if line == '$EndNodes\n':
                record_node = False
            elif record_node:
                #TBD
            elif line == '$Nodes\n':
                record_node = True
            else:
                pass
            
    def show_nodes(self):
        if self.nodes is None:
            raise ValueError('Nodes have not been assigned yet. Please create'
                             ' nodes using Node2D.get_ref_nodes()')
        elif type(self.nodes) == dict:
            print(len(self.nodes), "nodes in total.")
            for i in self.nodes.keys():
                print(self.nodes[i])
