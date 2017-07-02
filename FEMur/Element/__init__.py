from Element.Element1D import *
from Element.Element2D import *
from Element.Element3D import *


'Common class for all elements of the Finite Element Method.'
total_elements = 0

def __init__(self, node_table):
    self.number = Element.total_elements
    Element.total_elements += 1

    self.nodes = {}  # Dictionary with nodes
    for i in range(self.num_nodes):
        self.nodes[str(i)] = node_table[i]
