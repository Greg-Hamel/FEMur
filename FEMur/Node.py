import numpy as np
from scipy import pi
from scipy.special.orthogonal import p_roots
import sympy as sy
from sympy.abc import x, y
from sympy import pprint
import matplotlib.pyplot as plt
from math import *


# NODE
class Node(object):
    'Common class for all nodes of the Finite Element Method.'
    total_nodes = 0              # total number of node initialization

    def __init__(self):
        self.number = Node.total_nodes  # Defines the number of this node.
        Node.total_nodes += 1           # Counter of total number of node.

    def displayTotal():
        # Displays the total number of nodes.
        # This method should be transfered (in its current state to the
        # child classes) and be replaced by a method that gets the
        # information from the child methods and returns the total of all
        # nodes.
        if Node.total_nodes == 1:
            ('There is', Node.total_nodes, 'node.\n')
        elif Node.total_nodes > 1:
            print('There are', Node.total_nodes, 'nodes.\n')
        else:
            print('There are no nodes.\n')


class Node1D(Node):
    'Class for all 1-D Nodes'
    total_nodes = 0

    def __init__(self, x_coord, index):
        self.number = Node1D.total_nodes
        self.x = x_coord          # X coordinate of the node.
        self.dof = 1        # Degree of freedom of the node.
        self.index = index  # Index of this node in the Global Matrix.
        Node1D.total_nodes += 1

    def __sub__(self, other):
        # Substracts a Node 'x' value from another Node 'x' value
        return Node1D(self.x - other.x, self.dof, self.index)

    def __add__(self, other):
        # Add a Node 'x' value with another Node 'x' value
        return Node1D(self.x + other.x, self.dof, self.index)

    def nodeDistance(self, other):
        # Return the distance between two 1D nodes.
        return (self.x - other.x)

    def showSelf(self):
        # Print-out of all relevent information about the element.
        print(f'''Node "{self.number}" is located at:\n\tx = {self.x}\nIt has
        {self.dof} degrees of freedom.\nIt has the following
        index:\t{self.index}\n''')


class Node2D(Node):
    'Class for all 2-D Nodes'
    total_nodes = 0

    def __init__(self, x_coord, y_coord, index):
        self.number = Node2D.total_nodes
        self.x = x_coord          # X coordinate of the node.
        self.y = y_coord          # Y coordinate of the node.
        self.dof = 2        # Degree of freedom of the node.
        self.index = index  # Index of this node in the Global Matrix.
        Node2D.total_nodes += 1

    def __sub__(self, other):
        # Substracts a Node 'x' and 'y' values from another Node 'x' and 'y'
        # values
        return Node2D(self.x - other.x, self.y - other.y, self.dof, self.index)

    def __add__(self, other):
        # Add a Node 'x' and 'y' values with another Node 'x' and 'y'
        # values
        return Node2D(self.x + other.x, self.y + other.y, self.dof, self.index)

    def nodeDistance(self, other):
        # Return the distance between two 2D nodes.
        return (((self.x - other.x) ** 2) + ((self.y - other.y) ** 2)) ** 0.5

    def showSelf(self):
        # Print-out of all relevent information about the element.
        print(f'''Node "{self.number}" has the following
        coordinates:\n\t({self.x}, {self.y})\nIt has {self.dof} degrees of
        freedom.\nIt has the following index:\t{self.index}\n''')


class Node3D(Node):
    'Class for all 3-D Nodes'
    total_nodes = 0

    def __init__(self, x_coord, y_coord, z_coord, index):
        self.number = Node3D.total_nodes
        self.x = x_coord          # X coordinate of the node.
        self.y = y_coord          # Y coordinate of the node.
        self.z = z_coord          # Z coordinate of the node.
        self.dof = 3        # Degree of freedom of the node.
        self.index = index  # Index of this node in the Global Matrix.
        Node3D.total_nodes += 1

    def __sub__(self, other):
        # Substracts a Node 'x', 'y' and 'z' values from another Node 'x', 'y'
        # and 'z' values
        return Node(self.x - other.x, self.y - other.y, self.z - other.z,
                    self.dof, self.index)

    def __add__(self, other):
        # Add a Node 'x', 'y' and 'z' values with another Node 'x', 'y' and 'z'
        # values
        return Node(self.x + other.x, self.y + other.y, self.z + other.z,
                    self.dof, self.index)

    def nodeDistance(self, other):
        # Return the distance between two 3D nodes.
        return (((self.x - other.x) ** 2) + ((self.y - other.y) ** 2)
                + ((self.z - other.z) ** 2)) ** 0.5

    def showSelf(self):
        # Print-out of all relevent information about the element.
        print(f'''Node "{self.number}" has thefollowing coordinates:\n\t
                  ({self.x}, {self.y},{self.z})\nIt has {self.dof} degrees of
                  freedom.\nIt has the following index:\t{self.index}\n''')
