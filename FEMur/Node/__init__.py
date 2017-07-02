from Element.Node1D import *
from Element.Node2D import *
from Element.Node3D import *

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
