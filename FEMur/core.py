"""
FEM.py

This module introduces a few concepts of the Finite Element Method (FEM)and
aims at providing a number of tools for solving FEM-related problems.

Its development was started in order to solve problems and projects related to
the SYS806 'Application of the Finite Element Method' Class at 'Ecole de
technolologie superieur de Montreal' for the 2017 summer semester. I intend to
continue its development after finishing the course in order to provide an
easy-to-use solver for FEM problems that is accessible for most people.

This is in no-way a complete project as of yet. There is a lot more to come.
"""

__version__ = '0.1'
__author__ = 'Gregory Hamel'

import numpy as np
from scipy import pi
from scipy.special.orthogonal import p_roots
import sympy as sy
from sympy.abc import x, y

# NODE


class Node(object):
    'Common class for all nodes of the Finite Element Method.'
    total_nodes = 0              # total number of node initialization

    def __init__(self):
        self.number = Node.total_nodes  # Defines the number of this node.
        Node.total_nodes += 1           # Counter of total number of node.

    def displayTotal():
        # Displays the total number of nodes.
        # This function should be transfered (in its current state to the
        # child classes) and be replaced by a function that gets the
        # information from the child functions and returns the total of all
        # nodes.
        if Node.total_nodes == 1:
            print('There is', Node.total_nodes, 'node.\n')
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
        return Node1D(self.x - other.x, self.dof, self.index)

    def __add__(self, other):
        return Node1D(self.x + other.x, self.dof, self.index)

    def nodeDistance(self, other):
        # Return the distance between two nodes.
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
        return Node2D(self.x - other.x, self.y - other.y, self.dof, self.index)

    def __add__(self, other):
        return Node2D(self.x + other.x, self.y + other.y, self.dof, self.index)

    def nodeDistance(self, other):
        # Return the distance between two nodes.
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
        return Node(self.x - other.x, self.y - other.y, self.z - other.z,
                    self.dof, self.index)

    def __add__(self, other):
        return Node(self.x + other.x, self.y + other.y, self.z + other.z,
                    self.dof, self.index)

    def nodeDistance(self, other):
        # Return the distance between two nodes.
        return (((self.x - other.x) ** 2) + ((self.y - other.y) ** 2)
                + ((self.z - other.z) ** 2)) ** 0.5

    def showSelf(self):
        # Print-out of all relevent information about the element.
        print(f'''Node "{self.number}" has thefollowing coordinates:\n\t
                  ({self.x}, {self.y},{self.z})\nIt has {self.dof} degrees of
                  freedom.\nIt has the following index:\t{self.index}\n''')


# ELEMENT


class Element(object):
    'Common class for all elements of the Finite Element Method.'
    total_elements = 0

    def __init__(self):
        self.number = Element.total_elements
        Element.total_elements += 1


class Element1D(Element):
    'Defines the Linear Elements with its nodes and shape functions'
    total_elements = 0

    def __init__(self, node_table, number_gauss_point=None):
        self.ngp = number_gauss_point       # Gauss Points to be evaluated.

        self.nodes = node_table             # Table with nodes
        self.nnodes = len(node_table)       # Linear elements = 2 nodes
        self.ndof = (self.nodes[0]).dof     # ndof-D analysis
        self.L_e = self.nodes[-1].nodeDistance(self.nodes[0])

        self.N = None

        self.number = Element1D.total_elements
        Element1D.total_elements += 1       # Counter for number of elements.

    def __str__(self):
        nodes_str = f''
        for i in range(self.nnodes):
            if nodes_str == '':
                nodes_str = f'{self.nodes[i].number}'
            else:
                nodes_str = nodes_str + f', {self.nodes[i].number}'
        output_str = f'Element({self.number}) composed of Nodes({nodes_str})'
        return output_str

    def displayTotal():
        # Displays the total number of elements.
        if Linearelement.total_elements == 1:
            print('There is', Linearelement.total_elements, 'element.\n')
        elif Linearelement.total_elements > 1:
            print('There are', Linearelement.total_elements, 'elements.\n')
        else:
            print('There are no elements.\n')

    def showSelf(self):
        # Print-out of all relevent information about the element.
        print(f'''Element "{self.number}" is linked to the following
        nodes:\n\t({self.nodes[0].number}, {self.nodes[1].number})''')
        print('K =', self.k)
        print('h =', self.h, '\n')

    def getIndex(self):
        # Gets the index for the particular element in the global matrix.
        index = np.zeros(self.nnodes * self.ndof)
        for i in range(self.nnodes):
            for n in range(self.ndof):
                index[i] = self.nodes[i].index + n

        return index

    def jacobien(self):
        return np.array([0.5 * self.nodes[0].nodeDistance(self.nodes[1])])

    def detJacobien(self, jacobien):
        return np.linalg.det(jacobien)

    def getK(self, E, A):
        # Gets the stiffeness matrix.
        # This should be modified to compute the matrix on its own.
        # (Automate the process)
        k_elements = np.zeros((self.nnodes * self.ndof,
                               self.nnodes * self.ndof))

        for i in range(2):
            for n in range(2):
                if i == n:
                    k_elements[i, n] = (-1) * (E * A) / self.L_e
                else:
                    k_elements[i, n] = E * A / self.L_e

        return k_elements

    def getF(self, rho, A, g):
        # This should be modified to compute the vecteur on its own.
        # (Automate the process)
        f_elements = np.zeros((self.nnodes * self.ndof, 1))

        for i in range(len(f_elements)):
            f_elements[i] = rho * A * g * self.L_e / 2

        return f_elements

    def getR(self, rho, M, g, E, A):
        r_elements = np.zeros((self.nnodes * self.ndof, 1))
        r_elements[0] = ((-1) * (rho * A * g * self.L_e/2)) + (E*A/self.L_e)
        r_elements[1] = M * g


class Element2D(Element):
    'Defines the Planar Elements with its nodes and shape functions'
    total_elements = 0
    triangle_nodes = [3, 4, 6, 7, 9]

    def __init__(self, element_type, nodes, edge_nodes=False,
                 center_node=False):
        self.number = Element2D.total_elements
        self.e_type = element_type        # 'T' for triangle, 'Q' for quad
        self.e_nodes_number = len(nodes)  # The number of nodes per element
        self.nodes = nodes                # Nodes table
        self.edge = edge_nodes            # Edge Nodes or not.
        self.element_type = self.e_type + str(self.e_nodes_number)
        self.centered = center_node
        Element2D.total_elements += 1

    def get_ref_values(self):
        """Provide an array with 'x' and 'y' nodal values of the shape
        function."""
        if self.e_type == 'T':   # Triangle element
            if self.e_nodes_number not in self.triangle_nodes:
                Err = 'The number of nodes is not resolvable for now. Only 3,'\
                      '4, 6, or 7 nodes possible per triangle elements.'
                print(Err)
                return None

            self.table_x_ref = np.array([0, 1, 0])
            self.table_y_ref = np.array([0, 0, 1])
            if self.edge is True and self.centered is False:  # Edge nodes
                nodes_per_side = int((self.e_nodes_number - 3) / 3)
                spacing = 1.0 / (nodes_per_side + 1.0)
                for i in range(3):   # Iterate over all sides
                    for n in range(nodes_per_side):  # Iterate over all nodes
                        if i == 0:
                            self.table_x_ref = np.append(
                                self.table_x_ref,
                                np.array(spacing * (n + 1))
                            )
                            self.table_y_ref = np.append(
                                self.table_y_ref, np.array(0)
                            )
                        elif i == 1:
                            self.table_x_ref = np.append(
                                self.table_x_ref, np.array(
                                    spacing * ((nodes_per_side - (n + 1)) + 1)
                                )
                            )
                            self.table_y_ref = np.append(
                                self.table_y_ref, np.array(spacing * (n + 1))
                            )
                        elif i == 2:
                            self.table_x_ref = np.append(
                                self.table_x_ref, np.array(0)
                            )
                            self.table_y_ref = np.append(
                                self.table_y_ref, np.array(
                                    spacing * ((nodes_per_side - (n + 1)) + 1)
                                )
                            )
                        else:
                            Err = "Nodes are trying to be created on a "\
                                  "fourth triangle side which doesn't exist."
                            print(Err)
                            return None

        return self.table_x_ref, self.table_y_ref

    def get_eta(self):
        pass

    def get_ksi(self):
        pass


class Triangular(Element2D):
    'Common class for all Triangular 2D elements'
    def __init__(self):
        pass

    def shape(self):
        for i in range(self.num_dots):
            self.shape[i, :] = self.basis  # sub-in ksi_ref and eta_ref

        # calculate inverse of self.shape
        # shape_inv =

        # Calculate the shape functions matrix (N)
        # self.N = self.basis * shape_inv

    def validate_shape(self):
        pass
        # To be completed later on. Based on main2D.m (Cours 6)
        # for i in range(self.num_dots):

    def derive_shape(self):
        B = zeros((2, self.num_dots))
        B[1, :] = sy.diff(self.shape)

        self.derived_shape


class T6(Triangular):
    eta = sy.symbols('eta')
    ksi = sy.symbols('ksi')

    def __init__(self):
        self.basis = np.array([1, ksi, eta, ksi * eta, ksi * ksi, eta * eta])
        self.ksi_ref = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
        self.eta_ref = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 0.5])
        self.num_dots = len(self.ksi_ref)
        self.shape = np.zeros((self.num_dots, self.num_dots))


class Quad(Element2D):
    'Common class for all Quad 2D elements'
    def __inti__(self):
        pass


# Mesh
class Mesh(object):
    'Common class for all Mesh of the Finite Element Method.'
    def __init__(self):
        pass


class Mesh1D(Mesh):
    '1 Dimensional Mesh.'
    def __init__(self, domain, Number_of_elements, Nodes_per_element):
        self.start = domain[0]                   # First 'x' of the domain
        self.end = domain[1]                     # Last 'x' of the domain
        self.num_elements = Number_of_elements   # Number of elements
        self.nodes_elements = Nodes_per_element  # Number of nodes per element

        self.length = self.end - self.start      # Length of the domain
        self.num_nodes = (
                          self.num_elements + 1
                          + self.num_elements * (self.nodes_elements - 2)
        )
        self.L_element = self.length / self.num_elements
        self.inside_node_distance = self.L_element / (self.nodes_elements - 1)

        self.meshing = None

    def __str__(self):
        if self.meshing is None:
            Err = 'Error, the one-dimension mesh has not been generated yet. '\
                  'Please use Mesh1D.mesh() to generate your mesh based on '\
                  'values provided to Mesh1D.'
            return Err

        else:
            n_nodes = len(self.meshing[0])     # Checks the new created values
            n_elements = len(self.meshing[1])  # Checks the new created values
            output = f'The mesh contains {n_nodes} nodes and {n_elements} '\
                     'elements.'
            print(output)

            str_output = ""
            for i in range(n_elements):
                str_output = str_output + str(self.meshing[1][str(i)]) + "\n"

            return str_output

    def mesh_poly(self):
        nodes = {}
        elements = {}

        for i in range(self.num_nodes):          # Nodes Creation
            nodes[str(i)] = (
                Node1D((self.inside_node_distance * i) + self.start, i)
                )

        for i in range(self.num_elements):  # Elements Creation
            element_nodes = range(
                        i * (self.nodes_elements - 1),
                        i * (self.nodes_elements - 1) + self.nodes_elements,
                        1
                        )
            nodes_table = []
            for j in list(element_nodes):
                nodes_table.append(nodes[str(j)])

            elements[str(i)] = Element1D(nodes_table)

        self.nodes = nodes
        self.elements = elements
        self.meshing = [nodes, elements]
        return nodes, elements


class Mesh2D(Mesh):
    '2 Dimensional Mesh.'
    def __init__(self):
        pass


class Mesh3D(Mesh):
    '3 Dimensional Mesh.'
    def __init__(self):
        pass


class ElementSolver(object):
    'Common class for all element-based solvers'
    def __init__(self):
        pass


class ElementSolver1D(ElementSolver):
    '1 Dimensional Element-based Solver.'
    def __init__(self, mesh):
        self.mesh = mesh

        self.nnodes = len(self.mesh.nodes)
        self.nodes = self.mesh.nodes        # Dictionary of nodes

    def get_p(self):
        # Gets the P matrix
        p = [None] * self.nnodes  # create empty list of nnodes size

        for i in range(self.nnodes):
            if i == 0:
                p[i] = 1
            else:
                p[i] = x ** i

        p = np.array(p)
        self.p = p[0:self.nnodes]

    def get_Me(self):
        # Gets the M_e Matrix
        Me = np.zeros((self.nnodes, self.nnodes))
        expr = x
        for i in range(self.nnodes):
            for j in range(self.nnodes):
                Me[i, j] = int(self.nodes[str(i)].x ** j)

        self.Me = Me

    def get_inv_Me(self):
        self.get_Me()
        self.inv_Me = np.linalg.inv(self.Me)
        tol = 1e-15
        self.inv_Me.real[np.abs(self.inv_Me.real) < tol] = 0.0

    def get_N(self):
        # Get the shape functions for the element
        self.get_p()
        self.get_inv_Me()

        self.N = np.dot(self.p, self.inv_Me)

    def get_N_prime(self):
        if self.N is None:
            self.get_N()
        N_prime = [None] * self.nnodes
        for i in range(self.nnodes):
            N_prime[i] = sy.diff(self.N[i], x)

        self.N_prime = N_prime

    def set_conditions(self, conditions):
        if len(conditions) == self.nnodes:
            self.conditions = np.array(conditions)
        else:
            raise ValueError(
                'Given conditions do not match the number of nodes'
                )

    def get_function(self):
        if self.conditions is None:
            raise ValueError(
                'No conditions were given, please provide conditions using'
                'provide_conditions().'
                )
        else:
            function = np.dot(self.N, self.conditions)
            function2 = function
            for i in sy.preorder_traversal(function):
                if isinstance(i, sy.Float) and abs(i) < 1e-15:
                    function2 = function2.subs(i, round(i, 1))

            self.function = function2

    def get_function_prime(self):
        if self.conditions is None:
            raise ValueError(
                'No conditions were given, please provide conditions using'
                'provide_conditions().'
                )
        else:
            function_prime = np.dot(self.N_prime, self.conditions)
            function2 = function_prime
            for i in sy.preorder_traversal(function_prime):
                if isinstance(i, sy.Float) and i < 1e-15:
                    function2 = function2.subs(i, round(i, 1))

            self.function_prime = function2

    def get_approximation(self, coordinate, round_to=4):
        return round(self.function.subs(x, coordinate), round_to)

    def validate_N(self):

        validation_matrix = np.zeros((self.nnodes, self.nnodes))

        for i in range(self.nnodes):
            for j in range(self.nnodes):
                validation_matrix[i, j] = self.N[i].subs(
                    x, self.nodes[str(j)].x
                    )

        if validation_matrix.all() == np.identity(self.nnodes).all():
            return True
        else:
            return False


# ASSEMBLER

class Assembler(object):
    'Common class for all Assembler of the Finite Element Method.'
    def __init__(self):
        pass


class Assembler1D(Assembler):
    '1 Dimensional Assembler.'
    def __init__(self):
        pass


class Assembler2D(Assembler):
    '2 Dimensional Assembler.'
    def __init__(self):
        pass


class Assembler3D(Assembler):
    '3 Dimensional Mesh.'
    def __init__(self):
        pass


# SOLVER

class Solver(object):
    'Common class for all Solver of the Finite Element Method.'
    def __init__(self):
        pass


class Solver1D(Solver):
    '1 Dimensional Solver.'
    def __init__(self):
        pass


class Solver2D(Solver):
    '2 Dimensional Solver.'
    def __init__(self):
        pass


class Solver3D(Solver):
    '3 Dimensional Solver.'
    def __init__(self):
        pass
