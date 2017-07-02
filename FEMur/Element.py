import numpy as np
from scipy import pi
from scipy.special.orthogonal import p_roots
import sympy as sy
from sympy.abc import x, y
from sympy import pprint
import matplotlib.pyplot as plt
from math import *


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

        self.node_table = node_table        # Table with nodes
        self.L_e = self.node_table[-1].nodeDistance(self.node_table[0])
        self.start = node_table[0].x
        self.end = node_table[-1].x

        self.nodes = {}                     # Dictionary with nodes
        for i in range(len(node_table)):
            self.nodes[str(i)] = node_table[i]

        self.nnodes = len(self.nodes)       # Linear elements = 2 nodes
        self.ndof = (self.nodes[str(0)]).dof     # ndof-D analysis

        self.Ne = None
        self.Be = None
        self.trial = None
        self.de = None

        self.number = Element1D.total_elements
        Element1D.total_elements += 1       # Counter for number of elements.

    def __str__(self):
        # Define the print function for Element1D
        nodes_str = f''
        for i in range(self.nnodes):
            if nodes_str == '':
                key = str(i)
                nodes_str = f'{self.nodes[key].number}'
            else:
                key = str(i)
                nodes_str = nodes_str + f', {self.nodes[key].number}'
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

    def getIndex(self):
        # Gets the index for the particular element in the global matrix.
        index = np.zeros(self.nnodes * self.ndof)
        for i in range(self.nnodes):
            for n in range(self.ndof):
                index[i] = self.nodes[str(i)].index + n

        return index

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
        for i in range(self.nnodes):
            for j in range(self.nnodes):
                Me[i, j] = self.nodes[str(i)].x ** j

        self.Me = Me

    def get_inv_Me(self):
        # Get the inverse of the M_e Matrix
        self.get_Me()
        self.inv_Me = np.linalg.inv(self.Me)
        tol = 1e-15
        self.inv_Me.real[np.abs(self.inv_Me.real) < tol] = 0.0

    def get_Ne(self):
        # Get the shape functions for the element.
        self.get_p()
        self.get_inv_Me()

        self.Ne = np.dot(self.p, self.inv_Me)

    def get_Be(self):
        # Get the Shape function derivatives for the element.
        if self.Ne is None:
            self.get_Ne()

        N_prime = [None] * self.nnodes
        for i in range(self.nnodes):
            N_prime[i] = sy.diff(self.Ne[i], x)

        self.Be = N_prime

    def get_we(self, weight):
        # Get the we matrix based on the weight matrix provided
        if self.Ne is None:
            self.get_Ne()

        we = np.dot(self.Ne, weight)

        self.we = we

    def get_we_prime(self, weight):
        # Get the we_prime matrix based on the weight matrix provided
        if self.Be is None:
            self.get_Be()

        we_prime = np.dot(self.Be, weight)

        self.we_prime = we_prime

    def set_conditions(self, conditions):
        # Provide the elements conditions (d_e).
        if len(conditions) == self.nnodes:
            self.de = np.array(conditions)
        else:
            raise ValueError(
                'Given conditions do not match the number of nodes'
                )

    def get_trial(self):
        # Get the trial function
        if self.de is None:
            raise ValueError(
                'No conditions were given, please provide conditions using'
                'provide_conditions().'
                )
        else:
            trial = np.dot(self.Ne, self.de)
            trial2 = trial
            for i in sy.preorder_traversal(trial):
                if isinstance(i, sy.Float) and abs(i) < 1e-15:
                    trial2 = trial2.subs(i, round(i, 1))

            self.trial = trial2

    def get_trial_prime(self):
        # Get the trial function derivative
        if self.de is None:
            raise ValueError(
                'No conditions were given, please provide conditions using'
                'provide_conditions().'
                )
        else:
            trial_prime = np.dot(self.Be, self.de)
            trial2 = trial_prime
            for i in sy.preorder_traversal(trial_prime):
                if isinstance(i, sy.Float) and i < 1e-15:
                    trial2 = trial2.subs(i, round(i, 1))

            self.trial_prime = trial2

    def get_approximation(self, coordinate, round_to=4):
        # get the FE approximation between two nodes
        return round(self.trial.subs(x, coordinate), round_to)

    def validate_Ne(self):
        # Validate the N_e matrix by providing nodes 'x' values. In order for
        # this to be validated as "Good", it has to return the identity matrix.
        validation_matrix = np.zeros((self.nnodes, self.nnodes))

        for i in range(self.nnodes):
            for j in range(self.nnodes):
                validation_matrix[i, j] = self.Ne[i].subs(
                    x, self.nodes[str(j)].x
                    )

        if validation_matrix.all() == np.identity(self.nnodes).all():
            return True
        else:
            return False


class Element2D(Element):
    'Defines the Planar Elements with its nodes and shape functions'
    total_elements = 0

    def __init__(self, element_type, node_table):
        self.number = Element2D.total_elements
        self.e_type = element_type  # 'T' for triangle, 'Q' for quad
        self.num_nodes = len(node_table)  # The number of nodes per element
        self.nodes = {}  # Dictionary with nodes
        for i in range(self.num_nodes):
            self.nodes[str(i)] = node_table[i]
        Element2D.total_elements += 1

        self.p_ref = None
        self.Me_ref = None
        self.Ne_ref = None
        self.GN_ref = None
        self.de = None

    def provide_p_ref(self, p_matrix):
        self.p_ref = p_matrix

    def provide_xi_ref(self, xi_ref):
        self.xi_ref = xi_ref

    def provide_eta_ref(self, eta_ref):
        self.eta_ref = eta_ref

    def provide_xcoord(self, x_coord):
        self.x_coord = x_coord

    def provide_ycoord(self, y_coord):
        self.y_coord = y_coord

    def provide_de(self, de):
        self.de = de

    def p_function_ref(self, eval_coordinates):
        # 'eval_coordinates' is a table with [xi_coord, eta_coord]
        # Returns the p_matrix evaluated a the given point (xi, eta).

        xi, eta = sy.symbols('xi eta')

        returned_p = sy.zeros(1, self.num_nodes)
        for i in range(self.num_nodes):
            if i == 0:
                returned_p[i] = 1.0
            else:
                returned_p[i] = self.p_ref[i].subs([(xi, eval_coordinates[0]),
                                                   (eta, eval_coordinates[1])])

        return returned_p

    def get_Me_ref(self):
        # Gets the M_e Matrix in the xi and eta domain
        Me = sy.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            inp = self.p_function_ref([self.nodes[str(i)].x,
                                       self.nodes[str(i)].y])
            Me[i, :] = inp

        self.Me_ref = Me

    def get_inv_Me_ref(self):
        # Get the inverse of the M_e Matrix
        if self.Me_ref is None:
            self.get_Me_ref()

        self.inv_Me_ref = self.Me_ref.inv()

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if self.p_ref is None:
            self.get_p_ref()
        if self.Me_ref is None:
            self.get_inv_Me_ref()

        self.Ne_ref = sy.Matrix(self.p_ref * self.inv_Me_ref)

    def validate_Ne_ref(self):
        # Validate the N_e matrix by providing nodes 'x' values. In order for
        # this to be validated as "Good", it has to return the identity matrix.
        xi, eta = sy.symbols('xi eta')

        validation_matrix = sy.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                validation_matrix[i, j] = self.Ne_ref[i].subs(
                    [(xi, self.nodes[str(j)].x), (eta, self.nodes[str(j)].y)]
                    )

        if validation_matrix == sy.eye(self.num_nodes):
            return True
        else:
            return False

    def get_GN_ref(self):
        # Get the dot product of the gradient operator and the  shape functions
        xi, eta = sy.symbols('xi eta')

        if self.Ne_ref is None:
            self.get_Ne_ref()

        GN_xi = [None] * self.num_nodes
        GN_eta = [None] * self.num_nodes
        for i in range(2):
            for j in range(self.num_nodes):
                if i == 0:
                    GN_xi[j] = sy.diff(self.Ne_ref[j], xi)
                elif i == 1:
                    GN_eta[j] = sy.diff(self.Ne_ref[j], eta)
                else:
                    raise ValueError('Get_Be() tried to go over the number of'
                                     'dimensions.')

        self.GN_ref = sy.Matrix([GN_xi, GN_eta])

    def get_xy_coord_matrix(self):
        # Get a matrix contain the x coordinates at its first column and the
        # y coordinates as its second column.
        xy_coord = sy.zeros(self.num_nodes, 2)
        for i in range(self.num_nodes):
            xy_coord[i, :] = sy.Matrix([[self.x_coord[i], self.y_coord[i]]])

        self.xy_coord = xy_coord

    def get_Je(self):
        # Get the jacobien
        self.get_xy_coord_matrix()

        if self.GN_ref is None:
            self.get_GN_ref()

        jacobien = self.GN_ref * self.xy_coord

        self.Je = jacobien

    def get_detJe(self):
        if self.Je is None:
            self.get_Je()

        detJe = self.Je.det()

        self.detJe = detJe

    def get_Be(self):
        # Get the B_e matrix
        Be = self.Je.inv() * self.GN_ref

        self.Be = Be

    def get_trial(self):
        # Get the trial function
        if self.de is None:
            raise ValueError(
                'No conditions were given, please provide conditions using'
                'provide_de().'
                )
        else:
            trial = self.Ne_ref * self.de
            trial2 = trial
            for i in sy.preorder_traversal(trial):
                if isinstance(i, sy.Float) and abs(i) < 1e-15:
                    trial2 = trial2.subs(i, round(i, 1))

            self.trial = trial2


class Triangular(Element2D):
    'Common class for all Triangular 2D elements'
    def __init__(self, node_table, using_directly=None):
        Element2D.__init__(self, "T", node_table)
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.


class T3(Triangular):
    "Class representing the T3 shape."
    xi = sy.symbols('xi')
    eta = sy.symbols('eta')

    def __init__(self, node_table):
        Triangular.__init__(self, node_table)
        self.p_ref = sy.Matrix([1, xi, eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')


class T6(Triangular):
    "Class representing the T6 shape."
    eta = sy.symbols('eta')
    xi = sy.symbols('xi')

    def __init__(self, node_table):
        Triangular.__init__(self, node_table)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi * xi, eta * eta])
        self.xi_ref = sy.Matrix([0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 0.0, 0.5, 1.0, 0.5])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')


class Quad(Element2D):
    'Common class for all Quad 2D elements'
    def __init__(self, node_table):
        Element2D.__init__(self, "Q", node_table)
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.
