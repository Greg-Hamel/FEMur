from FEMur import Element
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

class Element2D(Element):
    'Defines the Planar Elements with its nodes and shape functions'
    total_elements = 0

    def __init__(self, element_type, node_table):
        Element.__init__(self)
        self.number = Element2D.total_elements
        Element2D.total_elements += 1

        self.e_type = element_type  # 'L' for line, 'T' for triangle, 'Q' for
                                    # quad
        self.num_nodes = len(node_table)  # The number of nodes per element

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
            return True # if the validation matrix is the identity matrix
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

class Line2(Element2D):
    'Class for 2D linear line elements with 2 nodes.'
    def __init__(self, node_table, index):
        Element2D.__init__(self, "L", node_table)
        self.p_ref = sy.Matrix([1.0, xi])
        self.xi_ref = sy.Matrix([-1.0, 1.0])
        self.eta_ref = sy.Matrix([0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')

class Line3(Element2D):
    'Class for 2D 2nd order line elements with 3 nodes.'
    def __init__(self, node_table, index):
        Element2D.__init__(self, "L", node_table)
        self.p_ref = sy.Matrix([1.0, xi, xi ** 2])
        self.xi_ref = sy.Matrix([-1.0, 0.0,1.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')

class Triangular(Element2D):
    'Common class for all Triangular 2D elements'
    def __init__(self, node_table, using_directly=None):
        Element2D.__init__(self, "T", node_table)
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.


class Tria3(Triangular):
    "Class representing the T3 shape."
    xi = sy.symbols('xi')
    eta = sy.symbols('eta')

    def __init__(self, node_table, index):
        Triangular.__init__(self, node_table)
        self.p_ref = sy.Matrix([1, xi, eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')


class Tria6(Triangular):
    "Class representing the T6 shape."
    eta = sy.symbols('eta')
    xi = sy.symbols('xi')

    def __init__(self, node_table, index):
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


class Quad4(Quad):
    "Class representing the CQUAD4 shape."
    eta = sy.symbols('eta')
    xi = sy.symbols('xi')

    def __init__(self, node_table, index):
        Triangular.__init__(self, node_table)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta])
        self.xi_ref = sy.Matrix([-1.0, 1.0, 1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, 1.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')

class Quad8(Quad):
    "Class representing the CQUAD8 shape."
    eta = sy.symbols('eta')
    xi = sy.symbols('xi')

    def __init__(self, node_table, index):
        Triangular.__init__(self, node_table)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi ** 2, eta ** 2, xi ** 3, eta ** 3])
        self.xi_ref = sy.Matrix([-1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')


class Quad9(Quad):
    "Class representing the CQUAD9 shape."
    eta = sy.symbols('eta')
    xi = sy.symbols('xi')

    def __init__(self, node_table, index):
        Triangular.__init__(self, node_table)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi ** 2, eta ** 2, xi ** 3, eta ** 3, (xi ** 2) * (eta ** 2)])
        self.xi_ref = sy.Matrix([-1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')
