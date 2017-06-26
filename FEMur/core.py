"""
FEMur.py

This module introduces a few concepts of the Finite Element Method (FEM) and
aims at providing a number of tools for solving FEM-related problems.

Its development was started in order to solve problems and projects related to
the SYS806 'Application of the Finite Element Method' Class at 'Ecole de
technolologie superieur de Montreal' for the 2017 summer semester. I intend to
continue its development after finishing the course in order to provide an
easy-to-use solver for FEM problems that is accessible for most people.

This is in no-way a complete project as of yet. There is a lot more to come.
"""

import numpy as np
from scipy import pi
from scipy.special.orthogonal import p_roots
import sympy as sy
from sympy.abc import x, y
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

        self.p = None
        self.Me = None
        self.Ne = None

    def provide_p(self, p_matrix):
        self.p = p_matrix

    def provide_ksi_ref(self, ksi_ref):
        self.ksi_ref = ksi_ref

    def provide_eta_ref(self, eta_ref):
        self.eta_ref = eta_ref

    def provide_eta_ref(self, num_dots):
        self.num_dots = num_dots

    def p_function(self, eval_coordinates):
        # 'eval_coordinates' is a table with [x_coord, y_coord]
        # Returns the p_matrix evaluated a the given point (x, y).
        returned_p = []
        for i in range(self.num_nodes):
            returned_p[i] = self.p[i].subs([(x, eval_coordinates[0]),
                                            (y, eval_coordinates[1])])
        return np.array(returned_p)

    def get_Me(self):
        # Gets the M_e Matrix
        Me = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            Me[i, :] = p_function(self.nodes[str(i)].x, self.nodes[str(i)].y)

        self.Me = Me

    def get_inv_Me(self):
        # Get the inverse of the M_e Matrix
        self.get_Me()
        self.inv_Me = np.linalg.inv(self.Me)
        tol = 1e-15
        self.inv_Me.real[np.abs(self.inv_Me.real) < tol] = 0.0

    def get_Ne(self):
        # Get the shape functions for the element.
        if self.p is None:
            self.get_p()
        if self.Me is None:
            self.get_inv_Me()

        self.Ne = np.dot(self.p, self.inv_Me)

    def get_Be(self):
        if self.Ne is None:
            self.get_Ne()

        N_prime = np.zeros((2, self.num_nodes))
        for i in range(2):
            for j in range(self.num_nodes):
                if i == 0:
                    N_prime[i, j] = sy.diff(self.Ne[i], x)
                elif i == 1:
                    N_prime[i, j] = sy.diff(self.Ne[i], y)
                else:
                    raise ValueError('Get_Be() tried to go over the number of'
                                     'dimensions.')


class Triangular(Element2D):
    'Common class for all Triangular 2D elements'
    def __init__(self, node_table, using_directly=None):
        Element2D.__init__(self, "T", node_table)
        # If using Triangular Directly, define self.p, self.ksi_ref,
        # self.eta_ref, self.num_dots in your script.


class T3(Triangular):
    "Class representing the T3 shape."
    y = sy.symbols('y')
    x = sy.symbols('x')

    def __init__(self, node_table):
        Triangular.__init__(self, node_table)
        self.p = np.array([1, x, y])
        self.ksi_ref = np.array([0.0, 1.0, 0.0])
        self.eta_ref = np.array([0.0, 0.0, 1.0])
        self.num_dots = len(self.ksi_ref)
        self.shape = np.zeros((self.num_dots, self.num_dots))

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')


class T6(Triangular):
    "Class representing the T6 shape."
    y = sy.symbols('y')
    x = sy.symbols('x')

    def __init__(self, node_table):
        Triangular.__init__(self, node_table)
        self.p = np.array([1, x, y, x * y, x * x, y * y])
        self.ksi_ref = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
        self.eta_ref = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 0.5])
        self.num_dots = len(self.ksi_ref)
        self.shape = np.zeros((self.num_dots, self.num_dots))

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             '{self.num_dots} expected.')


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
    def __init__(self, domain, Number_of_elements, Nodes_per_element,
                 conditions):
        self.start = domain[0]  # First 'x' of the domain
        self.end = domain[1]  # Last 'x' of the domain
        self.num_elements = Number_of_elements  # Number of elements
        self.nodes_elements = Nodes_per_element  # Number of nodes per element
        self.d = np.array(conditions)  # Conditions at the nodes (d vector)

        self.length = self.end - self.start      # Length of the domain
        self.num_nodes = (
                          self.num_elements
                          + 1
                          + self.num_elements * (self.nodes_elements - 2)
                          )
        self.L_element = self.length / self.num_elements
        self.node_distance = self.L_element / (self.nodes_elements - 1)

        self.meshing = None
        self.Le_container = None
        self.N = None
        self.calculated = False  # True when elements trials are calculated

    def __str__(self):
        # Prints the element-node interaction.
        if self.meshing is None:
            Err = 'Error, the one-dimension mesh has not been generated yet. '\
                  'Please use Mesh1D.mesh() to generate your mesh based on '\
                  'values provided to Mesh1D.'
            return Err

        else:
            n_nodes = len(self.meshing[0])  # Checks the new created values
            n_elements = len(self.meshing[1])  # Checks the new created values
            output = f'The mesh contains {n_nodes} nodes and {n_elements} '\
                     'elements.'
            print(output)

            str_output = ""
            for i in range(n_elements):
                str_output = str_output + str(self.meshing[1][str(i)]) + "\n"

            return str_output

    def mesh(self):
        # Create a mesh for linear models
        nodes = {}
        elements = {}

        for i in range(self.num_nodes):  # Nodes Creation
            nodes[str(i)] = (
                Node1D((self.node_distance * i) + self.start, i)
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

    def get_Le_container(self):
        # Get the dictionary which contains all L^e matrices
        Le = {}
        for i in self.elements.keys():
            Le[i] = np.zeros((self.nodes_elements, self.num_nodes))
            for j in range(self.nodes_elements):
                Le[i][j, (int(i) * (self.nodes_elements - 1)) + j] = 1

        self.Le_container = Le

    def get_de_container(self):
        # Get the dictionary which contains all d^e matrices
        de = {}
        if self.Le_container is None:
            self.get_Le_container()

        for i in self.elements.keys():
            de[i] = np.dot(self.Le_container[i], self.d)

        self.de = de

    def solve_elements(self):
        # Solve all current elements (shape functions, approximation, etc)
        self.get_de_container()
        for i in self.elements.keys():
            key = int(i)
            print(f"Calculating Element({key})'s shape functions")
            self.elements[i].get_Ne()

            validation = self.elements[i].validate_Ne()
            print(f'Validation of shape function is: {validation}')

            print(f"Calculating Element({key})'s shape functions derivatives")
            self.elements[i].get_Be()

            print(f"Injecting Conditions to Element({key})'s Shape Functions")
            self.elements[i].set_conditions(self.de[i])

            print(f"Calculating Element({key})'s trial functions")
            self.elements[i].get_trial()

            print(f"Calculating Element({key})'s trial derivative functions\n")
            self.elements[i].get_trial_prime()

            self.calculated = True

    def print_elements_trial(self):
        # Shows each element's trial function
        for i in self.elements.keys():
            if self.elements[i].trial is None:
                self.solve_elements()

            key = int(i)
            print(f'Element({key}) has a trial function of: '
                  f'{self.elements[i].trial}')

    def print_elements_Ne(self):
        # Shows each element's trial function
        for i in self.elements.keys():
            if self.elements[i].Ne is None:
                self.elements[i].get_Ne()

            key = int(i)
            print(f'Element({key}) has a trial function of: '
                  f'{self.elements[i].Ne}')

    def get_N(self):
        # Get the global shape function matrix (N)
        N = [0] * self.num_nodes

        if self.Le_container is None:
            self.get_Le_container()

        self.solve_elements()
        print("Calculating Global Shape functions")
        for i in range(self.num_nodes):
            for j in self.elements.keys():
                N[i] = N[i] + np.dot(self.elements[j].Ne,
                                     self.Le_container[j])[i]

        self.N = N

    def get_trial(self):
        # Get the global trial function
        # Warning: this does not seem to work properly.
        if self.N is None:
            self.get_N()

        omega = np.dot(self.N, self.d)

        self.omega = omega

    def get_global_weights(self):
        # Get the global weight function (omega)
        omega = np.zeros(self.num_elements)

        if self.Le_container is None:
            get_Le_container()

        for i in self.elements.keys():
            omega[int(i)] = np.dot(self.elements[i].Be,
                                   self.Le_container[i])

        self.omega = np.dot((np.sum(omega)), self.d)

    def get_trial_L2(self, expected):
        # Get the L2 norm error for the given expected function vs FEM results
        integral = 0
        for i in self.elements.keys():
            ksi = sy.symbols('ksi')
            expr_exp = sy.sympify(expected)
            expr_approx = self.elements[i].trial

            expr_error = (expr_exp - expr_approx) ** 2

            domain = [self.elements[i].start, self.elements[i].end]
            order = sy.degree(expr_error, x)

            length = domain[-1] - domain[0]
            npg = ceil((order + 1) / 2)

            new_x = (0.5 * (domain[0] + domain[1])
                     + 0.5 * ksi * (domain[1] - domain[0]))
            expr = expr_error.subs(x, new_x)

            [new_ksi, w] = p_roots(npg)

            for j in range(len(new_ksi)):
                integral = (integral
                            + (w[j] * length * 0.5 * expr.subs(ksi,
                                                               new_ksi[j]))
                            )

        print(integral)
        self.L2_error = integral

    def get_trial_derivative_L2(self, expected):
        # Get the L2 norm error for the given expected function vs FEM results
        integral = 0
        for i in self.elements.keys():
            ksi = sy.symbols('ksi')
            expr_exp = sy.sympify(expected)
            expr_approx = self.elements[i].trial_prime

            expr_error = (expr_exp - expr_approx) ** 2

            domain = [self.elements[i].start, self.elements[i].end]
            order = sy.degree(expr_error, x)

            length = domain[-1] - domain[0]
            npg = ceil((order + 1) / 2)

            new_x = (0.5 * (domain[0] + domain[1])
                     + 0.5 * ksi * (domain[1] - domain[0]))
            expr = expr_error.subs(x, new_x)

            [new_ksi, w] = p_roots(npg)

            for j in range(len(new_ksi)):
                integral = (integral
                            + (w[j] * length * 0.5 * expr.subs(ksi,
                                                               new_ksi[j]))
                            )

        print(integral)
        self.L2_error = integral

    def plot_trial_comparison(self, real_equation):
        if self.calculated is False:
            self.solve_elements()
        plot_x = np.linspace(self.start, self.end)
        equation = sy.sympify(real_equation)
        equation = sy.lambdify(x, equation)

        fem = []
        for i in self.elements.keys():
            f = sy.lambdify(x, self.elements[i].trial)
            for j in plot_x:
                if j >= self.elements[i].start and j <= self.elements[i].end:
                    if i == '0' and j == self.elements[i].start:
                        fem.append(f(j))
                    elif i != '0' and j == self.elements[i].start:
                        pass
                    else:
                        fem.append(f(j))
                else:
                    pass

        plt.plot(plot_x, equation(plot_x), 'b-', plot_x, fem, 'r--')
        plt.show()

    def plot_trial_derivative_comparison(self, real_equation):
        if self.calculated is False:
            self.solve_elements()

        plot_x = np.linspace(self.start, self.end)
        equation = sy.sympify(real_equation)
        equation = sy.lambdify(x, equation)

        fem = []
        for i in self.elements.keys():
            f_prime = sy.lambdify(x, self.elements[i].trial_prime)
            for j in plot_x:
                if j >= self.elements[i].start and j <= self.elements[i].end:
                    if i == '0' and j == self.elements[i].start:
                        fem.append(f_prime(j))
                    elif i != '0' and j == self.elements[i].start:
                        pass
                    else:
                        fem.append(f_prime(j))
                else:
                    pass

        plt.plot(plot_x, equation(plot_x), 'b-', plot_x, fem, 'r--')
        plt.show()


class Mesh2D(Mesh):
    '2 Dimensional Mesh.'
    def __init__(self):
        pass


class Mesh3D(Mesh):
    '3 Dimensional Mesh.'
    def __init__(self):
        pass
