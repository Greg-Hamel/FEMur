from FEMur import Element
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

class Element1D(Element):
    'Defines the Linear Elements with its nodes and shape functions'
    total_elements = 0

    def __init__(self, node_table):
        Element.__init__(self)  # Use methods defined in Element Class
        self.number = Element1D.total_elements
        Element1D.total_elements += 1  # Counter for number of elements.

        self.num_nodes = len(node_table)  # Linear elements = 2 nodes
        self.L_e = node_table[-1].nodeDistance(node_table[0])
        self.start = node_table[0].x
        self.end = node_table[-1].x

        self.ndof = (self.nodes[str(0)]).dof  # ndof-D analysis

        self.Ne = None
        self.Be = None
        self.trial = None
        self.de = None



    def __str__(self):
        # Define the print function for Element1D
        nodes_str = f''
        for i in range(self.num_nodes):
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
        index = np.zeros(self.num_nodes * self.ndof)
        for i in range(self.num_nodes):
            for n in range(self.ndof):
                index[i] = self.nodes[str(i)].index + n

        return index

    def get_p(self):
        # Gets the P matrix
        p = [None] * self.num_nodes  # create empty list of num_nodes size

        for i in range(self.num_nodes):
            if i == 0:
                p[i] = 1
            else:
                p[i] = x ** i

        p = np.array(p)
        self.p = p[0:self.num_nodes]

    def get_Me(self):
        # Gets the M_e Matrix
        Me = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
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

        N_prime = [None] * self.num_nodes
        for i in range(self.num_nodes):
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
        if len(conditions) == self.num_nodes:
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
        validation_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                validation_matrix[i, j] = self.Ne[i].subs(
                    x, self.nodes[str(j)].x
                    )

        if validation_matrix.all() == np.identity(self.num_nodes).all():
            return True
        else:
            return False
