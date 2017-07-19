from FEMur import Element
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

class Element2D(Element):
    'Defines the Planar Elements with its nodes and shape functions'

    def __init__(self, element_type, node_table, index):
        Element.__init__(self, node_table, index)
        self.number = index

        self.e_type = element_type  # 'L' for line, 'T' for triangle, 'Q' for
                                    # quad

        self.nodes = {}
        for i in range(self.num_nodes):
            self.nodes[str(i)] = node_table[i]

        self.x_coord = []  # Creates uni-column matrix for x_coords of nodes
        for i in self.nodes.keys():
            self.x_coord.append(self.nodes[i].x)
        self.x_coord = sy.Matrix(self.x_coord)

        self.y_coord = []  # Creates uni-column matrix for y_coords of nodes
        for i in self.nodes.keys():
            self.y_coord.append(self.nodes[i].y)
        self.y_coord = sy.Matrix(self.y_coord)

        self.p_ref = None
        self.xi_ref = None
        self.eta_ref = None
        self.Me_ref = None
        self.Ne_ref = None
        self.GN_ref = None
        self.x_coord = None
        self.y_coord = None
        self.de = None
        self.Je = None
        self.detJe = None

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
        output_str = f'Element({self.number}) is composed of Nodes({nodes_str})'
        return output_str

    def __repr__(self):
        return str(self)

    def provide_p_ref(self, p_matrix):
        self.p_ref = p_matrix

    def provide_xi_ref(self, xi_ref):
        self.xi_ref = xi_ref

    def provide_eta_ref(self, eta_ref):
        self.eta_ref = eta_ref

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
            inp = self.p_function_ref([self.xi_ref[i],
                                       self.eta_ref[i]])
            Me[i, :] = inp

        self.Me_ref = Me

    def get_inv_Me_ref(self):
        # Get the inverse of the M_e Matrix
        if self.Me_ref is None:
            self.get_Me_ref()

        self.inv_Me_ref = self.Me_ref.inv()

    def validate_Ne_ref(self):
        # Validate the N_e matrix by providing nodes 'x' values. In order for
        # this to be validated as "Good", it has to return the identity matrix.
        xi, eta = sy.symbols('xi eta')

        validation_matrix = sy.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                validation_matrix[i, j] = self.Ne_ref[i].subs(
                    [(xi, self.xi_ref[j]), (eta, self.eta_ref[j])]
                    )

        if validation_matrix == sy.eye(self.num_nodes):
            return True # if the validation matrix is the identity matrix
        else:
            return False

    def get_xy_ref(self):
        '''
        Gets x in function of 'xi' and 'eta'
        Gets y in function of 'xi; and 'eta'

        X is the Sum of N_i * X_i for each node of the element.
        Y is the Sum of N_i * Y_i for each node of the element.
        '''
        if self.Ne_ref is None:
            self.get_Ne_ref()

        x_coord = 0
        y_coord = 0

        for i in range(self.num_nodes):
            j = str(i)
            x_coord = x_coord + (self.nodes[j].x * self.Ne_ref[i])
            y_coord = y_coord + (self.nodes[j].y * self.Ne_ref[i])

        for i in sy.preorder_traversal(x_coord):
            if isinstance(i, sy.Float) and abs(i) < 1e-14:
                x_coord = x_coord.subs(i, round(i, 1))
            elif isinstance(i, sy.Float):
                x_coord = x_coord.subs(i, round(i, 15))

        for i in sy.preorder_traversal(y_coord):
            if isinstance(i, sy.Float) and abs(i) < 1e-14:
                y_coord = y_coord.subs(i, round(i, 1))
            elif isinstance(i, sy.Float):
                y_coord = y_coord.subs(i, round(i, 15))

        self.x_coord = x_coord
        self.y_coord = y_coord

    def get_Je(self):
        '''
        Get the Jacobien Matrix between the element and the reference element.

        Where J = [d x_coord/d xi   d y_coord/d xi]
                  [d x_coord/d eta  d y_coord/d eta]
        '''

        xi, eta = sy.symbols('xi eta')

        if self.x_coord is None or self.y_coord is None:
            self.get_xy_ref()

        jacobien = sy.zeros(2)

        jacobien[0, 0] = sy.diff(self.x_coord, xi)
        jacobien[0, 1] = sy.diff(self.y_coord, xi)
        jacobien[1, 0] = sy.diff(self.x_coord, eta)
        jacobien[1, 1] = sy.diff(self.y_coord, eta)

        self.Je = jacobien

    def get_detJe(self):
        # Get the determinant of the Jacobien Matrix
        if self.Je is None:
            self.get_Je()

        detJe = self.Je.det()
        # print('DetJe:\n', detJe)

        self.detJe = detJe

    def get_Be(self):
        # Get the B_e matrix
        if self.Je is None:
            self.get_Je() # will define GN_ref at the same time.

        # Be =

        # self.Be = Be

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

    def get_trial_prime(self):
        # Get the trial function
        if self.de is None:
            raise ValueError(
                'No conditions were given, please provide conditions using'
                'provide_de().'
                )
        else:
            trial = self.Be * self.de
            trial2 = trial
            for i in sy.preorder_traversal(trial):
                if isinstance(i, sy.Float) and abs(i) < 1e-15:
                    trial2 = trial2.subs(i, round(i, 1))

            self.trial_prime = trial2


class Point1(Element2D):
    'Class for all single-node elements.'
    Ne_ref = None

    def __init__(self, node, index):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')
        Element2D.__init__(self, "P", node, index)
        self.p_ref = sy.Matrix([1.0])
        self.xi_ref = sy.Matrix([0.0])
        self.eta_ref = sy.Matrix([0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "1-node Point"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Point1.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Point1.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Point1.Ne_ref
        else:
            self.Ne_ref = Point1.Ne_ref


class Line(Element2D):
    'Common class for all Line 2D elements.'
    def __init__(self, node_table, index, using_directly=None):
        Element2D.__init__(self, "L", node_table, index)

    def solve_heat_stiff(self):
        '''
        Get Heat Stiffness Matrix for Cauchy condition in Sympy format.
        Size: [Num_node x Num_node]

        'h' is the convection factor

        Returns the stiffness matrix for the element assuming convection on the
        contour of the element.
        '''

        if self.Ne_ref is None:
            self.get_Ne_ref()

        if self.detJe is None:
            self.get_detJe()

        K_e = self.h * self.integrate_domain(self.Ne_ref, 2)

        self.K_e = K_e

    def solve_heat_load(self):
        '''
        Get Heat Load Vector for Cauchy condition in Sympy format.
        Size: [Num_node x Num_node]

        'h' is the convection factor
        't_ext' is the external temperature

        Returns the heat load vector for the element assuming convection on the
        contour of the element.
        '''
        if self.Ne_ref is None:
            self.get_Ne_ref()

        if self.detJe is None:
            self.get_detJe()

        F_e = (self.h * self.t_ext
               * self.integrate_domain(self.Ne_ref, 1))

        self.F_e = F_e

    def integrate_domain(self, Ne_ref, type_int):
        '''
        Returns domain_integration

        Integrates over the domain of the element based on the pre-defined
        shape of the triangular element (Tria3, Tria6, etc).
        '''

        xi = sy.symbols('xi')



        if type_int == 1:
            M = sy.zeros(self.num_nodes, 1)
            for i in range(self.num_nodes):
                M[i] = sy.integrate(Ne_ref[i], (xi, -1, 1))

        elif type_int == 2:
            M = sy.zeros(self.num_nodes)
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    M[i,j] = sy.integrate(Ne_ref[i] * Ne_ref[j], (xi, -1, 1))

        return M * self.detJe


class Line2(Line):
    'Class for 2D linear line elements with 2 nodes.'
    Ne_ref = None

    def __init__(self, node_table, index):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')
        Line.__init__(self, node_table, index)
        self.p_ref = sy.Matrix([1.0, xi])
        self.xi_ref = sy.Matrix([-1.0, 1.0])
        self.eta_ref = sy.Matrix([0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "2-node line"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Line2.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Line2.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Line2.Ne_ref
        else:
            self.Ne_ref = Line2.Ne_ref


class Line3(Line):
    'Class for 2D 2nd order line elements with 3 nodes.'
    Ne_ref = None

    def __init__(self, node_table, index):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')
        Line.__init__(self, node_table, index)
        self.p_ref = sy.Matrix([1.0, xi, xi ** 2])
        self.xi_ref = sy.Matrix([-1.0, 0.0, 1.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "3-node 2nd-order line"

    def get_Ne_ref(self):
        '''
        Get the shape functions for the element in the xi and eta domain.
        '''
        if Line3.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Line3.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Line3.Ne_ref
        else:
            self.Ne_ref = Line3.Ne_ref


class Shell(Element2D):
    'Common class for all shell 2D elements types'
    def __init__(self, node_table, index, using_directly=None):
        Element2D.__init__(self, "T", node_table, index)

    def solve_heat_stiff(self):
        '''
        Get Heat Stiffness Matrix in Sympy format.
        Size: [Num_node x Num_node]

        'D' is the [D] Matrix containing the stiffness factors
        'h' is the convection factor
        'e' is the plate thickness

        Returns the stiffness matrix for the element assuming diffusion in the
        plane direction and convection through both sides of the surface
        created by the element.
        '''

        if self.Be is None:
            self.get_Be()

        if self.detJe is None:
            self.get_detJe()

        # print('Be.T:', self.Be.T.shape)
        # print('D:', self.D.shape)
        # print('Be:', self.Be.shape)

        K_e = ((self.Be.T * self.D * self.Be * self.detJe / 2)
              + ((2 * self.h / self.e) * self.integrate_domain(self.Ne_ref, 2))
              )

        self.K_e = K_e


    def solve_heat_load(self):
        '''
        Get Heat Load Vector in Sympy format.
        Size: [Num_node x Num_node]

        'h' is the convection factor
        'e' is the plate thickness
        't_ext' is the external temperature

        Returns the head load vector for the element assuming convection
        through both sides of the surface created by the element.
        '''
        F_e = ((2 * self.h * self.t_ext / self.e)
               * self.integrate_domain(self.Ne_ref, 1))

        self.F_e = F_e


class Triangular(Shell):
    'Common class for all Triangular 2D elements'
    def __init__(self, node_table, index, using_directly=None):
        Shell.__init__(self, node_table, index)
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.

    def integrate_domain(self, Ne_ref, type_int):
        '''
        Returns domain_integration

        Integrates over the domain of the element based on the pre-defined
        shape of the triangular element (Tria3, Tria6, etc).
        '''

        xi, eta = sy.symbols('xi eta')

        if type_int == 1:
            M = sy.zeros(self.num_nodes, 1)
            for i in range(self.num_nodes):
                M[i] = sy.integrate(sy.integrate(Ne_ref[i], (xi, 0, 1-eta)),
                                    (eta, 0, 1)
                                   )
        elif type_int == 2:
            M = sy.zeros(self.num_nodes)
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    M[i,j] = sy.integrate(sy.integrate(Ne_ref[i] * Ne_ref[j],
                                         (xi, 0, 1-eta)),
                                         (eta, 0, 1)
                                         )

        return M * self.detJe


class Tria3(Triangular):
    "Class representing the T3 shape."
    Ne_ref = None

    def __init__(self, node_table, index):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')

        Triangular.__init__(self, node_table, index)
        self.p_ref = sy.Matrix([1, xi, eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "3-node triangle"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Tria3.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Tria3.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Tria3.Ne_ref
        else:
            self.Ne_ref = Tria3.Ne_ref


class Tria6(Triangular):
    "Class representing the T6 shape."
    Ne_ref = None

    def __init__(self, node_table, index):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, index)

        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi * xi, eta * eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "6-node 2nd-order triangle"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Tria6.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Tria6.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Tria6.Ne_ref
        else:
            self.Ne_ref = Tria6.Ne_ref


class CTria6(Triangular):
    "Class representing the Curved-T6 shape."
    Ne_ref = None

    def __init__(self, node_table, index):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, index)

        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi * xi, eta * eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "6-node 2nd-order triangle"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if CTria6.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            CTria6.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = CTria6.Ne_ref
        else:
            self.Ne_ref = CTria6.Ne_ref


class Quad(Shell):
    'Common class for all Quad 2D elements'
    def __init__(self, node_table, index):
        Shell.__init__(self, node_table, index)
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.

    def integrate_domain(self, Ne_ref, type_int):
        '''
        Returns domain_integration

        Integrates over the domain of the element based on the pre-defined
        shape of the Quad element (Quad4, Quad8, etc).

        type_int is the matrix type to integrate
            '1' being a column matrix
            '2' being a square matrix
        '''

        xi, eta = sy.symbols('xi eta')

        if type_int == 1:
            M = sy.zeros(self.num_nodes, 1)
            for i in range(self.num_nodes):
                M[i] = sy.integrate(sy.integrate(Ne_ref[i],
                                                (xi, -1, 1)
                                                ), (eta, -1, 1))
        elif type_int == 2:
            M = sy.zeros(self.num_nodes)
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    M[i,j] = sy.integrate(sy.integrate(Ne_ref[i] * Ne_ref[j],
                                         (xi, -1, 1)),
                                         (eta, -1, 1)
                                         )

        return M * self.detJe


class Quad4(Quad):
    "Class representing the CQUAD4 shape."
    Ne_ref = None

    def __init__(self, node_table, index):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, index)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta])
        self.xi_ref = sy.Matrix([-1.0, 1.0, 1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, 1.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "4-node quad"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Quad4.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Quad4.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Quad4.Ne_ref
        else:
            self.Ne_ref = Quad4.Ne_ref


class Quad8(Quad):
    "Class representing the CQUAD8 shape."
    Ne_ref = None

    def __init__(self, node_table, index):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, index)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi ** 2, eta ** 2, xi ** 3, eta ** 3])
        self.xi_ref = sy.Matrix([-1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "8-node 2nd-order quad"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Quad8.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Quad8.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Quad8.Ne_ref
        else:
            self.Ne_ref = Quad8.Ne_ref


class Quad9(Quad):
    "Class representing the CQUAD9 shape."
    Ne_ref = None

    def __init__(self, node_table, index):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, index)
        self.p_ref = sy.Matrix([1, xi, eta, xi * eta, xi ** 2, eta ** 2, xi ** 3, eta ** 3, (xi ** 2) * (eta ** 2)])
        self.xi_ref = sy.Matrix([-1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "9-node 2nd-order quad"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Quad9.Ne_ref == None:
            if self.p_ref is None:
                self.get_p_ref()
            if self.Me_ref is None:
                self.get_inv_Me_ref()

            Quad9.Ne_ref = self.p_ref.T * self.inv_Me_ref
            self.Ne_ref = Quad9.Ne_ref
        else:
            self.Ne_ref = Quad9.Ne_ref
