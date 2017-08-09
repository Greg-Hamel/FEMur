from FEMur import Element
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from math import ceil

class Element2D(Element):
    'Defines the Planar Elements with its nodes and shape functions'

    def __init__(self, element_type, node_table, dof, index, analysis_type):
        Element.__init__(self, node_table, dof, index, analysis_type)
        self.number = index

        self.e_type = element_type  # 'L' for line, 'T' for triangle, 'Q' for
                                    # quad
        self.analysis_type = analysis_type

        self.nodes = {}
        for i in range(self.num_nodes):
            self.nodes[i] = node_table[i]

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
        self.GN = None
        self.GN_ref = None
        self.x_coord = None
        self.y_coord = None
        self.de = None
        self.Je = None
        self.detJe = None
        self.Be = None

    def __str__(self):
        # Define the print function for Element1D
        nodes_str = f''
        for i in range(self.num_nodes):
            if nodes_str == '':
                nodes_str = f'{self.nodes[i].number}'
            else:
                nodes_str = nodes_str + f', {self.nodes[i].number}'
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

    def get_npg(self, func = None):
        '''
        returns the number of gauss point to be used for the exact numerical
        integration of the order specified using the rule 'p <= 2n + 1' for all
        line element or the next higher value in the 'npg_list' of the element.
        '''
        x, y = sy.symbols('x y')

        if func is None:
            order = 0
            for i in self.p_ref:
                if isinstance(i, sy.Float):
                    pass
                else:
                    p_order = sy.degree(i)
                    if p_order > order:
                        order = p_order
            npg = ceil((order + 1)/ 2)  # from Belytchko p. 88
        else:
            npg= ceil((sy.degree(func, gen=x) + sy.degree(func, gen=y) + 1)/ 2)

        if npg in self.npg_list:
            return npg
        else:
            for i in self.npg_list:
                if i > npg:
                    return i

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

    def get_Me_ref(self, refs=None):
        '''
        Get M_e matrix in the xi and eta domain.

        'refs' - can be provided as a list of lists [xi_ref, eta_ref].

        If 'refs' is used, the output will be returned.
        '''
        Me = sy.zeros(self.num_nodes)

        if refs is not None:
            for i in range(len(refs[0])):
                inp = self.p_function_ref([refs[0][i],
                                           refs[1][i]])
                Me[i, :] = inp

            return Me

        else:
            for i in range(self.num_nodes):
                inp = self.p_function_ref([self.xi_ref[i],
                                           self.eta_ref[i]])
                Me[i, :] = inp

            self.Me_ref = Me

    def get_inv_Me_ref(self, refs=None):
        '''
        Get Inverse of M_e matrix

        'refs' - can be provided as a list of lists [xi_ref, eta_ref].

        If 'refs' is used, the output will be returned.
        '''
        if refs is not None:
            return self.get_Me_ref(refs=refs).inv()

        elif self.Me_ref is None:
            self.get_Me_ref()
            self.inv_Me_ref = self.Me_ref.inv()

    def validate_Ne_ref(self):
        # Validate the N_e matrix by providing nodes 'x' values. In order for
        # this to be validated as "Good", it has to return the identity matrix.
        xi, eta = sy.symbols('xi eta')

        validation_matrix = sy.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                i_j = round(self.Ne_ref[i].subs(
                    [(xi, self.xi_ref[j]), (eta, self.eta_ref[j])]
                    ), 4)

                if abs(i_j) < 1e-13:
                    i_j = 0

                if isinstance(i_j, sy.Float):
                    i_j = int(i_j)

                validation_matrix[i, j] = i_j

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
            x_coord = x_coord + (self.nodes[i].x * self.Ne_ref[i])
            y_coord = y_coord + (self.nodes[i].y * self.Ne_ref[i])

        for i in sy.preorder_traversal(x_coord):
            if isinstance(i, sy.Float) and abs(i) < 1e-10:
                x_coord = x_coord.subs(i, round(i, 1))
            elif isinstance(i, sy.Float):
                x_coord = x_coord.subs(i, round(i, 15))

        for i in sy.preorder_traversal(y_coord):
            if isinstance(i, sy.Float) and abs(i) < 1e-10:
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

        if self.e_type == 'L':
            # Line element does not have an area, using length instead.
            jacobien = self.length / 2

        elif self.e_type == 'T' or self.e_type == 'Q':
            self.get_jacob()

        self.Je = jacobien

    def get_detJe(self):
        # Get the determinant of the Jacobien Matrix
        if self.Je is None:
            self.get_Je()

        if self.e_type == 'L':
            # Determinant of a 1x1 matrix is the only number in the matrix
            detJe = self.Je
        else:
            detJe = self.Je.det()

        for i in sy.preorder_traversal(detJe):
            if isinstance(i, sy.Float) and abs(i) < 1e-10:
                detJe = detJe.subs(i, round(i, 1))

        try:
            if detJe <= 0:
                print('WARNING: Jacobien Matrix of an Element is non-positive.')
        except:
            pass

        self.detJe = detJe

    def get_GN_ref(self):

        xi, eta = sy.symbols('xi eta')

        if self.Ne_ref is None:
            self.get_Ne_ref()

        if self.analysis_type == 'SSHeat':
            GN_ref = sy.Matrix([sy.diff(self.Ne_ref, xi),
                                sy.diff(self.Ne_ref, eta)])

        elif self.analysis_type == 'SSMech':
            GN_ref = sy.Matrix([sy.diff(self.Ne_ref[0, :], xi),
                                sy.diff(self.Ne_ref[1, :], eta),
                                (sy.diff(self.Ne_ref[0, :], eta)
                                 + sy.diff(self.Ne_ref[1, :], xi))])

        self.GN_ref = GN_ref

    def get_GN(self):

        if self.GN_ref is None:
            self.get_GN_ref()

        if self.detJe is None:
            self.get_detJe()

        self.GN = self.detJe * self.GN_ref

    def get_Be(self):
        # Get the B_e matrix
        if self.GN is None:
            self.get_GN()

        if self.Je is None:
            self.get_Je() # will define GN_ref at the same time.

        Be = (self.Je ** -1) * self.GN

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
                if isinstance(i, sy.Float) and abs(i) < 1e-10:
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
                if isinstance(i, sy.Float) and abs(i) < 1e-10:
                    trial2 = trial2.subs(i, round(i, 1))

            self.trial_prime = trial2

    def process_Ne(self):
        if self.p_ref is None:
            self.get_p_ref()
        if self.Me_ref is None:
            self.get_inv_Me_ref()

        temp_Ne_ref = self.p_ref.T * self.inv_Me_ref

        if self.analysis_type == 'SSHeat':
            Ne_ref = temp_Ne_ref  # Ne_ref stays as calculated for SSHeat.

        elif self.analysis_type == 'SSMech':
            # Ne_ref must be changed into the correct format.
            Ne_ref = sy.zeros(2, len(temp_Ne_ref) * 2)

            for i in range(2):
                for j in range(len(temp_Ne_ref)):
                    Ne_ref[i, (j * 2) + i] = temp_Ne_ref[j]

        else:
            raise ValueError(f'Unkown analysis_type. The analysis type'
                              ' provided was "{self.analysis_type}".')

        return Ne_ref




class Point1(Element2D):
    'Class for all single-node elements.'
    Ne_ref = None

    def __init__(self, node, dof, index, analysis_type):
        xi, eta = sy.symbols('xi eta')
        Element2D.__init__(self, "P", node, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0])
        self.xi_ref = sy.Matrix([0.0])
        self.eta_ref = sy.Matrix([0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "1-node Point"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Point1.Ne_ref == None:
            Point1.Ne_ref = self.process_Ne()
            self.Ne_ref = Point1.Ne_ref
        else:
            self.Ne_ref = Point1.Ne_ref


class Line(Element2D):
    'Common class for all Line 2D elements.'

    def __init__(self, node_table, dof, index, analysis_type,
                 using_directly=None):
        xi, eta = sy.symbols('xi eta')
        Element2D.__init__(self, "L", node_table, dof, index, analysis_type)
        self.length = self.get_length()
        self.npg_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def get_length(self):

        length = (((self.nodes[0].x - self.nodes[self.num_nodes - 1].x)
                 ** 2 + (self.nodes[0].y - self.nodes[self.num_nodes -
                 1].y) ** 2) ** 0.5)

        return length

    def get_C(self):
        xi, eta = sy.symbols('xi eta')

        if self.Ne_ref is None:
            self.get_Ne_ref()

        if self.detJe is None:
            self.get_detJe()

        if self.detJe <=0:
            raise ValueError("Det is less or equal to zero!!")

        K_e = np.zeros((self.num_nodes, self.num_nodes))
        F_e = np.zeros((self.num_nodes, 1))

        coord, w = self.get_gauss() # coordinates and weight for gauss points

        for i in range(self.npg):
            N = self.Ne_ref.subs(xi, coord[i])
            K_e = K_e + (self.detJe * w[i] * N.T * N * self.h)
            F_e = F_e + (self.detJe * w[i] * self.h * self.t_ext * N.T)

        self.K_e = K_e
        self.F_e = F_e


class Line2(Line):
    'Class for 2D linear line elements with 2 nodes.'
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')
        Line.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi])
        self.xi_ref = sy.Matrix([-1.0, 1.0])
        self.eta_ref = sy.Matrix([0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "2-node line"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Line2.Ne_ref == None:
            Line2.Ne_ref = self.process_Ne()
            self.Ne_ref = Line2.Ne_ref
        else:
            self.Ne_ref = Line2.Ne_ref


class Line3(Line):
    'Class for 2D 2nd order line elements with 3 nodes.'
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')
        Line.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi, xi ** 2])
        self.xi_ref = sy.Matrix([-1.0, 1.0, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

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
            Line3.Ne_ref = self.process_Ne()
            self.Ne_ref = Line3.Ne_ref
        else:
            self.Ne_ref = Line3.Ne_ref


class Line4(Line):
    'Class for 2D 2nd order line elements with 3 nodes.'
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')
        Line.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi, xi ** 2, xi ** 3])
        self.xi_ref = sy.Matrix([-1.0, 1.0, -1/6, 1/6])
        self.eta_ref = sy.Matrix([0.0, 0.0, 0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "4-node 3nd-order line"

    def get_Ne_ref(self):
        '''
        Get the shape functions for the element in the xi and eta domain.
        '''
        if Line4.Ne_ref == None:
            Line4.Ne_ref = self.process_Ne()
            self.Ne_ref = Line4.Ne_ref
        else:
            self.Ne_ref = Line4.Ne_ref


class Shell(Element2D):
    'Common class for all shell 2D elements types'
    def __init__(self, e_type, node_table, dof, index, analysis_type,
                 using_directly=None):
        Element2D.__init__(self, e_type, node_table, dof, index, analysis_type)

    def get_jacob(self, dN):
        jacob = sy.zeros(2)

        for i in range(self.num_nodes):
            for j in range(2):
                jacob[j, 0] += dN[j, i]*self.nodes[i].x
                jacob[j, 1] += dN[j, i]*self.nodes[i].y

        return jacob

    def get_C(self):
        xi, eta = sy.symbols('xi eta')

        if self.Ne_ref is None:
            self.get_Ne_ref()

        if self.GN_ref is None:
            self.get_GN_ref()

        K_e = np.zeros((self.num_nodes, self.num_nodes))
        F_e = np.zeros((self.num_nodes, 1))

        coord, w = self.get_gauss() # coordinates and weight for gauss points

        for i in range(self.npg):
            # sy.pprint(self.Ne_ref)
            N = self.Ne_ref.subs([(xi, coord[i, 0]), (eta, coord[i, 1])])
            # print('N')
            # sy.pprint(N)
            dN = self.GN_ref.subs([(xi, coord[i, 0]), (eta, coord[i, 1])])
            # print('GN')
            # sy.pprint(dN)
            jacobien = self.get_jacob(dN)
            # print('J')
            # sy.pprint(jacobien)

            detJ = jacobien.det()
            # print('detJe')
            # print(detJ)

            if detJ <=0:
                raise ValueError("Det is less or equal to zero!!")

            Be =  jacobien.inv() * dN
            # print('Be')
            # sy.pprint(Be)

            K_e += (detJ * w[i] * ((Be.T * self.D * Be) + (2 *  N.T * N * self.h / self.e)))
            F_e += (detJ * w[i] * 2 * self.h * self.t_ext * N.T / self.e)

        for i in range(len(F_e)):
            if abs(F_e[i]) < 1e-9:
                F_e[i] = 0.0

        self.K_e = K_e
        self.F_e = F_e



class Triangular(Shell):
    'Common class for all Triangular 2D elements'

    def __init__(self, node_table, dof, index, analysis_type, using_directly=None):
        xi, eta = sy.symbols('xi eta')
        Shell.__init__(self, "T", node_table, dof, index, analysis_type)
        self.npg_list = [1, 3, 4, 6, 7, 12]
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.


class Tria3(Triangular):
    "Class representing the T3 shape."
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        xi = sy.symbols('xi')
        eta = sy.symbols('eta')

        Triangular.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi, eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "3-node triangle"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Tria3.Ne_ref == None:
            Tria3.Ne_ref = self.process_Ne()
            self.Ne_ref = Tria3.Ne_ref
        else:
            self.Ne_ref = Tria3.Ne_ref


class Tria6(Triangular):
    "Class representing the T6 shape."
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, dof, index, analysis_type)

        self.p_ref = sy.Matrix([1.0, xi, eta, xi * eta, xi * xi, eta * eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "6-node 2nd-order triangle"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Tria6.Ne_ref == None:
            Tria6.Ne_ref = self.process_Ne()
            self.Ne_ref = Tria6.Ne_ref
        else:
            self.Ne_ref = Tria6.Ne_ref


class CTria6(Triangular):
    "Class representing the Curved-T6 shape."
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Triangular.__init__(self, node_table, dof, index, analysis_type)

        self.p_ref = sy.Matrix([1.0, xi, eta, xi * eta, xi * xi, eta * eta])
        self.xi_ref = sy.Matrix([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
        self.eta_ref = sy.Matrix([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "6-node 2nd-order triangle"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if CTria6.Ne_ref == None:
            CTria6.Ne_ref = self.process_Ne()
            self.Ne_ref = CTria6.Ne_ref
        else:
            self.Ne_ref = CTria6.Ne_ref


class Quad(Shell):
    'Common class for all Quad 2D elements'

    def __init__(self, node_table, dof, index, analysis_type):
        xi, eta = sy.symbols('xi eta')
        Shell.__init__(self, 'Q', node_table, index, analysis_type)
        self.npg_list = [1, 4, 5, 8, 9]
        # If using Triangular Directly, define self.p, self.xi_ref,
        # self.eta_ref, self.num_dots in your script.


class Quad4(Quad):
    "Class representing the CQUAD4 shape."
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        xi, eta = sy.symbols('xi eta')
        Quad.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi, eta, xi * eta])
        self.xi_ref = sy.Matrix([-1.0, 1.0, 1.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, 1.0, 1.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "4-node quad"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Quad4.Ne_ref == None:
            Quad4.Ne_ref = self.process_Ne()
            self.Ne_ref = Quad4.Ne_ref
        else:
            self.Ne_ref = Quad4.Ne_ref


class Quad8(Quad):
    "Class representing the CQUAD8 shape."
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        xi, eta = sy.symbols('xi eta')
        Quad.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi, eta, xi * eta, xi ** 2, eta ** 2,
                                xi ** 3, eta ** 3])
        self.xi_ref = sy.Matrix([-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "8-node 2nd-order quad"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Quad8.Ne_ref == None:
            Quad8.Ne_ref = self.process_Ne()
            self.Ne_ref = Quad8.Ne_ref
        else:
            self.Ne_ref = Quad8.Ne_ref


class Quad9(Quad):
    "Class representing the CQUAD9 shape."
    Ne_ref = None

    def __init__(self, node_table, dof, index, analysis_type):
        eta = sy.symbols('eta')
        xi = sy.symbols('xi')
        Quad.__init__(self, node_table, dof, index, analysis_type)
        self.p_ref = sy.Matrix([1.0, xi, eta, xi * eta, xi ** 2, eta ** 2,
                                xi ** 3, eta ** 3, (xi ** 2) * (eta ** 2)])
        self.xi_ref = sy.Matrix([-1.0, 1.0, 1.0, -1.0,
                                 0.0, 1.0, 0.0, -1.0, 0.0])
        self.eta_ref = sy.Matrix([-1.0, -1.0, 1.0, 1.0,
                                 -1.0, 0.0, 1.0, 0.0, 0.0])
        self.num_dots = len(self.xi_ref)
        self.shape = sy.zeros(self.num_dots)
        self.Ne_ref = None
        self.npg = self.get_npg() # n-Gauss point for numerical integration

        if self.num_nodes != self.num_dots:
            raise ValueError(f'Number of nodes provided is {self.num_nodes},'
                             f' {self.num_dots} expected.')

    def __name__(self):
        return "9-node 2nd-order quad"

    def get_Ne_ref(self):
        # Get the shape functions for the element in the xi and eta domain
        if Quad9.Ne_ref == None:
            Quad9.Ne_ref = self.process_Ne()
            self.Ne_ref = Quad9.Ne_ref
        else:
            self.Ne_ref = Quad9.Ne_ref
