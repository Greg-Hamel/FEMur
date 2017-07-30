import sympy as sy

class Element(object):
    'Common class for all elements of the Finite Element Method.'
    total_elements = 0

    def __init__(self, node_table, index):
        self.num_nodes = len(node_table)  # The number of nodes per element

        self.nodes = {}  # Dictionary with nodes
        for i in range(self.num_nodes):
            self.nodes[str(i)] = node_table[i]

    def get_gauss(self):
        '''
        returns [coord, weight]

        Returns the Gauss weight and points coordinates based on

        'npg' Number of gauss point
        'order' 1D: 1, 2D: 2, 3D: 3
        '''

        if self.npg < 1:
            raise ValueError(f'Gauss point number requested is {self.npg},'
                             'must be 1 or higher.')
        elif self.e_type == 'T':
            weight = sy.ones(2, self.npg)
            coord = sy.zeros(self.npg, 2)

            if self.npg == 1:
                weight = weight * 1/2
                coord[0, 0] = 1/3
                coord[0, 1] = 1/3

            elif self.npg == 3:
                coord = sy.zeros(self.npg, 2)
                weight = sy.ones(2, self.npg) * 1/6
                coord[0, 0] = 1/6
                coord[0, 1] = 1/6
                coord[1, 0] = 2/3
                coord[1, 1] = 1/6
                coord[2, 0] = 1/6
                coord[2, 1] = 2/3

            elif self.npg in self.npg_list:
                raise ValueError(f"'npg' of {self.npg}, is not yet supported.")

            else:
                raise ValueError(f"'npg' provided is {npg}, expected to be one"
                                 f" of the following: {self.npg_list}")

        elif self.e_type == 'Q':
            coord = sy.zeros(self.npg, 2)
            weight = sy.ones(2, self.npg)

            if self.npg == 1:
                weight = weight * 1/2
                coord[0, 0] = 0
                coord[0, 1] = 0

            elif self.npg in self.npg_list:
                raise ValueError(f"'npg' of {self.npg}, is not yet supported.")

            else:
                raise ValueError(f"'npg' provided is {self.npg}, expected to be"
                                 f" one of the following: {self.npg_list}")

        elif self.e_type == 'L':
            coord = sy.zeros(self.npg, 1)
            weight = sy.ones(1, self.npg)

            if self.npg == 1:
                weight = weight * 2
                coord[0] = 0

            elif self.npg == 2:
                # weight stays 1 as defined
                coord[0] = -1 / (3 ** 0.5)
                coord[1] = 1 / (3 ** 0.5)

            elif self.npg == 3:
                weight = weight * 5/9  # Set all to 5/9
                weight[0] = 8/9  # Change first value to 8/9
                coord[0] = 0
                coord[1] = -((3 / 5) ** 0.5)
                coord[2] = (3 / 5) ** 0.5

            else:
                raise ValueError(f"'npg' of {self.npg}, is not yet supported.")

        return [coord, weight]
