import sympy as sy

class Element(object):
    'Common class for all elements of the Finite Element Method.'
    total_elements = 0

    def __init__(self, node_table, index):
        self.num_nodes = len(node_table)  # The number of nodes per element

        self.nodes = {}  # Dictionary with nodes
        for i in range(self.num_nodes):
            self.nodes[str(i)] = node_table[i]

    def get_gauss(self, ow_npg = None):
        '''
        returns [coord, weight]

        Returns the Gauss weight and points coordinates based on

        'npg' Number of gauss point
        'order' 1D: 1, 2D: 2, 3D: 3
        '''

        if ow_npg is None:
            npg = self.npg
        else:
            npg = ow_npg

        if npg < 1:
            raise ValueError(f'Gauss point number requested is {npg},'
                             'must be 1 or higher.')
        elif self.e_type == 'T':
            weight = sy.ones(1, npg)
            coord = sy.zeros(npg, 2)

            if npg == 1:
                print('npg = 1')
                weight = weight * 1/2
                coord[0, 0] = 1/3
                coord[0, 1] = 1/3

            elif npg == 3:
                print('npg=3')
                weight = weight * 1/6
                coord[0, 0] = 1/6
                coord[0, 1] = 1/6
                coord[1, 0] = 2/3
                coord[1, 1] = 1/6
                coord[2, 0] = 1/6
                coord[2, 1] = 2/3


            elif npg in npg_list:
                raise ValueError(f"'npg' of {npg}, is not yet supported.")

            else:
                raise ValueError(f"'npg' provided is {npg}, expected to be one"
                                 f" of the following: {self.npg_list}")

        elif self.e_type == 'Q':
            coord = sy.zeros(npg, 2)
            weight = sy.ones(2, npg)

            if npg == 1:
                weight = weight * 1/2
                coord[0, 0] = 0
                coord[0, 1] = 0

            elif npg == 4:
                weight = weight
                base = 1 / (3 ** 0.5)
                coord[0, 0] = -base
                coord[0, 1] = -base
                coord[1, 0] = base
                coord[1, 1] = -base
                coord[2, 0] = base
                coord[2, 1] = base
                coord[3, 0] = -base
                coord[3, 1] = base

            elif npg in self.npg_list:
                raise ValueError(f"'npg' of {npg}, is not yet supported.")

            else:
                raise ValueError(f"'npg' provided is {npg}, expected to be"
                                 f" one of the following: {self.npg_list}")

        elif self.e_type == 'L':
            coord = sy.zeros(npg, 1)
            weight = sy.ones(1, npg)

            if npg == 1:
                weight = weight * 2
                coord[0] = 0

            elif npg == 2:
                # weight stays 1 as defined
                coord[0] = -1 / (3 ** 0.5)
                coord[1] = 1 / (3 ** 0.5)

            elif npg == 3:
                weight = weight * 5/9  # Set all to 5/9
                weight[0] = 8/9  # Change first value to 8/9
                coord[0] = 0
                coord[1] = -((3 / 5) ** 0.5)
                coord[2] = (3 / 5) ** 0.5

            else:
                raise ValueError(f"'npg' of {npg}, is not yet supported.")

        return [coord, weight]
