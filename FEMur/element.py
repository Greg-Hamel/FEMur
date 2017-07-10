class Element(object):
    'Common class for all elements of the Finite Element Method.'
    total_elements = 0

    def __init__(self, node_table, index):
        self.num_nodes = len(node_table)  # The number of nodes per element

        self.nodes = {}  # Dictionary with nodes
        for i in range(self.num_nodes):
            self.nodes[str(i)] = node_table[i]
