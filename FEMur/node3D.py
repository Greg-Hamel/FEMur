from FEMur import Node

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
