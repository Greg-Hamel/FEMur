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
