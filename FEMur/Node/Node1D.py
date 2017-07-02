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
