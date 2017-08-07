from FEMur import *
from sympy import symbols, pprint

#
# nodetable = [
#     Node2D(0.1, 0.3, 0),
#     Node2D(0.0, 0.3, 1),
#     Node2D(0.05, 0.25, 2),
#     Node2D(0.05, 0.3, 3),
#     Node2D(0.025, 0.275, 4),
#     Node2D(0.075, 0.275, 5),
# ]

nodetable = [
    Node2D(0.0, 0.0, 0),
    Node2D(4.0, -1.0, 1),
    Node2D(3.0, 4.0, 2),
    Node2D(2.0, -0.5, 3),
    Node2D(3.5, 1.5, 4),
    Node2D(1.5, 2.0, 5),
]

a = Tria6(nodetable, 0)

a.get_Ne_ref()
a.get_Je()
a.get_detJe()
a.get_Be()

print(a.Ne_ref)
pprint(a.x_coord)
pprint(a.y_coord)
print(a.detJe)
pprint(a.Je)
pprint(a.Je.inv())
pprint(a.GN_ref)
pprint(a.GN)
pprint(a.Be)
