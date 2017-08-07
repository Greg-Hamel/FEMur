from FEMur import *  # Import all of FEMur
import sympy as sy

a = Quad4([Node2D(0, 1, 0), Node2D(3, 0, 1), Node2D(4, 3, 2), Node2D(1, 4, 3)], 0)

a.D = sy.Matrix([[170, 170], [170, 170]])
a.h = 50
a.t_ext = 20
a.e = 0.01

a.get_Ne_ref()
a.get_C()
sy.pprint(a.Ne_ref)

b = Quad8([Node2D(0, 1, 0), Node2D(3, 0, 1), Node2D(4, 3, 2), Node2D(1, 4, 3)], 0)
