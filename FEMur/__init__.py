"""
FEMur.py

This module introduces a few concepts of the Finite Element Method (FEM) and
aims at providing a number of tools for solving FEM-related problems.

Its development was started in order to solve problems and projects related to
the SYS806 'Application of the Finite Element Method' Class at 'Ecole de
technolologie superieur de Montreal' for the 2017 summer semester. I intend to
continue its development after finishing the course in order to provide an
easy-to-use solver for FEM problems that is accessible for most people.

This is in no-way a complete project as of yet. There is a lot more to come.
"""
from FEMur.node import *
from FEMur.node1D import *
from FEMur.node2D import *
from FEMur.node3D import *
from FEMur.element import *
from FEMur.element1D import *
from FEMur.element2D import *
from FEMur.element3D import *
from FEMur.mesh import *
from FEMur.mesh1D import *
from FEMur.mesh2D import *
from FEMur.mesh3D import *
from FEMur.solver2D import *
