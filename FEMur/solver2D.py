from FEMur import *
import sys
import sympy as sy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.interpolate
from math import ceil


class Solver(object):
    '''
    2-dimensional solver top class.

    Provides common initialization  to all child solver classes.
    '''
    def __init__(self, meshfile):
        self.meshfile = meshfile
        self.get_mesh()

    def weakform(self):
        '''
        Prints weak form used by the solver to approximate the results.
        '''

    def get_mesh(self):
        '''
        Call Mesh class to create the mesh.
        '''

        self.mesh = Mesh2D(self.meshfile)

        pass


class SteadyHeatSolver(Solver):
    '''
    2-dimensional steady state heat transfer solver.
    '''
    def __init__(self, meshfile):
        Solver.__init__(meshfile)

    def solve(self):
        '''
        Automatic Solver for heat transfer equation.
        '''
        pass


class SteadyStructureSolver(Solver):
    '''
    2-dimensional steady state structure solver.
    '''
    def __init__(self, meshfile):
        Solver.__init__(meshfile)

    def solve(self):
        '''
        Automatic Solver for structure equation.
        '''
        pass
