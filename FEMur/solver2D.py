from FEMur import *
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
    def __init__(self):
        pass


class SteadyHeatSolver(Solver):
    '''
    2-dimensional steady state heat transfer solver.
    '''
    def __init__(self):
        Solver.__init__()
        pass


class SteadyStructureSolver(Solver):
    '''
    2-dimensional steady state structure solver.
    '''
    def __init__(self):
        Solver.__init__()
        pass
