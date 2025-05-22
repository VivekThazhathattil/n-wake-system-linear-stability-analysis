import sympy as sp
import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import root
#from mpmath import findroot
from scipy.linalg import lu, solve_triangular, eig
from numpy.linalg import cond

from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from two_equation_solver import *
from claude1 import *

sf = StreamFunctions(n_wakes=1)
bc = BoundaryConditions(sf)
D  = bc.M_non_dim

# Obtain the dispersion relation
dispersion_relation = sp.simplify(sp.Eq(sp.det(D), 0))
dispersion_relation_latex_string = sp.latex(dispersion_relation)

# Substitute values into the dispersion relation
val_set = {'S': 0.2, 'Lambda': -3.1, 'delta': 0.1, 's':-1, 'c': f'{sf.w}/{sf.alpha}'}
dispersion_relation_sub = dispersion_relation.subs(val_set)
dispersion_relation_sub_latex_string = sp.latex(dispersion_relation_sub)

f1_eq = dispersion_relation_sub.evalf()
f2_eq = sp.Eq(sp.diff(f1_eq.lhs, sf.alpha), sp.diff(f1_eq.rhs, sf.alpha))
f2_eq = sp.simplify(f2_eq).evalf()

# Call the function with your equations
solutions = find_and_display_solutions(f1_eq, f2_eq, sf.alpha, sf.w)

# sort the list based on the imaginary part of the second term in the ordered pair
#filtered_solutions = [tup for tup in solutions if tup[0].imag < 0]
filtered_solutions = [tup for tup in solutions]
sorted_filtered_solutions = sorted(filtered_solutions, key=lambda x: x[1].imag)