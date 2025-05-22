import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from two_equation_solver import *
from claude3 import *

sf = StreamFunctions(n_wakes=1)
bc = BoundaryConditions(sf)
D  = bc.M_non_dim

# Obtain the dispersion relation
dispersion_relation = sp.simplify(sp.Eq(sp.det(D), 0))
dispersion_relation_latex_string = sp.latex(dispersion_relation)

#CHANGETHESE
case_no = 1
bluff_body_characteristic_length = 3.18 # mm
curr_mode = 'varicose'
if curr_mode is 'varicose':
    curr_mode_id = -1
else:
    curr_mode_id = 1

# extract columns:
dataset_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\case{case_no}_Lambda_delta.csv"
Lambda_list, delta_list = np.loadtxt(dataset_path, delimiter=',', skiprows=1, unpack=True)
delta_list = delta_list / bluff_body_characteristic_length
solution_pair_curr_mode_list = []

# Substitute values into the dispersion relation
#for idx in range(2):
for idx in range(len(Lambda_list)):
    print(f'Processing {idx+1} / {len(Lambda_list)}')
    val_set = {'S': 1, 'Lambda':Lambda_list[idx], 'delta': delta_list[idx], 's':curr_mode_id, 'c': f'{sf.w}/{sf.alpha}'}
    dispersion_relation_sub = dispersion_relation.subs(val_set)
    dispersion_relation_sub_latex_string = sp.latex(dispersion_relation_sub)

    f1_eq = dispersion_relation_sub.evalf()
    f2_eq = sp.Eq(sp.diff(f1_eq.lhs, sf.alpha), sp.diff(f1_eq.rhs, sf.alpha))
    f2_eq = sp.simplify(f2_eq).evalf()

    # Call the function with your equations
    #solutions = find_and_display_solutions(f1_eq, f2_eq, sf.alpha, sf.w)

    ## for generalized eigenvalue problem used in temporal stability analysis case
    #A_non_dim_num = bc.A_non_dim.subs(val_set)
    #B_non_dim_num = bc.B_non_dim.subs(val_set)
    #solutions = find_and_display_solutions(f1_eq, f2_eq, A_non_dim_num, B_non_dim_num, sf.alpha, sf.w)

    ## for newton iteration in temporal stability analysis case
    val_set_new = {'S': 1, 'Lambda':Lambda_list[idx], 'delta': delta_list[idx], 's':curr_mode_id}
    solutions = find_and_display_solutions(f1_eq, f2_eq, sf.alpha, sf.w, val_set)

    # sort the list based on the imaginary part of the second term in the ordered pair
    #filtered_solutions = [tup for tup in solutions if tup[0].imag < 0]
    filtered_solutions = [tup for tup in solutions]
    sorted_filtered_solutions = sorted(filtered_solutions, key=lambda x: x[1].imag)
    print(sorted_filtered_solutions)
    solution_pair_curr_mode_list.append(sorted_filtered_solutions[-1])

####################################################################
w_list = []
for ii in range(len(solution_pair_curr_mode_list)):
    w_list.append(np.imag(solution_pair_curr_mode_list[ii][1]))

savefile_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\w_list_case{case_no}_{curr_mode}_gep.txt"
np.savetxt(savefile_path, np.array(w_list))
####################################################################