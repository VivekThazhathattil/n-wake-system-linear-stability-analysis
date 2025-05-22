import sympy as sp
import numpy as np

# user module for finding streamfunctions
from stream_functions import StreamFunctions
# module for equations matching interface and outer boundary conditions
from boundary_conditions import BoundaryConditions
# module to iteratively solve for complex frequency based on D=0 & partial dD/dk = 0
from wi_spatiotemporal import *

# stream function object based on single wake
sf = StreamFunctions(n_wakes=1)

# equations obtained by applying interface and outer boundary conditions on stream functions
bc = BoundaryConditions(sf)

# non-dimensionalized matrix form of equations
D  = bc.M_non_dim

# Obtain the dispersion relation
dispersion_relation = sp.simplify(sp.Eq(sp.det(D), 0))

# latex version of the dispersion relation (to verify correctness with hand-derived equation)
#dispersion_relation_latex_string = sp.latex(dispersion_relation)

#======================================================================#
# case specific parameters
case_no = 1
density_ratio = 1 #(non-reacting case)
bluff_body_characteristic_length = 3.18 # mm
#curr_mode = 'sinuous'
curr_mode = 'varicose'
if curr_mode == 'varicose':
    curr_mode_id = -1
else:
    curr_mode_id = 1
#======================================================================#

# extract data arrays of shear ratio and shear layer thickness from PSU experimental dataset:
dataset_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\case{case_no}_Lambda_delta.csv"
Lambda_list, delta_list = np.loadtxt(dataset_path, delimiter=',', skiprows=1, unpack=True)
# delta_arr is in mm, have to non-dimensionalize it using bluff body diameter (D)
delta_list = delta_list / bluff_body_characteristic_length

# create empty array to which each solution pair, i.e. (alpha, omega) will be appended
solution_pair_curr_mode_list = []

#for idx in range(2):
for idx in range(len(Lambda_list)):
    print(f'Processing {idx+1} / {len(Lambda_list)}')

    # the values to be substituted into the dispersion relation
    val_set = {'S': density_ratio, 'Lambda':Lambda_list[idx], 'delta': delta_list[idx], 's':curr_mode_id, 'c': f'{sf.w}/{sf.alpha}'}

    # substiute the values in dispersion relation
    dispersion_relation_sub = dispersion_relation.subs(val_set)

    ## obtain the latex string of dispersion relation after value substitution (for DEBUGGING)
    # dispersion_relation_sub_latex_string = sp.latex(dispersion_relation_sub)

    # numerically evaluate expression to floating-point expression
    f1_eq = dispersion_relation_sub.evalf()
    # differentiate dispersion relation to obtain the equation dD/dk = 0
    f2_eq = sp.Eq(sp.diff(f1_eq.lhs, sf.alpha), sp.diff(f1_eq.rhs, sf.alpha))
    # simplifies the differential equation
    f2_eq = sp.simplify(f2_eq).evalf()

    ########################################################################
    ## generic case (older version)
    #solutions = find_and_display_solutions(f1_eq, f2_eq, sf.alpha, sf.w)

    ## for generalized eigenvalue problem used in temporal stability analysis case
    #A_non_dim_num = bc.A_non_dim.subs(val_set)
    #B_non_dim_num = bc.B_non_dim.subs(val_set)
    #solutions = find_and_display_solutions(f1_eq, f2_eq, A_non_dim_num, B_non_dim_num, sf.alpha, sf.w)
    ########################################################################

    ## for newton iteration in temporal stability analysis case
    val_set_new = {'S': 1, 'Lambda':Lambda_list[idx], 'delta': delta_list[idx], 's':curr_mode_id}
    dispersion_relation_sub = dispersion_relation.subs(val_set_new)
    f1_eq = dispersion_relation_sub.evalf()
    f2_eq = sp.Eq(sp.diff(f1_eq.lhs, sf.alpha), sp.diff(f1_eq.rhs, sf.alpha))
    f2_eq = sp.simplify(f2_eq).evalf()
    solutions = find_and_display_solutions(f1_eq, f2_eq, sf, val_set_new)

    # filter out those solutions where the w_i < 0 (i.e. filter out the stable modes)
    #filtered_solutions = [tup for tup in solutions if tup[0].imag < 0]

    ## stub function for filtered solutions
    filtered_solutions = [tup for tup in solutions]

    ## sort the list based on the imaginary part of the second term in the ordered pair (ascending order of w_i)
    sorted_filtered_solutions = sorted(filtered_solutions, key=lambda x: x[1].imag)
    print(sorted_filtered_solutions)
    # from sorted filtered solutions, take only the solution pair with maximum w_i
    solution_pair_curr_mode_list.append(sorted_filtered_solutions[-1])

####################################################################
# Save w_i values for each x/D
w_list = []
for ii in range(len(solution_pair_curr_mode_list)):
    w_list.append(np.imag(solution_pair_curr_mode_list[ii][1]))

savefile_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\w_list_case{case_no}_{curr_mode}_gep.txt"
np.savetxt(savefile_path, np.array(w_list))
####################################################################