
from stream_functions import StreamFunctions
import sympy as sp
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from claude3 import *

sf = StreamFunctions(n_wakes=1)
bc = BoundaryConditions(sf)
case_no = 3
bluff_body_characteristic_length = 3.18 # mm
curr_mode = 'varicose'
if curr_mode == 'sinuous':
    curr_mode_id = -1
else:
    curr_mode_id = 1

dataset_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\case{case_no}_Lambda_delta.csv"
Lambda_list, delta_list = np.loadtxt(dataset_path, delimiter=',', skiprows=1, unpack=True)
delta_list = delta_list / bluff_body_characteristic_length
solution_pair_curr_mode_list = []

idx = 25
val_set = {'S': 1, 'Lambda':Lambda_list[idx], 'L':1, 'delta': delta_list[idx], 's':curr_mode_id}
ts = TemporalStability(sf, bc, val_set)

alpha_vals = np.linspace(-1,3,100)
#alpha_vals = np.linspace(-1,3,1)
c_vals = []

for alpha_val in alpha_vals:
    c_val_list = ts.find_c(alpha_val)
    c_val_list[np.isinf(np.abs(c_val_list))] = -1000000000
    c_vals.append(np.max(np.imag(c_val_list)))

c_vals = np.array(c_vals)
w_vals = c_vals * alpha_vals

plt.ion()
plt.figure()
#plt.plot(x_D, w0i, color='blue', linestyle='--')
plt.plot(alpha_vals, w_vals, color='blue')
plt.scatter(alpha_vals, w_vals, color='blue', marker='x', s=50, label='Varicose mode - generalized eigenvalue problem')
plt.xlabel('$k_r$')
plt.ylabel('$\\omega_i$')

#########################################

#find_and_display_solutions(f1_eq, f2_eq, alpha_symbol, w_symbol, val_set)

###### Test
wi_vals_newt = []
D  = bc.M_non_dim
dispersion_relation = sp.simplify(sp.Eq(sp.det(D), 0))
dispersion_relation_sub = dispersion_relation.subs(val_set)
f1_eq = dispersion_relation_sub.evalf()
f1_eq_diff = sp.diff(f1_eq.lhs, sf.c)
f1_eq_sub = f1_eq.subs(val_set)
f1_eq_diff_sub = f1_eq_diff.subs(val_set)

f1 = sp.lambdify((sf.alpha, sf.c), f1_eq_sub.lhs - f1_eq_sub.rhs, 'numpy')
df1 = sp.lambdify((sf.alpha, sf.c), f1_eq_diff_sub, 'numpy')

c_guess = 1 + (1j * 1)

for alpha_val in alpha_vals:
    c_temp = newton_iteration(f1, df1, c_guess, alpha_val, max_iter=100, tolerance=1e-6)
    wi_vals_newt.append(np.max(np.imag(c_temp * alpha_val)))

wi_vals_newt = np.array(wi_vals_newt)
######

#plt.figure()
plt.plot(alpha_vals, wi_vals_newt, color='red')
plt.scatter(alpha_vals, wi_vals_newt, color='red', marker='o', s=50, label='Varicose mode - Newton iteration')
plt.xlabel('$k_r$')
plt.ylabel('$\\omega_i$')