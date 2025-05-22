from stream_functions import StreamFunctions
import sympy as sp
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt

sf = StreamFunctions(n_wakes=1)
bc = BoundaryConditions(sf)
case_no = 1
bluff_body_characteristic_length = 3.18 # mm
curr_mode = 'sinuous'
if curr_mode is 'varicose':
    curr_mode_id = -1
else:
    curr_mode_id = 1

dataset_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\case{case_no}_Lambda_delta.csv"
Lambda_list, delta_list = np.loadtxt(dataset_path, delimiter=',', skiprows=1, unpack=True)
delta_list = delta_list / bluff_body_characteristic_length
solution_pair_curr_mode_list = []

idx = 13
val_set = {'S': 1, 'Lambda':Lambda_list[idx], 'L':1, 'delta': delta_list[idx], 's':curr_mode_id}
ts = TemporalStability(sf, bc, val_set)

alpha_vals = np.linspace(-1,3,100)
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
plt.scatter(alpha_vals, w_vals, color='blue', marker='x', s=50, label='Varicose mode')
plt.xlabel('$k_r$')
plt.ylabel('$\omega_i$')