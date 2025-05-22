import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from two_equation_solver import *

sf = StreamFunctions(n_wakes=1)
bc = BoundaryConditions(sf)
D  = bc.M_non_dim

# Obtain the dispersion relation
dispersion_relation = sp.simplify(sp.Eq(sp.det(D), 0))
dispersion_relation_latex_string = sp.latex(dispersion_relation)

case_no = 4
bluff_body_characteristic_length = 3.18 # mm
curr_mode = 'sinuous'
#curr_mode = 'varicose'
if curr_mode is 'varicose':
    curr_mode_id = -1
else:
    curr_mode_id = 1

dataset_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\case{case_no}_Lambda_delta.csv"
fig_save_path =  f"C:\\Users\\tvivek\\n_wake_system\\figs\\wi_contourf\\case_{case_no}\\{curr_mode}"
Lambda_list, delta_list = np.loadtxt(dataset_path, delimiter=',', skiprows=1, unpack=True)
delta_list = delta_list / bluff_body_characteristic_length

#for idx in range(3):
for idx in range(len(Lambda_list)):
    print(f'Processing {idx+1}/{len(Lambda_list)}...')
    #idx = 13
    # Substitute values into the dispersion relation
    #val_set = {'S': 1, 'Lambda':Lambda_list[idx], 'delta': delta_list[idx], 's':1, 'c': f'{sf.w}/{sf.alpha}'}
    #val_set = {'S': 0.2, 'Lambda':-3.1, 'delta': 0.1, 's':1, 'c': f'{sf.w}/{sf.alpha}'}
    val_set = {'S': 1,'Lambda':Lambda_list[idx], 'delta': 0.1, 's':curr_mode_id, 'c': f'{sf.w}/{sf.alpha}'}
    dispersion_relation_sub = dispersion_relation.subs(val_set)
    #dispersion_relation_sub_latex_string = sp.latex(dispersion_relation_sub)

    f1_eq = dispersion_relation_sub.evalf()
    f1_eq_fn = sp.lambdify([sf.w, sf.alpha], f1_eq.lhs - f1_eq.rhs, 'numpy')

    def dispersion_real(w_vec, k_vec):
        wr, wi = w_vec
        kr, ki = k_vec
        w = wr + 1j * wi
        k = kr + 1j * ki
        D = f1_eq_fn(w, k)
        return [np.real(D), np.imag(D)]

    # get the contourf plot
    y_ext = 10 # ylim range 
    kr_values = np.arange(-y_ext, y_ext, 0.1)  #earlier: 0 to 2 with a step of 0.01
    ki_values = np.arange(-y_ext, y_ext, 0.1)  #earlier: -2 to 0 with a step of 0.01
    kr, ki = np.meshgrid(kr_values, ki_values)
    alpha_vals = kr + 1j * ki  # Element-wise complex addition
    max_w_vals = np.zeros_like(alpha_vals, dtype=np.complex128)

    # find the solution using fsolve
    w_initial_guess = 1 + 5j
    for ii in range(max_w_vals.shape[0]):
        #print(f'{ii+1}/{max_w_vals.shape[0]}')
        for jj in range(max_w_vals.shape[1]):
            alpha_val = alpha_vals[ii,jj]
            temp_sol = fsolve(dispersion_real, [1.0, 1.0], 
                              args=([np.real(alpha_val), np.imag(alpha_val)],))
            max_w_vals[ii,jj] = temp_sol[0] + 1j * temp_sol[1]

    plt.ion()

    #CLev = np.linspace(-40, 40, 50) 
    #CLim = (-40, 40) 

    CLev = np.linspace(-y_ext,y_ext, 50) 
    CLim = (-y_ext, y_ext) 

    fig, ax = plt.subplots(1,1)
    ax.set_title(r'$\omega_1(k)$' + f' i = {idx + 1}', fontsize=14)
    ax.set_xlabel(r'$k_r$', fontsize=14)
    ax.set_ylabel(r'$k_i$', fontsize=14)
    #cmap = plt.cm.gray
    cmap = plt.cm.jet
    cf = ax.contourf(kr, ki, np.imag(max_w_vals), levels=CLev, cmap=cmap)
    ax.contour(kr, ki, np.imag(max_w_vals), levels=CLev, colors='k', linewidths=1)
    ax.contour(kr, ki, np.real(max_w_vals), levels=CLev, colors=[(0.6, 0.6, 0.6)])
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax)
    cf.set_clim(CLim)
    #plt.show()
    plt.savefig(f"{fig_save_path}\\wi_kr_ki_xloc_{case_no}_{curr_mode}_{idx+1}.png")

## find the value at a location
#loc_of_interest = [1.51, 8.86]
#kr_idx = np.argmin(np.abs(kr_values - loc_of_interest[0]))  # Closest kr index to 1.0
#ki_idx = np.argmin(np.abs(ki_values - loc_of_interest[1]))  # Closest ki index to 0.5
#
## Extract value
#value = np.imag(max_w_vals)[ki_idx, kr_idx]
#print(f"Value of omega_i at (kr={kr_values[kr_idx]}, ki={ki_values[ki_idx]}) is {value}")