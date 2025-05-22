import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from two_equation_solver import *

val_set = {'S': 0.38, 'Lambda': -3.1, 'delta': 0.1, 'L': 1.01, 's':-1}
kr_values = np.arange(0, 2.01, 0.1)  # 0 to 2 with a step of 0.01
ki_values = np.arange(-2, 0.01, 0.1)  # -2 to 0 with a step of 0.01
kr, ki = np.meshgrid(kr_values, ki_values)
alpha_vals = kr + 1j * ki  # Element-wise complex addition
max_w_vals = np.zeros_like(alpha_vals, dtype=np.complex128)

sf = StreamFunctions(n_wakes=2)
bc = BoundaryConditions(sf)
D  = bc.M_non_dim

n_rows, n_cols = alpha_vals.shape
for i in range(n_rows):
    print(f'{i+1}/{n_rows}')
    for j in range(n_cols):
        k = alpha_vals[i][j]
        D_sub = D.subs(val_set)
        D_sub = D_sub.subs({sf.alpha: k})
        D_sub = D_sub.evalf()
        D_det = D_sub.det("berkowitz")
        #D_det = D_sub.det()
        #c_solutions = sp.solve(sp.Eq(D_sub,0), sf.c, rational=False)

        numerator, denominator = D_det.as_numer_denom()
        poly_numerator = sp.Poly(numerator, sf.c)
        coefficients_numerator = poly_numerator.all_coeffs()
        filtered_coeffs_numerator = [np.real(coeff) + 1j * np.imag(coeff) if np.imag(coeff) >= 1e-8 else np.real(coeff) for coeff in coefficients_numerator]
        filtered_coeffs_numerator = [coeff if abs(coeff) >= 1e-8 else 0 for coeff in filtered_coeffs_numerator]
        filtered_poly = sum(coeff * sf.c**i for i, coeff in enumerate(reversed(filtered_coeffs_numerator)))
        c_solutions = sp.solve(filtered_poly, sf.c, rational=False)

        if not c_solutions:
            max_w_vals[i,j] = np.nan
        else:
            #print(c_solutions)
            w_solutions = np.array(c_solutions, dtype=np.complex128) * k
            w_solutions[np.isinf(w_solutions)] = 0.0
            w_solutions[np.isnan(w_solutions)] = 0.0
            w_solutions[abs(w_solutions) > 1e8] = 0.0
            max_imag_w_idx = np.argmax(np.imag(w_solutions))
            max_w_vals[i,j] = w_solutions[max_imag_w_idx]
#        break
#    break

plt.ion()
CLev = np.linspace(-1, 1, 50) 
#CLim = (-1, 1) 
CLim = (-0.5, 0.5) 
fig, ax = plt.subplots(figsize=(6, 6))
#ax.set_title(r'$\omega_1(k)$', fontsize=14)
#ax.set_xlabel(r'$k_r$', fontsize=14)
#ax.set_ylabel(r'$k_i$', fontsize=14)
cmap = plt.cm.gray
cf = ax.contourf(kr, ki, np.imag(max_w_vals), levels=CLev, cmap=cmap)
ax.contour(kr, ki, np.imag(max_w_vals), levels=[0], colors='k', linewidths=2)
ax.contour(kr, ki, np.real(max_w_vals), levels=CLev, colors=[(0.6, 0.6, 0.6)])
cf.set_clim(CLim)
plt.show()