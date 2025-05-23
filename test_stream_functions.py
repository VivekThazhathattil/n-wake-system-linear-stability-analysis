import sympy as sp
import numpy as np
from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from plot_tools import *
from two_equation_solver import *

sf = StreamFunctions(n_wakes=1)
#print(sf.region_ids)

bc = BoundaryConditions(sf)

#print(bc.interfacial_equations)

#print('Obtaining latex form of interfacial equations...')
#bc_latex_eqns = bc.get_latex_expressions()  
#for ii in range(len(bc_latex_eqns)):
#    print(f'{bc_latex_eqns[ii]} \\\\')  

#print('Matrix M')
#print('--------')
#print(sp.latex(sp.simplify(bc.M, rational=True)))

#print('Vector X')
#print('--------')
#print(sp.latex(sp.simplify(bc.X, rational=True)))

### temporal stability analysis
##val_set = {'S': 0.2, 'Lambda': -3.1, 'delta': 0.1, 'L': 2, 's':1}
#val_set = {'S': 0.38, 'Lambda': -3.1, 'delta': 0.2, 'L': 1.5, 's':1}
#ts = TemporalStability(sf, bc, val_set)
#alpha_vals = np.linspace(0.001, 20, 100)
#max_w_vals = np.zeros_like(alpha_vals)
#for ii, alpha_val in enumerate(alpha_vals):
#    c_vals = ts.find_c(alpha_val) 
#    c_vals[np.isinf(c_vals)] = 0.0
#    c_vals[c_vals > 1e8] = 0.0
#    w_vals = c_vals * alpha_val
#    max_w_vals[ii] =  np.max(np.imag(w_vals))
#
#plt_line = line_plotter(alpha_vals, max_w_vals)
#plt_line.show()

### Effect of increasing the spacing
#L_vals = np.linspace(0.1,2.5,15)
#for L_val in L_vals:
#    val_set = {'S': 0.38, 'Lambda': -3.1, 'delta': 0.2, 'L': L_val, 's':1}
#    ts = TemporalStability(sf, bc, val_set)
#    alpha_vals = np.linspace(0.001, 20, 100)
#    max_w_vals = np.zeros_like(alpha_vals)
#    for ii, alpha_val in enumerate(alpha_vals):
#        c_vals = ts.find_c(alpha_val) 
#        c_vals[np.isinf(c_vals)] = 0.0
#        c_vals[c_vals > 1e8] = 0.0
#        w_vals = c_vals * alpha_val
#        max_w_vals[ii] =  np.max(np.imag(w_vals))
#
#    plt_line = line_plotter(alpha_vals, max_w_vals)
#    plt_line.show()

## Two wakes scenario get wi vs L
#nL = 25
#L_vals = np.linspace(1.01,5,nL)
#count = 0
#alpha_arr = np.zeros((nL,1), dtype=np.complex128)
#w_arr = np.zeros((nL,1), dtype=np.complex128)
#for L_val in L_vals:
#    D  = bc.M_non_dim
#    val_set = {'S': 0.38, 'Lambda': -3.1, 'delta': 0.2, 'L': L_val, 's':-1}
#    ts = TemporalStability(sf, bc, val_set)
#    alpha_vals = np.linspace(0.001, 20, 100)
#    max_imag_w_vals = np.zeros_like(alpha_vals)
#    max_w_vals = np.zeros_like(alpha_vals, dtype=np.complex128)
#    for ii, alpha_val in enumerate(alpha_vals):
#        c_vals = ts.find_c(alpha_val) 
#        c_vals[np.isinf(c_vals)] = 0.0
#        c_vals[c_vals > 1e8] = 0.0
#        w_vals = c_vals * alpha_val
#        max_imag_w_vals[ii] =  np.max(np.imag(w_vals))
#        max_imag_index = np.argmax(np.imag(w_vals))
#        max_w_vals[ii] = w_vals[max_imag_index]
#
#    max_w_val_idx = np.argmax(max_imag_w_vals)
#    w0 = max_w_vals[max_w_val_idx]
#    alpha0 = alpha_vals[max_w_val_idx]
#
#    D_sub = D.subs(val_set)
#    D_sub = D_sub.subs({sf.c: sf.w/sf.alpha})
#    D_sub = D_sub.evalf()
#    D_det = D_sub.det("berkowitz")
#    D_det_diff = D_det.diff(sf.alpha)
#
#    sol = newton_raphson_2eq(D_det, D_det_diff, (sf.alpha, sf.w), (alpha0, w0))
#    alpha_arr[count] = sol[0]
#    w_arr[count] = sol[1]
#    count += 1
#    print("Solution:", sol)

## Single wake scenario get wi
count = 0
nL = 50
alpha_arr = np.zeros((nL,1), dtype=np.complex128)
w_arr = np.zeros((nL,1), dtype=np.complex128)
D  = bc.M_non_dim
val_set = {'S': 0.38, 'Lambda': -3.1, 'delta': 0.5, 'L': 2, 's':-1}
alpha_vals = np.linspace(0.001, 3, nL)
max_imag_w_vals = np.zeros_like(alpha_vals)
max_w_vals = np.zeros_like(alpha_vals, dtype=np.complex128)
for ii, alpha_val in enumerate(alpha_vals):
    print(ii)
    D_sub = D.subs(val_set)
    D_sub = D_sub.subs({sf.c: sf.w / sf.alpha})
    D_sub = D_sub.subs({sf.alpha: alpha_val})
    D_sub = D_sub.evalf()
    #D_det = D_sub.det("berkowitz")
    D_det = D_sub.det()
    D_eq = sp.Eq(D_det.evalf(), 0)

    numerator, denominator = D_det.as_numer_denom()
    poly_numerator = sp.Poly(numerator, sf.w)
    coefficients_numerator = poly_numerator.all_coeffs()
    filtered_coeffs_numerator = [coeff if abs(coeff) >= 1e-8 else 0 for coeff in coefficients_numerator]
    filtered_poly = sum(coeff * sf.w**i for i, coeff in enumerate(reversed(filtered_coeffs_numerator)))
    w_solutions = sp.solve(filtered_poly, sf.w, rational=False)
    #w_solutions = sp.solve(D_eq, sf.w, rational=False)
    if w_solutions:
        w_vals = np.array([complex(w_val.evalf()) for w_val in w_solutions], dtype=np.complex128)
        w_vals[abs(w_vals) > 1e8] = 0.0
        max_imag_w_vals[ii] = np.max(np.imag(w_vals))
        max_imag_index = np.argmax(np.imag(w_vals))
        max_w_vals[ii] = w_vals[max_imag_index]

max_w_val_idx = np.argmax(max_imag_w_vals)
w0 = max_w_vals[max_w_val_idx]
alpha0 = alpha_vals[max_w_val_idx]

plt_line = line_plotter(alpha_vals, np.imag(max_w_vals))
plt_line.show()