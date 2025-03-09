import sympy as sp
import numpy as np
from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from plot_tools import *

sf = StreamFunctions(3)
print(sf.region_ids)

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

## temporal stability analysis
val_set = {'S': 0.2, 'Lambda': -3.1, 'delta': 0.1, 'L': 2, 's':-1}
ts = TemporalStability(sf, bc, val_set)
alpha_vals = np.linspace(0.001, 20, 100)
max_w_vals = np.zeros_like(alpha_vals)
for ii, alpha_val in enumerate(alpha_vals):
    c_vals = ts.find_c(alpha_val) 
    c_vals[np.isinf(c_vals)] = 0.0
    c_vals[c_vals > 1e8] = 0.0
    w_vals = c_vals * alpha_val
    max_w_vals[ii] =  np.max(np.imag(w_vals))

plt_line = line_plotter(alpha_vals, max_w_vals)
plt_line.show()