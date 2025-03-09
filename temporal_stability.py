from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from scipy.linalg import eig
import numpy as np

class TemporalStability:
    def __init__(self, sf, bc, val_set):
        self.sf = sf
        self.bc = bc
        self.S_val = val_set['S']
        self.s_val = val_set['s']
        self.Lambda_val = val_set['Lambda']
        self.delta_val = val_set['delta']
        self.L_val = val_set['L']
        self.M_non_dim_subbed = None
        self.A_non_dim_subbed = None
        self.B_non_dim_subbed = None

        self.substitute_values()

    def substitute_values(self):
        sf = self.sf
        sub_vals = {
            sf.Lambda: self.Lambda_val, 
            sf.S: self.S_val,
            sf.delta: self.delta_val,
            sf.L: self.L_val,
            sf.s: self.s_val
        }

        self.M_non_dim_subbed = self.bc.M_non_dim.subs(sub_vals) 
        self.A_non_dim_subbed = self.bc.A_non_dim.subs(sub_vals) 
        self.B_non_dim_subbed = self.bc.B_non_dim.subs(sub_vals) 

    def find_c(self, alpha_val):
        sf = self.sf
        self.A_temp = self.A_non_dim_subbed.subs({sf.alpha: alpha_val})
        self.B_temp = self.B_non_dim_subbed.subs({sf.alpha: alpha_val})

        A_num = np.array(self.A_temp.evalf()).astype(np.complex128)
        B_num = np.array(self.B_temp.evalf()).astype(np.complex128)

        c_vals, eig_vecs = eig(A_num, B_num)
        return c_vals