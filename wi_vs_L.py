import os 
import sys
import sympy as sp
import numpy as np
import scipy.io as sio
from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from sivp import SIVP
from datetime import datetime

now = datetime.now()
formatted_date = now.strftime("%Y%m%d%H%M%S")

m_path = os.getcwd()
sys.path.append(m_path)

# for case A,
SL_caseA = {}
SL_caseA['S'] = 1
SL_caseA['Lambda'] = -1.5

# for case B,
SL_caseB = {}
SL_caseB['S'] = 1
SL_caseB['Lambda'] = 2

# for case C,
SL_caseC = {}
SL_caseC['S'] = 1
SL_caseC['Lambda'] = -0.5

is_sinuous = -1
num_wakes = 3
wake_spacings = np.linspace(2,5,50)

val_set = {'S': SL_caseA['S'], 'Lambda': SL_caseA['Lambda'], 'L': 2.0, 'delta': 0.1, 's':is_sinuous}
max_threshold_for_c = 1e6
iter_max = 50
dk_thres = 1e-8

sf = StreamFunctions(n_wakes=num_wakes)
bc = BoundaryConditions(sf)
A = bc.A_non_dim
B = bc.B_non_dim
ts = TemporalStability(sf, bc, val_set)
k0j1 = 2.5 -6.1j
w_idx = 4 
sivp = SIVP(sf, bc, max_threshold_for_c, val_set)
ws, q_hats = sivp.get_ws_q_hats(val_set, k0j1)
ws, q_hats = sivp.get_filtered_ws_qhats(ws, q_hats, max_threshold_for_c)

print(f'ws:{ws}')
w = ws[w_idx]
print(f'Chosen w: {w}')
qhat = q_hats[:,w_idx]

w_vals = np.zeros(len(wake_spacings), dtype=np.complex128)
q_hat_vals = np.zeros((len(wake_spacings), len(qhat)), dtype=np.complex128)
k0j1_vals = np.zeros(len(wake_spacings), dtype=np.complex128)

for idx, wake_spacing in enumerate(wake_spacings):
    print(f'wake_spacing = {wake_spacing}')
    val_set['L'] = wake_spacing
    iter_num = 0
    dk = np.inf

    # For the first index, use initial values
    if idx == 0:
        k0j1_local = k0j1
        w_local = w
        qhat_local = qhat
        w_vals[idx] = w_local
        k0j1_vals[idx] = k0j1_local
        q_hat_vals[idx,:] = qhat_local
        continue
    else:
        # For other indices, use previous values
        k0j1_local = k0j1_vals[idx-1]
        w_local = w_vals[idx-1]
        qhat_local = q_hat_vals[idx-1, :]

    while iter_num < iter_max and dk > dk_thres:
        sivp.sub_all(val_set)
        sivp.sub_alpha(k0j1_local)
        sivp.eval_sub(w_local, qhat_local)
        xdot = sivp.eval_xdot()
        w_dot = xdot[-1]
        qhat_dot = xdot[:-1]
        qhat_dot_herm = np.conjugate(qhat_dot)
        qhat_dot_herm = qhat_dot_herm[:, np.newaxis].transpose()

        J2_inv = sivp.get_J2_inv(w_local, qhat_local)
        b2 = sivp.get_b2(w_local, w_dot, qhat_local, qhat_dot, qhat_dot_herm)

        x_ddot = J2_inv @ b2.ravel()
        d2wdk2 = x_ddot[-1]

        k0j2 = k0j1_local - w_dot/d2wdk2

        if np.abs(k0j2 - k0j1_local) > 15:
            print('Convergence failed')
            w_local = w
            qhat_local = qhat
            k0j1_local = k0j1
            k0j2 = k0j1
            break

        w_local, qhat_local = sivp.get_x_multi_SIVP(k0j1_local, k0j2, w_local, qhat_local)

        dk = np.abs(k0j2 - k0j1_local)
        k0j1_local = k0j2
        iter_num += 1

    w_vals[idx] = w_local
    qhat_local = np.squeeze(qhat_local)
    q_hat_vals[idx, :] = qhat_local
    k0j1_vals[idx] = k0j1_local

# Save results
sio.savemat(f'data\\out\\w_2wakes_1_{formatted_date}.mat',
            {'S': val_set['S'], 
             'L': val_set['Lambda'], 
             'w_vals': w_vals, 
             'q_hat_vals': q_hat_vals, 
             'k0j1_vals': k0j1_vals, 
             'is_sinuous': is_sinuous,
             'num_wakes': num_wakes,
             'wake_spacings': wake_spacings})