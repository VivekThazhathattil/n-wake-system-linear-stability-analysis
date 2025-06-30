# OBJECTIVE: Construct global map of wi(k) using SIVP and temporal stability analysis, but this script does so using parallel mpi.

import numpy as np

from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from sivp import SIVP

# construct global map of wi(k):
max_threshold = 1e6
num_wakes = 1
val_set = {'S': 0.1, 'Lambda': -2, 'delta': 0.1, 'L':0.0, 's':1}
k_re_beg = 0
k_re_end = 2
k_re_step = 0.05
k_im_beg = -2
k_im_end = 2
k_im_step = 0.05

alpha_r = np.arange(k_re_beg, k_re_end, k_re_step)
alpha_i = np.arange(k_im_beg, k_im_end, k_im_step)

sf = StreamFunctions(n_wakes=num_wakes)
bc = BoundaryConditions(sf)
ts = TemporalStability(sf, bc, val_set)

D  = bc.M_non_dim
A = bc.A_non_dim
B = bc.B_non_dim

A_sub = A.subs(val_set)
B_sub = B.subs(val_set)

alpha_R, alpha_I = np.meshgrid(alpha_r, alpha_i)
w_vals = np.zeros_like(alpha_R, dtype=complex)
visited = np.zeros_like(alpha_R, dtype=bool)

#-----------------------------------------------------------#
# using temporal stability analysis to find the initial point
ts = TemporalStability(sf, bc, val_set)
max_threshold = 1e6
alpha_r1 = np.arange(5,20,0.5)
w_vals = np.zeros_like(alpha_r1, dtype=complex)
w1_vals = np.zeros_like(alpha_r1, dtype=complex)
w2_vals = np.zeros_like(alpha_r1, dtype=complex)

for i in range(len(alpha_r1)):
    c_val_list, ignore_vals = ts.find_c(alpha_r1[i])
    temp_w_vals = c_val_list * alpha_r1[i]
    temp_w_vals = temp_w_vals[np.isfinite(temp_w_vals)]
    temp_w_vals = temp_w_vals[np.real(temp_w_vals) < max_threshold]
    if temp_w_vals.shape[0] == 0:
        w_vals[i] = np.nan
        w1_vals[i] = np.nan
        w2_vals[i] = np.nan
    else:
        w1_vals[i] = temp_w_vals[0]
        w2_vals[i] = temp_w_vals[1]
        max_w_val_idx = np.argmax(np.imag(temp_w_vals))
        w_vals[i] = temp_w_vals[max_w_val_idx]
max_imag_w_val = np.max(np.imag(w_vals))
max_imag_w_val_idx = np.argmax(np.imag(w_vals))
k_beg = alpha_r1[max_imag_w_val_idx]
w_beg = w_vals[max_imag_w_val_idx]

ts = TemporalStability(sf, bc, val_set)
max_threshold = 1e6
c_vals, q1_vals = ts.find_c(k_beg)
filter_for_c_vals = [np.isfinite(c_vals)] and c_vals < max_threshold
q1_vals_filtered = []
for i in range(q1_vals.shape[1]):
    if filter_for_c_vals[i]:
        q1_vals_filtered.append(q1_vals[:,i])
q1_hat = np.array(q1_vals_filtered, dtype=complex)
q1_hat = q1_hat.T
c_vals = c_vals[filter_for_c_vals]
w_vals = c_vals * k_beg
if np.imag(w_vals[0]) > np.imag(w_vals[1]):
    w1 = w_vals[0]
else:
    w1 = w_vals[1]
w_vals

sivp = SIVP(sf, bc)
sivp.sub_all(val_set)
sivp.sub_alpha(k_beg)
ws, q_hats = sivp.get_ws_q_hats(val_set, k_beg)
q_hat = q_hats[:,1]
q_hat = q_hat[:, np.newaxis]
w = ws[1]
w = w[np.newaxis]
x1 = np.vstack((q_hat, w))
#-----------------------------------------------------------#

# find the index of maximum growth rate from temporal stability analysis
kr_beg_idx = np.argmin(np.abs(alpha_r - k_beg))
ki_beg_idx = np.argmin(np.abs(alpha_i))
k1 = alpha_R[ki_beg_idx, kr_beg_idx] + 1j * alpha_I[ki_beg_idx, kr_beg_idx]
w_vals[ki_beg_idx, kr_beg_idx] = w_beg

# mark that index as visited
visited[ki_beg_idx, kr_beg_idx] = True 

val_set = {'S': 0.1, 'Lambda': -2.0, 'delta': 0.1, 'L':0.0, 's':1}
sivp = SIVP(sf, bc)

def get_matching_x(x1, k1, k2):
    sivp.sub_all(val_set)
    sivp.sub_alpha(k1)
    sivp.eval_sub(x1[-1], x1[:-1])
    x2 = sivp.eval_x2(x1, k2 - k1)
    ws, q_hats = sivp.get_ws_q_hats(val_set, k2)
    #print('DEBUG')
    #print(x2)
    result = sivp.two_norm_find_matching_x(x2, ws, q_hats)
    if result is None:
        return None, None
    matching_x = result[0]
    min_val = result[1]
    return matching_x, min_val

# to store all x1 values along the imaginary axis for each ki and fixed dominant alpha-mode kr
x1_dom = [x1]
x1_up_list = [] 
x1_down_list = [] 

# go upward in the ki direction
curr_imag_idx = ki_beg_idx + 1
while curr_imag_idx < alpha_I.shape[0]:
    print(f'curr_imag_idx = {curr_imag_idx}/ {alpha_I.shape[0] - 1}')
    k2 = alpha_R[curr_imag_idx, kr_beg_idx] + 1j * (alpha_I[curr_imag_idx, kr_beg_idx])
    result = get_matching_x(x1, k1, k2)
    if result is None:
        x2 = None
        min_val = -1
    else:
        x2 = result[0]
        min_val = result[1]
    if x2 is None:
        w_vals[curr_imag_idx, kr_beg_idx] = np.nan
    else:
        w_vals[curr_imag_idx, kr_beg_idx] = x2[-1]
    if min_val > 10:
        print(f'(ki,kr): ({curr_imag_idx},{kr_beg_idx})')
    w_vals[curr_imag_idx, kr_beg_idx] = x2[-1]
    visited[curr_imag_idx, kr_beg_idx] = True
    x1_up_list.append(x2)
    x1 = x2
    k1 = k2
    curr_imag_idx += 1

# go downward in the ki direction
curr_imag_idx = ki_beg_idx - 1
while curr_imag_idx >= 0:
    print(f'curr_imag_idx = {curr_imag_idx}/ {alpha_I.shape[0] - 1}')
    k2 = alpha_R[curr_imag_idx, kr_beg_idx] + 1j * (alpha_I[curr_imag_idx, kr_beg_idx])
    result = get_matching_x(x1, k1, k2)
    if result is None:
        x2 = None
        min_val = -1
    else:
        x2 = result[0]
        min_val = result[1]
    if x2 is None:
        w_vals[curr_imag_idx, kr_beg_idx] = np.nan
    else:
        w_vals[curr_imag_idx, kr_beg_idx] = x2[-1]
    if min_val > 10:
        print(f'(ki,kr): ({curr_imag_idx},{kr_beg_idx})')
    w_vals[curr_imag_idx, kr_beg_idx] = x2[-1]
    visited[curr_imag_idx, kr_beg_idx] = True
    x1_down_list.append(x2)
    x1 = x2
    k1 = k2
    curr_imag_idx -= 1

x1_down_list.reverse()
x1_list = sum([x1_down_list, x1_dom, x1_up_list], [])

curr_imag_idx = 0
while curr_imag_idx < alpha_I.shape[0]:
    print(f'curr_imag_idx = {curr_imag_idx}/ {alpha_I.shape[0] - 1}')
    # go left in the kr direction
    k1 = alpha_R[curr_imag_idx, kr_beg_idx] + 1j * (alpha_I[curr_imag_idx, kr_beg_idx])
    curr_real_idx = kr_beg_idx - 1
    x1 = x1_list[curr_imag_idx]
    while curr_real_idx >= 0:
        #print(f'curr_real_idx = {curr_real_idx}/ {alpha_I.shape[1] - 1}')
        k2 = alpha_R[curr_imag_idx, curr_real_idx] + 1j * (alpha_I[curr_imag_idx, curr_real_idx])
        result = get_matching_x(x1, k1, k2)
        if result is None:
            x2 = None
            min_val = -1
        else:
            x2 = result[0]
            if result[1] is None:
                min_val = -1
            else:
                min_val = result[1]
        if min_val > 10:
            print(f'(ki,kr): ({curr_imag_idx},{curr_real_idx})')
        if x2 is None:
            w_vals[curr_imag_idx, curr_real_idx] = np.nan
        else:
            w_vals[curr_imag_idx, curr_real_idx] = x2[-1]
            x1 = x2
        k1 = k2
        visited[curr_imag_idx, curr_real_idx] = True
        curr_real_idx -= 1
    # go right in the kr direction
    k1 = alpha_R[curr_imag_idx, kr_beg_idx] + 1j * (alpha_I[curr_imag_idx, kr_beg_idx])
    curr_real_idx = kr_beg_idx + 1
    x1 = x1_list[curr_imag_idx]
    while curr_real_idx < alpha_I.shape[1]:
        #print(f'curr_real_idx = {curr_real_idx}/ {alpha_I.shape[1] - 1}')
        k2 = alpha_R[curr_imag_idx, curr_real_idx] + 1j * (alpha_I[curr_imag_idx, curr_real_idx])
        result = get_matching_x(x1, k1, k2)
        if result is None:
            x2 = None
            min_val = -1
        else:
            x2 = result[0]
            if result[1] is None:
                min_val = -1
            else:
                min_val = result[1]
        if min_val > 10:
            print(f'(ki,kr): ({curr_imag_idx},{curr_real_idx})')
        if x2 is None:
            w_vals[curr_imag_idx, curr_real_idx] = np.nan
        else:
            w_vals[curr_imag_idx, curr_real_idx] = x2[-1]
            x1 = x2
        k1 = k2
        visited[curr_imag_idx, curr_real_idx] = True
        curr_real_idx += 1
    curr_imag_idx += 1
    