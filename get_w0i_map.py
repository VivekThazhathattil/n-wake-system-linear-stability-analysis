from mpi4py import MPI
import os 
import sys
import sympy as sp
import numpy as np
import scipy.io as sio
from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions
from temporal_stability import TemporalStability
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

m_path = os.getcwd()
sys.path.append(m_path)

now = datetime.now()
formatted_date = now.strftime("%Y%m%d%H%M%S")

S = 1
Lambda = -2
L = 2
delta = 0.1
num_wakes = 3
is_sinuous = -1
max_threshold_for_c = 1e6

alpha_r = np.arange(0,20,0.1)
alpha_i = np.arange(-10,10,0.1)
alpha_R, alpha_I = np.meshgrid(alpha_r, alpha_i)
alpha = alpha_R + 1j * alpha_I

val_set = {'S': S, 'Lambda': Lambda, 'L': L, 'delta': delta, 's':is_sinuous}

sf = StreamFunctions(n_wakes=num_wakes)
bc = BoundaryConditions(sf)
ts = TemporalStability(sf, bc, val_set)

A = bc.A_non_dim
B = bc.B_non_dim

# Split rows among processes
rows_per_proc = alpha.shape[0] // size
remainder = alpha.shape[0] % size
if rank < remainder:
    start = rank * (rows_per_proc + 1)
    end = start + rows_per_proc + 1
else:
    start = rank * rows_per_proc + remainder
    end = start + rows_per_proc

local_rows = end - start
local_w_vals = np.zeros((local_rows, alpha.shape[1]), dtype=complex)

for local_i, i in enumerate(range(start, end)):
    print(f'Rank {rank}: Row {i+1} of {alpha.shape[0]}')
    for j in range(alpha.shape[1]):
        alpha_val = alpha[i, j]
        c_val_list, eigvecs = ts.find_c(alpha_val)
        temp_w_vals = c_val_list * alpha_val
        temp_w_vals = temp_w_vals[~np.isnan(np.abs(temp_w_vals))]
        temp_w_vals = temp_w_vals[~np.isinf(np.abs(temp_w_vals))]
        temp_w_vals = temp_w_vals[np.abs(temp_w_vals) <= max_threshold_for_c]
        if len(temp_w_vals) == 0:
            local_w_vals[local_i, j] = np.nan
        else:
            max_w_val_idx = np.argmax(np.imag(temp_w_vals))
            local_w_vals[local_i, j] = temp_w_vals[max_w_val_idx]

# Gather results at root
if rank == 0:
    w_vals = np.zeros(alpha.shape, dtype=complex)
    counts = [(alpha.shape[0] // size + (1 if r < remainder else 0)) * alpha.shape[1] for r in range(size)]
    displs = [sum(counts[:r]) for r in range(size)]
else:
    w_vals = None
    counts = None
    displs = None

comm.Gatherv(local_w_vals.ravel(), [w_vals, counts, displs, MPI.COMPLEX], root=0)

if rank == 0:
    #sio.savemat(f'/work/home/vivek/008_ONR_2025/202505271747/data/out/w0i_map_{formatted_date}.mat',
    sio.savemat(f'data\out\w0i_map_{formatted_date}.mat',
                {'S': val_set['S'], 
                 'L': val_set['Lambda'], 
                 'w_vals': w_vals, 
                 'alpha_R': alpha_R, 
                 'alpha_I': alpha_I, 
                 'delta': delta, 
                 'is_sinuous': is_sinuous,
                 'num_wakes': num_wakes,
                 'wake_spacing': L})