import numpy as np
import sympy as sp
from stream_functions import StreamFunctions
from boundary_conditions import BoundaryConditions

def newton_iteration(f, df, x0, k, max_iter=100, tolerance=1e-6):
    x = complex(x0)
    for iterations in range(max_iter):
        try:
            fx = f(k, x)
            dfx = df(k, x)
            #print(f'CHECKPOINT1: {dfx}')
            if abs(dfx) < tolerance:
                return x
            x_new = x - fx / dfx
            #print(f'CHECKPOINT2: {abs(x_new - x)}')
            if abs(x_new - x) < tolerance:
                return x_new
            x = x_new
            #print('CHECKPOINT #3')
        
        except Exception as e:
            print(f"Error in iteration {iterations}: {e}")
            return x
    print(f"Number of iterations exceeded {max_iter}. Error: {(x_new - x)/(x)*100}%")
    return x