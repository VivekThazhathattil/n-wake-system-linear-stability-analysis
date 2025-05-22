import sympy as sp
import numpy as np

def newton_raphson_2eq(f1, f2, vars, init_vals, tol=1e-8, max_iter=250):
    alpha, w = vars
    # Build our Jacobian matrix
    J = sp.Matrix([[sp.diff(f1, alpha), sp.diff(f1, w)],
                   [sp.diff(f2, alpha), sp.diff(f2, w)]])

    F = sp.Matrix([f1, f2]) 
    
    # Convert to numerical functions
    F_lambdified = sp.lambdify((alpha, w), F, 'numpy')
    J_lambdified = sp.lambdify((alpha, w), J, 'numpy')
    
    alpha_val, w_val = init_vals
    for ii in range(max_iter):
        print(f'iter_num: {ii}')
        alpha_val = np.squeeze(alpha_val)
        w_val = np.squeeze(w_val)
        F_eval = np.array(F_lambdified(alpha_val, w_val), dtype=np.complex128)
        J_eval = np.array(J_lambdified(alpha_val, w_val), dtype=np.complex128)
        
        if np.linalg.det(J_eval) == 0:
            raise ValueError("Jacobian is singular; choose a different initial guess.")
        
        delta = -np.linalg.inv(J_eval).dot(F_eval) # Solve for updates
        alpha_val += np.squeeze(delta[0])
        w_val += np.squeeze(delta[1])
        
        if abs(delta[0]) < tol and abs(delta[1]) < tol:
            return alpha_val, w_val  # Converged
    
    raise ValueError(f"Newton-Raphson did not converge within the maximum iterations. Current delta[0]: {delta[0]}, delta[1]: {delta[1]}, alpha_val: {alpha_val}, w_val: {w_val}")
