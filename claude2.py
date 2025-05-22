import sympy as sp
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eig

def newton_two_equation_solver(f1_eq, f2_eq, alpha_symbol, w_symbol, 
                               alpha_guess, w_guess, max_iterations=50, 
                               tolerance=1e-8, verbose=False):
    f1 = sp.lambdify((alpha_symbol, w_symbol), f1_eq.lhs - f1_eq.rhs, 'numpy')
    f2 = sp.lambdify((alpha_symbol, w_symbol), f2_eq.lhs - f2_eq.rhs, 'numpy')

    df1_dalpha = sp.diff(f1_eq.lhs - f1_eq.rhs, alpha_symbol)
    df1_dw = sp.diff(f1_eq.lhs - f1_eq.rhs, w_symbol)
    df2_dalpha = sp.diff(f2_eq.lhs - f2_eq.rhs, alpha_symbol)
    df2_dw = sp.diff(f2_eq.lhs - f2_eq.rhs, w_symbol)
    
    J11 = sp.lambdify((alpha_symbol, w_symbol), df1_dalpha, 'numpy')
    J12 = sp.lambdify((alpha_symbol, w_symbol), df1_dw, 'numpy')
    J21 = sp.lambdify((alpha_symbol, w_symbol), df2_dalpha, 'numpy')
    J22 = sp.lambdify((alpha_symbol, w_symbol), df2_dw, 'numpy')

    alpha = complex(alpha_guess)
    w = complex(w_guess)
    
    for i in range(max_iterations):
        f1_val = f1(alpha, w)
        f2_val = f2(alpha, w)
        
        if abs(f1_val) < tolerance and abs(f2_val) < tolerance:
            if verbose:
                print(f"Converged after {i} iterations")
            return alpha, w, True, i
        
        try:
            J = np.array([
                [J11(alpha, w), J12(alpha, w)],
                [J21(alpha, w), J22(alpha, w)]
            ], dtype=complex)
            
            f_vals = np.array([f1_val, f2_val], dtype=complex)
            
            det_J = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            if abs(det_J) < 1e-10:
                if verbose:
                    print(f"Jacobian nearly singular at iteration {i}, det(J)={det_J}")
                break
                
            delta = np.linalg.solve(J, f_vals)
            
            alpha_new = alpha - delta[0]
            w_new = w - delta[1]

            if (abs(alpha_new) > 1e5 or abs(w_new) > 1e5 or 
                np.isnan(alpha_new) or np.isnan(w_new)):
                if verbose:
                    print(f"Diverged at iteration {i}")
                break
                
            alpha, w = alpha_new, w_new
            
            if verbose:
                print(f"Iteration {i}: alpha={alpha}, w={w}, |f1|={abs(f1_val)}, |f2|={abs(f2_val)}")
                
        except Exception as e:
            if verbose:
                print(f"Error at iteration {i}: {str(e)}")
            break
    
    if verbose:
        print(f"Did not converge after {max_iterations} iterations")
    return alpha, w, False, max_iterations

def find_all_saddle_points(f1_eq, f2_eq, A, B, alpha_symbol, w_symbol):
    solutions = []
    unique_solutions = []

    # instead of a grid search, we use the maximum of 
    # temporal stability analysis as the initial guess
    alpha_vals = np.linspace(-5,5,50)
    wi_vals = []
    for alpha_val in alpha_vals:
        A_temp =  A.subs({alpha_symbol: alpha_val})
        B_temp =  B.subs({alpha_symbol: alpha_val})
        A_num = np.array(A_temp.evalf()).astype(np.complex128)
        B_num = np.array(B_temp.evalf()).astype(np.complex128)
        c_val, eig_vecs = eig(A_num, B_num)
        print(c_val)
        c_val[np.isinf(c_val)] = -100000000
        wi_vals.append(np.max(np.imag(c_val * alpha_val)))

    w_chosen = np.max(wi_vals)
    alpha_chosen_idx = np.argmax(wi_vals)
    alpha_chosen = alpha_vals[alpha_chosen_idx]
    
    alpha, w, converged, _ = newton_two_equation_solver(
        f1_eq, f2_eq, alpha_symbol, w_symbol, 
        alpha_chosen, w_chosen, verbose=False
    )

    if converged:
        unique_solutions.append((alpha, w))
    else:
        print(f"Solution didnt converge for (alpha, w) = ({alpha_chosen},{w_chosen})")

    return unique_solutions

def find_and_display_solutions(f1_eq, f2_eq, A, B, alpha_symbol, w_symbol):
    solutions = find_all_saddle_points(
        f1_eq, f2_eq, A, B, alpha_symbol, w_symbol
    )
    return solutions