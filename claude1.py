import sympy as sp
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def newton_two_equation_solver(f1_eq, f2_eq, alpha_symbol, w_symbol, 
                               alpha_guess, w_guess, max_iterations=50, 
                               tolerance=1e-8, verbose=False):
    """
    Solve a system of two equations using Newton's method.
    
    Parameters:
    -----------
    f1_eq : sympy equation
        First equation in the system
    f2_eq : sympy equation
        Second equation in the system
    alpha_symbol : sympy symbol
        Symbol for alpha variable
    w_symbol : sympy symbol
        Symbol for w variable
    alpha_guess : complex
        Initial guess for alpha
    w_guess : complex
        Initial guess for w
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    verbose : bool
        Whether to print intermediate results
        
    Returns:
    --------
    alpha : complex
        Solution for alpha
    w : complex
        Solution for w
    convergence : bool
        Whether the method converged
    iterations : int
        Number of iterations performed
    """
    # Convert sympy equations to functions
    f1 = sp.lambdify((alpha_symbol, w_symbol), f1_eq.lhs - f1_eq.rhs, 'numpy')
    f2 = sp.lambdify((alpha_symbol, w_symbol), f2_eq.lhs - f2_eq.rhs, 'numpy')
    
    # Compute Jacobian symbolically
    df1_dalpha = sp.diff(f1_eq.lhs - f1_eq.rhs, alpha_symbol)
    df1_dw = sp.diff(f1_eq.lhs - f1_eq.rhs, w_symbol)
    df2_dalpha = sp.diff(f2_eq.lhs - f2_eq.rhs, alpha_symbol)
    df2_dw = sp.diff(f2_eq.lhs - f2_eq.rhs, w_symbol)
    
    # Convert Jacobian to functions
    J11 = sp.lambdify((alpha_symbol, w_symbol), df1_dalpha, 'numpy')
    J12 = sp.lambdify((alpha_symbol, w_symbol), df1_dw, 'numpy')
    J21 = sp.lambdify((alpha_symbol, w_symbol), df2_dalpha, 'numpy')
    J22 = sp.lambdify((alpha_symbol, w_symbol), df2_dw, 'numpy')
    
    # Initialize
    alpha = complex(alpha_guess)
    w = complex(w_guess)
    
    # Newton iteration
    for i in range(max_iterations):
        # Evaluate functions
        f1_val = f1(alpha, w)
        f2_val = f2(alpha, w)
        
        # Check if we've converged
        if abs(f1_val) < tolerance and abs(f2_val) < tolerance:
            if verbose:
                print(f"Converged after {i} iterations")
            return alpha, w, True, i
        
        # Compute Jacobian
        try:
            J = np.array([
                [J11(alpha, w), J12(alpha, w)],
                [J21(alpha, w), J22(alpha, w)]
            ], dtype=complex)
            
            # Compute update
            f_vals = np.array([f1_val, f2_val], dtype=complex)
            
            # Check if Jacobian is invertible
            det_J = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            if abs(det_J) < 1e-10:
                if verbose:
                    print(f"Jacobian nearly singular at iteration {i}, det(J)={det_J}")
                break
                
            delta = np.linalg.solve(J, f_vals)
            
            # Update values
            alpha_new = alpha - delta[0]
            w_new = w - delta[1]
            
            # Check for divergence
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
    
    # If we get here, we didn't converge
    if verbose:
        print(f"Did not converge after {max_iterations} iterations")
    return alpha, w, False, max_iterations

def find_all_saddle_points(f1_eq, f2_eq, alpha_symbol, w_symbol, 
                          alpha_range=(-2, 2), w_range=(-2, 2), 
                          n_alpha=5, n_w=5, complex_grid=True):
    """
    Find all saddle points by using multiple starting points.
    
    Parameters:
    -----------
    f1_eq, f2_eq : sympy equations
        The two equations to solve
    alpha_symbol, w_symbol : sympy symbols
        The symbols for alpha and w
    alpha_range, w_range : tuple
        Range of values for alpha and w initial guesses
    n_alpha, n_w : int
        Number of initial guesses in each dimension
    complex_grid : bool
        Whether to include complex initial guesses
        
    Returns:
    --------
    solutions : list
        List of (alpha, w) pairs that are solutions
    """
    solutions = []
    unique_solutions = []
    
    # Create grid of initial guesses
    if complex_grid:
        # Include complex values in grid
        alpha_real = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        alpha_imag = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        w_real = np.linspace(w_range[0], w_range[1], n_w)
        w_imag = np.linspace(w_range[0], w_range[1], n_w)
        
        for ar in alpha_real:
            for ai in alpha_imag:
                for wr in w_real:
                    for wi in w_imag:
                        alpha_guess = complex(ar, ai)
                        w_guess = complex(wr, wi)
                        
                        alpha, w, converged, _ = newton_two_equation_solver(
                            f1_eq, f2_eq, alpha_symbol, w_symbol, 
                            alpha_guess, w_guess, verbose=False
                        )
                        
                        if converged:
                            solutions.append((alpha, w))
    else:
        # Only real initial guesses
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        w_values = np.linspace(w_range[0], w_range[1], n_w)
        
        for alpha_guess in alpha_values:
            for w_guess in w_values:
                alpha, w, converged, _ = newton_two_equation_solver(
                    f1_eq, f2_eq, alpha_symbol, w_symbol, 
                    alpha_guess, w_guess, verbose=False
                )
                
                if converged:
                    solutions.append((alpha, w))
    
    # Group similar solutions (accounting for numerical precision)
    tolerance = 1e-5
    for sol in solutions:
        is_unique = True
        for unique_sol in unique_solutions:
            if (abs(sol[0] - unique_sol[0]) < tolerance and 
                abs(sol[1] - unique_sol[1]) < tolerance):
                is_unique = False
                break
        if is_unique:
            unique_solutions.append(sol)
    
    return unique_solutions

# Example usage:
# Assuming you have f1_eq, f2_eq defined as in your code
def find_and_display_solutions(f1_eq, f2_eq, alpha_symbol, w_symbol):
    # Find solutions
    solutions = find_all_saddle_points(
        f1_eq, f2_eq, alpha_symbol, w_symbol,
        alpha_range=(-3, 3), w_range=(-3, 3),
        n_alpha=4, n_w=4, complex_grid=True
    )
    
    # Display solutions
    ##print(f"Found {len(solutions)} unique solutions:")
    ##for i, (alpha, w) in enumerate(solutions):
    ##    print(f"Solution {i+1}:")
    ##    print(f"  alpha = {alpha:.8f}")
    ##    print(f"  w = {w:.8f}")
    ##    print(f"  Im(w) = {w.imag:.8f}")
    ##    
    ##    # Verify solution
    ##    f1 = sp.lambdify((alpha_symbol, w_symbol), f1_eq.lhs - f1_eq.rhs, 'numpy')
    ##    f2 = sp.lambdify((alpha_symbol, w_symbol), f2_eq.lhs - f2_eq.rhs, 'numpy')
    ##    
    ##    residual1 = abs(f1(alpha, w))
    ##    residual2 = abs(f2(alpha, w))
    ##    print(f"  Residuals: |f1| = {residual1:.2e}, |f2| = {residual2:.2e}")
    ##    print()
    ##    
    ### Plot solutions in the complex plane
    ##plt.ion()
    ##plt.figure(figsize=(12, 5))
    
    ##plt.subplot(1, 2, 1)
    ##plt.scatter([s[0].real for s in solutions], [s[0].imag for s in solutions], 
    ##            c='blue', marker='o', s=50)
    ##plt.grid(True)
    ##plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ##plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ##plt.title('Solutions in α-plane')
    ##plt.xlabel('Re(α)')
    ##plt.ylabel('Im(α)')
    
    ##plt.subplot(1, 2, 2)
    ##plt.scatter([s[1].real for s in solutions], [s[1].imag for s in solutions], 
    ##            c='red', marker='o', s=50)
    ##plt.grid(True)
    ##plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ##plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ##plt.title('Solutions in ω-plane')
    ##plt.xlabel('Re(ω)')
    ##plt.ylabel('Im(ω)')
    
    ##plt.tight_layout()
    ##plt.show()
    
    ### Check for absolute instability
    ##absolutely_unstable = any(s[1].imag > 0 for s in solutions)
    ##if absolutely_unstable:
    ##    print("The flow is ABSOLUTELY UNSTABLE")
    ##    max_growth_rate = max(s[1].imag for s in solutions)
    ##    print(f"Maximum temporal growth rate: {max_growth_rate:.6f}")
    ##else:
    ##    print("The flow is NOT absolutely unstable")

    return solutions