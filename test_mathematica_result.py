# For single wake case, we obtain the expression for c from mathematica
import sympy as sp
from sympy import symbols, sqrt
from sympy.parsing.mathematica import parse_mathematica
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
alpha, delta, s, Sr, c, lda = symbols('alpha delta s Sr c lda')

# mathematica expression
mathematica_expression_c1 = """
4 E^(-(1/
          2) alpha (-1 + delta) + alpha delta) alpha^2 
delta^2 + 
      4 E^(1/2 alpha (-1 + delta) + alpha delta)
        s alpha^2 delta^2 + 
      4 E^(-(1/2) alpha (-1 + delta) + alpha delta)
        Sr alpha^2 delta^2 - 
      4 E^(1/2 alpha (-1 + delta) + alpha delta)
        s Sr alpha^2 delta^2 + 
      2 E^(-(1/
          2) alpha (-1 + delta) - alpha delta) alpha 
delta lda - 
      2 E^(-(1/
          2) alpha (-1 + delta) + alpha delta) alpha 
delta lda + 
      2 E^(1/2 alpha (-1 + delta) - alpha delta)
        s alpha delta lda - 
      2 E^(1/2 alpha (-1 + delta) + alpha delta)
        s alpha delta lda - 
      2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
        Sr alpha delta lda + 
      2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
        Sr alpha delta lda + 
      2 E^(1/2 alpha (-1 + delta) - alpha delta)
        s Sr alpha delta lda - 
      2 E^(1/2 alpha (-1 + delta) + alpha delta)
        s Sr alpha delta lda - \\Sqrt((-4 E^(-(1/
               2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 - 
           4 E^(1/2 alpha (-1 + delta) + alpha delta)
             s alpha^2 delta^2 - 
           4 E^(-(1/2) alpha (-1 + delta) + alpha delta)
             Sr alpha^2 delta^2 + 
           4 E^(1/2 alpha (-1 + delta) + alpha delta)
             s Sr alpha^2 delta^2 - 
           2 E^(-(1/
               2) alpha (-1 + delta) - alpha delta) 
alpha delta lda + 
           2 E^(-(1/
               2) alpha (-1 + delta) + alpha delta) 
alpha delta lda - 
           2 E^(1/2 alpha (-1 + delta) - alpha delta)
             s alpha delta lda + 
           2 E^(1/2 alpha (-1 + delta) + alpha delta)
             s alpha delta lda + 
           2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
             Sr alpha delta lda - 
           
           2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
             Sr alpha delta lda - 
           2 E^(1/2 alpha (-1 + delta) - alpha delta)
             s Sr alpha delta lda + 
           2 E^(1/2 alpha (-1 + delta) + alpha delta)
             s Sr alpha delta lda)^2 - 
         4 (2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 + 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha^2 delta^2 + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha^2 delta^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha^2 delta^2) (2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 + 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha^2 delta^2 + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha^2 delta^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha^2 delta^2 + 
            2 E^(-(1/
                2) alpha (-1 + delta) - alpha delta) 
alpha delta lda - 
            2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha delta lda + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s alpha delta lda - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha delta lda - 
            2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
              Sr alpha delta lda + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha delta lda + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s Sr alpha delta lda - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha delta lda + 
            4 E^(-(1/
                2) alpha (-1 + delta) - alpha delta) 
lda^2 - 
            4 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
lda^2 + 
            4 E^(1/2 alpha (-1 + delta) - alpha delta)
              s lda^2 - 
            4 E^(1/2 alpha (-1 + delta) + alpha delta)
              s lda^2 + 
            2 E^(-(1/
                2) alpha (-1 + delta) - alpha delta) 
alpha delta lda^2 + 
            6 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha delta lda^2 + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s alpha delta lda^2 + 
            6 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha delta lda^2 - 
            2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
              Sr alpha delta lda^2 + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha delta lda^2 + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s Sr alpha delta lda^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha delta lda^2 - 
            2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 lda^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha^2 delta^2 lda^2 - 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha^2 delta^2 lda^2 + 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha^2 delta^2 lda^2))/(2 (2 
E^(-(1/2) alpha (-1 + delta) + alpha delta) alpha^2 
delta^2 + 
        2 E^(1/2 alpha (-1 + delta) + alpha delta)
          s alpha^2 delta^2 + 
        2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
          Sr alpha^2 delta^2 - 
        2 E^(1/2 alpha (-1 + delta) + alpha delta)
          s Sr alpha^2 delta^2))
"""
mathematica_expression_c2 = """
            4 E^(-(1/
          2) alpha (-1 + delta) + alpha delta) alpha^2 
delta^2 + 
      4 E^(1/2 alpha (-1 + delta) + alpha delta)
        s alpha^2 delta^2 + 
      4 E^(-(1/2) alpha (-1 + delta) + alpha delta)
        Sr alpha^2 delta^2 - 
      4 E^(1/2 alpha (-1 + delta) + alpha delta)
        s Sr alpha^2 delta^2 + 
      2 E^(-(1/
          2) alpha (-1 + delta) - alpha delta) alpha 
delta lda - 
      2 E^(-(1/
          2) alpha (-1 + delta) + alpha delta) alpha 
delta lda + 
      2 E^(1/2 alpha (-1 + delta) - alpha delta)
        s alpha delta lda - 
      2 E^(1/2 alpha (-1 + delta) + alpha delta)
        s alpha delta lda - 
      2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
        Sr alpha delta lda + 
      2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
        Sr alpha delta lda + 
      2 E^(1/2 alpha (-1 + delta) - alpha delta)
        s Sr alpha delta lda - 
      2 E^(1/2 alpha (-1 + delta) + alpha delta)
        s Sr alpha delta lda + \\Sqrt((-4 E^(-(1/
               2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 - 
           4 E^(1/2 alpha (-1 + delta) + alpha delta)
             s alpha^2 delta^2 - 
           4 E^(-(1/2) alpha (-1 + delta) + alpha delta)
             Sr alpha^2 delta^2 + 
           4 E^(1/2 alpha (-1 + delta) + alpha delta)
             s Sr alpha^2 delta^2 - 
           2 E^(-(1/
               2) alpha (-1 + delta) - alpha delta) 
alpha delta lda + 
           2 E^(-(1/
               2) alpha (-1 + delta) + alpha delta) 
alpha delta lda - 
           2 E^(1/2 alpha (-1 + delta) - alpha delta)
             s alpha delta lda + 
           2 E^(1/2 alpha (-1 + delta) + alpha delta)
             s alpha delta lda + 
           2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
             Sr alpha delta lda - 
           2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
             Sr alpha delta lda - 
           2 E^(1/2 alpha (-1 + delta) - alpha delta)
             s Sr alpha delta lda + 
           2 E^(1/2 alpha (-1 + delta) + alpha delta)
             s Sr alpha delta lda)^2 - 
         
         4 (2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 + 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha^2 delta^2 + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha^2 delta^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha^2 delta^2) (2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 + 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha^2 delta^2 + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha^2 delta^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha^2 delta^2 + 
            2 E^(-(1/
                2) alpha (-1 + delta) - alpha delta) 
alpha delta lda - 
            2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha delta lda + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s alpha delta lda - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha delta lda - 
            2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
              Sr alpha delta lda + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha delta lda + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s Sr alpha delta lda - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha delta lda + 
            4 E^(-(1/
                2) alpha (-1 + delta) - alpha delta) 
lda^2 - 
            4 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
lda^2 + 
            4 E^(1/2 alpha (-1 + delta) - alpha delta)
              s lda^2 - 
            4 E^(1/2 alpha (-1 + delta) + alpha delta)
              s lda^2 + 
            2 E^(-(1/
                2) alpha (-1 + delta) - alpha delta) 
alpha delta lda^2 + 
            6 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha delta lda^2 + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s alpha delta lda^2 + 
            6 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha delta lda^2 - 
            2 E^(-(1/2) alpha (-1 + delta) - alpha delta)
              Sr alpha delta lda^2 + 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha delta lda^2 + 
            2 E^(1/2 alpha (-1 + delta) - alpha delta)
              s Sr alpha delta lda^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha delta lda^2 - 
            2 E^(-(1/
                2) alpha (-1 + delta) + alpha delta) 
alpha^2 delta^2 lda^2 - 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s alpha^2 delta^2 lda^2 - 
            2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
              Sr alpha^2 delta^2 lda^2 + 
            2 E^(1/2 alpha (-1 + delta) + alpha delta)
              s Sr alpha^2 delta^2 lda^2))/(2 (2 
E^(-(1/2) alpha (-1 + delta) + alpha delta) alpha^2 
delta^2 + 
        2 E^(1/2 alpha (-1 + delta) + alpha delta)
          s alpha^2 delta^2 + 
        2 E^(-(1/2) alpha (-1 + delta) + alpha delta)
          Sr alpha^2 delta^2 - 
        2 E^(1/2 alpha (-1 + delta) + alpha delta)
          s Sr alpha^2 delta^2))
"""
c1 = parse_mathematica(mathematica_expression_c1)
c2 = parse_mathematica(mathematica_expression_c2)

val_set = {'Sr': 0.38, 'lda': -3.1, 'delta': 0.1, 's':-1}
kr_values = np.arange(0, 2.01, 0.1)  # 0 to 2 with a step of 0.01
ki_values = np.arange(-2, 0.01, 0.1)  # -2 to 0 with a step of 0.01
kr, ki = np.meshgrid(kr_values, ki_values)
alpha_vals = kr + 1j * ki  # Element-wise complex addition
max_w_vals = np.zeros_like(alpha_vals, dtype=np.complex128)
n_rows, n_cols = alpha_vals.shape

for i in range(n_rows):
    print(f'{i+1}/{n_rows}')
    for j in range(n_cols):
        k = alpha_vals[i][j]
        c1_sub1 = c1.subs(val_set)
        c1_sub = c1_sub1.subs({alpha: k})
        c1_sub = c1_sub.evalf()

        if not c1_sub:
            max_w_vals[i,j] = np.nan
        else:
            #print(c_solutions)
            w_solutions = np.array(c1_sub, dtype=np.complex128) * k
            max_imag_w_idx = np.argmax(np.imag(w_solutions))
            max_w_vals[i,j] = w_solutions[max_imag_w_idx]
#        break
#    break

plt.ion()
CLev = np.linspace(-1, 1, 50) 
#CLim = (-1, 1) 
CLim = (-0.5, 0.5) 
fig, ax = plt.subplots(figsize=(6, 6))
#ax.set_title(r'$\omega_1(k)$', fontsize=14)
#ax.set_xlabel(r'$k_r$', fontsize=14)
#ax.set_ylabel(r'$k_i$', fontsize=14)
cmap = plt.cm.gray
cf = ax.contourf(kr, ki, np.imag(max_w_vals), levels=CLev, cmap=cmap)
ax.contour(kr, ki, np.imag(max_w_vals), levels=[0], colors='k', linewidths=2)
ax.contour(kr, ki, np.real(max_w_vals), levels=CLev, colors=[(0.6, 0.6, 0.6)])
cf.set_clim(CLim)
plt.show()