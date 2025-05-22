import numpy as np
import matplotlib.pyplot as plt

# Parameters (adjust as needed)
S = -0.2      # Example value
Lambda = 1/1.5  # Example value
xi = 0.001      # Example value
z_over_t = 0.48  # From the third image

# Step 1: Solve for omega(k) numerically
def omega(k):
    # Solve the quadratic equation for v = omega/k
    a = (1 + S) + (1 - S)
    b = -2 * (1 + S) * (1/Lambda + 1) - 2 * (1 - S) * (1/Lambda - 1)
    c = (1 + S) * (1/Lambda + 1)**2 + (1 - S) * (1/Lambda - 1)**2 - xi
    discriminant = b**2 - 4 * a * c
    v = (-b + np.sqrt(discriminant)) / (2 * a)  # Take one root
    return k * v

# Step 2: Define g(k, x_over_t, z_over_t)
def g(k, x_over_t, z_over_t):
    return omega(k) - k * x_over_t - 1j * xi * z_over_t

# Step 3: Create a grid for k and x/t
k_vals = np.linspace(-1, 3, 100)       # Adjust range as needed
x_over_t_vals = np.linspace(0, 3, 100) # From the third image
K, X = np.meshgrid(k_vals, x_over_t_vals)

# Compute Re(g) and Im(g) on the grid
G = g(K, X, z_over_t)
G_real = np.real(G)
G_imag = np.imag(G)

# Step 4: Plot contours
plt.figure(figsize=(10, 6))
plt.contourf(K, X, G_real, levels=20, cmap='viridis')
plt.colorbar(label='Re(g)')
plt.xlabel('k')
plt.ylabel('x/t')
plt.title(f'Contours of Re(g) for z/t = {z_over_t}')
plt.show()

# Repeat for Im(g) if needed
plt.figure(figsize=(10, 6))
plt.contourf(K, X, G_imag, levels=20, cmap='plasma')
plt.colorbar(label='Im(g)')
plt.xlabel('k')
plt.ylabel('x/t')
plt.title(f'Contours of Im(g) for z/t = {z_over_t}')
plt.show()