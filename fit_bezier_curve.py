import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.special import comb

def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(t, *control_points):
    n = (len(control_points) // 2) - 1
    x_vals = np.array(control_points[0::2])
    y_vals = np.array(control_points[1::2])
    curve_x = sum(bernstein_poly(i, n, t) * x for i, x in enumerate(x_vals))
    curve_y = sum(bernstein_poly(i, n, t) * y for i, y in enumerate(y_vals))
    return np.vstack([curve_x, curve_y])

def residuals(control_points, x_data, y_data, t_values):
    bezier_points = bezier_curve(t_values, *control_points)
    return np.ravel(bezier_points - np.vstack([x_data, y_data]))

def fit_bezier_curve(x_data, y_data, degree=3):
    assert len(x_data) == len(y_data), "X and Y data must be of same length"
    num_points = len(x_data)
    t_values = np.linspace(0, 1, num_points)
    
    initial_guess = np.ravel(np.column_stack([x_data, y_data]))
    
    result = opt.least_squares(residuals, initial_guess, args=(x_data, y_data, t_values))
    optimized_control_points = result.x
    return optimized_control_points

def plot_bezier_fit(x_data, y_data, control_points, num_samples=100):
    t_values = np.linspace(0, 1, num_samples)
    bezier_points = bezier_curve(t_values, *control_points)
    
    plt.scatter(x_data, y_data, label='Data Points', color='red')
    plt.plot(bezier_points[0], bezier_points[1], label='Bézier Fit', color='blue')
    #plt.scatter(control_points[0::2], control_points[1::2], color='green', marker='x', label='Control Points')
    plt.legend()
    plt.xlabel('Downstream Distance')
    plt.ylabel('Shear Layer Thickness')
    plt.title('Bézier Curve Fit to Shear Layer Thickness Variation')
    plt.show()
