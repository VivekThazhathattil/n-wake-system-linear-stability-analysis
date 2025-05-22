import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.special import comb
from fit_bezier_curve import *

x_data = np.array([0, 1, 2, 3, 4, 5])
y_data = np.array([0.2, 0.5, 1.0, 1.5, 1.8, 2.0])

case_no = 1
bluff_body_characteristic_length = 3.18 # mm
dataset_path = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\case{case_no}_Lambda_delta.csv"
Lambda_arr, delta_arr= np.loadtxt(dataset_path, delimiter=',', skiprows=1, unpack=True)
delta_list = delta_arr/ bluff_body_characteristic_length

x_arr = np.linspace(1,30,30)

control_points = fit_bezier_curve(x_arr, delta_arr, degree=200)
plot_bezier_fit(x_arr, delta_arr, control_points)