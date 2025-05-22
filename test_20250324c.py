# used for line and scatter plots of w_i vs x/D
import numpy as np
import matplotlib.pyplot as plt

case_no = 4

curr_mode = 'varicose'
x_D = np.linspace(1,30,30)
input_file_name = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\w_list_case{case_no}_{curr_mode}_gep.txt"
w0i = np.loadtxt(input_file_name, unpack=True)

#plt.ion()
plt.figure()
#plt.plot(x_D, w0i, color='blue', linestyle='--')
plt.scatter(x_D, w0i, color='blue', marker='x', s=50, label='Varicose mode')

curr_mode = 'sinuous'
x_D = np.linspace(1,30,30)
input_file_name = f"C:\\Users\\tvivek\\Documents\\onr_20250324\\w_list_case{case_no}_{curr_mode}_gep.txt"
w0i = np.loadtxt(input_file_name, unpack=True)

#plt.ion()
plt.scatter(x_D, w0i, color='red', marker='+', s=50, label='Sinuous mode')
#plt.plot(x_D, w0i, color='red', linestyle='--')
plt.grid(True)
plt.legend()
plt.ylim((0,7))
plt.show()