"""
Created on Fri Mar 22 15:50:01 2024

@authors: diederick & konstantinos
"""
# Pistachio import
import numpy as np


# Things to change
temperature = 1.5 # k_B = 1
Nsize       = 50
rngseed     = 8

J_coupling = 1
H_ext      = 0


# Derived
critical_temp = (2/np.log(1+np.sqrt(2))) * J_coupling


# Plotting
cmap = "coolwarm"

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6 , 5]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'