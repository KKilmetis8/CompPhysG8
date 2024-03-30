"""
Created on Fri Mar 22 15:50:01 2024

@authors: diederick & konstantinos
"""
# Pistachio imports
import numpy as np


# Things to change
temperature = 2.269 # 2.269 # 2.269 # k_B = 1
Nsize       = 50
rngseed     = None
loud        = False
J_coupling = 1
H_ext      = 0
kind = 'sweep' # or sweep

# Derived
critical_temp = (2/np.log(1+np.sqrt(2))) * J_coupling
beta = 1/temperature

# Plotting 
cmap = "coolwarm" # coolwarm, PuOr, PiYG

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False # False makes it fast, true makes it slow.
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6 , 5]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'