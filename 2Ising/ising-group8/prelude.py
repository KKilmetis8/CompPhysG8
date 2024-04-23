"""
Created on Fri Mar 22 15:50:01 2024

@authors: diederick & konstantinos
"""
# Pistachio imports
import numpy as np
import config as c
from os import makedirs
import time

# Derived
critical_temp = (2/np.log(1+np.sqrt(2))) * c.J_coupling
beta = 1/c.temperature
MC_step = c.Nsize**2
rng  = np.random.default_rng(c.rngseed)

# Initial grid
if c.init_grid == '75% positive':
    flip = rng.integers(c.Nsize, size = (2, c.Nsize**2//4))
    init_grid = np.ones((c.Nsize, c.Nsize))
    init_grid[flip[0], flip[1]] = -1
elif c.init_grid == 'random':
    init_grid = np.sign(rng.random((c.Nsize, c.Nsize)) - 0.5)
else:
    raise ValueError('Unrecognized preset initial grid')

# simname
simname = c.simname
if simname is None:
    time_of_creation = time.strftime('%d-%h-%H:%M:%S', time.localtime())
    simname = f'{c.init_grid}_{c.kind}_at_{time_of_creation}'
makedirs(f"sims/{simname}/", exist_ok=True)

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