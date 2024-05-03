#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:27:11 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import numba

@numba.njit
def eq(x, rates, old, h):
    res = np.zeros(4)
    res[0] = rates[0] * 2 * x[0]**2 + rates[1] * x[0] * x[1]
    res[1] = - rates[0]  * x[0]**2 + rates[1] * x[0] * x[1]
    res[2] = -rates[1] * x[0] * x[1] + rates[2] *2* x[2]**2
    res[3] = -rates[2] * x[2]**2
    res += h * (x - old)
    return res


Hyd = 0.7
DtoH = 1.5e-5 # Black 79
Deut = Hyd * DtoH
He3 = 1e-6
He = 0.28
h = 0.01
timesteps = 100_000
Ys = np.zeros((timesteps, 4))
Ys[0] = [Hyd, Deut, He3, He]
rates = np.array([1e-6, 1e-2, 2e-2, 1e-2])

for i in range(1,timesteps):
    Ys[i] = root(eq, Ys[i-1], args = (rates, Ys[i-1], h)).x
    
    
plt.plot(np.arange(timesteps), Ys.T[0], c = 'k', label = 'H')
plt.plot(np.arange(timesteps), Ys.T[1], c = 'tab:red', ls = '--', label = 'D')
plt.plot(np.arange(timesteps), Ys.T[2], c = 'b', label = '3He')
plt.plot(np.arange(timesteps), Ys.T[3], c = 'darkorange', ls = '--', label = 'He')
plt.yscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time', fontsize = 14)
#plt.ylim(1e-1,1)
plt.legend(ncols = 4)