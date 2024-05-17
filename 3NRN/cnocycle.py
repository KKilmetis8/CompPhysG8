#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  10 13:03:39 2024

@author: diederick
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import numba
from tqdm import tqdm

@numba.njit
def eq(Y, rates, old, h):
    res = np.zeros(8)
    # H
    res[0] = rates[0] * Y[0] * Y[1] + rates[2] * Y[0] * Y[3] + rates[3] * Y[0] * Y[4] + rates[5] * Y[0] * Y[6]
    # 12C
    res[1] = rates[0] * Y[1] * Y[0] - rates[5] * Y[0] * Y[6]
    # 13N
    res[2] = -rates[0] * Y[0] * Y[1] + rates[1] * Y[2]
    # 13C
    res[3] = -rates[1] * Y[2] + rates[2] * Y[3] * Y[0]
    # 14N
    res[4] = -rates[2] * Y[3] * Y[0] + rates[3] * Y[4] * Y[0]
    # 15O
    res[5] = -rates[3] * Y[4] * Y[0] + rates[4] * Y[5]
    # 15N
    res[6] = -rates[4] * Y[5] + rates[5] * Y[6] * Y[0]
    # 4He
    res[7] = -rates[5] * Y[6] * Y[0]
    res += h * (Y - old)
    return res

h = 0.01
timesteps = 100_000
Ys = np.zeros((timesteps, 8))

# Initial abundances
#           H,     12C,  13N,  13C,      14N,  15O,     15N,  4He
Ys[0] = [0.75, 0.00297, 1e-7, 3e-5, 9.963e-4, 1e-7, 3.68e-6, 0.23]
# Reaction rates
rates = np.array([ 1e-6, 1e-4,1e-6,1e-5, 1e-4, 1e-5])

for i in tqdm(range(1,timesteps)):
    Ys[i] = root(eq, Ys[i-1], args = (rates, Ys[i-1], h)).x
    

labels = ["H", "$^{12}$C", "$^{13}$N", "$^{13}$C", "$^{14}$N", "$^{15}$O", "$^{15}$N", "$^{4}$He"]
plt.figure(tight_layout=True)
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps), abundances, label = labels[i], marker='')

plt.grid()
plt.yscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time', fontsize = 14)
#plt.ylim(1e-1,1)
plt.legend(ncols = 1, loc='upper left', bbox_to_anchor = (1,1))