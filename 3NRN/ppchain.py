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
from tqdm import tqdm

@numba.njit
def eq(Y, rates, old, h):
    res = np.zeros(4)
    res[0] = rates[0] * 2 * Y[0]**2 + rates[1] * Y[0] * Y[1] - 2 * rates[2] * Y[2]**2
    res[1] = - rates[0]  * Y[0]**2 + rates[1] * Y[0] * Y[1]
    res[2] = - rates[1] * Y[0] * Y[1] + rates[2] *2* Y[2]**2
    res[3] = - rates[2] * Y[2]**2
    res += h * (Y - old)
    return res

def inv_Jacobian(Ys, rates, old, h):
    reac_1 = [4*rates[0]*Ys[0]+rates[1]*Ys[1],
             rates[1]*Ys[0],
             -4*rates[2]*Ys[2],
             0]
    reac_2 = [-2*rates[0]*Ys[0]+rates[1]*Ys[1],
             rates[1]*Ys[0],
             0,
             0]
    reac_3 = [-rates[1]*Ys[1],
             -rates[1]*Ys[0],
             4*rates[2]*Ys[2] + h,
             0]
    reac_4 = [0,0,0,-2*rates[2]*Ys[3] + h]
    return np.linalg.inv(np.array([reac_1, reac_2, reac_3, reac_4]))

def newton_raphson(oldY, invJ, fsol, args, maxsteps = 10, tol=1e-25):
    
    for nr_step in range(maxsteps):
        newY = oldY - np.dot(invJ(args), fsol(args))
        diff = np.linalg.norm(oldY-newY)
        if diff<tol:
            break
        oldY = newY
    return newY
 

Hyd = 0.7
DtoH = 1.5e-5 # Black 79 
Deut = Hyd * DtoH
He3 = 1e-20
He = 0.28

Hyd = 0.7
DtoH = 2.1e-5 # Geiss Gloeckler 98
Deut = Hyd * DtoH
He3toH = 1.5e-5 # Geiss Gloeckler 98
He3 = Hyd * He3toH
He = 0.28

h = 1e-17
timesteps = 10000
Ys = np.zeros((timesteps, 4))
Ys[0] = [Hyd, Deut, He3, He]
#Ys[0] = [1, 0, 0, 0]
rates = np.array([1.25e-19,	1.9e-2, 1e-9])

for i in tqdm(range(1,timesteps)):
    Ys[i] = root(eq, Ys[i-1], args = (rates, Ys[i-1], h), method='lm', jac=inv_Jacobian).x
    #print(f"({np.round((i+1)/len(Ys))}%): {Ys[i]}",end='\r')
    
labels = ["H", "D", "$^{3}$He", "$^{4}$He"]
colors = ["k", "tab:red", "b", "green"]
linestyles = ["-","-","-","--"]
plt.figure(tight_layout=True)
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps), abundances, label = labels[i], ls=linestyles[i], color=colors[i], marker='')

#plt.plot(np.arange(timesteps), Ys.sum(axis=1), 'k--')

plt.grid()
plt.yscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time', fontsize = 14)
#plt.ylim(1e-1,1)
plt.legend(ncols = 1, loc='upper left', bbox_to_anchor = (1,1))