#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:27:11 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [4 , 4]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
from scipy.optimize import root
import numba
from tqdm import tqdm
from density_interp import den_of_T

@numba.njit
def eq(Ys, old, rates, h):
    res = np.zeros(len(Ys))
    res[0] = -2*rates[0]*Ys[0]**2 - rates[1]*Ys[0]*Ys[1] + rates[2]*Ys[2]**2
    res[1] = rates[0]*Ys[0]**2 - rates[1]*Ys[1]*Ys[0]
    res[2] = rates[1]*Ys[0]*Ys[1] - 2*rates[2]*Ys[2]**2
    res[3] = rates[2]*Ys[2]**2
    return res - h*(Ys-old)

@numba.njit
def inv_Jacobian(Ys, rates, h):
    reac_1 = [-4*rates[0]*Ys[0]-rates[1]*Ys[1]-h,
              -rates[1]*Ys[0],
              2*rates[2]*Ys[2],
              0]
    reac_2 = [2*rates[0]*Ys[0]-rates[1]*Ys[1],
              -rates[1]*Ys[0]-h,
              0,0]
    reac_3 = [rates[1]*Ys[1],
              rates[1]*Ys[0],
              -4*rates[2]*Ys[2]-h,
              0]
    reac_4 = [0,0,2*rates[2]*Ys[2], -h]
    return np.linalg.inv(np.array([reac_1, reac_2, reac_3, reac_4]))

@numba.njit
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 3, tol=1e-6):
    As = np.array([1,2,3,4])
    prevY = oldY.copy()
    rates, h = args
    conv_flag = False
    timestep_increase = 0
    # while not conv_flag:
    for nr_step in range(maxsteps):
        try:
            newY = oldY - np.dot(invJ(oldY, rates, h), 
                                 fsol(oldY, prevY, rates, h))
        except:
            break
        
        if newY[1]<1e-6:
            newY[1] = newY[0] * 5e-6
                
        oldY = newY
    return newY, h

Hyd = 0.91
DtoH = 2e-3# Geiss Gloeckler 98
Deut = Hyd * DtoH
He = 0.089
He3toHe = 1e-2 # Geiss Gloeckler 98
He3 = He * He3toHe

year = 365*24*60*60 # [s]
hmax = 1/(1e7*year)
dT = 1e4*year
hinit = 1/dT # 1/dT, 
h = hinit
timesteps = int(1e7)
Ys = np.zeros((timesteps, 4))
sols = np.zeros((timesteps, 4))
As = np.array([1,2,3,4])
Ys[0] = [Hyd, Deut, He3, He]
Ys[0] /= np.sum(As * Ys)
rates = np.array([7.9e-20,	1.01e-2, 2.22e-10])
density = den_of_T(0.015) # sun
rates *= density

elapsed_time = 0
max_time = 12e9*year
for i in tqdm(range(1,timesteps)):
    Ys[i], h = newton_raphson(Ys[i-1], inv_Jacobian, eq,
                                    args = (rates, h),)
    elapsed_time += 1/h
    if elapsed_time > max_time:
        break

print('Evo time', elapsed_time/(1e9*year), 'Gyrs')
#%%
labels = ["H", "D", "$^{3}$He", "$^{4}$He"]
colors = ["k", "tab:red", "b", "darkorange"]
linestyles = ["-","-","-","--"]
plt.figure(tight_layout=True)

step_plot = 100
try:
    stop = np.where(Ys.T[0] == 0)[0][0]
except:
    stop = -1
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps)[:stop:step_plot]*1e-4, abundances[:stop:step_plot], 
             label = labels[i], ls=linestyles[i], color=colors[i], marker='')

plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Gyrs]', fontsize = 14)
plt.legend(ncols = 1)
