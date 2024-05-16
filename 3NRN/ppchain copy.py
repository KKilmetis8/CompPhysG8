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

#@numba.njit
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 3, tol=1e-10, minsteps = 1):
    As = np.array([1,2,3,4])
    prevY = oldY.copy()
    for nr_step in range(maxsteps):
        newY = oldY - np.dot(invJ(oldY, *args), fsol(oldY, prevY, *args))
        #print(oldY[1], newY[1])
        
        #crit = np.abs(1 - np.sum(newY*As))
        # print(nr_step, crit)
        # print('---')
        if newY[1]<1e-10:
            newY[1]=1e-6
        # if crit < tol:
        #     print('hi')
        #     break
        oldY = newY
    
    #print(fsol(newY, *args))
    sol = fsol(newY, prevY, *args)
    #print(diff)

    return newY, sol


Hyd = 0.7
DtoH = 2.1e-5 # Geiss Gloeckler 98
Deut = Hyd * DtoH
He3toH = 1.5e-5 # Geiss Gloeckler 98
He3 = Hyd * He3toH
He = 0.28

year = 365*24*60*60 # [s]
h = 1/(1e3*year) # 1/dT, 
timesteps = int(10e6)
Ys = np.zeros((timesteps, 4))
sols = np.zeros((timesteps, 4))
As = np.array([1,2,3,4])
Ys[0] = [Hyd, Deut, He3, He]
# Ys[0] /= np.sum(Ys[0] * As)
rates = np.array([7.9e-20,	1.01e-2, 2.22e-10])
#rates /= rates[0]
#rates = np.array([1e-8, 1e-2, 5e-6])

#print( np.sum(Ys[0]))

for i in tqdm(range(1,timesteps)):
    Ys[i], sols[i] = newton_raphson(Ys[i-1], inv_Jacobian, eq,
                                    args = (rates, h),)
    #print( np.sum(Ys[i])) # * As))
    # Ys[i] /= np.sum(Ys[i] * As)
    #print(f"({np.round((i+1)/len(Ys))}%): {Ys[i]}",end='\r')
#%%
labels = ["H", "D", "$^{3}$He", "$^{4}$He"]
colors = ["k", "tab:red", "b", "green"]
linestyles = ["-","-","-","--"]
plt.figure(tight_layout=True)

step = 1000
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps)[::step]*1e-3, abundances[::step], label = labels[i], ls=linestyles[i], color=colors[i], marker='')

#plt.plot(np.arange(timesteps), Ys.sum(axis=1), 'k--')

plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Myr]', fontsize = 14)
plt.xlim(0.3e2,10e3)
plt.legend(ncols = 1, loc='upper left', bbox_to_anchor = (1,1))
#%%
fig, axs = plt.subplots(len(sols[0]),1, sharex=True)
for i,sol in enumerate(sols.T):
    axs[i].plot(np.arange(timesteps)[1:], sol[1:], label = labels[i], ls=linestyles[i], color=colors[i], marker='')