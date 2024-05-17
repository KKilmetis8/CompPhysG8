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

@numba.njit
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 3, tol=1e-2):
    As = np.array([1,2,3,4])
    prevY = oldY.copy()
    rates, h = args
    conv_flag = False
    #while not conv_flag:
    for nr_step in range(maxsteps):
        try:
            newY = oldY - np.dot(invJ(oldY, rates, h), 
                                 fsol(oldY, prevY, rates, h))
        except:
            break
        if newY[1]<1e-6:
            newY[1]= 1e-6 # newY[0] * DtoH # rates[0] * newY[0]**2 / h
        oldY = newY

        # crit = np.abs(1 - np.sum(newY))
        # if crit < tol:
        #     break
        # conv_flag = True
        # h = h*2
        # print('timestep decreased')
    return newY


Hyd = 0.91
DtoH = 2.1e-5 # Geiss Gloeckler 98
Deut = Hyd * DtoH
He3toH = 1.5e-5 # Geiss Gloeckler 98
He3 = Hyd * He3toH
He = 0.089

year = 365*24*60*60 # [s]
hmax = 1/(1e4*year) # 1/dT, 
h = hmax
timesteps = int(1e6)
Ys = np.zeros((timesteps, 4))
sols = np.zeros((timesteps, 4))
As = np.array([1,2,3,4])
Ys[0] = [Hyd, Deut, He3, He]
rates = np.array([7.9e-20,	1.01e-2, 2.22e-10])

elapsed_time = 0
max_time = 10e9*year
for i in tqdm(range(1,timesteps)):
    Ys[i] = newton_raphson(Ys[i-1], inv_Jacobian, eq,
                                    args = (rates, h),)
    elapsed_time += 1/h
    if elapsed_time > max_time:
        break
    # h = np.min( [hmax, h/2] )
print('Evo time', elapsed_time/(1e9*year), 'Gyrs')
#%%
labels = ["H", "D", "$^{3}$He", "$^{4}$He"]
colors = ["k", "tab:red", "b", "darkorange"]
linestyles = ["-","-","-","--"]
plt.figure(tight_layout=True)

step = 1000
try:
    stop = np.where(Ys.T[0] == 0)[0][0]
except:
    stop = -1
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps)[:stop:step]*1e-5, abundances[:stop:step], 
             label = labels[i], ls=linestyles[i], color=colors[i], marker='')

plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Gyrs]', fontsize = 14)
plt.legend(ncols = 1)
