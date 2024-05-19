#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  10 13:03:39 2024

@author: diederick
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [5 , 5]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
import numba
from tqdm import tqdm


@numba.njit
def eq(Ys, old, rates, h):
    res = np.zeros(len(Ys))
    # H
    res[0] = -rates[0]*Ys[0]*Ys[1] - rates[2]*Ys[0]*Ys[3] - rates[3]*Ys[0]*Ys[4]*rates[5]*Ys[0]*Ys[6]
    # 12C
    res[1] = -rates[0]*Ys[1]*Ys[0] + rates[5]*Ys[0]*Ys[6]
    # 13N 
    res[2] = rates[0]*Ys[0]*Ys[1] - rates[1]*Ys[2]
    # 13C
    res[3] = rates[1]*Ys[2] - rates[2]*Ys[3]*Ys[0]
    # 14N
    res[4] = rates[2]*Ys[3]*Ys[0] - rates[3]*Ys[4]*Ys[0]
    # 15O
    res[5] = rates[3]*Ys[4]*Ys[0] - rates[4]*Ys[5]
    # 15N
    res[6] = rates[4]*Ys[5] - rates[5]*Ys[6]*Ys[0]
    # 4He
    res[7] = rates[5]*Ys[6]*Ys[0]
    return res - h*(Ys-old)

@numba.njit
def inv_Jacobian(Ys, rates, h):
    reac_1 = [-rates[0]*Ys[1] - rates[2]*Ys[3] - rates[3]*Ys[4] - rates[5]*Ys[6] - h,
              -rates[0]*Ys[0],
              0,
              -rates[2]*Ys[0],
              -rates[3]*Ys[0],
              0,
              -rates[5]*Ys[0],
              0]
    reac_2 = [-rates[0]*Ys[1] + rates[5]*Ys[6],
              -rates[0]*Ys[0] - h,
              0, 0, 0, 0, 
              rates[5]*Ys[0],
              0]
    reac_3 = [rates[0]*Ys[1],
              rates[0]*Ys[0],
              -rates[1]-h,
              0,0,0,0,0]
    reac_4 = [-rates[2]*Ys[3],
              0,
              rates[1],
              -rates[2]*Ys[0]-h,
              0,0,0,0]
    reac_5 = [rates[2]*Ys[3]-rates[3]*Ys[4],
              0,0,
              rates[2]*Ys[0],
              -rates[3]*Ys[0] - h,
              0,0,0]
    reac_6 = [rates[3]*Ys[4],
              0,0,0,
              rates[3]*Ys[0],
              -rates[4]-h,
              0,0]
    reac_7 = [-rates[5]*Ys[6],
              0,0,0,0,
              rates[4],
              -rates[5]*Ys[0]-h,
              0]
    reac_8 = [rates[5]*Ys[6],
              0,0,0,0,0,
              rates[5]*Ys[0],
              -h]
    return np.linalg.inv(np.array([reac_1, reac_2, reac_3, reac_4, reac_5, reac_6, reac_7, reac_8]))

@numba.njit
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 2, tol=1e-10,):
    prevY = oldY.copy()
    for nr_step in range(maxsteps):
        newY = oldY - np.dot(invJ(oldY, *args), fsol(oldY, prevY, *args))
        oldY = newY
        if newY[2]<1e-9:
            newY[2] =  1e-9
        # if newY[4]<5e-4:
        #     newY[4] =  5e-4

        if newY[5]<1e-9:
            newY[5] =  1e-9
    return newY

year = 365*24*60*60 # [s]
hmax = 1/(1e7*year)
step = 1e4
dT = step*year
hinit = 1/dT # 1/dT, 
h = hinit
timesteps = int(1e7)
Ys = np.zeros((timesteps, 8))
sols = np.zeros((timesteps, 8))

# Initial abundances
#           H,     12C,  13N,  13C,      14N,  15O,     15N,  4He
# Ys[0] = [0.91, 0.00297, 1e-7, 3e-5,      5e-3, 1e-7, 3.68e-6, 0.09]
Ys[0] = [0.91,    1e-3, 0, 1e-5,      5e-3,  0,   1e-4, 0.09]
As =    [1,         12,  13,   13,       14,   15,      15,   4]
As = np.array(As)
#Ys[0] = [ 0.9,     0.1,    0,    0,        0,    0,       0,    0]
Ys[0] /= np.sum(As * Ys) # normalize

# Reaction rates (at T9 = 0.03)
rates = np.array([2.85e-16,  # 12C + H -> 13N
                  1.67e-03,  # 13N     -> 13C # Half-life time 
                  7.38e-11,  # 13C + H -> 14N
                  2.48e-13,  # 14N + H -> 15O
                  8.18e-03,  # 15O     -> 15N # Half-life time
                  7.48e-14]) # 15N + H -> 12C + 4He 
rates *= 5 # density

elapsed_time = 0
max_time = 12e9*year
for i in tqdm(range(1,timesteps)):
    Ys[i] = newton_raphson(Ys[i-1], inv_Jacobian, eq,
                                    args = (rates, h),)
    elapsed_time += 1/h
    if elapsed_time > max_time:
        break
print('Evo time', elapsed_time/(1e9*year), 'Gyrs')
#%%
labels = ["H", "$^{12}$C", "$^{13}$N", "$^{13}$C", "$^{14}$N", "$^{15}$O", "$^{15}$N", "$^{4}$He"]
plt.figure(tight_layout=True)
step_plot = 100
try:
    stop = np.where(Ys.T[0] == 0)[0][0]
except:
    stop = -1
    
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps)[:stop:step_plot]/step, 
             abundances[:stop:step_plot], 
             label = labels[i], marker='')

plt.yscale('log')
plt.xscale('log')
plt.ylim(1e-15,1.2)
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Gyr]', fontsize = 14)
plt.legend(ncols = 1, bbox_to_anchor = (1.1,1))
# #%%
# fig, axs = plt.subplots(len(sols[0]),1, sharex=True)
# for i,sol in enumerate(sols.T):
#     axs[i].plot(np.arange(timesteps)[1:], sol[1:], label = labels[i], marker='').legend(ncols = 1, loc='upper left', bbox_to_anchor = (1,1))
