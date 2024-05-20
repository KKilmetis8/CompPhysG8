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

import ISM_abudances as ism
from density_interp import den_of_T


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
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 10, tol=1e-10):
    prevY = oldY.copy()
    rates, h = args
    conv_flag = False
    timestep_increase = 0
    while not conv_flag:
        for nr_step in range(maxsteps):
            #try:
            newY = oldY - np.dot(invJ(oldY, rates, h), 
                                     fsol(oldY, prevY, rates, h))
            sol = fsol(newY, prevY, rates, h)
            
            if newY[1]<1e-6:
                newY[1] = newY[0] * 2e-4
                
            if np.all(sol < tol):
                conv_flag = True
                break
            #except:
            #    break              
            oldY = newY
        if not conv_flag:
            h *= 2
            timestep_increase += 1
        if timestep_increase > 25:
            return prevY, h, conv_flag
    return newY, h, conv_flag

# Timesteps
year = 365*24*60*60 # [s]
step = 1e-4
dT = step*year
hinit = 1/dT # 1/dT, 
h = hinit
max_step = 1e6
hmax = 1/(max_step * year)
max_time = 12e9*year
timesteps = int(max_time/(step*year))
save_step = max_step * year
Ys = np.zeros(( int(max_time / save_step) + 1 , 8))

# Initial abundances
#           H,     12C,  13N,  13C,      14N,  15O,     15N,  4He
Ys[0] = [ism.H,  ism.C12, 0, ism.C13, ism.N14,  0,   ism.N15, ism.He]
As =    [1,         12,  13,   13,       14,   15,      15,   4]
As = np.array(As)
Ys[0] /= np.sum(As * Ys) # normalize

# Reaction rates (at T9 = 0.03)
rates = np.array([2.85e-16,  # 12C + H -> 13N
                  1.67e-03,  # 13N     -> 13C # Half-life time 
                  7.38e-11,  # 13C + H -> 14N
                  2.48e-13,  # 14N + H -> 15O
                  8.18e-03,  # 15O     -> 15N # Half-life time
                  7.48e-14]) # 15N + H -> 12C + 4He 
rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')
T9 = rates_table[:,0][-1]
density = den_of_T(T9)
rates = rates * density

#%%
oldYs = np.array(Ys[0].copy())
currentYs = np.zeros_like(Ys[0])
elapsed_time = 0

save_counter = 1
savetimes = np.zeros(len(Ys))
for i in tqdm(range(1,timesteps)):
    currentYs, h, conv_flag = newton_raphson(oldYs, inv_Jacobian, eq,
                                    args = (rates, h),)
    if conv_flag:
        elapsed_time += 1/h
        rel_change = (currentYs - oldYs ) / currentYs
        max_change = np.max(rel_change)
        oldYs = currentYs
        dT = np.min([1/hmax, 2/h, 10/h /max_change])
        h = 1/dT
        
    if elapsed_time > save_counter * save_step :
        savetimes[save_counter] = elapsed_time
        Ys[save_counter] = currentYs
        save_counter += 1
        
    if elapsed_time > max_time:
        break
print('\n Evo time', elapsed_time/(1e9*year), 'Gyrs')
#%%
labels = ["H", "$^{12}$C", "$^{13}$N", "$^{13}$C", "$^{14}$N", "$^{15}$O", "$^{15}$N", "$^{4}$He"]
plt.figure(tight_layout=True)
step_plot = 1
try:
    stop = np.where(Ys.T[0] < 1e-4)[0][0]
except:
    stop = -1

unit = 1e9 / step
for i,abundances in enumerate(Ys.T):
    plt.plot(savetimes[:stop] / unit, 
             abundances[:stop], 
             label = labels[i], marker='')

plt.yscale('log')
plt.xscale('log')
#plt.ylim(1e-10,1.2)
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Gyr]', fontsize = 14)
plt.legend(ncols = 1, bbox_to_anchor = (1.1,1))
# #%%
# fig, axs = plt.subplots(len(sols[0]),1, sharex=True)
# for i,sol in enumerate(sols.T):
#     axs[i].plot(np.arange(timesteps)[1:], sol[1:], label = labels[i], marker='').legend(ncols = 1, loc='upper left', bbox_to_anchor = (1,1))
