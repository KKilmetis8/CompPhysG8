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
import ISM_abudances as ism

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
            
            if newY[1]<4e-7:
                newY[1] = newY[0] * 6e-8
                
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

year = 365*24*60*60 # [s]
step = 1
dT = step*year
hinit = 1/dT # 1/dT, 
h = hinit
max_step = 1e7
hmax = 1/(max_step * year)
max_time = 1e12*year
timesteps = int( max_time / (year*max_step))
save_step = max_step * year

Ys = np.zeros(( int(max_time / save_step) + 1 , 4))
As = np.array([1,2,3,4])
Ys[0] = [ism.H, ism.Deut, ism.He3, ism.He]
Ys[0] /= np.sum(As * Ys)
rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')

pick = 0
rates = rates_table[pick][1:4]
T9 = rates_table[:,0][pick]
density = den_of_T(T9)
rates = rates * density


#%%
oldYs = np.array(Ys[0].copy())
currentYs = np.zeros_like(Ys[0])
elapsed_time = 0

save_counter = 1
savetimes = np.zeros(len(Ys))
equality_time = 0
equality_flag = False
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

    if currentYs[-1] >= currentYs[0] and equality_flag == False:
        equality_flag = True
        equality_time = elapsed_time / (year * 1e9)
        
    if elapsed_time > max_time:
        break

print('\n Evo time', elapsed_time/(1e9*year), 'Gyrs')
print('\n Eq time', equality_time, 'Gyrs')

#%%
AEK = '#F1C410'
labels = ["H", "D", "$^{3}$He", "$^{4}$He"]
colors = ["dodgerblue", "dodgerblue", AEK, AEK]
linestyles = ["-","--","--","-"]
plt.figure(tight_layout=True)

step_plot = 100
try:
    stop = np.where(Ys.T[0] == 0)[0][0]
except:
    stop = -1

unit = 1 / (year * 1e9)
for i,abundances in enumerate(Ys.T):
    plt.plot(savetimes[:stop]*unit, abundances[:stop], 
             label = labels[i], ls=linestyles[i], color=colors[i], marker='', linewidth=2.5)

eq_idx =  np.argmin(np.abs( equality_time - savetimes*unit))
plt.scatter(equality_time, Ys.T[-1][eq_idx]
            , marker = 'h', c = 'gold', ec = 'dodgerblue', 
            linewidth = 2, 
            s = 200, zorder = 4)


plt.grid()
plt.ylim(1e-8,10)
plt.xlim(1e-3,1e3)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Gyrs]', fontsize = 14)
# plt.legend(ncols = 1)

# %%
