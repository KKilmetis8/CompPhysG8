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
plt.rcParams['figure.figsize'] = [4 , 4]
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
        # if newY[1]< newY[0]*DtoH:
        #     newY[1]=newY[0]*DtoH
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
timesteps = int(5e6)
Ys = np.zeros((timesteps, 8))
sols = np.zeros((timesteps, 8))
As = np.array([1,2,3,4])

# Initial abundances
#           H,     12C,  13N,  13C,      14N,  15O,     15N,  4He
Ys[0] = [0.75, 0.00297, 1e-7, 3e-5, 9.963e-4, 1e-7, 3.68e-6, 0.23]
#Ys[0] = [ 0.9,     0.1,    0,    0,        0,    0,       0,    0]

# Reaction rates (at T9 = 0.015)
rates = np.array([2.85e-16,  # 12C + H -> 13N
                  1.67e-03,  # 13N     -> 13C # Half-life time 
                  1.18e-15,  # 13C + H -> 14N
                  1.19e-18,  # 14N + H -> 15O
                  8.20e-03,  # 15O     -> 15N # Half-life time
                  2.98e-14]) # 15N + H -> 12C + 4He

for i in tqdm(range(1,timesteps)):
    Ys[i], sols[i] = newton_raphson(Ys[i-1], inv_Jacobian, eq,
                                    args = (rates, h),)
    #print( np.sum(Ys[i])) # * As))
    # Ys[i] /= np.sum(Ys[i] * As)
    #print(f"({np.round((i+1)/len(Ys))}%): {Ys[i]}",end='\r')
    
#%%
labels = ["H", "$^{12}$C", "$^{13}$N", "$^{13}$C", "$^{14}$N", "$^{15}$O", "$^{15}$N", "$^{4}$He"]
plt.figure(tight_layout=True)

step = 1000
for i,abundances in enumerate(Ys.T):
    plt.plot(np.arange(timesteps)[::step]*1e-3, abundances[::step], label = labels[i], marker='')

plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Abundance', fontsize = 14)
plt.xlabel('time [Myr]', fontsize = 14)
#plt.xlim(0.3e2,10e3)
plt.legend(ncols = 1, loc='upper left', bbox_to_anchor = (1,1))

#%%
fig, axs = plt.subplots(len(sols[0]),1, sharex=True)
for i,sol in enumerate(sols.T):
    axs[i].plot(np.arange(timesteps)[1:], sol[1:], label = labels[i], marker='')