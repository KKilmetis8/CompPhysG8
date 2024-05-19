#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  19 22:37:31 2024

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

# Functions
@numba.njit
def pp_eq(Ys, old, rates, h):
    res = np.zeros(len(Ys))
    res[0] = -2*rates[0]*Ys[0]**2 - rates[1]*Ys[0]*Ys[1] + rates[2]*Ys[2]**2
    res[1] = rates[0]*Ys[0]**2 - rates[1]*Ys[1]*Ys[0]
    res[2] = rates[1]*Ys[0]*Ys[1] - 2*rates[2]*Ys[2]**2
    res[3] = rates[2]*Ys[2]**2
    return res - h*(Ys-old)

@numba.njit
def pp_inv_Jacobian(Ys, rates, h):
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
def pp_newton_raphson(oldY, invJ, fsol, args, maxsteps = 3, tol=1e-6):
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
        
        if newY[1]<1e-8:
            newY[1] = newY[0] * 1e-6
                
        oldY = newY
    return newY

@numba.njit
def cno_eq(Ys, old, rates, h):
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
def cno_inv_Jacobian(Ys, rates, h):
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
def cno_newton_raphson(oldY, invJ, fsol, args, maxsteps = 2, tol=1e-10,):
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

# Table
rates_table = np.loadtxt("3NRN/NRN_Rates.csv", skiprows=1, delimiter=',')
T9s = rates_table[:,0]
pp_rates_all  = rates_table[:,1:4]
cno_rates_all = rates_table[:,4:]

# Initial abundances
Hyd = 0.91
DtoH = 2.1e-5 # Geiss Gloeckler 98
Deut = Hyd * DtoH
He3toH = 1.5e-5 # Geiss Gloeckler 98
He3 = Hyd * He3toH
He = 0.089

pp_Y0 = [Hyd, Deut, He3, He]
pp_As = np.array([1,2,3,4])
pp_Y0 /= np.sum(pp_As * pp_Y0)

cno_Y0 = [0.91, 1e-3, 0, 1e-5, 5e-3, 0, 1e-4, 0.09]
cno_As = np.array([1, 12, 13, 13, 14, 15, 15, 4])
cno_Y0 /= np.sum(cno_As * cno_Y0)

year = 365*24*60*60 # [s]
hmax = 1/(1e7*year)
timesteps = int(1e7)

# elapsed_time = 0
# max_time = 12e9*year
times_of_equality = np.zeros((len(T9s), 2))
for i, temp in enumerate(T9s):
    # something something prefactor stuff for light/heavy stars
    pp_rates, cno_rates = pp_rates_all[i], cno_rates_all[i]
    for j in range(2):
        dT = [1e4, 1e5][j]*year
        hinit = 1/dT # 1/dT, 
        h = hinit

        # Do we really need all the abundances?
        # Can just overwrite it each time?
        Y0 = [pp_Y0, cno_Y0][j]
        Ys = np.zeros((timesteps, len(Y0)))
        sols = np.zeros((timesteps, len(Y0)))
        Ys[0] = Y0

        newton_raphson = [pp_newton_raphson, cno_newton_raphson][j]
        inv_Jacobian   = [pp_inv_Jacobian, cno_inv_Jacobian][j]
        eq             = [pp_eq, cno_eq][j]
        rates          = [pp_rates, cno_rates][j]
        
        for k in tqdm(range(1,timesteps), desc=f"T9 = {temp}, {('pp' if j == 0 else 'cno'):>3}"):
            Ys[k] = newton_raphson(Ys[k-1], inv_Jacobian, 
                                            eq, args = (rates, h))
            
            if Ys[k,-1] > Ys[k,0]:
                times_of_equality[i, j] = k*dT/year
                break
            
            # elapsed_time += 1/h
            # if elapsed_time > max_time:
            #     break

# 0 = pp-chain, 1 = cno-cycle
dominated = np.argmin(times_of_equality, axis=1)
plt.plot(T9s, dominated)