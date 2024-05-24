#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:34:43 2024

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


    
def eq(t, Ys):
    rates = np.array([7.9e-20,	1.01e-2, 2.22e-10])
    rates /= rates[0]
    rates = np.array([1e-4, 1e-1, 5e-2])
    res = np.zeros(len(Ys))
    res[0] = -2*rates[0]*Ys[0]**2 - rates[1]*Ys[0]*Ys[1] + rates[2]*Ys[2]**2 #- h*(Ys[0] - self.old[0])
    res[1] = rates[0]*Ys[0]**2 - rates[1]*Ys[1]*Ys[0]# - h*(Ys[1] - self.old[1])
    res[2] = rates[1]*Ys[0]*Ys[1] - 2*rates[2]*Ys[2]**2 #- h*(Ys[2] - self.old[2])
    res[3] = rates[2]*Ys[2]**2# - h*(Ys[3] - self.old[3])
    
    return res

from scipy.integrate import solve_ivp
Hyd = 0.7
DtoH = 2.1e-5 # Geiss Gloeckler 98
Deut = Hyd * DtoH
He3toH = 1.5e-5 # Geiss Gloeckler 98
He3 = Hyd * He3toH
He = 0.28

Ys = [Hyd, Deut, He3, He]
year = 365*24*60*60 # [s]

sol = solve_ivp(eq, [0, 100000 ], Ys, method = 'RK23',
                )
#%%

plt.plot(sol.y[0])
plt.plot(sol.y[1])
plt.plot(sol.y[2])
plt.plot(sol.y[3])
plt.yscale('log')
plt.xscale('log')