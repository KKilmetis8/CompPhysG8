#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  24 16:54:46 2024

@author: diederick & konstantinos
"""
# Vanilla
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [4 , 4]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# Choc
from simulation import run_network

#%%
rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')

T9s = rates_table[:14,0]
Zs  = np.logspace(-1, 2, 10)#%%
eq_pps  = -np.ones((len(Zs), len(T9s)))
eq_cnos = -np.ones((len(Zs), len(T9s)))

for i,metallicity in enumerate(Zs):
    for j,T9 in tqdm(enumerate(T9s)):
        _, eq_pps[i,j] = run_network('pp', T9, max_step = 1e6, max_time=2e10)
        try:
            _, eq_cnos[i,j] = run_network('cno', T9, initY = float(metallicity), 
                                   max_step = 1e6, max_time=2e10)
        except:
            print(j)
            continue

#%%
pp_dominance = (eq_pps <= eq_cnos)
plt.figure()
plt.pcolormesh(T9s*1e3, Zs, pp_dominance)

plt.yscale('log')
plt.ylabel('Metallicity $Z/Z_\\mathrm{ISM}$', fontsize = 14)
plt.xlabel('$t_\\mathrm{eq}$ [Gyr]', fontsize = 14)

# %%
