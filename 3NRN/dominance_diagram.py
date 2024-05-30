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
from matplotlib.colors import ListedColormap

#%%
rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')

T9s_table = rates_table[:11,0]

T9s = np.linspace(T9s_table[0], T9s_table[-1], 30)
T9s = np.linspace(12, 25, 60)*1e-3
Zs  = np.linspace(1, 1e2, 30)

eq_pps  = -np.ones((len(Zs), len(T9s)))
eq_cnos = -np.ones((len(Zs), len(T9s)))

for i,metallicity in tqdm(enumerate(Zs)):
    for j,T9 in enumerate(T9s):
        _, eq_pps[i,j] = run_network('pp', T9, max_step = 1e6, max_time=1e11)
        try:
            _, eq_cnos[i,j] = run_network('cno', T9, initY = float(metallicity), 
                                   max_step = 1e6, max_time=1e11)
        except:
            #print(j)
            continue

np.save('eq_pps4', eq_pps)
np.save('eq_cnos4', eq_cnos)

#%%
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(eq_pps, cmap='coolwarm', origin='lower')
ax1.set_ylabel('Metallicity $Z/Z_\\mathrm{ISM}$', fontsize = 14)
ax1.set_xlabel('Temperature [MK]', fontsize = 14)
ax1.set_title('pp-chain')

ax2.imshow(eq_cnos, cmap='coolwarm', origin='lower')
ax2.set_ylabel('Metallicity $Z/Z_\\mathrm{ISM}$', fontsize = 14)
ax2.set_xlabel('Temperature [MK]', fontsize = 14)
ax2.set_title('CNO-Cycle')


#%%
pp_dominance = (eq_pps <= eq_cnos)
plt.figure()

T9s_mask = (12 < T9s*1e3) & (T9s*1e3 < 25)

custom_cmap = ListedColormap(['firebrick', plt.get_cmap('cividis')(255)])
plt.pcolormesh(T9s[T9s_mask]*1e3, Zs, pp_dominance[:,T9s_mask], cmap=custom_cmap)#, ec='k', lw=0.1)
#plt.colorbar()
plt.ylabel('Metallicity $Z/Z_\\mathrm{ISM}$', fontsize = 14)
plt.xlabel('Temperature [MK]', fontsize = 14)
plt.show()

# %%
plt.figure()

for i in range(len(pp_dominance)):
    pp_dom = pp_dominance[i]
    colors = ['r' if dom else 'b' for dom in pp_dom]
    Z = np.log10(Zs[i])

    plt.scatter(np.log10(T9s*1e3), Z*np.ones(len(T9s)), c = colors)

plt.show()