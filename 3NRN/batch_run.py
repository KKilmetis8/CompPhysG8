#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  19 22:37:31 2024

@author: konstantinos & diederick
"""
# Vanilla
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [4 , 4]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# Choc
from simulation import run_network


# Table
rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')
T9s = rates_table[:,0]

cno_eqs = []
cno_eqs_bad = []
cno_T9 = []
cno_T9s_bad = []
pp_eqs = []
for T9 in tqdm(T9s):
    _, eqpp = run_network('pp', T9, max_step = 1e6)
    pp_eqs.append(eqpp)
    try:
        _, eqcno = run_network('cno', T9, initY = 10, max_step = 1e6, max_time=20e9)
        cno_eqs.append(eqcno)
        cno_T9.append(T9)
    except NameError:
        if T9<70*1e3:
            cno_T9s_bad.append(T9)
            cno_eqs_bad.append(20)
    except np.linalg.LinAlgError:
        continue

cno_T9 = np.array(cno_T9) * 1e3
cno_T9s_bad = np.array(cno_T9s_bad) * 1e3
#%%

fig, ax = plt.subplots(tight_layout=True)

AEK = '#F1C410'
ax.plot(T9s * 1e3, pp_eqs, c = 'k', marker = 'h', ls=':')
ax.plot(cno_T9, cno_eqs, c = AEK, marker = 'h', ls=':', 
         markeredgecolor = 'k')
ax.plot(cno_T9s_bad, cno_eqs_bad, c = AEK, marker = '^', ls=':', 
         markeredgecolor = 'r')
ax.plot( [cno_T9s_bad[-1], cno_T9[0]], [cno_eqs_bad[-1], cno_eqs[0]], 
         c = AEK, marker ='', ls =':')
ax.axvline(18, color = 'maroon', ls = '--')
ax.set_xscale('log')
# plt.yscale('log')
ax.set_xlabel('Temperature [MK]')
ax.set_ylabel('$t_\\mathrm{eq}$ [Gyrs]')

# Inset
left, bottom, width, height = [0.55, 0.4, 0.35, 0.5]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(T9s[14:] * 1e3, pp_eqs[14:], c = 'k', marker = 'h', ls=':')
ax2.plot(cno_T9[3:], cno_eqs[3:], c = AEK, marker = 'h', ls=':', markeredgecolor='k')
ax2.set_yscale('log')
ax2.set_facecolor('snow')
ax.indicate_inset_zoom(ax2, edgecolor="#3d3c3c")


# %%
