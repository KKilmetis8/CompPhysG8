#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:46:45 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import prelude as c

fig, axs = plt.subplots(2,2, figsize = (8,6))
# Mean absolute spin
obs1 = np.zeros(len(ms))
sigma1 = np.zeros(len(ms))
for i in range(len(ms)):
    m = np.array(ms[i])
    obs1[i] = np.abs(m.mean())
    var1 = np.abs( np.mean(m**2) - np.mean(m)**2)
    sigma1[i] = np.sqrt(2*taus[i]/len(m) * var)
axs[0,0].errorbar(Ts, obs1, yerr = sigma1, 
             ls=':', marker='h', c='k', capsize = 4)
axs[0,0].set_xlabel('Temperature', fontsize = 14)
axs[0,0].set_ylabel('Mean absolute spin', fontsize = 14)
#%% Energy per spin
obs2 = np.zeros(len(ms))
sigma2 = np.zeros(len(ms))
for i in range(len(es)):
    e = np.array(es[i]) / N**2  
    obs2[i] = e.mean()
    var2 = np.abs( np.mean(e**2) - np.mean(e)**2)
    sigma2[i] = np.sqrt(2*taus[i]/len(e) * var2)
axs[0,1].errorbar(Ts, obs2, yerr = sigma2, 
             ls=':', marker='h', c='k', capsize = 4)
axs[0,1].set_xlabel('Temperature', fontsize = 14)
axs[0,1].set_ylabel('Energy per spin', fontsize = 14)
#%% Magnetic susceptibility
obs3 = np.zeros(len(ms))
sigma3 = np.zeros(len(ms))
for i in range(len(ms)): # loops over runs
    m = np.array(ms[i])
    chunksize = int(16*taus[i])
    chunk_chi_ms = 0
    prefactor = 1/Ts[i] * 1/N**2
    for j in range(chunksize, len(m), chunksize): # loops over tau chunks
        chunk_m = m[j-chunksize:j]
        chunk_chi_ms += np.mean(chunk_m**2) - np.mean(chunk_m)**2
    sigma3[i] = prefactor * chunk_chi_ms / j
    obs3 = prefactor * (np.mean(chunk_m**2) - np.mean(chunk_m)**2)
axs[1,0].errorbar(Ts, obs3, yerr = sigma3, 
             ls=':', marker='h', c='k', capsize = 4)
axs[1,0].set_xlabel('Temperature', fontsize = 14)
axs[1,0].set_ylabel('Magnetic susceptibility', fontsize = 14)





