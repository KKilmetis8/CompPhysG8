#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:21:35 2024

@author: konstantinos & Diederick
"""
# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorcet
import pandas as pd
# Chocolate
import prelude as c

#%%

# Load Data
df = pd.read_csv('past_sims.tsv', header = 0, delimiter = '\s+')

# Make marker shapes, colours
markers  = []
colors = []
max_den = np.max(df['Density'])
cmap = plt.get_cmap('cet_rainbow4')
for state, den in zip(df['State'], df['Density']):
    if state == 'S':
        marker =  's'
    elif state == 'L':
        marker = 'o'
    elif state == 'G':
        marker = '^'
    else:
        marker = '*'
    markers.append(marker)
    colors.append(cmap(den / max_den))

# Plot
fig, ax = plt.subplots(1,1)
ax.grid(zorder = 1)
for i in range(len(markers)):
    if df['Pressure'][i] != 'negative':
        ax.scatter(df['Temperature'][i], float(df['Pressure'][i]),
                c = colors[i], ec = 'k', 
                marker = markers[i], s = 75, 
                zorder = 3)
    
# CB
cbar_range = (np.min(df['Density']), max_den)
sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(*cbar_range))
cbar = fig.colorbar(sm, ax= ax)
cbar.set_label('Density $\\left[ m_\\mathrm{Ar}/\\sigma^{3} \\right] $', 
               rotation=90, fontsize = 14, labelpad = 5)

# Pretty
ax.set_xlabel('Temperature $\\left[ \\varepsilon / k_{\\mathrm B} \\right] $', fontsize = 14)    
ax.set_ylabel('Pressure $\\left[ \\varepsilon / \\sigma^3 \\right] $', fontsize = 14)    
ax.set_title('Phase Diagram of Argon', fontsize = 15)
ax.set_yscale('log')
# Legend
custom_scatter = [ Line2D([0], [0], color = 'white', linestyle = '',
                          markeredgecolor = 'k', marker = 's', markersize = 8),
                   Line2D([0], [0], color = 'white',  linestyle = '',
                          markeredgecolor = 'k', marker = 'o', markersize = 8),
                   Line2D([0], [0], color = 'white',  linestyle = '',
                          markeredgecolor = 'k', marker = '^', markersize = 8),
                   Line2D([0], [0], color = 'white',  linestyle = '',
                          markeredgecolor = 'k', marker = '*', markersize = 12)
                 ]
labels = ['Solid', 'Liquid', 'Gas', 'Co-existance']
ax.legend(custom_scatter, labels, fontsize = 10, ncols = 4,
          bbox_to_anchor=(0.83, 0.03), bbox_transform = fig.transFigure,)

# Triple point
#ax.scatter(0.69, 0.0012, marker="P", ec='k', fc='purple', s=75, zorder=4)

# %%
