#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:18:14 2024

@author: konstantinos
"""

# Simulation parameters
Nbodies = 9
h = 0.01
boxL = 100 # Units??
inv_boxL = 1 / boxL
dims = 2 
timesteps = 10000
plot_number = 1000
steps_per_plot = timesteps / plot_number 

# Constants
epsilon = 1 # 119.8 # [K]
sigma = 1 #3.405 # [Angstrom]
m_argon = 1 # 39.792 # [amu] 
inv_m_argon = 1/m_argon

# Plotting
AEK = '#F1C410'

# 9 palette
c91 = '#e03524'
c92 = '#f07c12'
c93 = '#ffc200'
c94 = '#90bc1a'
c95 = '#21b534'
c96 = '#0095ac'
c97 = '#1f64ad'
c98 = '#4040a0'
c99 = '#903498'
colors = [c91, c92, c93, c94, c95, c96, c97, c98, c99]

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6 , 5]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'