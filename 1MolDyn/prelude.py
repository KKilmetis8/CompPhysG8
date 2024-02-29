#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:18:14 2024

@authors: konstantinos & diederick
"""
import numpy as np

# N-body units
epsilon = 1
sigma   = 1 
m_argon = 1 # 39.792# * amu_to_gram
inv_m_argon = 1/m_argon

rngseed = 8
# Converters | ALWAYS multiply by the converter to go from sim->cgs
amu_to_gram = 1.66054e-24
angstrom_to_cm = 1e-8

# Derived
EPSILON = 119.8 # [K]
SIGMA = 3.405 # [Angstrom]
M_ARGON = 39.792 # [amu] 
K_BOLTZMANN = 1.3803658e-16 # [cgs]
time_to_cgs = np.sqrt((M_ARGON * amu_to_gram * (SIGMA * angstrom_to_cm)**2 / (EPSILON * K_BOLTZMANN) ))
time_to_sim = 1/time_to_cgs
vel_to_cgs =  SIGMA * angstrom_to_cm / time_to_cgs



# Simulation parameters
# USER INPUT
Nbodies = 9
time = 100 # [ps] 
timestep = 1 # [ps]
boxL = 20 # [nm]
dims = 2 
plot_number = 100

# we say
time *= 1e-12 # [s]
time *= time_to_sim # [sim units]
h_sim_units = timestep * 1e-12 * time_to_sim # [sim units]
boxL *= 1e-7 / (SIGMA * angstrom_to_cm)
inv_boxL = 1 / boxL
timesteps = int(time / h_sim_units)
steps_per_plot = timesteps / plot_number 

# Pretty stuff
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
plt.rcParams['text.usetex'] = False # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6 , 5]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'