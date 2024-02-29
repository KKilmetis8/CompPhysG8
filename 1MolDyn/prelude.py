#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:18:14 2024

@authors: konstantinos & diederick
"""
import numpy as np
import astropy.constants as c
import astropy.units as u

# Simulation parameters
Nbodies = 9
time = 200 * u.s # [ps]
h = 0.01 * u.s
boxL = 20 * u.m # Units??
inv_boxL = 1 / boxL
dims = 2 
timesteps = int(time / h)
plot_number = 2

rngseed = 8

# N-body units
epsilon = 1
sigma   = 1 
m_argon = 1

# Argon constants
EPSILON = 1 * u.J # 119.8 * u.K * c.k_B
SIGMA = 1 * u.m # 3.405 * u.AA 
M_ARGON = 39.792 * u.u

# Normalization constants
to_norm_dist = 1 / SIGMA 
to_norm_mass = 1 / M_ARGON 
to_norm_energy = 1 / EPSILON
to_norm_time = np.sqrt(EPSILON/(M_ARGON * SIGMA**2)).decompose()

# Normalized inputs
time_norm = (time * to_norm_time).decompose().value
h_norm = (h * to_norm_time).decompose().value
boxL_norm = (boxL * to_norm_dist).decompose().value

# Derived
EPSILON = 119.8 # [K]
SIGMA = 3.405 # [Angstrom]
M_ARGON = 39.792 # [amu] 
K_BOLTZMANN = 1.3803658e-16 # [cgs]
steps_per_plot = timesteps / plot_number 
inv_m_argon = 1 / m_argon
t_tilde = 1 / np.sqrt((M_ARGON * SIGMA**2 / EPSILON ))

# Converters | ALWAYS multiply by the converter
amu_to_gram = 1.66054e-24
angstrom_to_cm = 1e-8
time_to_cgs = np.sqrt(amu_to_gram/K_BOLTZMANN) * angstrom_to_cm * t_tilde
vel_to_cgs =  t_tilde / SIGMA

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