#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:06:53 2024

@author: konstantinos
"""
# Vanilla imports
import numpy as np
from scipy.signal import convolve2d
from importlib import reload

# Chocolate imports
import prelude as c
import auxiliary as aux
import plotters as plot
import ising

if c.kind == 'single':
    rng  = np.random.default_rng(seed=c.rngseed)
    grid = np.sign(rng.random((c.Nsize, c.Nsize)) - 0.5)
    
    grids, energies = ising.Metropolis(grid, 1_000*c.Nsize**2)
    plot.energy(energies)
    plot.avg_magnetization(grids)
    
if c.kind == 'sweep':
    Temperatures = np.arange(1,4,0.2)
    Magnetizations = np.zeros(len(Temperatures))
    
    for i, temperature in enumerate(Temperatures):
        print(f'T:{temperature}')
        beta = 1/temperature
        rng  = np.random.default_rng(seed=c.rngseed)
        grid = np.sign(rng.random((c.Nsize, c.Nsize)) - 0.5)
        
        grids, energies = ising.Metropolis(grid, 1_000*c.Nsize**2, beta)
        Magnetizations[i] = aux.avg_magnetization(grids[-1])
    
    plot.mag_temp(Temperatures, Magnetizations)

    
    