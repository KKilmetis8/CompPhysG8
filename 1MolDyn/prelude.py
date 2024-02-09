#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:18:14 2024

@author: konstantinos
"""
import numpy as np

# Simulation parameters
Nbodies = 10
h = 0.1
boxL = 10 # Units??
inv_boxL = 1 / boxL
dims = 2 

# Constants
epsilon = 119.8 # [K]
sigma = 3.405 # [Angstrom]
m_argon = 39.792 # [amu] 
inv_m_argon = 1/m_argon

# Plotting
AEK = '#F1C410'