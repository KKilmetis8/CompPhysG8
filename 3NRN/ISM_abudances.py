#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:50:08 2024

@author: konstantinos

ISM abundance data
"""
import numpy as np


# Hydrogen
H = 0.91 # Ferreire '01

# Deuterium
DtoH = 2e-3 # Rosman & Taylor '98 table
Deut = H * DtoH

# Helium
He = 0.089  # Ferreire '01
He3toHe = 1e-2 # Rosman & Taylor '98 table
He3 = He * He3toHe

# Carbon
CtoH = 225 * 1e-6  # Snow & Witt '95
C = CtoH * H
C12 = 0.988938 * C # Rosman & Taylor '98 table
C13 = 1.1062 * C

# Oxygen
logOtoH = -3.0 # Nomoto + '06
OtoH = np.power(10,logOtoH)
O = OtoH * H
O16 = 99.7621 * O # Rosman & Taylor '98 table
# O17 = 0.0379 * O

# Nitrogen
logNtoO = -2.5 # Nomoto + '06
NtoO = np.power(10,logNtoO)
N = NtoO * O
N14 = 99.771 * N # Rosman & Taylor '98 table
N15 = 0.229 * N # Rosman & Taylor '98 table



