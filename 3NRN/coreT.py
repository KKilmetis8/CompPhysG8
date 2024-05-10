#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:45:22 2024

@author: konstantinos
"""

def CoreT(M, R):
    ''' 
    Parameters
    ---
    M : stellar mass, in Msol
    R : stellar radius, in Rsol
    
    Returns,
    ---
    Tcore, in K

    '''
    Rsol_to_cm = 6.957e10 # [cm]
    Msol_to_g =  1.989e33 # [g]
    Mcgs = M  * Msol_to_g
    Rcgs = R * Rsol_to_cm
    G = 6.67e-8
    ideal = 8.31e7
    mu = 0.5
    constants = 8/3 * G*mu/ideal
    Tc = constants * Mcgs / Rcgs
    T9 = Tc * 1e-9
    return T9
#%%
import mesa_reader as mr
import numpy as np
for i in range(1, 20):
    p = mr.MesaData(f'profile{i}.data')
    
    print(p.star_age / 1e6)
    print(np.power(10, p.logT[-1]) / 1e6)
    #print(np.power(10, p.logM[0]))
    print('---')
