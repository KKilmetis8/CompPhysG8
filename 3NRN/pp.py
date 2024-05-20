#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:13:51 2024

@author: konstantinos
"""
import numba
import numpy as np

@numba.njit
def eq(Ys, old, rates, h):
    res = np.zeros(len(Ys))
    res[0] = -2*rates[0]*Ys[0]**2 - rates[1]*Ys[0]*Ys[1] + rates[2]*Ys[2]**2
    res[1] = rates[0]*Ys[0]**2 - rates[1]*Ys[1]*Ys[0]
    res[2] = rates[1]*Ys[0]*Ys[1] - 2*rates[2]*Ys[2]**2
    res[3] = rates[2]*Ys[2]**2
    return res - h*(Ys-old)

@numba.njit
def inv_Jacobian(Ys, rates, h):
    reac_1 = [-4*rates[0]*Ys[0]-rates[1]*Ys[1]-h,
              -rates[1]*Ys[0],
              2*rates[2]*Ys[2],
              0]
    reac_2 = [2*rates[0]*Ys[0]-rates[1]*Ys[1],
              -rates[1]*Ys[0]-h,
              0,0]
    reac_3 = [rates[1]*Ys[1],
              rates[1]*Ys[0],
              -4*rates[2]*Ys[2]-h,
              0]
    reac_4 = [0,0,2*rates[2]*Ys[2], -h]
    return np.linalg.inv(np.array([reac_1, reac_2, reac_3, reac_4]))

@numba.njit
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 10, tol=1e-10):
    prevY = oldY.copy()
    rates, h = args
    conv_flag = False
    timestep_increase = 0
    while not conv_flag:
        for nr_step in range(maxsteps):
            #try:
            newY = oldY - np.dot(invJ(oldY, rates, h), 
                                     fsol(oldY, prevY, rates, h))
            sol = fsol(newY, prevY, rates, h)
            
            if newY[1]<1e-6:
                newY[1] = newY[0] * 2e-4
                
            if np.all(sol < tol):
                conv_flag = True
                break
            #except:
            #    break              
            oldY = newY
        if not conv_flag:
            h *= 2
            timestep_increase += 1
        if timestep_increase > 25:
            return prevY, h, conv_flag
    return newY, h, conv_flag