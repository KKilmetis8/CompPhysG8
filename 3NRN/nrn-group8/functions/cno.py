#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:15:30 2024

@author: konstantinos
"""
import numpy as np
import numba 

@numba.njit
def eq(Ys, old, rates, h):
    """The function to find the root for, which 
    gives the next abundances for each species
    in the CNO-cycle.

    Parameters
    ----------
    Ys : array[floats]
        Current abundances
    old : array[floats]
        Previous abundances
    rates : array[floats]
        The reaction rates
    h : array[floats]
        The reciprocal of the timestep

    Returns
    -------
    Ys_new : 
        The new abundances
    """    
    res = np.zeros(len(Ys))
    # H
    res[0] = -rates[0]*Ys[0]*Ys[1] - rates[2]*Ys[0]*Ys[3] - rates[3]*Ys[0]*Ys[4]*rates[5]*Ys[0]*Ys[6]
    # 12C
    res[1] = -rates[0]*Ys[1]*Ys[0] + rates[5]*Ys[0]*Ys[6]
    # 13N 
    res[2] = rates[0]*Ys[0]*Ys[1] - rates[1]*Ys[2]
    # 13C
    res[3] = rates[1]*Ys[2] - rates[2]*Ys[3]*Ys[0]
    # 14N
    res[4] = rates[2]*Ys[3]*Ys[0] - rates[3]*Ys[4]*Ys[0]
    # 15O
    res[5] = rates[3]*Ys[4]*Ys[0] - rates[4]*Ys[5]
    # 15N
    res[6] = rates[4]*Ys[5] - rates[5]*Ys[6]*Ys[0]
    # 4He
    res[7] = rates[5]*Ys[6]*Ys[0]
    return res - h*(Ys-old)

@numba.njit
def inv_Jacobian(Ys, rates, h):
    """The inverse Jacobian needed
    for the Newton-Raphson method 
    for the CNO-cycle.

    Parameters
    ----------
    Ys : array[floats]
        Current abundances
    rates : array[floats]
        The reaction rates
    h : array[floats]
        The reciprocal of the timestep

    Returns
    -------
    inv_Jac : 
        The inverse Jacobian
    """    
    reac_1 = [-rates[0]*Ys[1] - rates[2]*Ys[3] - rates[3]*Ys[4] - rates[5]*Ys[6] - h,
              -rates[0]*Ys[0],
              0,
              -rates[2]*Ys[0],
              -rates[3]*Ys[0],
              0,
              -rates[5]*Ys[0],
              0]
    reac_2 = [-rates[0]*Ys[1] + rates[5]*Ys[6],
              -rates[0]*Ys[0] - h,
              0, 0, 0, 0, 
              rates[5]*Ys[0],
              0]
    reac_3 = [rates[0]*Ys[1],
              rates[0]*Ys[0],
              -rates[1]-h,
              0,0,0,0,0]
    reac_4 = [-rates[2]*Ys[3],
              0,
              rates[1],
              -rates[2]*Ys[0]-h,
              0,0,0,0]
    reac_5 = [rates[2]*Ys[3]-rates[3]*Ys[4],
              0,0,
              rates[2]*Ys[0],
              -rates[3]*Ys[0] - h,
              0,0,0]
    reac_6 = [rates[3]*Ys[4],
              0,0,0,
              rates[3]*Ys[0],
              -rates[4]-h,
              0,0]
    reac_7 = [-rates[5]*Ys[6],
              0,0,0,0,
              rates[4],
              -rates[5]*Ys[0]-h,
              0]
    reac_8 = [rates[5]*Ys[6],
              0,0,0,0,0,
              rates[5]*Ys[0],
              -h]
    return np.linalg.inv(np.array([reac_1, reac_2, reac_3, reac_4, reac_5, reac_6, reac_7, reac_8]))

@numba.njit
def newton_raphson(oldY, invJ, fsol, args, maxsteps = 10, tol=1e-10):
    """Performs the Newton-Raphson method for the CNO-cycle.

    Parameters
    ----------
    oldY : array of floats 
        Old abundances
    invJ : array of floats
        Inverse Jacobian of oldY
    fsol : function
        The function to solve, eq()
    args : (array of floats, float)
        First entry: The rates of the reactions.
        Second entry: The reciprocal of the timestep.
    maxsteps : int, optional
        Maximum number of steps to perform the
        Newton-Raphson method for, by default 10
    tol : float, optional
        The tolerance within which the method
        is considered to be converged, by default 1e-10

    Returns
    -------
    (newY, h, conv_flag) : (array of floats, float, bool)
        First entry: The new abundances
        Second entry: The new reciprocal of the timestep
        Third entry: Wether or not the method converged
    """    
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