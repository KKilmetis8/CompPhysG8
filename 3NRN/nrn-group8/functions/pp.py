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
    """The function to find the root for, which 
    gives the next abundances for each species
    in the pp-chain.

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
    res[0] = -2*rates[0]*Ys[0]**2 - rates[1]*Ys[0]*Ys[1] + rates[2]*Ys[2]**2
    res[1] = rates[0]*Ys[0]**2 - rates[1]*Ys[1]*Ys[0]
    res[2] = rates[1]*Ys[0]*Ys[1] - 2*rates[2]*Ys[2]**2
    res[3] = rates[2]*Ys[2]**2
    return res - h*(Ys-old)

@numba.njit
def inv_Jacobian(Ys, rates, h):
    """The inverse Jacobian needed
    for the Newton-Raphson method 
    for the pp-chain.

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
    """Performs the Newton-Raphson method for the pp-chain.

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
            
            # if newY[1]<1e-8:
            #     newY[1] = newY[0] * 5e-8
                
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