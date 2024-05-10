#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:08:49 2024

@author: konstantinos
"""
import numpy as np

def f(Y):
    f = np.array([ 
        Y[0]*Y[1] + Y[0]**2 - Y[1]**3 - 1,
        Y[0] + 2*Y[1] - Y[0]*Y[1]**2 - 2,
        ])
    return f

def invJ(Y):
    eq1 = [ Y[1]+2*Y[0], Y[0]-3*Y[1]**2 ] 
    eq2 = [ 1-Y[1]**2, 2-2*Y[0]*Y[1] ]
    J = np.array([eq1, eq2])
    return np.linalg.inv(J)

def newton_raphson(oldY, invJ, fsol, maxsteps = 20, tol=1e-25):
    for nr_step in range(maxsteps):
        newY = oldY - np.dot(invJ(oldY), fsol(oldY))
        print(newY)
        diff = np.linalg.norm(oldY-newY)
        if diff<tol:
            print('HI')
            break
        oldY = newY
        
    return newY

Y0 = np.array([0.34, 0.5])
Y1 = newton_raphson(Y0, invJ, f,)