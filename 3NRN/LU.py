#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:42:33 2023

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import copy # Needed for row swaps

def row_swapper(A, row_1, row_2, loud = False):
    if len(A.shape) < 2: # 1d
        temp = copy.deepcopy(A[row_1])
        A[row_1] = A[row_2]
        A[row_2] = temp
        return A
    
    temp = copy.deepcopy(A[row_1,:]) # Equals just creates pointers!!!!!!
    A[row_1,:] = A[row_2,:]
    A[row_2,:] = temp
    if loud:
        print('Swapping row', row_1, 'with row', row_2)
        print(A)
    return A

def partial_pivoter(A, k, loud=False):
    '''Finds the pivot, then calls row_swapper'''
    # Prints for debugging
    if loud:
        print('Initial')
        print(A)
    rows, cols = A.shape
    biggest_guy = 0
    i_max = k
    for i in range(k, rows):
        if np.abs(A[i,k]) > biggest_guy:
            biggest_guy = np.abs(A[i,k])
            i_max = i
    if loud:
        print('|Pivot| is:', A[i_max, k])
    if biggest_guy == 0:
        print('Critical Error: Division by 0, \n Universe is shutting down')
    return i_max

def LU_mine(A, loud = False):
    rows, cols = A.shape
    LU = copy.deepcopy(A)
    bookeeper = np.zeros(rows)
    for k in range(cols):
        imax = partial_pivoter(LU, k, loud)
        bookeeper[k] = imax
        if imax != k:
            LU = row_swapper(LU, k, imax, loud)

        for i in range(k+1, rows): # i>k
            LU[i,k] /= LU[k,k]
            if loud:
                print('Scaled \n', LU)
            for j in range(k+1, cols): # j>k
                LU[i,j] -= LU[i,k]*LU[k,j]
                if loud:
                    print('Reduced \n', LU)
        if loud:
            print('------------------------------------')
        
    # Split to L and U, sacrifices speed for legibility.
    L = np.zeros_like(LU)
    for i in range(rows):
        for j in range(i):
            # Does not do diagonals
            L[i,j] = LU[i,j]
    # Get U
    U = np.add(LU, -L)
    # Fix Diagonal
    for i in range(rows):
        L[i,i] = 1
        
    return bookeeper, L, U
            

def forward_sub(L, b, bookeeper):
    # Scramble
    # b = bookeeper @ b
    y = np.zeros(len(b))
    # First step
    y[0] = b[0]/L[0,0]
    # Forward loop
    for i in range(1,len(b)):
        y[i] = b[i] - np.sum(L[i,:] * y[:])
    return y

def back_sub(U, y):
    x = np.zeros(len(y))
    # Last step
    x[-1] = y[-1] / U[-1,-1]
    # Backwards loop
    for i in range(-2, -(len(x)+1), -1): # loop from [N-2, 0]
        x[i] = (y[i] - np.sum(U[i,:]*x[:]) ) / U[i,i]
    return x

def unscrambler(bookeeper, LU):
    for i in range(len(bookeeper)-1, -1, -1): # backwards
        LU = row_swapper(LU, i, int(bookeeper[i]), loud = True) 
    return LU

def scrambler(bookeeper, b):
    for i in range(len(b)):
        b = row_swapper(b, i,int(bookeeper[i]))
    return b

def LU_solve(A,b):
    bookeeper, L, U = LU_mine(A)
    b2 = copy.deepcopy(b)
    b2 = scrambler(bookeeper, b2)
    y = forward_sub(L, b2, bookeeper)
    x = back_sub(U, y)
    return x

def LU_iterative(A,b, iteration_cap = 10_000):
    sol = LU_solve(A,b)
    error = 1 # get in the while loop
    small = 1e-40
    counter = 0 
    while np.sum(error) > small:
        delta_b = (A @ sol) - b
        error = LU_solve(A,delta_b)
        new_sol = sol - error
        counter += 1
        # Caps the while loop
        if counter > iteration_cap:
            return new_sol
    return new_sol

if __name__ == '__main__':
    A = np.array([
                  [3, 8, 1, -12, -4 ],
                  [1, 0, 0, -1,   0 ],
                  [4, 4, 3, -40, -3 ],
                  [0, 2, 1, -3,  -2 ],
                  [0, 1, 0, -12,  0 ]
                  ], dtype = np.float64)
    
    b = np.array(( [2, 0, 1, 0 ,0]), dtype = np.float64)
    b2 = copy.deepcopy(b)
    np.set_printoptions(precision=2) # 2 decimal points
    sol = LU_solve(A,b)
    real_sol = np.linalg.solve(A, b2)
    print('Is it correct ?', np.allclose(sol, real_sol))
