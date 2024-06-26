#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:02:28 2024

@author: konstantinos
"""
# Vanilla
import numpy as np
#import numba

# Choc
import ISM_abudances as ism
from density_interp import den_of_T

def normalize_abuds(Ys, cycle):
    '''
    Normalizes initial abundances so that sum(Y_i) = 1

    Parameters
    ----------
    Ys : float
        Initial abundances, len 4 for pp len 8 for cno.
    cycle : str
        The cycle to simulate.

    Returns
    -------
    Ys: arr
        Normalized abundances.

    '''
    if cycle == 'pp':
        As = np.array([1,2,3,4])
    elif cycle == 'cno':
        As = np.array([1,12, 13, 13, 14, 15, 15, 4])
    else:
        print(f'Cycle {cycle} not available, supported cycles \n pp, \n cno')
        return 1
    Ys /= np.sum(As * Ys)
    return Ys 

def interp_rates(Tcore, deg = 6):
    '''
    Finds the reaction rates for a given Tcore temperature 
    not equal to the ones in the table.

    Parameters
    ----------
    Tcore : float
        Core temperature in GK.
    def : int
        Degree of the interpolation polynomial.

    Returns
    -------
    interp_rates: arr
        Interpolated rates.
    '''
    rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')

    T9s = rates_table[:,0]
    rates_pp = rates_table[:,1:4]
    rates_cno = rates_table[:,4:]

    interp_rates = []
    for reac_rates in [rates_pp, rates_cno]:
        for rates in reac_rates.T:
            coeff = np.polyfit(np.log10(T9s), np.log10(rates), deg)
            interp_rates.append(10**np.poly1d(coeff)(np.log10(Tcore)))
    
    return np.array(interp_rates)


#@numba.njit
def run_network(cycle, coreT, initY = None,
                init_step = 1e-2, max_step = 1e6, 
                save_step = None, max_time = 20e9):
    '''

    Parameters
    ----------
    cycle : str,
        The cycle to simulate. Either `pp` or `cno`.
    coreT : float,
        core temperature in Giga Kelvin.
    initY : arr or int, optional
        Initial abundances. Length 4 for pp length 8 for cno. If not provided
        ISM values will be used.
        For `cycle=cno`: If `initY` is an integer, use it as a 
        metallicity multiplier so that Z = initY * Z_ISM.
    init_step : float, optional
        Initial timestep, in years. The default is 1e-2.
    max_step : float, optional
        Maximum allowed timestep, in years. The default is 1e6.
    save_step : float, optional
        How often to save. The default is max_step.
    max_time : float, optional
        Time to evolve for, in years. The default is 12e9.

    Returns
    -------
    Ys: arr,
        Array containing the evolution of the abundances.
    cross_
    '''
    
    # Set up steps ------------------------------------------------------------
    year = 365*24*60*60 # [s]
    if save_step == None: 
        save_step = max_step * year
    
    h = 1 / (init_step * year)
    hmax = 1/(max_step * year)
    max_time = max_time*year
    timesteps = int(max_time/(init_step*year))
    
    # Look up rates from table ------------------------------------------------
    # rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')
    # T9s = rates_table[:,0]
    # closest_available_T_index = np.argmin(np.abs(T9s - coreT))
    all_rates = interp_rates(coreT, deg=6)
    
    # Get the desired cycle ---------------------------------------------------

    if cycle == 'pp':
        from pp import eq, inv_Jacobian, newton_raphson
        Ys = np.zeros(( int(max_time / save_step) + 1 , 4))
        # rates = rates_table[:,1:4][closest_available_T_index]
        rates = all_rates[:3]
        if initY is None:
            Ys[0] = [ism.H, ism.Deut, ism.He3, ism.He]
        else:
            Ys[0] = initY
    elif cycle == 'cno':
        from cno import eq, inv_Jacobian, newton_raphson
        Ys = np.zeros(( int(max_time / save_step) + 1 , 8))
        # rates = rates_table[:,4:][closest_available_T_index]
        rates = all_rates[3:]
        if initY is None:
            Ys[0] = [ism.H, ism.C12, 0, ism.C13, ism.N14, 0, ism.N15, ism.He]
        elif type(initY) in [int, float]:
            Ys[0] = [ism.H, ism.C12, 0, ism.C13, ism.N14, 0, ism.N15, ism.He]
            Z_weights = np.array([1]+list(initY*np.ones(6))+[1])
            Ys[0] *= Z_weights
        else:
            Ys[0] = initY
    else:
        print(f"Cycle '{cycle}' not available, supported cycles \n 'pp', \n 'cno'")
        return 1
    Ys[0] = normalize_abuds(Ys[0], cycle)
    
    # Get stellar density  ----------------------------------------------------
    density = den_of_T(coreT) 
    rates *= density

    # Main Simulation  --------------------------------------------------------
    oldYs = np.array(Ys[0].copy())
    currentYs = np.zeros_like(Ys[0])
    elapsed_time = 0
    save_counter = 1
    savetimes = np.zeros(len(Ys))
    equality_flag = False
    equality_time = max_time*year
    for i in range(1,timesteps):
        currentYs, h, conv_flag = newton_raphson(oldYs, inv_Jacobian, eq,
                                        args = (rates, h),)
        if conv_flag:
            elapsed_time += 1/h
            rel_change = (currentYs - oldYs ) / currentYs
            max_change = np.max(rel_change)
            max_change = np.nan_to_num(max_change, nan = 1e-20, posinf = 1e-20, neginf= 1e-20)
            oldYs = currentYs
            dT = np.min([1/hmax, 2/h, 10/h /max_change])
            h = 1/dT
            
        if elapsed_time > save_counter * save_step :
            savetimes[save_counter] = elapsed_time
            Ys[save_counter] = currentYs
            save_counter += 1
            
        if currentYs[-1] >= currentYs[0] and not equality_flag:
            equality_flag = True
            equality_time = elapsed_time
            
        if elapsed_time > max_time:
            break
        
    # print('Evo time', elapsed_time/(1e9*year), 'Gyrs')
    

    return Ys, equality_time / (year * 1e9)

if __name__ == '__main__':
    # import config as c
    # Ys, eq = run_network(c.cycle, c.coreT, c.initY,
    #                      c.init_step, c.max_step, 
    #                      c.save_step, c.max_time)
    Ys, eq = run_network('cno', 30e-3, 1, max_step = 1e5, max_time=1e9)