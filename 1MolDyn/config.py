#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:54:30 2024

@author: konstantinos, diederick

Argon Simulation Config.  
This is the only file any user should touch.
Specifies parameters of simulation.

    state_of_matter, str: specifies density and temperature.
    accepts Solid, Liquid or Gas
    Pass Custom, if you desire to set your own.
    The units are: 
                  [density] = m_argon / sigma^3
                  where sigma = 3.405 Angstorm
                  - - - - - - - - - - - - - - - - - - - - - - - -
                  [temperature] = epsilon/k_boltzman
                  where epsilon/k_boltzman = 119 K
                  
    Nbodies, int: Number of particles. 108 is the default value
    
    time, float: Total duration of simulation. 
                 In picoseconds
    
    timestep, float: How often will positions and velocities be updated.
                     In picoseconds.
                    
    loud, bool: Set to true for tracking progress.
    
    plot, bool: Set to true to generate a movie of the simulation
"""

state_of_matter = 'Gas'
Nbodies = 108
time = 15 # [ps] 
timestep = 0.1 # [ps]
loud = True
plot = False

if state_of_matter == 'Solid':
    density = 1.2 # [M_ARGON/SIGMA^3]
    temperature = 0.5 # [EPSILON / K_BOLTZMAN]
elif state_of_matter == 'Liquid':
    density = 0.8 # [M_ARGON/SIGMA^3]
    temperature = 1.0 # [EPSILON / K_BOLTZMAN]
elif state_of_matter == 'Gas':
    density = 0.3 # [M_ARGON/SIGMA^3]
    temperature = 3.0 # [EPSILON / K_BOLTZMAN]
elif state_of_matter == 'Custom':
    density = 13 # [M_ARGON/SIGMA^3]
    temperature = 12 # [EPSILON / K_BOLTZMAN]
else:
    raise ValueError('This state of matter does not exist')
