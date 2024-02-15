#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:52:31 2024

@author: konstantinos
"""

# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate Imports
import prelude as c
from Atom import Atom

# NOTE: Use os to mkdir and check biggest simnumber
# this is a saving fucntion inside simulation class
simnum = 1

plt.ioff()
def make_plot(figno, particles):
    fig, ax = plt.subplots(figsize = (6,5))
    
    # NOTE: Make particles into a class so this isn't trash
    for particle in particles:
        ax.scatter(particle.pos[0], particle.pos[1], c = particle.color)
        # Add arrows
     
    ax.set_xlim(-0.1 * c.boxL, 1.1 * c.boxL)
    ax.set_ylim(-0.1 * c.boxL, 1.1 * c.boxL)
    ax.axhline(0, c = 'k', linestyle = '--')
    ax.axhline(c.boxL, c = 'k', linestyle = '--')
    ax.axvline(0, c = 'k', linestyle = '--')
    ax.axvline(c.boxL, c = 'k', linestyle = '--')

    plt.savefig(f'figs/sim{simnum}-{plotcounter}')
    plt.close(fig)
# Init random number generator
rng = np.random.default_rng(seed=8)

# Make set of particles
particles = []
plotcounter = 0 # NOTE: This trash, make it better
for i in range(c.Nbodies):
    pos = rng.random(c.dims)*c.boxL
    vel = rng.random(c.dims)*2 - 1 # -1 to 1
    temp = Atom(pos, vel, c.colors[i])
    particles.append(temp)
    
for i in range(c.timesteps):
    for particle in particles:
        particle.euler_step(particles)
        
    if not i % c.steps_per_plot:
        make_plot(plotcounter, particles)
        plotcounter += 1