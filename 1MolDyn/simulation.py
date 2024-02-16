#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:52:31 2024

@authors: konstantinos & diederick
"""

# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate Imports
import prelude as c
from Atom import Atom
from Particles import Particles
from os import listdir, makedirs, system

# this is a saving function inside simulation class
simnum = int(max([0]+[float(file[3:])for file in listdir('sims')])+1)

# make a folder within sims where the figures are saved
# easier to organise each simulation
makedirs(f"sims/sim{simnum}", exist_ok=True)

def make_movie(simnum=simnum, name='moive'):
    system(f'ffmpeg -i sims/sim{simnum}/fig%d.png -c:v libx264 -r 30 sims/sim{simnum}/{name}.mp4')


plt.ioff()
def make_plot(index, particles):
    figno = int(index/c.steps_per_plot)
    fig, ax = plt.subplots(figsize = (6,5))
    
    #get relevant info
    pos, vels, colors = particles.all_positions[index], particles.all_velocities[index], particles.colors
    #calculate the normalised velocity coordinates for arrows
    vnorms = vels/np.linalg.norm(vels, axis=0)
    
    #plot particles
    ax.scatter(pos[0], pos[1], c = colors, zorder=3)
    
    #plot arrows
    ax.quiver(pos[0], pos[1], vnorms[0], vnorms[1], angles='xy', zorder=2)
    
    #plot the box
    ax.set_xlim(-0.1 * c.boxL, 1.1 * c.boxL), ax.set_ylim(-0.1 * c.boxL, 1.1 * c.boxL)
    ax.axhline(0, c = 'k', linestyle = '--'), ax.axhline(c.boxL, c = 'k', linestyle = '--')
    ax.axvline(0, c = 'k', linestyle = '--'), ax.axvline(c.boxL, c = 'k', linestyle = '--')
    

    plt.savefig(f'sims/sim{simnum}/fig{figno}')
    plt.close(fig)

#Testing
#Collision
# particles = Particles([Atom(pos = [0.1*c.boxL, 0.5*c.boxL], vel=[0.5 , 0], color=c.colors[0]),
#                        Atom(pos = [0.5*c.boxL, 0.5*c.boxL], vel=[-0.5, 0], color=c.colors[1])])


# Make set of particles
particles = Particles(c.Nbodies, seed=c.rngseed)


# Moved this to be inside the Particles class
# Init random number generator
# rng = np.random.default_rng(seed=c.rngseed)
# particles = []
# for i in range(c.Nbodies):
#     pos = rng.random(c.dims)*c.boxL
#     vel = rng.random(c.dims)*2 - 1 # -1 to 1
#     temp = Atom(pos, vel, c.colors[i])
#     particles.append(temp)

# First calculate all positions/velocities
for i in range(c.timesteps):
    particles.update()

#Then make all the plots
for i in np.arange(0, c.timesteps, c.steps_per_plot):
    make_plot(int(i), particles)

make_movie()