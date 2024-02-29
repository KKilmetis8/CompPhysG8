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
    system(f'ffmpeg -i sims/sim{simnum}/fig%d.png -c:v libx264 -r 30 sims/sim{simnum}/{name}.mp4 -loglevel panic')


def make_plot(index, particles):
    plt.ioff()
    figno = int(index/c.steps_per_plot)
    fig, ax = plt.subplots(figsize = (6,5))
    
    # get relevant info
    pos, vels, colors = particles.all_positions[index], particles.all_velocities[index], particles.colors
    
    # calculate the normalised velocity coordinates for arrows
    vnorms = vels/np.linalg.norm(vels, axis=0)
    
    # plot particles
    ax.scatter(pos[0], pos[1], c = colors, zorder=3)
    
    # plot arrows, maybe this should be with ax.arrow so we can change the length
    ax.quiver(pos[0], pos[1], vnorms[0], vnorms[1], 
              headwidth = 5, headaxislength = 5, width = 0.003,
              angles='xy', zorder=2)
    
    # plot the box
    ax.set_xlim(-0.1 * c.boxL, 1.1 * c.boxL), ax.set_ylim(-0.1 * c.boxL, 1.1 * c.boxL)
    ax.axhline(0, c = 'k', linestyle = '--'), ax.axhline(c.boxL, c = 'k', linestyle = '--')
    ax.axvline(0, c = 'k', linestyle = '--'), ax.axvline(c.boxL, c = 'k', linestyle = '--')
    

    plt.savefig(f'sims/sim{simnum}/fig{figno}')
    plt.close(fig)

# Testing
# Collision
# particles = Particles([Atom(pos = [0.1*c.boxL, 0.5*c.boxL], vel=[0.5 , 0], color=c.colors[0]),
#                        Atom(pos = [0.5*c.boxL, 0.5*c.boxL], vel=[-0.5, 0], color=c.colors[1])])

# From slides
particles = Particles([Atom(pos = [0.3*c.boxL, 0.51*c.boxL], vel=[0.09 , 0], color=c.colors[0]),
                       Atom(pos = [0.7*c.boxL, 0.49*c.boxL], vel=[-0.09, 0], color=c.colors[1])])



# Make set of particles


#%% First calculate all positions/velocities
particles = Particles(c.Nbodies, seed=c.rngseed)
for i in range(c.timesteps):
    particles.update()
#%%
def energy_plot():
    time = np.arange(0, c.timesteps + 1) * c.h * c.time_to_cgs * 1e12
    plt.ion()
    energies = particles.all_energies
    kinetic = [ np.sum(energy[0]) for energy in energies]
    potential = [ np.abs(np.sum(energy[1])) for energy in energies]
    total = [ np.sum(energy[2]) for energy in energies]

    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(time, kinetic, c = c.c93)
    ax.plot(time, potential, c = c.c97)
    ax.plot(time, total, c = 'k', linestyle = '-.')

    # Make pretty
    ax.set_yscale('log')
    # ax.set_xscale('')
    # ax.set_ylim(1e-12, 1e2)
    # ax.set_xlim(0, 0.5)
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Energy [sim units]')
energy_plot()
# Then make all the plots
for i in np.arange(0, c.timesteps, c.steps_per_plot):
    make_plot(int(i), particles)

make_movie()