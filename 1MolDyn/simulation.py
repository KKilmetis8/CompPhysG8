#%%
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
particles = Particles([Atom(pos = [0.7*c.boxL, 0.49*c.boxL], vel=[-0.09, 0], color=c.colors[1]), 
                       Atom(pos = [0.3*c.boxL, 0.51*c.boxL], vel=[0.09 , 0], color=c.colors[0])],
                       )


# Make set of particles
particles = Particles(c.Nbodies, seed=c.rngseed)

#%% First calculate all positions/velocities

for i in range(c.timesteps):
    particles.update(step = 'leapfrog')

#%%
def energy_plot():
    time = np.arange(c.timesteps + 1) * c.h_sim_units * c.time_to_cgs * 1e12 # ps
    plt.ion()
    energies = np.array(particles.all_energies)
    kinetic, potential, total = np.sum(energies, axis=2).T

    fig, ax = plt.subplots(1,2, figsize = (6,4), tight_layout = True)
    ax[0].plot(time, kinetic, c = c.c93)
    ax[0].plot(time, potential, c = c.c97)
    ax[0].plot(time, total, c = 'k', linestyle = '-.')
    
    ax[0].set_xlabel('Time [ps]')
    ax[0].set_ylabel('Energy [sim units]')
    
    # Error
    error = total - total[0]
    ax[1].plot(time, error, c='k')
    ax[1].grid()
    ax[1].set_xlabel('Time [ps]')
    ax[1].set_ylabel('Energy Error [sim units]')

    # Make pretty
    # ax.set_yscale('log')
    # ax.set_xscale('')
    # ax.set_ylim(1e-12, 1e2)
    #ax.set_xlim(0, 0.5)

energy_plot()

#%% Then make all the plots
for i in np.arange(0, c.timesteps, c.steps_per_plot):
    make_plot(int(i), particles)

make_movie()