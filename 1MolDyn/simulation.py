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
from os import listdir, makedirs, system
import time

# Chocolate Imports
import config
import prelude as c
from Particles import Particles
from importlib import reload

reload(c)

# Simulation ID for saving
time = time.strftime('%d-%h-%H:%M:%S', time.localtime())
#simname = f'{config.state_of_matter}_at_{time}'
simname = f'{c.density, c.temperature}'
print(simname)


# Folder for figs
makedirs(f"sims/{simname}/", exist_ok=True)
#%% Make set of particles
particles = Particles(c.Nbodies)
particles.equilibriate()

#%% Simulate
current_progress = 0
for i in range(c.timesteps):
    particles.update(step = 'leapfrog')
    if not i % c.n_frequency: # Calculate n(r) every n_frequency timesteps
        particles.n_pair_correlation()
        particles.pressure_sum_part()
    
    # Progress check
    if config.loud:
        progress = int(np.round(i/c.timesteps,1) * 100)
        if i % 100  == 0 and progress != current_progress:
            print('Simulation is', progress, '% done')
            current_progress = progress            
if config.loud:
    print('Simulation is done')
    
pressure = particles.pressure()
pressure_string = f'{pressure}' 
print('Pressure:', pressure_string)
with open(f'sims/{simname}/params.txt', 'w') as f:    
    f.write(f'{c.temperature} {c.density} {pressure}')
    f.close()

#%% Plotting   
def pair_correlation_plot(simname):
    plt.figure()
    r, g = particles.g_pair_correlation()
    plt.plot(r * c.SIGMA, g, c='k',marker='h')
    plt.xlabel('r [Angstrom]', fontsize = 14)
    plt.ylabel('g(r)', fontsize = 14)
    plt.title(config.state_of_matter, fontsize = 14)
    plt.ylim(-0.1, 5)
    plt.savefig(f'sims/{simname}/pair_corr.pdf', format = 'pdf')
pair_correlation_plot(simname)
#%%
def energy_plot(simname):
    time = np.arange(c.timesteps + 1) * c.h_sim_units * c.time_to_cgs * 1e12 # ps
    # Only use actual simulation, not equilibriation energies.
    energies = np.array(particles.all_energies)[-len(time):] 
    kinetic, potential, total = np.sum(energies, axis=2).T

    fig, ax = plt.subplots(1,2, figsize = (6,4), tight_layout = True)
    ax[0].plot(time, kinetic, c = c.c93, label = 'Kinetic')
    ax[0].plot(time, potential, c = c.c97, label = 'Potential')
    ax[0].plot(time, total, c = 'k', linestyle = '-.', label = 'Total')
    
    ax[0].set_xlabel('Time [ps]')
    ax[0].set_ylabel('Energy [sim units]')
    ax[0].legend()
    
    # Error
    error = total - total[0]
    ax[1].plot(time, error, c='k')
    ax[1].grid()
    ax[1].set_xlabel('Time [ps]')
    ax[1].set_ylabel('Energy Error [sim units]')
    fig.suptitle(config.state_of_matter, fontsize = 14)

    # Make pretty
    # ax.set_yscale('log')
    # ax.set_xscale('')
    # ax.set_ylim(1e-12, 1e2)
    # ax.set_xlim(0, 0.5)
    plt.savefig(f'sims/{simname}/energy_err.pdf', format = 'pdf')
energy_plot(simname)

#%% Then make all the plots

def make_movie(simnum=simname, name='moive'):
    system(f'ffmpeg -i sims/{simname}/fig%d.png -c:v libx264 -r 30 sims/{simname}/{name}.mp4 -loglevel panic')

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
    
    plt.savefig(f'sims/{simname}/fig{figno}')
    plt.close(fig)

if config.plot:
    for i in np.arange(0, c.timesteps, c.steps_per_plot):
        make_plot(int(i), particles)
    make_movie()