#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:57:27 2024

@author: konstantinos
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from os import system, makedirs

# Choc
import prelude as c
import auxiliary as aux 

def grid(grid_p: np.ndarray, cmap: str = c.cmap, title: str | None = None) -> tuple[Axes, Axes]:
    '''
    Plot the grid.

    Parameters
    ----------
    grid_p: np.ndarray, the (Nsize, Nsize) grid to plot.

    cmap: string, which colormap to use.

    title: string or None, title to put above the plot.
           If set to None, does not put a title.

    Returns
    -------
    (ax,cbar): matplotlib.Axes objects, references the main figure axis
               and colorbar axis, respectively.
    '''
    # plt.ioff()
    unique = np.unique(grid_p)
    colors = plt.get_cmap(cmap, len(unique))

    fig, ax = plt.subplots(1,1)
    img = ax.imshow(grid_p, colors, origin='lower', vmax = 1.5, vmin = -1.5)
    cbar = fig.colorbar(img, ax = ax, cmap=colors, fraction=0.046, pad=0.04)
    cbar.ax.set_yticks(unique*(1-1/len(unique)))
    cbar.ax.set_yticklabels(unique.astype(int))
    ax.set_title(title)
    return ax, cbar

def avg_magnetization(grids: np.ndarray) -> Axes:
    '''
    Creates a figure showing the average magnetization over time.

    Parameters
    ----------
    grids: A 3D numpy array which has the grids.
    
    Returns
    -------
    ax: matplotlib.Axes object, references the main figure axis.
    '''
    avg_mags = [aux.avg_magnetization(grid) for grid in grids]
    time = np.arange(len(grids)) / c.Nsize**2

    fig, ax = plt.subplots(1,1)
    ax.plot(time, avg_mags)

    ax.set_xlim(time[0], time[-1])

    ax.set_ylabel('Average magnetization $\\left(m=M/N^2\\right)$')
    ax.set_xlabel('Monte Carlo steps per lattice site $\\left(\\mathrm{steps}/N^2 \\right)$')

    return ax

def energy(energies:np.array) -> Axes:
    '''
    Creates a figure showing the energy over time.

    Parameters
    ----------
    energies: arr, An 1D array which contains the energies

    Returns
    -------
    ax: matplotlib.Axes object, references the main figure axis.

    '''
    
    time = np.arange(len(energies)) / c.Nsize**2

    fig, ax = plt.subplots(1,1, figsize = (3,3))
    ax.plot(time, energies, c = 'k')

    ax.text(0.70, 0.82, f'T: {c.temperature}', 
            fontsize = 12, transform=fig.transFigure)
    ax.set_xlim(time[0], time[-1])
    # ax.set_yscale('log')
    ax.set_ylabel('Energy [sim units]')
    ax.set_xlabel('Monte Carlo steps per lattice site $\\left(\\mathrm{steps}/N^2 \\right)$')
    return ax

def mag_temp(temp, mag):
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(temp, mag, c = 'k', ls = ':', marker = 'h', ms = 3)
    # ax.set_yscale('log')
    ax.set_ylabel('Average Magnetization [sim units]')
    ax.set_xlabel('Temperature [sim units]')
    return ax

def make_movie(grids: np.ndarray, nplots: int):
    '''
    Creates a movie showing how the grid evolves over time.

    Parameters
    ----------
    grids: A 3D numpy array which has the grids.

    nplots: The number of plots to put in the movie, equally spaced.
    '''
    makedirs('movie', exist_ok=True)

    plot_indices = np.linspace(0,len(grids)-1, nplots, dtype=int)

    for i, index in enumerate(plot_indices):
        grid_temp = grids[index]
        grid(grid_temp)
        plt.savefig(f'movie/{i+1}grid.png')
        plt.close()
    system('ffmpeg -i movie/%dgrid.png -c:v libx264 -r 30 movie/movie.mp4 -loglevel panic')