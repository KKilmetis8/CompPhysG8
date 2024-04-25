"""
Created on Tue Apr 23 17:55:18 2024

@authors: diederick & konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import config as c
import prelude as p

def eq_mags(final_mags: list[int], convergence: list[bool]):
    '''
    Creates an equilibrated average magnitudes versus temperature plot.

    Parameters
    ----------
    final_mags: list of integers, the equilibrated magnitudes of the simulated
                grids over the temperatures given in `config.py`

    convergence: list of bools, wether the simulations converged or not,
                 determines the color of the datapoints.
    '''

    colors = ["k" if converged else "r" for converged in convergence]
    plt.scatter(c.temperatures, final_mags, 
                c = colors, marker = 'h', s = 100, zorder=3)
    plt.axhline(0, c = 'maroon', ls ='--', zorder=2)
    plt.axvline(p.critical_temp, c = 'maroon', ls ='--', zorder=2)
    plt.xlabel('Temperature', fontsize = 14)
    plt.ylabel('Mean spin', fontsize = 14)
    plt.ylim(-1.2, 1.2)
    plt.xlim(np.min(c.temperatures)*0.8, np.max(c.temperatures)*1.2)
    plt.savefig(f'sims/{p.simname}/eq_mags.pdf', format = 'pdf')
    plt.close()

def main_result(observables: list[np.ndarray[float]], sigmas: list[np.ndarray[float]]):
    """
    Creates a figure with 2x2 subplots showing the mean absolute spin, energy per spin, 
    magnetic susceptibility, specific heat susceptibility, as a function of temperature.
    
    Includes errorbars for each datapoint, the error corresponding to the values given in sigmas

    Parameters
    ----------
    observables : list[np.ndarray[float]]
        list of arrays of equal length with the values for the observables, in the order 
        mean absolute spin, energy per spin, magnetic susceptibility, specific heat susceptibility.

    sigmas : list[np.ndarray[float]]
        list of arrays of equal length with the standard deviations on the observables, in the order 
        mean absolute spin, energy per spin, magnetic susceptibility, specific heat susceptibility.
    """

    fig, axs = plt.subplots(2,2, figsize = (8,6), tight_layout = True)

    ylabels = ['Mean absolute spin', 'Energy per spin', 'Magnetic susceptibility', 'Specific heat per spin']

    for i, (obs, sigma) in enumerate(zip(observables, sigmas)):
        ax = axs.flatten()[i]

        ax.errorbar(c.temperatures, obs, yerr = sigma, ls=':', marker='h', c='k', capsize = 4)
        ax.axvline(p.critical_temp, c = 'maroon', ls ='--', zorder=2)
        ax.set_xlabel('Temperature', fontsize = 14)
        ax.set_ylabel(ylabels[i], fontsize = 14)

    fig.savefig(f'sims/{p.simname}/main_result.pdf', format = 'pdf')
    plt.close(fig)