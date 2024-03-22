"""
Created on Fri Mar 22 12:09:45 2024

@authors: diederick & konstantinos
"""

#%%
# Strawberry imports
import numpy as np
import matplotlib.pyplot as plt

# Stracciatella import
from scipy.signal import convolve2d
import prelude as c
from importlib import reload
reload(c)

# Convolve kernel
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])

#%%
# Functions
def plot_grid(grid: np.ndarray, cmap: str = c.cmap, title: str | None = None):
    '''
    Plot the grid.

    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid to plot.

    cmap: string, which colormap to use.

    title: string or None, title to put above the plot.
           If set to None, does not put a title.
    '''
    unique = np.unique(grid)
    colors = plt.get_cmap(cmap, len(unique))

    fig, ax = plt.subplots(1,1)
    img = ax.imshow(grid, colors, origin='lower')
    cbar = fig.colorbar(img, ax = ax, cmap=colors, fraction=0.046, pad=0.04)
    cbar.ax.set_yticks(unique*(1-1/len(unique)))
    cbar.ax.set_yticklabels(unique.astype(int))
    ax.set_title(title)

def total_magnetization(grid: np.ndarray) -> float:
    '''
    Calculates the total magnetization of the grid.

    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid.

    Returns
    -------
    M: float, total magnetization.
    '''
    return grid.sum()

def neighbor_sum(grid: np.ndarray) -> np.ndarray:
    '''
    Finds the sum of all direct neighbors in grid for each cell.

    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid.

    Returns
    -------
    summed: neighbors-summed version of grid.
    '''
    return convolve2d(grid, kernel, boundary='wrap', mode='same')

def Hamiltonian(grid: np.ndarray, J: float = c.J_coupling, H: float = c.H_ext) -> float:
    '''
    Calculates the Hamiltonian of the grid.

    Hamiltonian = -J * sum(s_i, s_j) - H * sum(s_j)
    
    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid.

    J: float, coupling constant between spins,
       default to J = k_B = 1.

    H: float, external magnetic field.

    Returns
    -------
    fancy_H: Calculated Hamiltonian of the grid
    '''
    coupling_term      = -J * (grid*neighbor_sum(grid)).sum()
    ext_mag_field_term = -H * total_magnetization(grid)
    return coupling_term + ext_mag_field_term

def avg_magnetization(grid: np.ndarray) -> float:
    '''
    Calculates the average magnetization per cell.

    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid.

    Returns
    -------
    avg_magnetization: float, average magnetization per cell.
    '''
    return total_magnetization(grid)/(c.Nsize**2)

def probability_ratio(grid1: np.ndarray, grid2: np.ndarray) -> float:
    '''
    Calculates the probability ratio p(grid1)/p(grid2) = exp[(H(grid2) - H(grid1))/T]

    Parameters
    ----------
    grid1: np.ndarray, (Nsize, Nsize) grid.

    grid2: np.ndarray, (Nsize, Nsize) grid.

    Returns
    -------
    ratio: The probability ratio p(grid1)/p(grid2)
    '''
    Ham1, Ham2 = Hamiltonian(grid1), Hamiltonian(grid2)
    ratio = np.exp((Ham2-Ham1)/c.temperature)
    return ratio

def trial_prob(grid_now: np.ndarray, grid_next: np.ndarray) -> float:
    '''
    Calculates the trial probability parameter w,
    specifically for a single flip.

    Parameters
    ----------
    grid_now:  np.ndarray, the current (Nsize, Nsize) grid.

    grid_next: np.ndarray, the next (Nsize, Nsize) grid.

    Returns
    -------
    w: float, trial probability
    '''
    # if single flip -> single_flip = 1
    single_flip = (grid_now != grid_next).sum()

    if single_flip == 1:
        return 1/(c.Nsize**2)
    else: 
        return 0

def accept_prob(grid_now: np.ndarray, grid_next: np.ndarray) -> float:
    '''
    Calculates the acceptance probability parameter A,
    specifically for a single flip.

    Parameters
    ----------
    grid_now:  np.ndarray, the current (Nsize, Nsize) grid.

    grid_next: np.ndarray, the next (Nsize, Nsize) grid.

    Returns
    -------
    A: float, acceptance probability
    '''
    prob_ratio = probability_ratio(grid_next, grid_now)

    if prob_ratio > 1:
        return 1
    else:
        return prob_ratio

#%%
# Grid initialization
rng  = np.random.default_rng(seed=c.rngseed)
grid = np.sign(rng.random((c.Nsize, c.Nsize)) - 0.5)

# random spin-flip
flip_indices = rng.integers(c.Nsize, size=2)
flipped = grid.copy()
flipped[*flip_indices] *= -1

plot_grid(grid, title='Initial grid')
plot_grid(flipped, title=f"{flip_indices} flipped")
plot_grid(neighbor_sum(grid), title='Neighbors summed')
# %%
