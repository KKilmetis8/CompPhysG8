"""
Created on Fri Mar 22 12:09:45 2024

@authors: diederick & konstantinos
"""

# Strawberry imports
import numpy as np
import matplotlib.pyplot as plt

# Stracciatella import
from scipy.signal import convolve2d 

# constants
temperature = 1.5 # k_B
Nbodies     = 10
rngseed     = 8
cmap        = "coolwarm"

# Grid initialization
rng  = np.random.default_rng(seed=rngseed)
grid = np.sign(rng.random((Nbodies, Nbodies)) - 0.5)

# Convolve kernel
kernel = np.array([[0, 1, 0], 
                   [1, 1, 1], 
                   [0, 1, 0]])

# Functions
def plot_grid(grid: np.ndarray, cmap: str = cmap):
    '''
    Plot the grid.

    Parameters
    ----------
    grid: np.ndarray, the (Nbodies, Nbodies) grid to plot.

    cmap: string, which colormap to use.
    '''
    fig, ax = plt.subplots(1,1)
    ax.imshow(grid, cmap)

def total_magnetization(grid: np.ndarray) -> float:
    '''
    Calculates the total magnetization of the grid.

    Parameters
    ----------
    grid: np.ndarray, the (Nbodies, Nbodies) grid.

    Returns
    -------
    M: float, total magnetization.
    '''
    return grid.sum()

def neighbor_sum(grid: np.ndarray) -> np.ndarray:
    '''
    Finds the sum of all direct neighbors in grid for each cell.
    Sum includes the cell itself.

    Parameters
    ----------
    grid: np.ndarray, the (Nbodies, Nbodies) grid.

    Returns
    -------
    summed: neighbors-summed version of grid.
    '''
    return convolve2d(grid, kernel, boundary='wrap', mode='same')

print(total_magnetization(grid))
plot_grid(grid)
plt.show()
