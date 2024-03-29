"""
Created on Fri Mar 22 12:09:45 2024

@authors: diederick & konstantinos
"""

#%%
# Strawberry imports
import numpy as np
import matplotlib.pyplot as plt
import numba

# Stracciatella import
from scipy.signal import convolve2d, fftconvolve
import prelude as c
from matplotlib.axes import Axes
from importlib import reload
from os import system, makedirs
reload(c)

# Convolve kernel
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])

#%%
# Functions
def plot_grid(grid: np.ndarray, cmap: str = c.cmap, title: str | None = None) -> tuple[Axes, Axes]:
    '''
    Plot the grid.

    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid to plot.

    cmap: string, which colormap to use.

    title: string or None, title to put above the plot.
           If set to None, does not put a title.

    Returns
    -------
    (ax,cbar): matplotlib.Axes objects, references the main figure axis
               and colorbar axis, respectively.
    '''
    # plt.ioff()
    unique = np.unique(grid)
    colors = plt.get_cmap(cmap, len(unique))

    fig, ax = plt.subplots(1,1)
    img = ax.imshow(grid, colors, origin='lower', vmax = 1.5, vmin = -1.5)
    cbar = fig.colorbar(img, ax = ax, cmap=colors, fraction=0.046, pad=0.04)
    cbar.ax.set_yticks(unique*(1-1/len(unique)))
    cbar.ax.set_yticklabels(unique.astype(int))
    ax.set_title(title)
    return ax, cbar

@numba.njit
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
    
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    return Conv 

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
    Ham2: The new Hamiltionian, for tracking the energy.
    '''
    Ham1, Ham2 = Hamiltonian(grid1), Hamiltonian(grid2)
    ratio = np.exp((Ham2-Ham1) * c.beta)
    return ratio, Ham2

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
    new_energy: float, the energy of the new configuration
    '''
    prob_ratio, new_energy = probability_ratio(grid_next, grid_now)

    if prob_ratio > 1:
        return 1, new_energy
    else:
        return prob_ratio, new_energy
    
def transition_prob(grid_now: np.ndarray, grid_next: np.ndarray) -> float:
    '''
    T(x->x')
    Calculates the probability of changing from grid_now to grid_next.

    Parameters
    ----------
    grid_now:  np.ndarray, the current (Nsize, Nsize) grid.

    grid_next: np.ndarray, the next (Nsize, Nsize) grid.

    Returns
    -------
    T: float, state change probability.
    '''
    return trial_prob(grid_now, grid_next) * accept_prob(grid_now, grid_next)

def Equilibrium_check(last_step, energies, rtol = 1e-2, atol = 1e-4):
    ''' 
    Fits a line to the energy of the last 5 MC steps. If the lines slope does 
    not deviate a lot from the line fitted to the 5 MC steps before, 
    stops the simulation
    '''
    MC_step = c.Nsize**2
    x_array = np.arange(5*MC_step)

    old_line = np.polyfit(x_array, 
                          energies[last_step - 5*MC_step : last_step]
                          , deg = 1)
    new_line = np.polyfit(x_array,
                          energies[last_step - 10*MC_step : last_step-5*MC_step]
                          , deg = 1)
 
    abs_diff = np.abs(new_line[0] - old_line[0])
    rel_diff = np.abs(abs_diff / old_line[0])
    
    if c.loud:
        print(f'absolute diff: {abs_diff}')
        print(f'relative diff: {rel_diff}')
        print('---')
        
    if rel_diff < rtol:
        return True
    elif abs_diff < atol:
        return True
    else:
        return False



def Metropolis(grid: np.ndarray, steps: int = c.Nsize**2) -> np.ndarray:
    '''
    Performs the Metropolis algorithm.

    Parameters
    ----------
    grid: np.ndarray, the starting (Nsize, Nsize) grid. (x_0)

    steps: int, how many steps to perform the Metropolis algorithm.
           Default value of c.Nsize**2 as to give every spin a chance
           to flip.

    Returns
    -------
    all_grids: array, all the used grids in the sequence.
    all_energies: array, all the energies in the sequence
    '''
    eq_flag = False
    MC_step = c.Nsize * c.Nsize
    # all_grids = np.zeros((steps+1, *grid.shape))
    all_energies = np.zeros(steps+1)
    # Init energies
    all_energies[0] = Hamiltonian(grid)
    # all_grids[0] = grid

    for i in range(steps):
        old_grid = grid
        
        # Generate single-flip grid
        flip_row, flip_col = rng.integers(c.Nsize, size=2)
        new_grid = old_grid.copy()
        new_grid[flip_row][flip_col] *= -1

        A_param, new_energy = accept_prob(old_grid, new_grid)
        
        if (A_param >= 1) or (np.random.random() < A_param):
            # all_grids[i+1] = new_grid
            grid = new_grid.copy()
            all_energies[i+1] = new_energy

        else:
            # all_grids[i+1] = old_grid
            grid = old_grid.copy()
            all_energies[i+1] = all_energies[i]
        
        # Check for Eq.
        if i % MC_step == 0 and i!=0 and i > 10*MC_step:
            if c.loud:
                print(f'MC_step: {i / MC_step} out of {steps / MC_step}')
            am_I_in_EQ = Equilibrium_check(i, all_energies)
            if am_I_in_EQ:
                # Remove zeros at the end
                all_energies = all_energies[:i]
                eq_flag = True
                print('System Equilibriated')
                break

    if not eq_flag:
        print(f'System did NOT equilibriate in {i*MC_step} MC steps' +\
              '\n, simulation is untrustworthy.')
    return 0, all_energies

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
        grid = grids[index]
        plot_grid(grid)
        plt.savefig(f'movie/{i+1}grid.png')
        plt.close()
    system('ffmpeg -i movie/%dgrid.png -c:v libx264 -r 30 movie/movie.mp4 -loglevel panic')

def avg_magnetization_plot(grids: np.ndarray) -> Axes:
    '''
    Creates a figure showing the average magnetization over time.

    Parameters
    ----------
    grids: A 3D numpy array which has the grids.
    
    Returns
    -------
    ax: matplotlib.Axes object, references the main figure axis.
    '''
    avg_mags = [avg_magnetization(grid) for grid in grids]
    time = np.arange(len(grids)) / c.Nsize**2

    fig, ax = plt.subplots(1,1)
    ax.plot(time, avg_mags)

    ax.set_xlim(time[0], time[-1])

    ax.set_ylabel('Average magnetization $\\left(m=M/N^2\\right)$')
    ax.set_xlabel('Monte Carlo steps per lattice site $\\left(\\mathrm{steps}/N^2 \\right)$')

    return ax

def energy_plot(energies:np.array) -> Axes:
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
#%%
# Grid initialization
rng  = np.random.default_rng(seed=c.rngseed)
grid = np.sign(rng.random((c.Nsize, c.Nsize)) - 0.5)

# random spin-flip
flip_row, flip_col = rng.integers(c.Nsize, size=2)
flipped = grid.copy()
flipped[flip_row][flip_col] *= -1

plot_grid(grid, title='Initial grid')
ax, _ = plot_grid(flipped, title=f"{flip_row},{flip_col} flipped")

flipped_coords_x = [flip_col-0.5, flip_col+0.5, flip_col+0.5, flip_col-0.5, 
                    flip_col-0.5]
flipped_coords_y = [flip_row+0.5, flip_row+0.5, flip_row-0.5, flip_row-0.5, 
                    flip_row+0.5]

ax.plot(flipped_coords_x, flipped_coords_y, "yellow", lw=1)
#*plot_grid(neighbor_sum(grid), title='Neighbors summed')
#%%

_, energies = Metropolis(grid, 1_000*c.Nsize**2)
energy_plot(energies)