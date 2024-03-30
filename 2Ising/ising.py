"""
Created on Fri Mar 22 12:09:45 2024

@authors: diederick & konstantinos
"""

#%%
# Strawberry imports
import numpy as np
from scipy.signal import convolve2d
from importlib import reload

# Stracciatella import
import prelude as c
import auxiliary as aux
reload(c)

# Convolve kernel
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])
# RNG
rng  = np.random.default_rng(seed=c.rngseed)
#%%
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


def probability_ratio(grid1: np.ndarray, grid2: np.ndarray, beta) -> float:
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
    if c.kind == 'single':
        beta = c.beta
    Ham1, Ham2 = Hamiltonian(grid1), Hamiltonian(grid2)
    ratio = np.exp((Ham2-Ham1) * beta)
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

def accept_prob(grid_now: np.ndarray, grid_next: np.ndarray, beta) -> float:
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
    prob_ratio, new_energy = probability_ratio(grid_next, grid_now, beta)

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

def Metropolis(grid: np.ndarray, steps: int = c.Nsize**2, beta = c.beta) -> np.ndarray:
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
    # Steps
    eq_flag = False
    MC_step = c.Nsize * c.Nsize

    # Initilize holders
    all_grids = np.zeros((steps // MC_step +1, *grid.shape))
    all_energies = np.zeros(steps+1)
    all_energies[0] = Hamiltonian(grid)
    all_grids[0] = grid

    for i in range(steps):
        old_grid = grid
        
        # Generate single-flip grid
        flip_row, flip_col = rng.integers(c.Nsize, size=2)
        new_grid = old_grid.copy()
        new_grid[flip_row][flip_col] *= -1

        A_param, new_energy = accept_prob(old_grid, new_grid, beta)
        
        if (A_param >= 1) or (np.random.random() < A_param):
            grid = new_grid.copy()
            all_energies[i+1] = new_energy

        else:
            grid = old_grid.copy()
            all_energies[i+1] = all_energies[i]
        
        # Check for Eq.
        if i % MC_step == 0 and i!=0:
            # Save grid
            all_grids[i // MC_step] = grid
            
            if c.loud:
                print(f'MC_step: {i / MC_step} out of {steps / MC_step}')
            
            # Let it run a bit, then check for EQ
            if i > 10*MC_step:
                am_I_in_EQ = Equilibrium_check(i, all_energies)
                
                if am_I_in_EQ:
                    # Remove zeros at the end
                    all_energies = all_energies[:i]
                    all_grids = all_grids[:i // MC_step]
                    eq_flag = True
                    print('System Equilibriated')
                    break

    if not eq_flag:
        print(f'System did NOT equilibriate in {i*MC_step} MC steps' +\
              '\n, simulation is untrustworthy.')
    return all_grids, all_energies

