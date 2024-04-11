import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from plotters import grid as plot_grid

#%%
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])

rng  = np.random.default_rng(None)

N = 20
T = 2
steps = 1000_000

J = 1
H = 0

beta = 1/T

init_grid = np.sign(rng.random((N, N)) - 0.5)
init_grid = np.ones((N, N))
init_grid[:N//2] = -1
#init_grid = np.ones((N,N))

def Hamiltonian(grid: np.ndarray[float]) -> float:
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*Conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

def calc_iteration(init_grid : np.ndarray[float], flips_indices: np.ndarray[int, int], iter: int) -> np.ndarray[float]:
    """
    Calculates the the grid at step `iter` by consecutively flipping 
    the spins that were flipped before during the simulation.

    Parameters
    ----------
    init_grid : np.ndarray[float]
        The initial grid the simulation started with.
    flips_indices : np.ndarray[int, int]
        A (steps, 2) array with the indices of the spin flipped at each
        simulation step, if that grid was not chosen in the simulation step,
        the indices equal (-1, -1).
    iter : int
        The iteration for which to calculate the grid.

    Returns
    -------
    grid : np.ndarray[float]
        The resulting grid.
    """    
    grid = init_grid.copy()
    valid_flips = flips_indices[np.where(flips_indices[:iter+1,0] >= 0)]
    for flip_row, flip_col in valid_flips:
        grid[flip_row, flip_col] *= -1
    return grid

def energy_diff(last_grid: np.ndarray[float], flip_row: int, flip_col: int, J: float=J, H: float=H) -> float:
    """
    Calculates the energy difference from a single spin flip,
    as by dE = H_new - H_old = 4*J*s_flip*neighbors_spin_sum + 2*H*s_flip.

    Parameters
    ----------
    last_grid : np.ndarray[float]
        The grid as it was before the spin-flip.
    flip_row : int
        The row-index of the flipped spin.
    flip_col : int
        The column-index of the flipped spin.
    J : float, optional
        The coupling constant J.
    H : float, optional
        The strength of the external magnetic field.

    Returns
    -------
    float
        Resulting energy difference between last_grid
        and same grid with one spin flipped.
    """    
    flipped_spin    = last_grid[flip_row,flip_col]

    top    = last_grid[(flip_row-1)%len(last_grid), flip_col]
    left   = last_grid[flip_row, (flip_col-1)%len(last_grid)]
    right  = last_grid[flip_row, (flip_col+1)%len(last_grid)]
    bottom = last_grid[(flip_row+1)%len(last_grid), flip_col]

    neighbors_sum   = top + left + right + bottom
    energy_diff     = 4*J*flipped_spin*neighbors_sum + 2*H*flipped_spin
    return energy_diff

def moving_average(a, n=3):
    '''https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy'''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Initialize holders
all_energies = np.zeros(steps+1)
all_avg_mags = np.zeros(steps+1)

all_energies[0] = Hamiltonian(init_grid)
all_avg_mags[0] = init_grid.mean()

last_grid     = init_grid.copy()
flips_indices = -np.ones((steps,2), dtype=int)
for i in range(steps):
    # Generate single-flip grid 
    flip_row, flip_col = rng.integers(N, size = 2)
    new_grid = last_grid.copy()
    new_grid[flip_row,flip_col] *= -1
    
    delta_E = energy_diff(last_grid, flip_row, flip_col)
    prob_ratio = np.exp(-delta_E * beta)

    if delta_E < 0 or np.random.random() < prob_ratio:
        all_energies[i+1] = all_energies[i] + delta_E
        all_avg_mags[i+1] = new_grid.mean()
        last_grid = new_grid.copy()
        flips_indices[i] = [flip_row,flip_col]
        continue

    all_energies[i+1] = all_energies[i]
    all_avg_mags[i+1] = last_grid.mean()



#%%
fig,ax = plt.subplots(1,2, tight_layout=True, sharex=True, sharey=True)
ax[0].plot(np.arange(steps+1)/N**2, all_avg_mags)
ax[0].set_ylim(-1,1)
ax[0].set_xlim(0,steps/N**2)

ax[0].set_xticks(np.linspace(0, steps/N**2, 6, dtype=int))

ax[0].set_ylabel('Average magnetization $\\left(M/N^2\\right)$')
ax[1].set_ylabel('Normalized energy $\\left(\\mathcal{H}/\\mathcal{H}_\\mathrm{max} \\right)$')
fig.text(ax[0].get_position().xmax + fig.subplotpars.wspace/2, 0, 'Monte Carlo steps per lattice site $\\left(\\mathrm{steps}/N^2 \\right)$', ha='center')

max_energy = (4*J + H)*N**2

ax[1].plot(np.arange(steps+1)/N**2, all_energies/max_energy)
fig.suptitle(f"$T={T}$", y=0.9, x=ax[0].get_position().xmax + fig.subplotpars.wspace/2, ha='center')

fig,ax = plt.subplots(1,2, tight_layout=True, sharex=True, sharey=True)
ax[0].imshow(init_grid, cmap='coolwarm', extent=[1-0.5, N+0.5, 1-0.5, N+0.5])
ax[1].imshow(last_grid, cmap='coolwarm', extent=[1-0.5, N+0.5, 1-0.5, N+0.5])

fig.suptitle(f"$T={T}$", y=0.8)

ax[0].set_xticks([1]+list(np.arange(5,N+5, 5)))
ax[0].set_yticks([1]+list(np.arange(5,N+5, 5)))

ax[0].grid(False)
ax[1].grid(False)

plt.figure()
split = np.split(all_energies/max_energy, np.arange(steps, step=steps//10)[1:])
lines = np.zeros((len(split),2))
for i in range(len(split)):
    section = split[i]
    x_range = np.linspace(i,i+1,len(section))/len(split)
    plt.plot(x_range, section)

    line = np.polyfit(x_range, section, deg=1)
    lines[i] = line
    plt.plot(x_range, line[0]*x_range + line[1], lw=2, c='k', ls='--')

plt.axhline(y=-1, c='k', ls=':')

plot_grid(calc_iteration(init_grid, flips_indices, int(steps*0.4)))

plt.show()
# %%
