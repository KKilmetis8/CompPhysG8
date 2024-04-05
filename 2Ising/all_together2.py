import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

#%%
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])

rng  = np.random.default_rng(None)

N = 20
T = 10
steps = int(1e5)

J = 1
H = 0

beta = 1/T

#init_grid = np.sign(rng.random((N, N)) - 0.5)
init_grid = np.ones((N,N))

def Hamiltonian(grid: np.ndarray) -> float:
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*Conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

# Initialize holders
all_energies = np.zeros(steps+1)
all_avg_mags = np.zeros(steps+1)

all_energies[0] = Hamiltonian(init_grid)
all_avg_mags[0] = init_grid.mean()

last_grid = init_grid.copy()
for i in range(steps):
    # Generate single-flip grid 
    flip_row, flip_col = rng.integers(N, size = 2)
    new_grid = last_grid.copy()
    new_grid[flip_row,flip_col] *= -1

    old_energy, new_energy = Hamiltonian(last_grid), Hamiltonian(new_grid)
    prob_ratio = np.exp((old_energy-new_energy) * beta)

    if new_energy < old_energy or np.random.random() < prob_ratio:
        all_energies[i+1] = new_energy
        all_avg_mags[i+1] = new_grid.mean()
        last_grid = new_grid.copy()
        continue

    all_energies[i+1] = old_energy
    all_avg_mags[i+1] = last_grid.mean()



#%%
fig,ax = plt.subplots(1,2, tight_layout=True)
ax[0].plot(np.arange(steps+1)/N**2, all_avg_mags)
ax[0].set_ylim(-1,1)
ax[0].set_xlim(0,(steps+1)/N**2)

ax[1].plot(np.arange(steps+1)/N**2, all_energies)
ax[1].set_xlim(0, (steps+1)/N**2)
fig.suptitle(f"T={T}")

fig,ax = plt.subplots(1,2, tight_layout=True)
ax[0].imshow(init_grid, cmap='coolwarm')
ax[1].imshow(last_grid, cmap='coolwarm')
fig.suptitle(f"T={T}")
#plt.show()
# %%
