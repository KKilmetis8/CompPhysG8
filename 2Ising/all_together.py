import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

#%%
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])

rng  = np.random.default_rng(None)

N = 20
T = 0.2
steps = 1000*N**2

J = 1
H = 0

beta = 1/T

grid = np.sign(rng.random((N, N)) - 0.5)

chunk_size = 5
MC_step = N**2
x_array = np.arange(chunk_size*MC_step)

def Hamiltonian(grid: np.ndarray) -> float:
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*Conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

def mag_close(avg_mag: float, mag_tol: float = 0.01) -> bool:
    if avg_mag > 1-mag_tol:
        return True
    if avg_mag < -1+mag_tol:
        return True
    if avg_mag < mag_tol and avg_mag > -mag_tol:
        return True
    else: 
        return False

# Initialize holders
all_grids = np.zeros((steps +1, *grid.shape))
all_energies = np.zeros(steps+1)

all_energies[0] = Hamiltonian(grid)
all_grids[0] = grid

patience = 0
patience_threshold = 25

kick = False

for i in range(steps):
    
    old_grid = grid
    if kick:
        flip_row, flip_col = rng.integers(N, size = (2, 10))
        old_grid[flip_row,flip_col] *= -1
        kick = False
    
    # Generate single-flip grid
    flip_row, flip_col = rng.integers(N, size = 2)
    new_grid = old_grid.copy()
    new_grid[flip_row,flip_col] *= -1
    

    old_energy, new_energy = Hamiltonian(old_grid), Hamiltonian(new_grid)

    if i > 2*chunk_size*MC_step  and i % chunk_size*MC_step == 0:
        old_line = np.polyfit(x_array, 
                            all_energies[i - chunk_size*MC_step : i]
                            , deg = 1)
        new_line = np.polyfit(x_array,
                            all_energies[i - 2*chunk_size*MC_step : i-chunk_size*MC_step]
                            , deg = 1)
        abs_diff = np.abs(new_line[1] - old_line[1])
        rel_diff = np.abs(np.abs(new_line[0] - old_line[0]) / old_line[0])
        

        # old_mean = np.mean(all_energies[i - chunk_size*MC_step : i])
        # new_mean = np.mean(all_energies[i - 2*chunk_size*MC_step : i-chunk_size*MC_step])
        # abs_diff = np.abs(new_mean - old_mean)
        # rel_diff = np.abs(abs_diff / old_mean)

        # am i close to the desired mags

        if (rel_diff < 1e-6 or abs_diff < 1e-10) and ~mag_close(grid.mean()):
            if patience < patience_threshold:
                patience += 1
                kick = True
                print(f"KICK {i}")
            else:
                break

    if new_energy < old_energy:
        grid = new_grid.copy()
        all_energies[i+1] = new_energy
        all_grids[i+1] = grid
        continue
    
    prob_ratio = np.exp((old_energy-new_energy) * beta)

    if np.random.random() < prob_ratio:
        grid = new_grid.copy()
        all_energies[i+1] = new_energy
        all_grids[i+1] = grid
        continue

    grid = old_grid.copy()
    all_energies[i+1] = all_energies[i]
    all_grids[i+1] = grid

avg_magnetizations = all_grids.mean(axis=(1,2))

#%%
fig,ax = plt.subplots(1,2, tight_layout=True)
ax[0].plot(np.arange(i)/MC_step, avg_magnetizations[:i])
ax[0].set_ylim(-1,1)
ax[0].set_xlim(0,steps/MC_step)

ax[1].plot(np.arange(i)/MC_step, all_energies[:i])
#ax[1].set_xlim(i/MC_step-2*chunk_size,i/MC_step)
ax[1].set_xlim(0, steps/MC_step)

ax[1].axvline(x=i/MC_step, c='k', ls='--')

x_old = np.arange(i-2*chunk_size*MC_step, i-chunk_size*MC_step)/MC_step
x_new = np.arange(i-chunk_size*MC_step, i)/MC_step

# x_old = x_array/MC_step
# x_new = x_array/MC_step

# y_old  = x_old * old_line[0] + old_line[1]
# y_new  = x_new * new_line[0] + new_line[1]

# ax[1].plot(x_old, y_old, c='k'), ax[1].plot(x_new, y_new, c='r')

plt.suptitle(f"T={T}")

fig,ax = plt.subplots(1,2, tight_layout=True)
ax[0].imshow(all_grids[0], cmap='coolwarm')
ax[1].imshow(all_grids[i], cmap='coolwarm')
plt.suptitle(f"T={T}")
#plt.show()
# %%
