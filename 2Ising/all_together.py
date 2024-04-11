import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

#%%
kernel = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]])

# 9, T=0.01 is a river
rng  = np.random.default_rng(None)

N = 50
T = 2
steps = 200*N**2
J = 1
H = 0
beta = 1/T

grid = np.sign(rng.random((N, N)) - 0.5)
chunk_size = 15
MC_step = N**2
x_array = np.arange(chunk_size*MC_step)

def Hamiltonian(grid: np.ndarray) -> float:
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*Conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

def energy_diff(last_grid: np.ndarray[float], flip_row: int, flip_col: int, J: float=J, H: float=H) -> float:
    flipped_spin    = last_grid[flip_row,flip_col]

    top    = last_grid[(flip_row-1)%len(last_grid), flip_col]
    left   = last_grid[flip_row, (flip_col-1)%len(last_grid)]
    right  = last_grid[flip_row, (flip_col+1)%len(last_grid)]
    bottom = last_grid[(flip_row+1)%len(last_grid), flip_col]

    neighbors_sum   = top + left + right + bottom
    energy_diff     = 4*J*flipped_spin*neighbors_sum + 2*H*flipped_spin
    return energy_diff

def middle_finder(river, buffer):
    try:
        idx = river[ len(river) // 2]
    except IndexError:
        idx = buffer
    return idx

def bridge_builder(grid_in:np.ndarray[float], patience) -> np.ndarray[float]:
    grid = grid_in.copy()
    buffer = 1 # 2*buffer+1 lines will be flipped
    tol = 0.15
    mean_mag = np.mean(grid)    
    # Check if there is a river there, these can be combined if we get smart
    if mean_mag > tol:
        # Grid is mostly positive, find a river of negative spins
        flip_to = 1
    if mean_mag < -tol:
        # Grid is mostly negative, find a river of positive spins
        flip_to = -1
    else:
        # Do nothing, grid is not close to eq
        return grid, patience-1
    
    # Check the difference between each row/col
    row_sums_diffs = [ np.abs(np.sum(grid[i]) - np.sum(grid[i-1]))
                       for i in range(1, N)]
    col_sums_diffs = [ np.abs(np.sum(grid.T[i]) - np.sum(grid.T[i-1]))
                       for i in range(1, N)]
    # Find the river
    river_row = [] 
    river_col = [] 
    for i in range(N-1):
        if row_sums_diffs[i] > (2*N)*0.9:
            river_row.append(i)
        if col_sums_diffs[i] > (2*N)*0.9:
            river_col.append(i)
    row_idx = middle_finder(river_row, buffer) 
    col_idx =  middle_finder(river_col, buffer) 
    
    # Flip spins in a cross, make it wide
    grid[row_idx - buffer :row_idx + buffer] = flip_to
    grid.T[col_idx - buffer :col_idx + buffer] = flip_to
    return grid, patience

#%%
# Initialize holders
# all_grids = np.zeros((steps +1, *grid.shape))
all_energies = np.zeros(steps+1)
all_magnetizations = np.zeros(steps+1)

all_energies[0] = Hamiltonian(grid)
all_magnetizations[0] = grid.sum()/N**2

# all_grids[0] = grid
patience = 0
patience_threshold = 100

kick = False
for i in range(steps):
    old_grid = grid
    if patience == patience_threshold:
        print('patience break')
        break
    if kick:
        # flip_row, flip_col = rng.integers(N, size = (2, 10))
        # old_grid[flip_row,flip_col] *= -1
        old_grid, patience = bridge_builder(grid, patience)
        kick = False
        plt.figure()
        plt.imshow(old_grid)
        
    # Generate single-flip grid
    flip_row, flip_col = rng.integers(N, size = 2)
    new_grid = old_grid.copy()
    new_grid[flip_row,flip_col] *= -1
    old_energy, new_energy = Hamiltonian(old_grid), Hamiltonian(new_grid)
    # delta_E = energy_diff(last_grid, flip_row, flip_col)
    if i > 0.5*steps and i % (chunk_size*MC_step) == 0:
        old_line = np.polyfit(x_array, 
                            all_energies[i - chunk_size*MC_step : i]
                            , deg = 1)
        new_line = np.polyfit(x_array,
                            all_energies[i - 2*chunk_size*MC_step : i-chunk_size*MC_step]
                            , deg = 1)
        abs_diff = np.abs(new_line[1] - old_line[1])
        rel_diff = np.abs(np.abs(new_line[0] - old_line[0]) / old_line[0])

        if (rel_diff > 1e-2 or abs_diff > 1e-2):
            if patience < patience_threshold:
                patience += 1
                kick = True
                print(f"KICK {i/MC_step}")
                plt.figure()
                plt.imshow(grid)
            else:
                print('patience break')
                break
        else:
            print('tol break')
            break
        
    prob_ratio = np.exp((old_energy-new_energy) * beta)
    if new_energy < old_energy or np.random.random() < prob_ratio:
        grid = new_grid.copy()
        all_energies[i+1] = new_energy
        all_magnetizations[i+1] = grid.sum()/N**2
        continue
    grid = old_grid.copy()
    all_energies[i+1] = all_energies[i]
    all_magnetizations[i+1] = grid.sum()/N**2

    # all_grids[i+1] = grid


#%%
fig,ax = plt.subplots(1,2, tight_layout=True)
ax[0].plot(np.arange(i)/MC_step, all_magnetizations[:i])
# ax[0].set_ylim(-1,1)
# ax[0].set_xlim(0,steps/MC_step)

ax[1].plot(np.arange(i)/MC_step, all_energies[:i])
#ax[1].set_xlim(i/MC_step-2*chunk_size,i/MC_step)
# ax[1].set_xlim(0, steps/MC_step)

ax[1].axvline(x=i/MC_step, c='k', ls='--')

x_old = np.arange(i-2*chunk_size*MC_step, i-chunk_size*MC_step)/MC_step
x_new = np.arange(i-chunk_size*MC_step, i)/MC_step

# x_old = x_array/MC_step
# x_new = x_array/MC_step

# y_old  = x_old * old_line[0] + old_line[1]
# y_new  = x_new * new_line[0] + new_line[1]

# ax[1].plot(x_old, y_old, c='k'), ax[1].plot(x_new, y_new, c='r')

plt.suptitle(f"T={T}")

# fig,ax = plt.subplots(1,2, tight_layout=True)
# ax[0].imshow(all_grids[0], cmap='coolwarm')
# ax[1].imshow(all_grids[i], cmap='coolwarm')
# plt.suptitle(f"T={T}")
#plt.show()
# %%
