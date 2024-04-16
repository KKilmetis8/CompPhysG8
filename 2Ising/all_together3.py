import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from plotters import grid as plot_grid
from tqdm import tqdm
import numba
#%%
kernel = 0.5 * np.array([[0, 1, 0], 
                         [1, 0, 1], 
                         [0, 1, 0]])

rng  = np.random.default_rng(None)

def Hamiltonian(grid: np.ndarray[float]) -> float:
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*Conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

@numba.njit( nopython=True, nogil=True)
def metropolis(init_grid, steps, T, init_energy, J, H):
    chunk_size = 20
    beta = 1/T
    N = len(init_grid)
    MC_step = N**2
    trial = chunk_size*MC_step
    
    # Initialize holders
    all_energies = np.zeros(steps+1)
    all_avg_mags = np.zeros(steps+1)
    all_energies[0] = init_energy
    all_avg_mags[0] = init_grid.mean()
    
    last_grid     = init_grid.copy()
    for i in range(steps):
        # Equilibriation Check ----------------------------------------------->
        converged = False
        if i > (2*trial): # I do not know why but it never goes through the second if
            if np.mod(i,2*trial)==0:
                converged = True
                # We can write a numba polyfit but I'd rather not
                rtol = 1e-1
                atol = 1e-3
                old_energy = np.abs(np.mean(all_energies[i-2*trial:i - trial]))
                new_energy = np.abs(np.mean(all_energies[-trial:]))
                abs_diff = np.abs(old_energy - new_energy)
                rel_diff = abs_diff/old_energy
                if rel_diff < rtol or abs_diff < atol:
                    converged = True
                    break
            
        # Generate single-flip grid ------------------------------------------>
        flip_row = np.random.randint(0,N)
        flip_col = np.random.randint(0,N)
        new_grid = last_grid.copy()
        new_grid[flip_row,flip_col] *= -1
        
        # Calc new energy  --------------------------------------------------->
        current_spin = last_grid[flip_row,flip_col]
        candidate_spin = current_spin * -1

        # # Periodic Boundary Conditions
        # if flip_row > 0 and flip_row < N:
        #     top = last_grid[flip_row-1, flip_col]
        # elif flip_row == 0:
        #     top = last_grid[-1, flip_col]
        # else: # flip_row == N
        #     top = last_grid[N-1-1, flip_col] # indexing begins at 0
            
        # if flip_row > 0 and flip_row < N:
        #     bottom = last_grid[flip_row+1, flip_col]
        # elif flip_row == 0:
        #     bottom = last_grid[1, flip_col]
        # else: # flip_row == N
        #     bottom = last_grid[0, flip_col]
            
        # if flip_col > 0 and flip_col < N:
        #     right = last_grid[flip_row, flip_col+1]
        # elif flip_col == 0:
        #     right = last_grid[flip_row, 1]
        # else: # flip_col == N
        #     right = last_grid[flip_row, 0]
        
        # if flip_col > 0 and flip_col < N:
        #     left = last_grid[flip_row, flip_col-1]
        # elif flip_col == 0:
        #     left = last_grid[flip_row, -1]
        # else: # flip_col == N
        #     left = last_grid[flip_row, N-1-1]
            
        # current_energy = -current_spin * (top + left + right + bottom)
        # candidate_energy = -candidate_spin * (top + left + right + bottom)
        # delta_E = candidate_energy - current_energy
        
        top    = last_grid[(flip_row-1)%len(last_grid), flip_col]
        left   = last_grid[flip_row, (flip_col-1)%len(last_grid)]
        right  = last_grid[flip_row, (flip_col+1)%len(last_grid)]
        bottom = last_grid[(flip_row+1)%len(last_grid), flip_col]


        neighbors_sum = top + left + right + bottom
        delta_E = 2*(J*neighbors_sum + H)*current_spin
        
        # delta_E = 2*current_spin * neighbors_sum

        # Accept or Deny ----------------------------------------------------->
        prob_ratio = np.exp(-delta_E * beta)
        if delta_E < 0 or np.random.random() < prob_ratio:
            all_energies[i+1] = all_energies[i] + delta_E
            all_avg_mags[i+1] = new_grid.mean()
            last_grid = new_grid.copy()
            continue
        
        # Store
        all_energies[i+1] = all_energies[i]
        all_avg_mags[i+1] = last_grid.mean()
    return all_energies, all_avg_mags, converged

def chi(step, m, t_max):
    t_diff = t_max - step
    term1  = 1/t_diff*(m[:t_diff+1] * m[step:]).sum()
    term2  = (1/t_diff**2) * (m[:t_diff+1].sum()*m[step:].sum())
    return term1 - term2

#%%
N = 20
MC_step = N**2
steps = 10_000*MC_step
J = 1
H = 0
chunk_size = 20
x_array = np.arange(chunk_size*MC_step)
Ts = np.arange(1, 5, step = 0.1)
mags = []
convergance = []

# ----- 75% Positive ----- #
#flip = rng.integers(N, size = (2, N**2//4))
# init_grid = np.ones((N, N))
# init_grid[flip[0], flip[1]] = -1

# ----- Random ----- #
init_grid = np.sign(rng.random((N, N)) - 0.5)

init_energy = Hamiltonian(init_grid)


ms = []
for T in tqdm(Ts):
    e, m, c = metropolis(init_grid, steps, T, init_energy, J, H)
    
    mags.append( m[-50*MC_step:].mean() )
    convergance.append(c)
    ms.append(m[-50*MC_step:])
#plt.plot(convergance)
# mags2 = []
# init_grid = np.ones((N, N))*(-1)
# init_grid[flip[0], flip[1]] = 1
# init_energy = Hamiltonian(init_grid)
# print(init_grid.mean())
# for T in tqdm(Ts):
#     e, m = metropolis(init_grid, steps, T, init_energy, H)
#     mags2.append( m[-50*MC_step:].mean() )
#%%    
plt.plot(Ts, mags, 
         c = 'k', marker = 'h', ls = '', markersize = 10, zorder=3)
# plt.plot(Ts, mags2, c = '#F1C410', 
#           marker = 'h', ls = ':', markeredgecolor = 'k', markersize = 10)
plt.axhline(0, c = 'maroon', ls ='--', zorder=2)
plt.axvline(2.269, c = 'maroon', ls ='--', zorder=2)
plt.xlabel('T', fontsize = 14)
plt.ylabel('Avg mag', fontsize = 14)
plt.title(f'2D ising model for N:{N}', fontsize = 15)
plt.ylim(-1.2,1.2)
plt.xlim(Ts.min()*0.8, Ts.max()*1.2)

#%%
ts = np.arange(steps//MC_step-10, steps//MC_step)*MC_step
for m in ms:
    chis = np.array([chi(t, m, steps) for t in ts])

    a, b = np.polyfit(ts, np.log(chis), deg=1)
    tau  = -1/a
    chi0 = np.exp(b)

    avg_rel_error = (np.abs(chis - chi0*np.exp(-ts/tau))/chis).mean()
    print(tau, avg_rel_error)

    # ts_fit = np.linspace(0, ts[-1], 1000)
    # plt.figure()
    # plt.plot(ts, chis, marker='o')
    # plt.plot(ts_fit, chi0*np.exp(-ts_fit/tau))
    # #plt.axhline(0)

# %%
